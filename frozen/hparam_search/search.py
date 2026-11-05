"""Hyperparameter search: MLP on frozen RoBERTa <s> embeddings, full 20 Newsgroups.

Run from this folder (it reads the cached embeddings in ../roberta_cls_20ng.npz; keep the PC awake):
    nohup ~/venvs/cs423/bin/python -u search.py --hours 7.5 >> search_log.txt 2>&1 &

How models are compared: 5-fold cross-validation on the training set. Each model trains on 4/5 of train,
early-stops on the other 1/5 (lowest dev loss, as in the notebook) and is scored on that 1/5 at its best epoch,
so every training message gets scored once. A single 10% dev set (1,110 messages) is too noisy to rank hundreds
of configs: its accuracy moves by about +-1.4 points from chance alone.
The test set is only used in phase 3, after everything has been chosen.

Phases:
  0. Logistic regression baseline (sklearn): a C sweep, with and without standardized inputs.
  1. Random search over 2-hidden-layer MLPs, one seed each. It stops early enough to leave time for phase 2.
  2. Re-run the top 20 with two more seeds and take the best as the center. Then, all with 3 seeds:
     one-change-at-a-time sweeps from the center, dropout placement, a hidden-size grid, a depth ablation
     (0/1/2 hidden layers), and a second sweep round if the center has changed.
  3. Final test: retrain the chosen models on the notebook's 90/10 split with 10 seeds each and score them on test.

Results: lr_baselines.jsonl and trials.jsonl (one JSON object per line, appended as each one finishes) and
final_test.json. Re-running the script skips everything already in the files.
"""
import argparse
import copy
import json
import random
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "roberta_cls_20ng.npz"
device = torch.device("cuda")

# ---------------------------------------------------------------------------------------------------------------
# What gets searched
# ---------------------------------------------------------------------------------------------------------------
SEARCH_SPACE = {
    "hidden1":         [256, 512, 1024, 2048, 4096],    # first hidden layer width
    "hidden2":         [64, 128, 256, 512, 1024],       # second hidden layer width
    "dropout_in":      [0.0, 0.1, 0.2, 0.3, 0.5],       # dropout on the 768-d input embedding
    "dropout1":        [0.0, 0.1, 0.2, 0.3, 0.5, 0.7],  # dropout after hidden layer 1
    "dropout2":        [0.0, 0.1, 0.2, 0.3, 0.5],       # dropout after hidden layer 2
    "activation":      ["relu", "gelu", "tanh"],
    "weight_decay":    [0.0, 0.01, 0.1, 0.3, 1.0],      # AdamW's pull of the weights toward 0
    "batch_size":      [32, 64, 128, 256],
    "standardize":     [False, True],   # rescale each input dimension to mean 0, std 1 (train-fold statistics)
    "label_smoothing": [0.0, 0.1],      # train toward 0.9 on the right class instead of 1.0
    "lr_schedule":     ["constant", "plateau"],  # plateau: halve the learning rate when dev loss stalls
}
# Standardized inputs vary about 40x more than raw ones (median per-dimension std 0.025 -> 1), so the same
# learning rate is a much bigger step for them. Measured: standardized at 1e-3 is at its best after 1 epoch,
# while raw at 1e-4 still hadn't converged after 300. So each gets its own range.
LEARNING_RATES = {False: [3e-4, 1e-3, 3e-3, 1e-2], True: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]}

# The first run in Frozen Roberta MLP.ipynb, so the search has it as a reference point
NOTEBOOK_CFG = {"hidden1": 1024, "hidden2": 256, "dropout_in": 0.0, "dropout1": 0.2, "dropout2": 0.0,
                "activation": "relu", "learning_rate": 1e-3, "weight_decay": 0.01, "batch_size": 64,
                "standardize": False, "label_smoothing": 0.0, "lr_schedule": "constant"}

MAX_EPOCHS = 300
PATIENCE = 10          # early stopping: stop after 10 epochs without a new lowest dev loss
PLATEAU_PATIENCE = 3   # "plateau" schedule: halve the learning rate after 3 epochs without a new lowest dev loss
N_FOLDS = 5
SEEDS = (0, 1, 2)        # phase 2 scores each config by its mean over these seeds
TOP_K = 20               # phase 2 re-runs the 20 best random-search configs
FINAL_SEEDS = range(10)
LR_CS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]  # sklearn's C = 1 / regularization
LR_MAX_ITER = 5000       # A2 used 1000; more so that large C values converge

# Several training processes share the GPU: each small model spends most of its time launching GPU work from
# Python, not computing, so 4 processes gave about 3x the throughput of one (measured; 6 was no faster than 4).
# 3 here, because each process holds ~1.8 GB of GPU memory and other programs on this GPU need room too.
N_WORKERS = 3
PHASE2_JOBS = 450        # roughly how many (config, seed) runs phase 2 needs, for the time estimate

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def log(msg):
    print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)


def n_hidden(cfg):
    return (cfg["hidden1"] > 0) + (cfg["hidden2"] > 0)


def describe(cfg):
    sizes = {0: "linear", 1: f"{cfg['hidden1']}", 2: f"{cfg['hidden1']}/{cfg['hidden2']}"}[n_hidden(cfg)]
    return (f"{sizes:>9}  drop {cfg['dropout_in']}/{cfg['dropout1']}/{cfg['dropout2']}  {cfg['activation']:<4}  "
            f"lr {cfg['learning_rate']:g}  wd {cfg['weight_decay']:g}  bs {cfg['batch_size']}  "
            f"{'std' if cfg['standardize'] else 'raw'}  ls {cfg['label_smoothing']:g}  {cfg['lr_schedule']}")


def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------
def load_data():
    saved = np.load(CACHE)
    X_train, X_test = saved["X_train"], saved["X_test"]
    # Labels: the same loading and empty-message removal as the notebook
    y = {}
    for subset in ("train", "test"):
        s = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
        y[subset] = np.array([label for doc, label in zip(s.data, s.target) if len(doc) > 0])
    assert len(X_train) == len(y["train"]) and len(X_test) == len(y["test"])
    return X_train, y["train"], X_test, y["test"]


def make_folds(y_train):
    """The same 5 stratified folds in every process and every run."""
    return list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=423).split(np.zeros(len(y_train)), y_train))


def to_device(X_tr, y_tr, X_dev, y_dev, standardize, X_test=None, y_test=None):
    """GPU tensors for one train/dev(/test) split. Standardizing uses the mean/std of the training part only."""
    if standardize:
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_dev = scaler.transform(X_tr), scaler.transform(X_dev)
        X_test = None if X_test is None else scaler.transform(X_test)
    split = {"X_tr": X_tr, "y_tr": y_tr, "X_dev": X_dev, "y_dev": y_dev, "X_test": X_test, "y_test": y_test}
    return {name: torch.tensor(a, device=device, dtype=torch.float32 if name.startswith("X") else torch.long)
            for name, a in split.items() if a is not None}


# ---------------------------------------------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------------------------------------------
def make_mlp(cfg, in_dim=768, n_classes=20):
    """Input dropout, then up to two hidden layers (a width of 0 leaves the layer out), then one logit per class."""
    act = ACTIVATIONS[cfg["activation"]]
    layers, width = [nn.Dropout(cfg["dropout_in"])], in_dim
    for size, p in [(cfg["hidden1"], cfg["dropout1"]), (cfg["hidden2"], cfg["dropout2"])]:
        if size:
            layers += [nn.Linear(width, size), act(), nn.Dropout(p)]
            width = size
    layers.append(nn.Linear(width, n_classes))  # no softmax: cross_entropy applies it
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, X, y):
    """(cross-entropy loss, accuracy in %) on a whole split, with dropout off."""
    model.eval()
    logits = model(X)
    return nn.functional.cross_entropy(logits, y).item(), (logits.argmax(dim=1) == y).float().mean().item() * 100


def train_model(cfg, split, seed, keep_best_weights=False):
    """The notebook's training loop with early stopping on dev loss, plus the searched options.
    Returns the model, its metrics at the best epoch, and per-epoch curves."""
    torch.manual_seed(seed)
    model = make_mlp(cfg).to(device)
    # fused=True: the same AdamW update done in one GPU kernel, which is faster for small models
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"], fused=True)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=PLATEAU_PATIENCE)
                 if cfg["lr_schedule"] == "plateau" else None)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    X_tr, y_tr = split["X_tr"], split["y_tr"]
    n, batch_size = len(X_tr), cfg["batch_size"]
    curve = {"train_loss": [], "dev_loss": [], "dev_acc": []}
    best = {"dev_loss": float("inf"), "epoch": 0}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()  # dropout on
        # The data already sits on the GPU, so a shuffled index list replaces the notebook's DataLoader
        order = torch.randperm(n, device=device)
        total_loss = torch.zeros((), device=device)  # summed on the GPU, so there's no CPU sync per batch
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(idx)

        dev_loss, dev_acc = evaluate(model, split["X_dev"], split["y_dev"])
        for name, value in [("train_loss", total_loss.item() / n), ("dev_loss", dev_loss), ("dev_acc", dev_acc)]:
            curve[name].append(value)
        if scheduler is not None:
            scheduler.step(dev_loss)

        if dev_loss < best["dev_loss"]:
            _, train_acc = evaluate(model, X_tr, y_tr)  # to see overfitting: train vs. dev accuracy
            best = {"dev_loss": dev_loss, "dev_acc": dev_acc, "train_acc": train_acc, "epoch": epoch}
            if keep_best_weights:
                best_state = copy.deepcopy(model.state_dict())
        elif epoch - best["epoch"] >= PATIENCE:
            break

    best["epochs_run"] = epoch
    if keep_best_weights:
        model.load_state_dict(best_state)
    return model, best, curve


# ---------------------------------------------------------------------------------------------------------------
# Worker processes: each one keeps the 5 folds on the GPU and cross-validates the configs it's sent
# ---------------------------------------------------------------------------------------------------------------
def init_worker(max_epochs):
    global MAX_EPOCHS, FOLD_SPLITS
    MAX_EPOCHS = max_epochs
    torch.set_num_threads(1)
    X_train, y_train, _, _ = load_data()
    FOLD_SPLITS = [{std: to_device(X_train[tr], y_train[tr], X_train[dev], y_train[dev], std) for std in (False, True)}
                   for tr, dev in make_folds(y_train)]


def run_cv(cfg, seed):
    """Train one model per fold. Returns the per-fold results and their mean."""
    start, folds = time.time(), []
    for k, fold_splits in enumerate(FOLD_SPLITS):
        model, best, curve = train_model(cfg, fold_splits[cfg["standardize"]], seed=seed * 1000 + k)
        folds.append({name: round(value, 4) for name, value in best.items()})
        if k == 0:
            curve_fold0 = {name: [round(v, 4) for v in values] for name, values in curve.items()}
            n_params = sum(p.numel() for p in model.parameters())
    torch.cuda.empty_cache()  # hand cached GPU memory back between trials, so other programs on the GPU have room
    accs = [f["dev_acc"] for f in folds]
    return {"mean_dev_acc": round(float(np.mean(accs)), 3), "std_dev_acc": round(float(np.std(accs)), 3),
            "mean_dev_loss": round(float(np.mean([f["dev_loss"] for f in folds])), 4),
            "mean_train_acc": round(float(np.mean([f["train_acc"] for f in folds])), 3),
            "mean_best_epoch": round(float(np.mean([f["epoch"] for f in folds])), 1),
            "n_params": n_params, "seconds": round(time.time() - start, 1), "folds": folds,
            "curve_fold0": curve_fold0}


# ---------------------------------------------------------------------------------------------------------------
# Bookkeeping (main process): every (config, seed) is cross-validated once and saved; later phases reuse it
# ---------------------------------------------------------------------------------------------------------------
RESULTS = {}           # (config as JSON, seed) -> record
BEST_ONE_SEED = None   # best seed-0 two-layer record so far


def key(cfg, seed):
    return json.dumps(cfg, sort_keys=True), seed


def record_trial(cfg, seed, phase, note, result):
    global BEST_ONE_SEED
    record = {"phase": phase, "note": note, "seed": seed, "cfg": cfg, **result,
              "finished": datetime.now().isoformat(timespec="seconds")}
    append_jsonl(TRIALS_FILE, record)
    RESULTS[key(cfg, seed)] = record
    if seed == 0 and n_hidden(cfg) == 2 and (BEST_ONE_SEED is None
                                             or record["mean_dev_acc"] > BEST_ONE_SEED["mean_dev_acc"]):
        BEST_ONE_SEED = record
    log(f"[{phase:<9}] {record['mean_dev_acc']:6.2f} +- {record['std_dev_acc']:4.2f}  "
        f"ep {record['mean_best_epoch']:5.1f}  {record['seconds']:5.1f}s  seed {seed}  {describe(cfg)}"
        + (f"  ({note})" if note else ""))


def run_jobs(jobs, stop):
    """Cross-validate (cfg, seed, phase, note) jobs that aren't done yet, N_WORKERS at a time, until stop() is
    true. Jobs already running when it stops are allowed to finish. Returns True if every job got done."""
    jobs, pending, submitted, exhausted = iter(jobs), {}, set(), False
    while True:
        while not exhausted and len(pending) < 2 * N_WORKERS and not stop():
            job = next(jobs, None)
            if job is None:
                exhausted = True
            elif key(job[0], job[1]) not in RESULTS and key(job[0], job[1]) not in submitted:
                submitted.add(key(job[0], job[1]))
                pending[POOL.submit(run_cv, job[0], job[1])] = job
        if not pending:
            return exhausted
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            job = pending.pop(future)
            try:
                record_trial(*job, future.result())
            except BrokenProcessPool:
                raise
            except Exception as e:  # e.g. one config running out of GPU memory: log it and keep going
                log(f"FAILED seed {job[1]}  {describe(job[0])}: {e!r}")


def score(cfg):
    """Mean dev accuracy over SEEDS, or None if some seed hasn't run."""
    keys = [key(cfg, s) for s in SEEDS]
    return float(np.mean([RESULTS[k]["mean_dev_acc"] for k in keys])) if all(k in RESULTS for k in keys) else None


def best_cfg(depth):
    """The best config with this many hidden layers (0, 1 or 2) among those run with every seed."""
    cfgs = {json.dumps(r["cfg"], sort_keys=True): r["cfg"] for r in RESULTS.values() if n_hidden(r["cfg"]) == depth}
    scored = [(score(c), c) for c in cfgs.values() if score(c) is not None]
    return max(scored, key=lambda sc: sc[0])[1] if scored else None


# ---------------------------------------------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------------------------------------------
def phase0_logistic_regression():
    done = {(r["C"], r["standardize"]) for r in load_jsonl(LR_FILE)}
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # n_iter is recorded instead
    for standardize in (False, True):
        for C in LR_CS:
            if (C, standardize) in done:
                continue
            start, accs, n_iters = time.time(), [], []
            for tr_idx, dev_idx in FOLDS:
                X_tr, X_dev = X_train[tr_idx], X_train[dev_idx]
                if standardize:
                    scaler = StandardScaler().fit(X_tr)
                    X_tr, X_dev = scaler.transform(X_tr), scaler.transform(X_dev)
                # 1 thread: on this machine, multithreaded LR is ~60x slower (1.8 s vs 110 s for one fit)
                with threadpool_limits(1):
                    clf = LogisticRegression(C=C, max_iter=LR_MAX_ITER).fit(X_tr, y_train[tr_idx])
                accs.append(clf.score(X_dev, y_train[dev_idx]) * 100)
                n_iters.append(int(clf.n_iter_[0]))
            record = {"C": C, "standardize": standardize, "mean_dev_acc": round(float(np.mean(accs)), 3),
                      "std_dev_acc": round(float(np.std(accs)), 3), "fold_dev_acc": [round(a, 3) for a in accs],
                      "n_iter": n_iters, "converged": max(n_iters) < LR_MAX_ITER,
                      "seconds": round(time.time() - start, 1)}
            append_jsonl(LR_FILE, record)
            log(f"[LR       ] {record['mean_dev_acc']:6.2f} +- {record['std_dev_acc']:4.2f}  C {C:g}  "
                f"{'std' if standardize else 'raw'}  {record['seconds']}s"
                + ("" if record["converged"] else "  (hit max_iter)"))


def random_cfg(rng):
    cfg = {name: rng.choice(values) for name, values in SEARCH_SPACE.items()}
    cfg["learning_rate"] = rng.choice(LEARNING_RATES[cfg["standardize"]])
    return cfg


def phase1_random_search():
    def jobs():
        rng = random.Random(423)  # the same sequence every run, so a relaunch continues where it stopped
        yield NOTEBOOK_CFG, 0, "random", "notebook config"
        while True:
            yield random_cfg(rng), 0, "random", ""

    last_report = [len(RESULTS)]

    def stop():
        """Stop when the time left only covers phase 2 (estimated from how long the best config takes) and 3."""
        estimate = PHASE2_JOBS * BEST_ONE_SEED["seconds"] / N_WORKERS * 1.5 if BEST_ONE_SEED else 0
        estimate = min(max(estimate, args.min_phase2_hours * 3600), args.max_phase2_hours * 3600)
        if len(RESULTS) >= last_report[0] + 100 and BEST_ONE_SEED:
            last_report[0] = len(RESULTS)
            log(f"--- {len(RESULTS)} trials. Best so far {BEST_ONE_SEED['mean_dev_acc']:.2f}: "
                f"{describe(BEST_ONE_SEED['cfg'])}. Phase 2 starts around "
                f"{datetime.fromtimestamp(PHASE2_END - estimate):%H:%M}")
        return time.time() > PHASE2_END - estimate

    run_jobs(jobs(), stop)


def sweep_configs(center):
    """Every one-hyperparameter change from center. Switching standardize also tries every learning rate in the
    other range, so raw vs. standardized isn't decided by a learning rate that only suits one of them."""
    cfgs = [({**center, "learning_rate": lr}, f"learning_rate={lr:g}") for lr in LEARNING_RATES[center["standardize"]]]
    for name, values in SEARCH_SPACE.items():
        for value in values:
            if name == "standardize" and value != center["standardize"]:
                cfgs += [({**center, "standardize": value, "learning_rate": lr}, f"standardize={value}, lr={lr:g}")
                         for lr in LEARNING_RATES[value]]
            else:
                cfgs.append(({**center, name: value}, f"{name}={value}"))
    return cfgs


def seeded(cfgs_and_notes, phase):
    return [(cfg, seed, phase, note) for cfg, note in cfgs_and_notes for seed in SEEDS]


def phase2_refine():
    stop = lambda: time.time() > PHASE2_END

    # 2a. Re-run the 20 best one-seed configs (and the notebook's config) with seeds 1 and 2
    one_seed = sorted((r for r in RESULTS.values() if r["seed"] == 0 and n_hidden(r["cfg"]) == 2),
                      key=lambda r: -r["mean_dev_acc"])
    top = [(r["cfg"], "top random-search config") for r in one_seed[:TOP_K]] + [(NOTEBOOK_CFG, "notebook config")]
    if not run_jobs(seeded(top, "confirm"), stop):
        return

    # 2b. One change at a time from the best
    center = best_cfg(2)
    log(f"Center for the sweeps: {score(center):.2f}  {describe(center)}")
    if not run_jobs(seeded(sweep_configs(center), "sweep1"), stop):
        return

    # 2c. Dropout placement: the same rate at every combination of the three places
    center = best_cfg(2)
    log(f"Center for placement, sizes and depth: {score(center):.2f}  {describe(center)}")
    places = {"in": "dropout_in", "1": "dropout1", "2": "dropout2"}
    combos = [(), ("in",), ("1",), ("2",), ("in", "1"), ("in", "2"), ("1", "2"), ("in", "1", "2")]
    placement = [({**center, **{field: (rate if place in combo else 0.0) for place, field in places.items()}},
                  f"{'+'.join(combo) or 'none'} @ {rate}") for rate in (0.1, 0.3, 0.5) for combo in combos]
    if not run_jobs(seeded(placement, "placement"), stop):
        return

    # 2d. Hidden-size grid: every hidden1 x hidden2
    sizes = [({**center, "hidden1": h1, "hidden2": h2}, f"{h1}/{h2}")
             for h1 in SEARCH_SPACE["hidden1"] for h2 in SEARCH_SPACE["hidden2"]]
    if not run_jobs(seeded(sizes, "sizes"), stop):
        return

    # 2e. Depth: 0 hidden layers (a linear softmax classifier, like LR, at each learning rate) and 1 hidden layer
    linear = {**center, "hidden1": 0, "hidden2": 0, "dropout1": 0.0, "dropout2": 0.0}
    depth = [({**linear, "learning_rate": lr}, f"0 hidden layers, lr {lr:g}") for lr in LEARNING_RATES[center["standardize"]]]
    depth += [({**center, "hidden1": h1, "hidden2": 0, "dropout2": 0.0}, f"1 hidden layer, {h1}")
              for h1 in SEARCH_SPACE["hidden1"]]
    if not run_jobs(seeded(depth, "depth"), stop):
        return

    # 2f. A second sweep round if the best config has changed since the first
    new_center = best_cfg(2)
    if new_center != center:
        log(f"New center: {score(new_center):.2f}  {describe(new_center)}")
        run_jobs(seeded(sweep_configs(new_center), "sweep2"), stop)


def phase3_final_test():
    """Train the chosen models on the notebook's 90/10 split (10 seeds each) and score them on test, once."""
    chosen = {"MLP, notebook config": NOTEBOOK_CFG, "MLP, best 2 hidden layers": best_cfg(2),
              "MLP, best 1 hidden layer": best_cfg(1), "Linear, 0 hidden layers": best_cfg(0)}
    X_fit, X_dev, y_fit, y_dev = FINAL_NUMPY
    # Created only now, so the main process holds no GPU memory while the workers run
    final_splits = {std: to_device(X_fit, y_fit, X_dev, y_dev, std, X_test, y_test) for std in (False, True)}
    out = {"models": {}, "sklearn_lr": {}}
    for name, cfg in chosen.items():
        if cfg is None:
            continue
        split, runs = final_splits[cfg["standardize"]], []
        for seed in FINAL_SEEDS:
            model, best, _ = train_model(cfg, split, seed=10_000 + seed, keep_best_weights=True)
            model.eval()
            with torch.no_grad():
                pred = model(split["X_test"]).argmax(dim=1).cpu().numpy()
            runs.append({"seed": seed, "best_epoch": best["epoch"], "dev_acc": round(best["dev_acc"], 3),
                         "test_acc": round(metrics.accuracy_score(y_test, pred) * 100, 3),
                         "test_macro_f1": round(metrics.f1_score(y_test, pred, average="macro") * 100, 3)})
        test_accs = [r["test_acc"] for r in runs]
        out["models"][name] = {"cfg": cfg, "cv_dev_acc": score(cfg),
                               "test_acc_mean": round(float(np.mean(test_accs)), 3),
                               "test_acc_std": round(float(np.std(test_accs)), 3), "runs": runs}
        log(f"[final    ] test {np.mean(test_accs):6.2f} +- {np.std(test_accs):4.2f}  {name}: {describe(cfg)}")

    # sklearn LR on the same 90%: the notebook's setup (C=1, raw, max_iter=1000) and the best from phase 0
    lr_records = load_jsonl(LR_FILE)
    variants = {"LR, notebook setup (C=1, raw)": (1.0, False, 1000)}
    if lr_records:
        best_lr = max(lr_records, key=lambda r: r["mean_dev_acc"])
        variants["LR, best C from phase 0"] = (best_lr["C"], best_lr["standardize"], LR_MAX_ITER)
    for name, (C, standardize, max_iter) in variants.items():
        X_a, X_t = X_fit, X_test
        if standardize:
            scaler = StandardScaler().fit(X_fit)
            X_a, X_t = scaler.transform(X_fit), scaler.transform(X_test)
        with threadpool_limits(1):
            pred = LogisticRegression(C=C, max_iter=max_iter).fit(X_a, y_fit).predict(X_t)
        cv = next((r["mean_dev_acc"] for r in lr_records if (r["C"], r["standardize"]) == (C, standardize)), None)
        out["sklearn_lr"][name] = {"C": C, "standardize": standardize, "cv_dev_acc": cv,
                                   "test_acc": round(metrics.accuracy_score(y_test, pred) * 100, 3),
                                   "test_macro_f1": round(metrics.f1_score(y_test, pred, average="macro") * 100, 3)}
        log(f"[final    ] test {out['sklearn_lr'][name]['test_acc']:6.2f}          {name}: C {C:g}, "
            f"{'std' if standardize else 'raw'}")

    out["finished"] = datetime.now().isoformat(timespec="seconds")
    json.dump(out, open(FINAL_FILE, "w"), indent=1)


# ---------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=7.5, help="total time budget, from now")
    parser.add_argument("--min-phase2-hours", type=float, default=1.0)
    parser.add_argument("--max-phase2-hours", type=float, default=3.5)
    parser.add_argument("--final-reserve-hours", type=float, default=0.25, help="time kept back for phase 3")
    parser.add_argument("--out", default=str(HERE), help="folder for the result files")
    parser.add_argument("--quick", action="store_true", help="smoke test: 2 epochs, fewer seeds, 1 LR value")
    args = parser.parse_args()
    if args.quick:
        MAX_EPOCHS, SEEDS, TOP_K, FINAL_SEEDS, LR_CS = 2, (0, 1), 3, range(2), [1.0]

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    TRIALS_FILE, LR_FILE, FINAL_FILE = out_dir / "trials.jsonl", out_dir / "lr_baselines.jsonl", out_dir / "final_test.json"
    START = time.time()
    DEADLINE = START + args.hours * 3600
    PHASE2_END = DEADLINE - args.final_reserve_hours * 3600

    X_train, y_train, X_test, y_test = load_data()
    FOLDS = make_folds(y_train)
    # The notebook's split (90% train / 10% dev, random_state=423), for phase 3
    FINAL_NUMPY = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=423)

    for record in load_jsonl(TRIALS_FILE):
        RESULTS[key(record["cfg"], record["seed"])] = record
        if record["seed"] == 0 and n_hidden(record["cfg"]) == 2 and (
                BEST_ONE_SEED is None or record["mean_dev_acc"] > BEST_ONE_SEED["mean_dev_acc"]):
            BEST_ONE_SEED = record
    log(f"Start: {len(RESULTS)} trials already done. Deadline {datetime.fromtimestamp(DEADLINE):%H:%M}.")

    phase0_logistic_regression()
    POOL = ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn"), initializer=init_worker,
                               initargs=(MAX_EPOCHS,))
    log("Phase 1: random search")
    phase1_random_search()
    log(f"Phase 2: refinement ({len(RESULTS)} trials so far)")
    phase2_refine()
    POOL.shutdown()
    log("Phase 3: final test")
    phase3_final_test()
    log(f"Done after {(time.time() - START) / 3600:.2f} h. {len(RESULTS)} trials in {TRIALS_FILE.name}.")
