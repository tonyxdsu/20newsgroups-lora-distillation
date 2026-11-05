"""Hyperparameter search: MLP on frozen *mean-pooled* RoBERTa or ModernBERT embeddings, full 20 Newsgroups.

    ~/venvs/cs423/bin/python -u search_mean.py --encoder roberta --hours 5.25     (from this folder)

Re-running it skips every run already in <encoder>/trials.jsonl (--deadline / --random-until keep the original
schedule after a restart).

The protocol is the same as ../hparam_search/search.py, so scores are comparable: 5-fold cross-validation (CV) on
train with the same folds, early stopping on dev loss (patience 10, at most 300 epochs), each model scored on its
dev fold at its best epoch. The test set is not used here; see final_test.py.

What's new is speed. One small MLP is "launch-bound": the GPU spends ~1 ms per batch waiting for Python to hand it
work, whatever the layer sizes. So this trains many MLPs at once as one "stacked" model: G independent copies
whose weights for each layer sit in one tensor of shape (G, ...), so each layer is one batched matmul
(torch.baddbmm) for all of them. Each copy has its own fold, dropout rates, label smoothing, learning rate and
weight decay. AdamW works element by element, so one AdamW step on the stacked weights is the same as a separate
AdamW step per model. Typically G = 8 configs x 5 folds = 40 models train together, and a model that early-stops
is dropped from the stack. The limit is now GPU memory bandwidth (every step reads and writes all weights).

Checked against search.py on RoBERTa <s> features (roberta_validate/, roberta_compact_validate/): its best
2-layer config scored 72.91-72.98 here in three separate checks vs. 73.08 there, and its best 1-layer config
72.64-72.75 vs. 72.78 (3 seeds each; seed-to-seed spread is about +-0.15). If the ~0.1 gap is real rather than
noise, it shifts every config alike, so it doesn't change which config wins; final_test.py uses search.py's code.

What was learned from the RoBERTa <s> search is built in: inputs are always standardized, GELU, batch size 32,
constant learning rate (those either mattered a lot and were settled, or barely mattered). Ranges are wider where
RoBERTa's winners sat at an edge (dropout after layer 1, second-layer width).

Phases (results are appended to <encoder>/trials.jsonl as they finish):
  1. Random search, 1 run per config. Configs are drawn in groups that share layer sizes and learning rate,
     so a group can train together.
  2. Re-run the 16 best with 2 more seeds; the best 3-seed mean becomes the center.
  3. One change at a time from the center (3 seeds each). If that finds a better config, sweep again from it.

--group-by-thread makes the folds keep each thread (posts with the same subject line once "Re:" is stripped) in one
fold. With random folds, 65% of dev posts have a reply or parent in their fold's training part, but only 32% of test
posts have one in train (the test set is later in time), so random-fold CV rewards remembering threads
(probes/thread_cv.py). The headers are only read to form the groups, never used as features.
"""
import argparse
import json
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
device = torch.device("cuda")

SPACE = {
    "hidden1":         [1024, 2048, 4096],
    "hidden2":         [0, 32, 64, 128, 256],             # 0 = one hidden layer
    "dropout_in":      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],    # dropout on the input embedding
    "dropout1":        [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8],
    "dropout2":        [0.0, 0.2, 0.3, 0.5, 0.6, 0.7],    # only used with 2 hidden layers
    "learning_rate":   [3e-5, 1e-4, 3e-4],
    "weight_decay":    [0.0, 0.01, 0.1],
    "label_smoothing": [0.0, 0.05, 0.1, 0.2],
}
FIXED = {"activation": "gelu", "batch_size": 32, "standardize": True, "lr_schedule": "constant"}
GROUP_KEYS = ("hidden1", "hidden2", "learning_rate")  # configs that share these can train together
# The best config of the RoBERTa <s> search, run first as a reference point
ROBERTA_S_BEST = {"hidden1": 2048, "hidden2": 64, "dropout_in": 0.2, "dropout1": 0.7, "dropout2": 0.5,
                  "learning_rate": 1e-4, "weight_decay": 0.01, "label_smoothing": 0.1}

MAX_EPOCHS, PATIENCE, N_FOLDS, BATCH = 300, 10, 5, 32
# Configs trained together (x 5 folds = models per stacked group). Narrow networks need bigger groups to keep
# the GPU busy; wide ones smaller groups to fit in memory.
CONFIGS_PER_GROUP = {1024: 16, 2048: 8, 4096: 4}
TOP_K = 16


def log(msg):
    print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)


def canonical(cfg):
    """Full config in a fixed form. With one hidden layer, dropout2 doesn't exist, so it's stored as 0."""
    cfg = {**cfg, **FIXED}
    if cfg["hidden2"] == 0:
        cfg["dropout2"] = 0.0
    return {k: cfg[k] for k in sorted(cfg)}


def key(cfg):
    return json.dumps(canonical(cfg), sort_keys=True)


def describe(cfg):
    sizes = f"{cfg['hidden1']}/{cfg['hidden2']}" if cfg["hidden2"] else f"{cfg['hidden1']}"
    return (f"{sizes:>9}  drop {cfg['dropout_in']}/{cfg['dropout1']}/{cfg['dropout2']}  lr {cfg['learning_rate']:g}  "
            f"wd {cfg['weight_decay']:g}  ls {cfg['label_smoothing']:g}")


# ---------------------------------------------------------------------------------------------------------------
# Data: every fold's training part standardized with its own mean/std, all kept on the GPU
# ---------------------------------------------------------------------------------------------------------------
def thread_subjects(subset="train"):
    """The normalized subject line ("Re: " stripped, lowercase) of every non-empty post, in the feature files' order:
    posts with the same one belong to the same thread."""
    raw = fetch_20newsgroups(subset=subset)  # with headers
    stripped = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"))
    subject = lambda post: re.search(r"^Subject:(.*)$", post, re.M).group(1)
    norm = lambda s: re.sub(r"^((re|fwd?)(\^\d+|\[\d+\])?\s*:\s*)+", "", s.strip(), flags=re.I).strip().lower()
    return [norm(subject(post)) for post, doc in zip(raw.data, stripped.data) if len(doc) > 0]


class FoldData:
    def __init__(self, X, y, groups=None):
        """groups: one thread id per row, to keep each thread in one fold (None: plain stratified folds)."""
        if groups is None:
            folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=423).split(np.zeros(len(y)), y))
        else:
            folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=423).split(np.zeros(len(y)), y, groups))
        # X[f] = the whole training set, standardized with fold f's training statistics (dev rows included,
        # so a model on fold f reads its dev rows from the same array)
        self.X = torch.tensor(np.stack([StandardScaler().fit(X[tr]).transform(X).astype(np.float32)
                                        for tr, _ in folds]), device=device)
        self.y = torch.tensor(y, device=device)
        self.tr = [torch.tensor(tr, device=device) for tr, _ in folds]
        # Dev folds differ in size by one row: pad to the same length and mask the padding out
        m = max(len(dev) for _, dev in folds)
        self.dev = torch.stack([torch.tensor(np.pad(dev, (0, m - len(dev))), device=device) for _, dev in folds])
        self.dev_mask = torch.stack([torch.arange(m, device=device) < len(dev) for _, dev in folds]).float()
        self.steps = min(len(tr) for tr, _ in folds) // BATCH  # full batches per epoch (the last partial one is dropped)
        self.D, self.C = X.shape[1], int(y.max()) + 1


# ---------------------------------------------------------------------------------------------------------------
# G independent MLPs in one set of tensors
# ---------------------------------------------------------------------------------------------------------------
class StackedMLP:
    """Each layer's weights for all G models in one tensor: W of shape (G, fan_in, fan_out), b of shape (G, 1, fan_out)."""

    def __init__(self, G, sizes):
        self.G, self.params = G, []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = 1 / math.sqrt(fan_in)  # same initialization as nn.Linear: uniform in +-1/sqrt(fan_in)
            self.params += [torch.empty(G, fan_in, fan_out, device=device).uniform_(-bound, bound).requires_grad_(),
                            torch.empty(G, 1, fan_out, device=device).uniform_(-bound, bound).requires_grad_()]

    def forward(self, x, keep=None, models=slice(None)):
        """x: (g, batch, D) for the models in `models`. keep: per-layer keep probabilities (G, 1, 1) for dropout
        on each layer's input, or None for evaluation (dropout off)."""
        n = len(self.params) // 2
        for i in range(n):
            if keep is not None:
                k = keep[i][models]
                x = x * (torch.rand_like(x) < k) / k  # inverted dropout, a different rate per model
            x = torch.baddbmm(self.params[2 * i + 1][models], x, self.params[2 * i][models])
            if i < n - 1:
                x = F.gelu(x)
        return x

    def select(self, mask):
        """Keep only the models where mask (G,) is True."""
        self.params = [p.detach()[mask].requires_grad_() for p in self.params]
        self.G = len(self.params[0])


@torch.compile(dynamic=True)
def _adamw(ps, gs, ms, vs, step_size, decay, bc2_sqrt):
    """torch.optim.AdamW's update, written out so that lr and weight decay can differ per model (shape (G, 1, 1)).
    Compiled, so each tensor gets one fused GPU kernel instead of ~10 passes over its weights."""
    for p, g, m, v in zip(ps, gs, ms, vs):
        m.lerp_(g, 0.1)                            # m = 0.9 m + 0.1 g
        v.mul_(0.999).addcmul_(g, g, value=0.001)  # v = 0.999 v + 0.001 g^2
        p.mul_(decay)                              # decoupled weight decay: p *= 1 - lr * wd
        p.sub_(step_size * m / (v.sqrt() / bc2_sqrt + 1e-8))


class StackedAdamW:
    def __init__(self, params, lr, wd):  # lr, wd: (G, 1, 1)
        self.params, self.lr, self.wd, self.t = params, lr, wd, 0
        self.m, self.v = [torch.zeros_like(p) for p in params], [torch.zeros_like(p) for p in params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        _adamw(self.params, [p.grad for p in self.params], self.m, self.v, self.lr / (1 - 0.9 ** self.t),
               1 - self.lr * self.wd, torch.tensor(math.sqrt(1 - 0.999 ** self.t), device=device))
        for p in self.params:
            p.grad = None

    def select(self, mask, params):
        """Keep only the models where mask is True (params: the model's already-selected weights)."""
        self.params, self.lr, self.wd = params, self.lr[mask], self.wd[mask]
        self.m, self.v = [m[mask] for m in self.m], [v[mask] for v in self.v]


def smoothed_ce(logits, y, ls):
    """Cross-entropy with label smoothing, one value per model. logits (G, B, C), y (G, B), ls (G, 1).
    Same as nn.CrossEntropyLoss(label_smoothing=ls): (1 - ls) * NLL + ls * mean over classes of -log p."""
    logp = logits.log_softmax(-1)
    nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return ((1 - ls) * nll - ls * logp.mean(-1)).mean(dim=1)


@torch.no_grad()
def evaluate(model, fd, folds, chunk=8):
    """Plain cross-entropy (no smoothing) and accuracy of every model on its own dev fold."""
    losses, accs = [], []
    for a in range(0, model.G, chunk):
        sl = slice(a, a + chunk)
        idx = fd.dev[folds[sl]]                                      # (g, M)
        logits = model.forward(fd.X[folds[sl, None], idx], models=sl)  # (g, M, C)
        mask = fd.dev_mask[folds[sl]]
        ce = F.cross_entropy(logits.transpose(1, 2), fd.y[idx], reduction="none")
        losses.append((ce * mask).sum(1) / mask.sum(1))
        accs.append(((logits.argmax(-1) == fd.y[idx]).float() * mask).sum(1) / mask.sum(1) * 100)
    return torch.cat(losses), torch.cat(accs)


def train_group(runs, fd, seed):
    """Train every (config, fold) in `runs` at once. All configs share hidden1, hidden2 and learning rate.
    A model that early-stops is dropped from the stack (its result is final), so the others train faster.
    Returns, per run, its dev loss / accuracy at its best epoch (lowest dev loss), as in search.py."""
    torch.manual_seed(seed)
    G, c0 = len(runs), runs[0]["cfg"]
    sizes = [fd.D] + [h for h in (c0["hidden1"], c0["hidden2"]) if h] + [fd.C]
    model = StackedMLP(G, sizes)
    col = lambda name: torch.tensor([float(r["cfg"][name]) for r in runs], device=device).view(G, 1)
    keep = [1 - col(name).view(G, 1, 1) for name in ("dropout_in", "dropout1", "dropout2")[:len(sizes) - 1]]
    ls = col("label_smoothing")
    opt = StackedAdamW(model.params, col("learning_rate").view(G, 1, 1), col("weight_decay").view(G, 1, 1))
    folds = torch.tensor([r["fold"] for r in runs], device=device)
    ids = torch.arange(G, device=device)  # which run each row of the stack belongs to

    best_loss = torch.full((G,), float("inf"), device=device)
    best_acc, best_epoch = torch.zeros(G, device=device), torch.zeros(G, device=device)
    for epoch in range(1, MAX_EPOCHS + 1):
        # A fresh shuffle of its own fold's training rows for every model
        order = torch.empty(len(ids), fd.steps * BATCH, dtype=torch.long, device=device)
        for f in range(N_FOLDS):
            members = (folds == f).nonzero().squeeze(1)
            if len(members):
                perm = torch.rand(len(members), len(fd.tr[f]), device=device).argsort(dim=1)
                order[members] = fd.tr[f][perm[:, :fd.steps * BATCH]]
        for s in range(fd.steps):
            idx = order[:, s * BATCH:(s + 1) * BATCH]                       # (g, B)
            logits = model.forward(fd.X[folds[:, None], idx], keep)          # (g, B, C)
            smoothed_ce(logits, fd.y[idx], ls).sum().backward()              # each model gets its own gradient
            opt.step()

        dev_loss, dev_acc = evaluate(model, fd, folds)
        improved = dev_loss < best_loss[ids]
        best_loss[ids[improved]] = dev_loss[improved]
        best_acc[ids[improved]] = dev_acc[improved]
        best_epoch[ids[improved]] = epoch
        running = (epoch - best_epoch[ids]) < PATIENCE  # early stopping, per model
        if not running.all():
            if not running.any():
                break
            ids, folds, ls, keep = ids[running], folds[running], ls[running], [k[running] for k in keep]
            model.select(running)
            opt.select(running, model.params)
    return [{"dev_loss": round(l, 4), "dev_acc": round(a, 4), "epoch": int(e)}
            for l, a, e in zip(best_loss.tolist(), best_acc.tolist(), best_epoch.tolist())], epoch


# ---------------------------------------------------------------------------------------------------------------
# Bookkeeping: every (config, seed) is cross-validated once and saved
# ---------------------------------------------------------------------------------------------------------------
RESULTS = {}  # (config key, seed) -> record


def run_configs(cfgs_seeds, fd, phase, note=""):
    """Cross-validate (config, seed) pairs that haven't run yet, in stacked groups. Returns when all are done
    or the deadline passes (a group that has started always finishes)."""
    todo = [(canonical(c), s) for c, s in cfgs_seeds if (key(c), s) not in RESULTS]
    todo = list({(key(c), s): (c, s) for c, s in todo}.values())   # no duplicates
    by_group = {}
    for c, s in todo:
        by_group.setdefault(tuple(c[k] for k in GROUP_KEYS), []).append((c, s))
    for group_key, items in by_group.items():
        per = CONFIGS_PER_GROUP[group_key[0]]
        for i in range(0, len(items), per):
            if time.time() > DEADLINE:
                log("deadline reached")
                return False
            chunk = items[i:i + per]
            runs = [{"cfg": c, "seed": s, "fold": f} for c, s in chunk for f in range(N_FOLDS)]
            start = time.time()
            results, epochs = train_group(runs, fd, seed=random.randrange(2**31))
            secs = time.time() - start
            for j, (c, s) in enumerate(chunk):
                folds = results[j * N_FOLDS:(j + 1) * N_FOLDS]
                accs = [f["dev_acc"] for f in folds]
                rec = {"phase": phase, "note": note, "seed": s, "cfg": c, "mean_dev_acc": round(float(np.mean(accs)), 3),
                       "std_dev_acc": round(float(np.std(accs)), 3),
                       "mean_dev_loss": round(float(np.mean([f["dev_loss"] for f in folds])), 4),
                       "mean_best_epoch": round(float(np.mean([f["epoch"] for f in folds])), 1),
                       "folds": folds, "group_seconds": round(secs, 1), "group_size": len(runs),
                       "group_epochs": epochs, "finished": datetime.now().isoformat(timespec="seconds")}
                RESULTS[(key(c), s)] = rec
                with open(TRIALS, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                log(f"[{phase:<7}] {rec['mean_dev_acc']:6.2f} +- {rec['std_dev_acc']:4.2f}  ep {rec['mean_best_epoch']:5.1f}  "
                    f"seed {s}  {describe(c)}")
            log(f"          group of {len(runs)} models: {epochs} epochs in {secs:.0f} s")
    return True


def score(cfg, seeds=(0, 1, 2)):
    ks = [(key(cfg), s) for s in seeds]
    return float(np.mean([RESULTS[k]["mean_dev_acc"] for k in ks])) if all(k in RESULTS for k in ks) else None


def best_center():
    cfgs = {k: r["cfg"] for (k, s), r in RESULTS.items()}
    scored = [(score(c), c) for c in cfgs.values() if score(c) is not None]
    return max(scored, key=lambda t: t[0]) if scored else (None, None)


def sweep(center):
    out = []
    for name, values in SPACE.items():
        for v in values:
            c = canonical({**center, name: v})
            if key(c) != key(center) and not (name == "dropout2" and center["hidden2"] == 0):
                out.append((c, f"{name}={v}"))
    return out


def random_group(rng):
    """One group of configs for the random search: shared layer sizes and learning rate, the rest random."""
    shared = {k: rng.choice(SPACE[k]) for k in GROUP_KEYS}
    n = CONFIGS_PER_GROUP[shared["hidden1"]]
    return [canonical({**shared, **{k: rng.choice(v) for k, v in SPACE.items() if k not in GROUP_KEYS}})
            for _ in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True,
                        help="name of the output folder; reads ../<encoder>_mean_20ng.npz unless --features is given")
    parser.add_argument("--features", help="override the feature file (e.g. ../roberta_cls_20ng.npz to validate)")
    parser.add_argument("--hours", type=float, default=3.5)
    parser.add_argument("--random-share", type=float, default=0.45, help="share of the time for random search")
    parser.add_argument("--validate", action="store_true", help="only run RoBERTa <s>'s best 2- and 1-layer configs")
    parser.add_argument("--deadline", help="HH:MM today: overrides --hours (e.g. when restarting a search)")
    parser.add_argument("--random-until", help="HH:MM today: overrides --random-share")
    parser.add_argument("--wider", action="store_true",
                        help="also try learning rate 1e-5 and label smoothing 0.3 (the RoBERTa-mean winners sat at the edges)")
    parser.add_argument("--group-by-thread", action="store_true", help="keep each thread in one fold (see the top)")
    args = parser.parse_args()
    if args.wider:
        SPACE["learning_rate"] = [1e-5] + SPACE["learning_rate"]
        SPACE["label_smoothing"] = SPACE["label_smoothing"] + [0.3]
    at = lambda hhmm: datetime.combine(datetime.now().date(), datetime.strptime(hhmm, "%H:%M").time()).timestamp()

    out_dir = HERE / (args.encoder + ("_validate" if args.validate else ""))
    out_dir.mkdir(exist_ok=True)
    TRIALS = out_dir / "trials.jsonl"
    for line in (TRIALS.read_text().splitlines() if TRIALS.exists() else []):
        r = json.loads(line)
        RESULTS[(key(r["cfg"]), r["seed"])] = r
    START = time.time()
    DEADLINE = at(args.deadline) if args.deadline else START + args.hours * 3600

    features = Path(args.features) if args.features else HERE.parent / f"{args.encoder}_mean_20ng.npz"
    X = np.load(features)["X_train"]
    s = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    y = np.array([label for doc, label in zip(s.data, s.target) if len(doc) > 0])  # same empty-post removal
    assert len(X) == len(y)
    groups = np.unique(thread_subjects(), return_inverse=True)[1] if args.group_by_thread else None
    fd = FoldData(X, y, groups)
    log(f"{features.name}: {X.shape}, {fd.steps} batches per epoch, {'thread-grouped' if args.group_by_thread else 'random'} "
        f"folds. {len(RESULTS)} runs already done.")

    if args.validate:
        best1 = {**ROBERTA_S_BEST, "hidden2": 0}
        run_configs([(c, s) for c in (ROBERTA_S_BEST, best1) for s in (0, 1, 2)], fd, "validate")
        for name, c in (("2 layers (search.py: 73.08)", ROBERTA_S_BEST), ("1 layer (search.py: 72.78)", best1)):
            log(f"{name}: {score(c):.2f}")
        raise SystemExit

    # Phase 1: random search (plus RoBERTa <s>'s best config as a reference), one seed per config
    run_configs([(ROBERTA_S_BEST, s) for s in (0, 1, 2)], fd, "ref", "RoBERTa <s> best")
    rng = random.Random(423)
    random_end = at(args.random_until) if args.random_until else START + args.random_share * (DEADLINE - START)
    while time.time() < random_end:
        run_configs([(c, 0) for c in random_group(rng)], fd, "random")
    log(f"Phase 1 done: {len(RESULTS)} runs")

    # Phase 2: the 16 best one-seed configs get seeds 1 and 2
    one_seed = sorted((r for (k, s), r in RESULTS.items() if s == 0), key=lambda r: -r["mean_dev_acc"])
    run_configs([(r["cfg"], s) for r in one_seed[:TOP_K] for s in (1, 2)], fd, "confirm")
    best, center = best_center()
    if center is None:
        raise SystemExit("deadline reached before any config had 3 seeds")
    log(f"Center: {best:.2f}  {describe(center)}")

    # Phase 3: one change at a time from the center, 3 seeds each; again from a new center if one wins
    for round_ in (1, 2, 3):
        done = run_configs([(c, s) for c, note in sweep(center) for s in (0, 1, 2)], fd, f"sweep{round_}")
        new_best, new_center = best_center()
        log(f"After sweep {round_}: best {new_best:.2f}  {describe(new_center)}")
        if not done or key(new_center) == key(center):
            break
        center = new_center
    log(f"Done after {(time.time() - START) / 3600:.2f} h. {len(RESULTS)} runs in {TRIALS}")
