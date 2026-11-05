"""Final test for the mean-pooling searches: retrain each encoder's best MLP on the notebooks' 90/10 split with
10 seeds and score it on test, exactly as ../hparam_search/search.py's phase 3 did for RoBERTa <s>.

    ~/venvs/cs423/bin/python final_test.py roberta      (and modernbert; then: final_test.py merge)

The MLP training is search.py's own train_model (one model at a time, nn.Sequential, torch.optim.AdamW), not the
stacked trainer the search used, so these numbers come from the same code as hparam_search/final_test.json.
The best config is the one with the highest 3-seed CV mean in <encoder>/trials.jsonl; nothing here looks at
test until the config is fixed.

--group-by-thread holds out 10% of the threads (not of the posts) as the dev set for early stopping, to match a
search that was tuned with thread-grouped folds (search_mean.py --group-by-thread).
"""
import functools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hparam_search"))
import search  # noqa: E402  (the original search: train_model, to_device, load-time constants)
from search_mean import FIXED, key, thread_subjects  # noqa: E402

MAKE_MLP_768 = search.make_mlp

SEEDS = range(10)


def best_config(encoder, depth=None):
    """Highest 3-seed CV mean among configs run with seeds 0, 1 and 2 (optionally only 1- or 2-hidden-layer)."""
    runs = {}
    for line in (HERE / encoder / "trials.jsonl").read_text().splitlines():
        r = json.loads(line)
        runs.setdefault(key(r["cfg"]), {})[r["seed"]] = r
    scored = []
    for seeds in runs.values():
        if all(s in seeds for s in (0, 1, 2)):
            cfg = seeds[0]["cfg"]
            if depth is None or (cfg["hidden2"] > 0) == (depth == 2):
                scored.append((np.mean([seeds[s]["mean_dev_acc"] for s in (0, 1, 2)]), cfg))
    return max(scored, key=lambda t: t[0])


def labels():
    from sklearn.datasets import fetch_20newsgroups
    out = []
    for subset in ("train", "test"):
        s = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"))
        out.append(np.array([t for d, t in zip(s.data, s.target) if len(d) > 0]))
    return out


def run(encoder, seeds=SEEDS, out_path=None, features=None, group_by_thread=False):
    """Final test of one search's best MLP configs. Each search writes its own file (so two can run at once);
    merge() combines them into final_test.json. features: the cached vectors, one file or a list of files to put side
    by side (default ../<encoder>_mean_20ng.npz)."""
    out_path = out_path or HERE / f"final_test_{encoder}.json"
    y_train, y_test = labels()
    files = [features] if isinstance(features, (str, Path)) else features or [HERE.parent / f"{encoder}_mean_20ng.npz"]
    saved = [np.load(f) for f in files]
    X_train, X_test = (np.hstack([s[k] for s in saved]) for k in ("X_train", "X_test"))
    # search.py's make_mlp assumes 768-d inputs; use this feature file's width instead
    search.make_mlp = functools.partial(MAKE_MLP_768, in_dim=X_train.shape[1])
    if group_by_thread:  # one tenth of the threads as the dev set
        groups = np.unique(thread_subjects(), return_inverse=True)[1]
        fit, dev = next(StratifiedGroupKFold(10, shuffle=True, random_state=423).split(X_train, y_train, groups))
        X_fit, X_dev, y_fit, y_dev = X_train[fit], X_train[dev], y_train[fit], y_train[dev]
    else:
        X_fit, X_dev, y_fit, y_dev = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=423)
    split = search.to_device(X_fit, y_fit, X_dev, y_dev, True, X_test, y_test)
    out = {}
    for name, depth in (("MLP, best", None), ("MLP, best 1 hidden layer", 1)):
        cv, cfg = best_config(encoder, depth)
        cfg = {**cfg, **FIXED}
        runs, probs = [], 0
        for seed in seeds:
            model, best, _ = search.train_model(cfg, split, seed=10_000 + seed, keep_best_weights=True)
            model.eval()
            with torch.no_grad():
                p = model(split["X_test"]).softmax(dim=1).cpu().numpy()
            probs, pred = probs + p, p.argmax(axis=1)
            runs.append({"seed": seed, "best_epoch": best["epoch"], "dev_acc": round(best["dev_acc"], 3),
                         "test_acc": round(metrics.accuracy_score(y_test, pred) * 100, 3),
                         "test_macro_f1": round(metrics.f1_score(y_test, pred, average="macro") * 100, 3)})
        accs = [r["test_acc"] for r in runs]
        if depth is None:  # keep the ensemble's test predictions (for a confusion matrix)
            np.save(HERE / f"final_test_{encoder}_ensemble_pred.npy", probs.argmax(axis=1))
        out[name] = {"cfg": cfg, "cv_acc": round(float(cv), 3), "dev_split": "threads" if group_by_thread else "random",
                     "test_acc_mean": round(float(np.mean(accs)), 3),
                     "test_acc_std": round(float(np.std(accs)), 3),
                     "test_macro_f1_mean": round(float(np.mean([r["test_macro_f1"] for r in runs])), 3),
                     # all seeds together: average their predicted probabilities
                     "ensemble_test_acc": round(metrics.accuracy_score(y_test, probs.argmax(axis=1)) * 100, 3),
                     "runs": runs}
        print(f"{encoder:10s} {name:26s} CV {cv:.2f}  test {np.mean(accs):.2f} +- {np.std(accs):.2f}  "
              f"{len(runs)}-seed ensemble {out[name]['ensemble_test_acc']:.2f}", flush=True)
        out_path.write_text(json.dumps(out, indent=1))


def merge():
    out = {f.stem.removeprefix("final_test_"): json.loads(f.read_text()) for f in sorted(HERE.glob("final_test_*.json"))}
    (HERE / "final_test.json").write_text(json.dumps(out, indent=1))
    print("wrote", HERE / "final_test.json")


if __name__ == "__main__":
    # python final_test.py roberta | modernbert | [--group-by-thread] <search folder> <feature file> [<more>] | merge
    args = [a for a in sys.argv[1:] if a != "--group-by-thread"]
    merge() if args[0] == "merge" else run(args[0], features=args[1:] or None,
                                           group_by_thread="--group-by-thread" in sys.argv)
