"""Logistic regression on every frozen-transformer feature set: tune C and input scaling by 5-fold CV on train,
then refit and score on test once.

    ~/venvs/cs423/bin/python lr_search.py      (from this folder; skips feature files that don't exist yet)

For each feature file (../<name>_20ng.npz and ../<name>_sports.npz):
  1. Coarse grid: C = 1e-4 ... 1e3 in half-decade steps, raw and standardized inputs (StandardScaler fit on
     each fold's training part). Same folds as the MLP searches (StratifiedKFold(5, shuffle, random_state=423)).
  2. Fine grid: quarter-decade steps around the best coarse C, for the best scaling.
  3. The best (scaling, C) by CV accuracy is refit on 100% of train (the best this LR can do) and on the
     notebooks' 90% split (to compare with the MLPs, which hold out 10% for early stopping). Both are scored on test.
Results: lr_search.json.

    ~/venvs/cs423/bin/python lr_search.py --group-by-thread qwen3_4b_last qwen3_4b_2048_last ...

does the same with folds that keep each thread in one fold (search_mean.py's --group-by-thread), for the named
20NG feature sets only, and writes lr_search_threads.json.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / "lr_search.json"
FEATURES = ["roberta_cls", "roberta_mean", "modernbert_cls", "modernbert_mean", "qwen3_0.6b_last", "qwen3_4b_last",
            "qwen3_4b_2048_last", "qwen3_4b_2048_instrB_last"]
DATASETS = {"20ng": None, "sports": ["rec.sport.baseball", "rec.sport.hockey"]}
COARSE = [10 ** e for e in np.arange(-4, 3.01, 0.5)]
MAX_ITER = 5000


def labels(categories):
    out = {}
    for subset in ("train", "test"):
        s = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"), categories=categories)
        out[subset] = np.array([t for d, t in zip(s.data, s.target) if len(d) > 0])  # same empty-post removal
    return out["train"], out["test"]


def fit(X_tr, y_tr, C, standardize):
    """One LR fit, single-threaded (much faster on this machine). Returns (scaler or None, model)."""
    scaler = StandardScaler().fit(X_tr) if standardize else None
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # n_iter is recorded instead
        clf = LogisticRegression(C=C, max_iter=MAX_ITER).fit(scaler.transform(X_tr) if scaler else X_tr, y_tr)
    return scaler, clf


def cv_one(X, y, tr, dev, C, standardize):
    scaler, clf = fit(X[tr], y[tr], C, standardize)
    X_dev = scaler.transform(X[dev]) if scaler else X[dev]
    return clf.score(X_dev, y[dev]) * 100, int(clf.n_iter_[0])


def cv_grid(X, y, folds, grid):
    """grid: list of (standardize, C). Returns {(standardize, C): (mean acc, accs, n_iters)}."""
    jobs = [(st, C, k) for st, C in grid for k in range(len(folds))]
    res = Parallel(n_jobs=8)(delayed(cv_one)(X, y, *folds[k], C, st) for st, C, k in jobs)
    out = {}
    for (st, C, k), (acc, it) in zip(jobs, res):
        out.setdefault((st, C), ([], []))
        out[(st, C)][0].append(acc)
        out[(st, C)][1].append(it)
    return {g: (float(np.mean(a)), a, its) for g, (a, its) in out.items()}


def test_score(X_tr, y_tr, X_te, y_te, C, standardize):
    scaler, clf = fit(X_tr, y_tr, C, standardize)
    pred = clf.predict(scaler.transform(X_te) if scaler else X_te)
    return {"test_acc": round(metrics.accuracy_score(y_te, pred) * 100, 3),
            "test_macro_f1": round(metrics.f1_score(y_te, pred, average="macro") * 100, 3),
            "n_iter": int(clf.n_iter_[0])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-by-thread", nargs="+", metavar="FEATURES")
    args = parser.parse_args()
    if args.group_by_thread:
        from search_mean import thread_subjects
        OUT, FEATURES, DATASETS = HERE / "lr_search_threads.json", args.group_by_thread, {"20ng": None}
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for data, cats in DATASETS.items():
        y_train, y_test = labels(cats)
        if args.group_by_thread:
            groups = np.unique(thread_subjects(), return_inverse=True)[1]
            folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=423).split(np.zeros(len(y_train)), y_train, groups))
        else:
            folds = list(StratifiedKFold(5, shuffle=True, random_state=423).split(np.zeros(len(y_train)), y_train))
        for feat in FEATURES:
            name, path = f"{feat}_{data}", HERE.parent / f"{feat}_{data}.npz"
            if name in results or not path.exists():
                continue
            start = time.time()
            saved = np.load(path)
            X_train, X_test = saved["X_train"], saved["X_test"]
            grid = cv_grid(X_train, y_train, folds, [(st, C) for st in (False, True) for C in COARSE])
            (st, C0), _ = max(grid.items(), key=lambda kv: kv[1][0])
            fine = [C0 * 10 ** d for d in (-0.25, 0.25)]
            grid.update(cv_grid(X_train, y_train, folds, [(st, C) for C in fine]))
            (st, C), (cv, accs, its) = max(grid.items(), key=lambda kv: kv[1][0])
            X_90, _, y_90, _ = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=423)
            results[name] = {
                "best": {"standardize": st, "C": C, "cv_acc": round(cv, 3), "cv_fold_accs": [round(a, 3) for a in accs],
                         "cv_n_iter": its},
                "test_100pct_train": test_score(X_train, y_train, X_test, y_test, C, st),
                "test_90pct_train": test_score(X_90, y_90, X_test, y_test, C, st),
                "grid": [{"standardize": g[0], "C": g[1], "cv_acc": round(v[0], 3), "max_n_iter": max(v[2])}
                         for g, v in sorted(grid.items())],
                "seconds": round(time.time() - start, 1),
            }
            OUT.write_text(json.dumps(results, indent=1))
            r = results[name]
            print(f"{name:28s} best {'std' if st else 'raw'} C={C:.3g}: CV {cv:.2f} | test (100%) "
                  f"{r['test_100pct_train']['test_acc']:.2f} | test (90%) {r['test_90pct_train']['test_acc']:.2f} "
                  f"| {r['seconds']:.0f} s", flush=True)
