"""Does random-split CV leak threads? Replies share a subject line ("Re: ..."), and random folds put posts of one
thread in both the training and the dev part, while the test set (later in time) is mostly new threads. Compare LR on
the A and B vectors with random folds vs. folds grouped by thread (search_mean.thread_subjects: the headers are only
read to form the groups, never used as features). CV on train only; the test set is only read for its subject lines.
    ~/venvs/cs423/bin/python -u thread_cv.py      (a few minutes on the CPU)"""
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_mean import thread_subjects  # noqa: E402

FW = Path(__file__).resolve().parents[2]
s = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
y = np.array([t for d, t in zip(s.data, s.target) if d])
subjects = thread_subjects("train")
groups = np.unique(subjects, return_inverse=True)[1]
sizes = np.bincount(groups)
print(f"train: {len(y)} posts in {len(sizes)} threads; {(sizes[groups] > 1).mean() * 100:.0f}% of posts share a thread")

splits = {"random": list(StratifiedKFold(5, shuffle=True, random_state=423).split(np.zeros(len(y)), y)),
          "by thread": list(StratifiedGroupKFold(5, shuffle=True, random_state=423).split(np.zeros(len(y)), y, groups))}
for name, folds in splits.items():
    seen = np.mean([np.isin(groups[dev], groups[tr]).mean() for tr, dev in folds]) * 100
    print(f"{name} folds: {seen:.0f}% of dev posts have a thread in their fold's training part")
print(f"test: {np.isin(thread_subjects('test'), subjects).mean() * 100:.0f}% of test posts have a thread in train")

GRID = [(False, 0.0003), (False, 0.001), (False, 0.003), (True, 0.003), (True, 0.01), (True, 0.03)]
res = {}
for name, file in (("A", "qwen3_4b_2048_last_20ng.npz"), ("B", "qwen3_4b_2048_instrB_last_20ng.npz")):
    X = np.load(FW / file)["X_train"]
    for split, folds in splits.items():
        for standardize, C in GRID:
            clf = LogisticRegression(C=C, max_iter=5000)
            with threadpool_limits(1):
                accs = cross_val_score(make_pipeline(StandardScaler(), clf) if standardize else clf, X, y, cv=folds,
                                       n_jobs=10) * 100
            res[(name, split, standardize, C)] = accs
            print(f"{name} {split:9s} {'std' if standardize else 'raw'} C={C:<6g} CV {accs.mean():.2f}  "
                  f"folds {np.round(accs, 2)}", flush=True)
for split in splits:
    best = {n: max((k for k in res if k[:2] == (n, split)), key=lambda k: res[k].mean()) for n in "AB"}
    d = res[best["B"]] - res[best["A"]]
    print(f"{split} folds: best A {best['A'][2:]} {res[best['A']].mean():.2f}, best B {best['B'][2:]} "
          f"{res[best['B']].mean():.2f}; B - A {d.mean():+.2f}, B better on {(d > 0).sum()}/5")
