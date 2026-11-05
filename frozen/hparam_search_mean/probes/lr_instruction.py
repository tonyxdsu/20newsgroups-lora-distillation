"""Instruction A (the notebooks') vs. B on Qwen3-Embedding-4B at 2,048 tokens: LR, 5-fold CV on train only, same
folds, raw and standardized inputs. Needs ../../qwen3_4b_2048_instrB_last_20ng.npz (embed_instruction_b.py).
    ~/venvs/cs423/bin/python -u lr_instruction.py      (a minute or two on the CPU)"""
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FW = Path(__file__).resolve().parents[2]
s = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
y = np.array([t for d, t in zip(s.data, s.target) if d])
folds = StratifiedKFold(5, shuffle=True, random_state=423)
GRID = [(False, 0.0003), (False, 0.001), (False, 0.003), (True, 0.001), (True, 0.003), (True, 0.01), (True, 0.03)]
per_fold = {}
for name, file in (("A", "qwen3_4b_2048_last_20ng.npz"), ("B", "qwen3_4b_2048_instrB_last_20ng.npz")):
    X = np.load(FW / file)["X_train"]
    for std, C in GRID:
        clf = LogisticRegression(C=C, max_iter=5000)
        with threadpool_limits(1):
            accs = cross_val_score(make_pipeline(StandardScaler(), clf) if std else clf, X, y, cv=folds, n_jobs=10) * 100
        per_fold[(name, std, C)] = accs
        print(f"{name}  {'std' if std else 'raw'}  C={C:<7g} CV {accs.mean():.2f}  folds {np.round(accs, 2)}", flush=True)
best = {n: max((k for k in per_fold if k[0] == n), key=lambda k: per_fold[k].mean()) for n in "AB"}
d = per_fold[best["B"]] - per_fold[best["A"]]
print(f"best A {best['A']} {per_fold[best['A']].mean():.2f} | best B {best['B']} {per_fold[best['B']].mean():.2f} | "
      f"B - A {d.mean():+.2f}, per fold {np.round(d, 2)}, B better on {(d > 0).sum()}/5")
