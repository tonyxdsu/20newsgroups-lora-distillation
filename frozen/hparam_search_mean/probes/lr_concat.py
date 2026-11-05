"""LR (standardized) on several encoders' vectors side by side, 5-fold CV on train only (the same folds as everything
else, or grouped by thread with --group-by-thread). Feature files are names in the Further Wark folder.
    ~/venvs/cs423/bin/python -u lr_concat.py [--C 0.001,0.003,0.01] [--group-by-thread] <a.npz> [<b.npz> ...]"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

parser = argparse.ArgumentParser()
parser.add_argument("--C", default="0.001,0.003,0.01")
parser.add_argument("--group-by-thread", action="store_true")
parser.add_argument("files", nargs="+")
args = parser.parse_args()
FW = Path(__file__).resolve().parents[2]
s = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
y = np.array([t for d, t in zip(s.data, s.target) if d])
X = np.hstack([np.load(FW / f)["X_train"] for f in args.files])
if args.group_by_thread:
    sys.path.insert(0, str(FW / "hparam_search_mean"))
    from search_mean import thread_subjects
    groups = np.unique(thread_subjects(), return_inverse=True)[1]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=423).split(np.zeros(len(y)), y, groups))
else:
    folds = StratifiedKFold(5, shuffle=True, random_state=423)
print(" + ".join(args.files), f"({X.shape[1]}-d, {'thread-grouped' if args.group_by_thread else 'random'} folds)", flush=True)
for C in map(float, args.C.split(",")):
    with threadpool_limits(1):
        accs = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)), X, y,
                               cv=folds, n_jobs=5) * 100
    print(f"  C={C:<6g} CV {accs.mean():.2f}  folds {np.round(accs, 2)}", flush=True)
