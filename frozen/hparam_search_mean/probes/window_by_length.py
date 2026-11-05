"""Did the 2,048-token window help the long posts? Qwen3-Embedding-4B + LR (standardized, C = 0.003), 5-fold CV
predictions on train only, at 512 vs. 2,048 tokens: error rate per length bucket, same folds.
    ~/venvs/cs423/bin/python -u window_by_length.py      (a couple of minutes on the CPU)"""
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

FW = Path(__file__).resolve().parents[2]
INSTR = "Instruct: Identify the topic or theme of the given news articles\nQuery:"
s = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
docs = [d for d in s.data if d]
y = np.array([t for d, t in zip(s.data, s.target) if d])
n = np.array([len(x) for x in AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")([INSTR + d for d in docs])["input_ids"]])
wrong = {}
for name in ("qwen3_4b_last", "qwen3_4b_2048_last"):
    X = np.load(FW / f"{name}_20ng.npz")["X_train"]
    with threadpool_limits(1):
        pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.003, max_iter=3000)), X, y,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=423), n_jobs=5)
    wrong[name] = pred != y
    print(f"{name}: CV accuracy {100 * (1 - wrong[name].mean()):.2f}", flush=True)
a, b = wrong["qwen3_4b_last"], wrong["qwen3_4b_2048_last"]
print(f"{'tokens':>16} {'msgs':>6} {'err 512':>8} {'err 2048':>9} {'fixed':>6} {'broken':>7}")
for lo, hi in [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 10**6), (0, 512), (512, 10**6)]:
    m = (n > lo) & (n <= hi)
    print(f"  ({lo:4d}, {hi:7d}] {m.sum():6d} {100 * a[m].mean():7.1f}% {100 * b[m].mean():8.1f}% "
          f"{(a & ~b & m).sum():6d} {(~a & b & m).sum():7d}")
