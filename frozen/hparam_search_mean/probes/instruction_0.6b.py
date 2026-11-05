"""Does the instruction matter? Qwen3-Embedding-0.6B, 512 tokens, 20NG train only: 5-fold CV of LR per instruction.
    ~/venvs/cs423/bin/python -u instruction_0.6b.py      (writes instruction_0.6b.json; about 10 min on the GPU)"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoTokenizer

s = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
docs = [d for d in s.data if d]
y = np.array([t for d, t in zip(s.data, s.target) if d])
groups = ", ".join(s.target_names)
VARIANTS = {
    "A: Qwen's MTEB 20NG prompt (current)": "Instruct: Identify the topic or theme of the given news articles\nQuery:",
    "B: classification prompt": "Instruct: Classify the given Usenet newsgroup post by the newsgroup it was posted to\nQuery:",
    "C: classification + the 20 newsgroup names": f"Instruct: Classify the given Usenet newsgroup post into one of these newsgroups: {groups}\nQuery:",
    "D: no instruction": "",
}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
qwen = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", torch_dtype=torch.bfloat16).to("cuda").eval()


def embed(texts, max_length=512, budget=16384):
    """Last-token vectors, shortest messages first in batches of up to `budget` tokens (as the 2,048-token notebook)."""
    lengths = [len(ids) for ids in tok(texts, truncation=True, max_length=max_length)["input_ids"]]
    order, vecs, i = np.argsort(lengths, kind="stable"), np.empty((len(texts), 1024), np.float32), 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * lengths[order[j]] <= budget:
            j += 1
        idx = order[i:j]
        b = tok([texts[k] for k in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
        with torch.no_grad():
            vecs[idx] = qwen(**b).last_hidden_state[:, -1].float().cpu().numpy()
        i = j
    return vecs


folds = StratifiedKFold(5, shuffle=True, random_state=423)
results = {}
for name, instr in VARIANTS.items():
    t = time.time()
    X = embed([instr + d for d in docs])
    with threadpool_limits(1):
        cv = {C: cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)), X, y,
                                 cv=folds, n_jobs=5).mean() * 100 for C in (0.001, 0.003, 0.01)}
    best_C = max(cv, key=cv.get)
    results[name] = {"cv": {str(k): round(v, 2) for k, v in cv.items()}, "best": round(cv[best_C], 2),
                     "instruction_tokens": len(tok(instr)["input_ids"]) if instr else 0}
    print(f"{name:45s} CV {cv[best_C]:.2f} (C={best_C})  all: {results[name]['cv']}  "
          f"[{results[name]['instruction_tokens']} instruction tokens, {time.time() - t:.0f} s]", flush=True)
(Path(__file__).parent / "instruction_0.6b.json").write_text(json.dumps(results, indent=1))
