"""Re-embed 20NG with Qwen3-Embedding-4B at 2,048 tokens with instruction B, the best of four on the 0.6B model
(instruction_0.6b.py). Same data, batching and pooling as Frozen Qwen3-Embedding-4B 2048 tokens.ipynb; only the
instruction differs. About 10 minutes on the GPU, peak about 10 GB.
    ~/venvs/cs423/bin/python -u embed_instruction_b.py      (writes ../../qwen3_4b_2048_instrB_last_20ng.npz)"""
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoModel, AutoTokenizer

OUT = Path(__file__).resolve().parents[2] / "qwen3_4b_2048_instrB_last_20ng.npz"
INSTRUCTION = "Instruct: Classify the given Usenet newsgroup post by the newsgroup it was posted to\nQuery:"
MAX_LENGTH, TOKENS_PER_BATCH = 2048, 8192
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B", padding_side="left")
qwen = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-4B", torch_dtype=torch.bfloat16).to("cuda").eval()


def embed(X):
    texts = [INSTRUCTION + x for x in X]
    lengths = [len(ids) for ids in tok(texts, truncation=True, max_length=MAX_LENGTH)["input_ids"]]
    order, vecs, i = np.argsort(lengths, kind="stable"), np.empty((len(X), qwen.config.hidden_size), np.float32), 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * lengths[order[j]] <= TOKENS_PER_BATCH:
            j += 1
        idx = order[i:j]
        batch = tok([texts[k] for k in idx], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
        with torch.no_grad():
            vecs[idx] = qwen(**batch).last_hidden_state[:, -1].float().cpu().numpy()
        i = j
    return vecs


start = time.time()
out = {}
for subset in ("train", "test"):
    s = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"))
    out[f"X_{subset}"] = embed([d for d in s.data if d])
    print(subset, out[f"X_{subset}"].shape, f"{time.time() - start:.0f} s", flush=True)
np.savez(OUT, **out)
print("peak GPU memory", torch.cuda.max_memory_allocated() / 1e9, "GB")
