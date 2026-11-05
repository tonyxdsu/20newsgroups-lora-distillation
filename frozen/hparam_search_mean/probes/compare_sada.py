"""Our final model vs. Sada et al. (2026) on the same 20NG test posts.

Their code (github.com/groundsada/llms-or-naive-bayes, data/prepare.py) loads 20NG exactly as we do:
fetch_20newsgroups(remove=("headers", "footers", "quotes")), default shuffle with seed 42, so their published
per-post predictions line up with our test order. They keep the 162 test posts that end up empty; we drop them.
Their result files (copied from their repo's results/ into ../../../papers/Sada2026_results/): the LoRA Qwen3-8B on all
7,532 test posts, and the zero-/few-shot LLMs, Naive Bayes and LR on their 500-post LLM sample, which is the first 500
posts of the same order.
    ~/venvs/cs423/bin/python -u compare_sada.py"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.datasets import fetch_20newsgroups

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "papers" / "Sada2026_results"
READABLE = {"alt.atheism": "atheism", "comp.graphics": "computer graphics", "comp.os.ms-windows.misc": "Microsoft Windows OS",
            "comp.sys.ibm.pc.hardware": "PC hardware", "comp.sys.mac.hardware": "Mac hardware", "comp.windows.x": "X Window System",
            "misc.forsale": "items for sale", "rec.autos": "automobiles", "rec.motorcycles": "motorcycles",
            "rec.sport.baseball": "baseball", "rec.sport.hockey": "hockey", "sci.crypt": "cryptography",
            "sci.electronics": "electronics", "sci.med": "medicine", "sci.space": "outer space",
            "soc.religion.christian": "Christianity", "talk.politics.guns": "gun politics",
            "talk.politics.mideast": "Middle East politics", "talk.politics.misc": "general politics",
            "talk.religion.misc": "general religion"}  # their label names (data/prepare.py)

s = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
names = np.array([READABLE[n] for n in s.target_names])
y = names[s.target]
nonempty = np.array([len(d) > 0 for d in s.data])
ours = np.full(len(y), "", dtype=object)  # our prediction for every non-empty post ("" = dropped, counts as wrong)
ours[nonempty] = names[np.load(HERE.parent / "final_test_qwen3_4b_2048_instrB_threads_ensemble_pred.npy")]
right_ours = ours == y


def compare(name, right_theirs, posts):
    """Accuracy of theirs and ours on `posts` (a boolean mask over the test set) and McNemar's test."""
    a, b = right_ours[posts], right_theirs[posts]
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    print(f"  {name:32s} {posts.sum():5d} posts: theirs {b.mean() * 100:5.2f}, ours {a.mean() * 100:5.2f} "
          f"({(a.mean() - b.mean()) * 100:+.2f}; {only_a} vs. {only_b} right only in one, McNemar p = {binomtest(only_a, only_a + only_b).pvalue:.2g})")


full = json.loads((RESULTS / "20newsgroups_qwenft_preds.json").read_text())
assert list(y) == full["y_true"], "test order differs"
right_ft = np.array(full["models"]["Qwen/Qwen3-8B-ft-discriminative"]) == y
print(f"LoRA Qwen3-8B on all {len(y):,} posts: {right_ft.mean() * 100:.2f} "
      f"({right_ft[~nonempty].sum()} of the {(~nonempty).sum()} empty posts right)")
print(f"Ours on all {len(y):,} posts, every empty post counted wrong: {right_ours.mean() * 100:.2f}")
compare("LoRA Qwen3-8B (fine-tuned)", right_ft, nonempty)

files = ["20newsgroups_preds.json"] + [f"secondfamily_{d}_20newsgroups_preds.json" for d in ("gpt_oss_", "glm_5_", "kimi")]
done = set()
for file in files:
    sample = json.loads((RESULTS / file).read_text())
    n = sample["n_compared"]
    assert list(y[:n]) == sample["y_true"], "the 500-post sample isn't the first 500 posts"
    in_sample = np.zeros(len(y), bool)
    in_sample[:n] = True
    if not done:
        print(f"Their 500-post LLM sample: the first {n} test posts, {(in_sample & ~nonempty).sum()} of them empty")
    for model, preds in sample["models"].items():
        if model in done:  # Naive Bayes and LR are repeated in every file
            continue
        done.add(model)
        right = np.zeros(len(y), bool)
        right[:n] = np.array(preds) == y[:n]
        print(f"  {model:32s} all {n} posts: {right[:n].mean() * 100:.1f}")
        compare(model, right, in_sample & nonempty)
