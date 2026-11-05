"""McNemar's exact test on the test set for the main comparisons in Qwen4B2048Search.md: a binomial test on the
messages that exactly one of the two models gets right. It only re-scores models that were already tested (same
settings, refit on all of train); it doesn't choose anything.
    ~/venvs/cs423/bin/python -u significance.py      (a minute on the CPU)"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from lr_search import fit, labels  # noqa: E402

y_train, y_test = labels(None)


def lr_pred(features, results_file):
    """Test predictions of the tuned LR in results_file (lr_search.json or lr_search_threads.json)."""
    best = json.loads((HERE / results_file).read_text())[f"{features}_20ng"]["best"]
    saved = np.load(HERE.parent / f"{features}_20ng.npz")
    scaler, clf = fit(saved["X_train"], y_train, best["C"], best["standardize"])
    return clf.predict(scaler.transform(saved["X_test"]) if scaler else saved["X_test"])


preds = {
    "512, A, LR (random-fold tuning)": lr_pred("qwen3_4b_last", "lr_search.json"),
    "512, A, LR (grouped tuning)": lr_pred("qwen3_4b_last", "lr_search_threads.json"),
    "2048, A, LR": lr_pred("qwen3_4b_2048_last", "lr_search.json"),
    "2048, B, LR (random-fold tuning)": lr_pred("qwen3_4b_2048_instrB_last", "lr_search.json"),
    "2048, B, LR (grouped tuning)": lr_pred("qwen3_4b_2048_instrB_last", "lr_search_threads.json"),
    "2048, A, MLP ensemble": np.load(HERE / "final_test_qwen3_4b_2048_ensemble_pred.npy"),
}
final = HERE / "final_test_qwen3_4b_2048_instrB_threads_ensemble_pred.npy"
if final.exists():
    preds["2048, B, MLP ensemble"] = np.load(final)
for name, p in preds.items():
    print(f"{name:34s} test {np.mean(p == y_test) * 100:.2f}")

PAIRS = [("512, A, LR (grouped tuning)", "512, A, LR (random-fold tuning)"),
         ("2048, B, LR (grouped tuning)", "2048, B, LR (random-fold tuning)"),
         ("2048, A, LR", "512, A, LR (random-fold tuning)"),
         ("2048, B, LR (grouped tuning)", "2048, A, LR"),
         ("2048, A, MLP ensemble", "2048, A, LR"),
         ("2048, B, MLP ensemble", "2048, B, LR (grouped tuning)"),
         ("2048, B, MLP ensemble", "2048, A, MLP ensemble"),
         ("2048, B, MLP ensemble", "2048, A, LR")]
print()
for a, b in PAIRS:
    if a in preds and b in preds:
        right_a, right_b = preds[a] == y_test, preds[b] == y_test
        only_a, only_b = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
        p = binomtest(only_a, only_a + only_b).pvalue
        print(f"{a} vs. {b}: {(right_a.mean() - right_b.mean()) * 100:+.2f} points "
              f"({only_a} vs. {only_b} messages right only in one), p = {p:.2g}")
