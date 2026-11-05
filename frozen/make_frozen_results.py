"""Draw figures/frozen_results.png for FrozenResults.md: the best test accuracy of each model so far.

    ~/venvs/cs423/bin/python make_frozen_results.py      (from this folder)

Same style as hparam_search/make_figures.py. Read from result files: LR on frozen features (hparam_search_mean/
lr_search.json: C and scaling tuned by 5-fold CV, refit on 100% of train) and the 10-seed MLP final tests
(hparam_search/final_test.json, hparam_search_mean/final_test.json). The other numbers are copied from the
notebook named next to each one.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent

# Colors from a validated palette, light chart surface (as in hparam_search/make_figures.py)
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # MLPs, linear classifiers
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlesize": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
})

lr = json.loads((HERE / "hparam_search_mean" / "lr_search.json").read_text())
final_s = json.loads((HERE / "hparam_search" / "final_test.json").read_text())
final_mean = json.loads((HERE / "hparam_search_mean" / "final_test.json").read_text())
lr_threads = json.loads((HERE / "hparam_search_mean" / "lr_search_threads.json").read_text())


def lr_test(features):
    """Tuned LR refit on 100% of train: (20NG, sports) test accuracy, None where it wasn't run."""
    keys = (f"{features}_{data}" for data in ("20ng", "sports"))
    return tuple(lr[k]["test_100pct_train"]["test_acc"] if k in lr else None for k in keys)


# (label, is MLP, settings tuned for these features?, 20NG test acc, baseball-vs-hockey test acc).
# Groups are separated by None.
ROWS = [
    # Assignment models, default settings, trained on all of train.
    # 20NG: PLAYGROUND notebook. Sports: Assignment 2 notebook. BoW + LR is the better of unigram and
    # unigram + bigram features (bigram on 20NG: 64.45 vs. 63.74; unigram on sports: 89.06 vs. 87.77)
    ("BoW counts + LR", False, True, 64.45, 89.06),
    ("BoW counts + Naive Bayes", False, True, 64.08, 91.12),
    ("CBoW (average word vector) + LR", False, True, 64.55, 91.51),
    None,
    # MLPs train on 90% of train (10% is the dev set for early stopping), with the settings searched on full
    # 20NG. 20NG: mean of 10 seeds (final_test.json). Sports: one seed, from the notebook named.
    ("RoBERTa <s> + LR", False, True, *lr_test("roberta_cls")),
    ("RoBERTa <s> + MLP", True, True, final_s["models"]["MLP, best 2 hidden layers"]["test_acc_mean"],
     92.92),  # sports: Frozen ModernBERT 2 class.ipynb's code on roberta_cls_sports.npz
    ("RoBERTa mean + LR", False, True, *lr_test("roberta_mean")),
    ("RoBERTa mean + MLP", True, True, final_mean["roberta"]["MLP, best"]["test_acc_mean"],
     91.38),  # sports: Frozen Roberta MLP 2 Class.ipynb, last section
    None,
    ("ModernBERT [CLS] + LR", False, True, *lr_test("modernbert_cls")),
    # RoBERTa <s>'s settings, one seed: Frozen ModernBERT.ipynb, Frozen ModernBERT 2 class.ipynb
    ("ModernBERT [CLS] + MLP", True, False, 57.75, 76.32),
    ("ModernBERT mean + LR", False, True, *lr_test("modernbert_mean")),
    ("ModernBERT mean + MLP", True, True, final_mean["modernbert"]["MLP, best"]["test_acc_mean"],
     87.64),  # sports: Frozen ModernBERT 2 class.ipynb, last section
    None,
    ("Qwen3-Embedding-0.6B + LR", False, True, *lr_test("qwen3_0.6b_last")),
    # RoBERTa <s>'s settings, one seed: Frozen Qwen3-Embedding-0.6B.ipynb and its 2-class version
    ("Qwen3-Embedding-0.6B + MLP", True, False, 74.15, 91.38),
    ("Qwen3-Embedding-4B + LR", False, True, *lr_test("qwen3_4b_last")),
    # RoBERTa <s>'s settings, one seed: Frozen Qwen3-Embedding-4B.ipynb and its 2-class version
    ("Qwen3-Embedding-4B + MLP", True, False, 77.54, 94.72),
    None,
    # A 2,048-token window instead of 512, 20NG only (no baseball-vs-hockey test error was a post over 512 tokens).
    # MLP settings searched on these features (hparam_search_mean/qwen3_4b_2048/)
    ("4B, 2,048 tokens + LR", False, True, *lr_test("qwen3_4b_2048_last")),
    ("4B, 2,048 tokens + MLP", True, True, final_mean["qwen3_4b_2048"]["MLP, best"]["test_acc_mean"], None),
    # Same, with the new instruction, tuned with folds grouped by thread (hparam_search_mean/Qwen4B2048Search.md):
    # LR from lr_search_threads.json, MLP from the grouped search (hparam_search_mean/qwen3_4b_2048_instrB_threads/)
    ("4B, 2,048 tokens, new instruction + LR", False, True,
     lr_threads["qwen3_4b_2048_instrB_last_20ng"]["test_100pct_train"]["test_acc"], None),
    ("4B, 2,048 tokens, new instruction + MLP", True, True,
     final_mean["qwen3_4b_2048_instrB_threads"]["MLP, best"]["test_acc_mean"], None),
]
PANELS = [("Full 20 Newsgroups\n20 classes, 7,370 test messages (1 point ≈ 74)", 3, (53.5, 81)),
          ("Baseball vs. hockey\n2 classes, 777 test messages (1 point ≈ 8)", 4, (73.5, 97.5))]

# y positions top to bottom, with an extra half row between groups
ys, y = [], 0.0
for row in ROWS:
    if row is None:
        y -= 0.5
    else:
        ys.append(y)
        y -= 1
rows = [r for r in ROWS if r is not None]

fig, axes = plt.subplots(1, 2, figsize=(10.2, 6.9), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (title, col, xlim) in zip(axes, PANELS):
    for yy, row in zip(ys, rows):
        acc, color = row[col], BLUE if row[1] else ORANGE
        if acc is None:
            continue
        if row[2]:
            ax.scatter(acc, yy, s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        else:  # hollow: borrowed settings, not tuned for these features
            ax.scatter(acc, yy, s=58, facecolor=SURFACE, edgecolor=color, linewidth=1.8, zorder=3)
        ax.annotate(f"{acc:.2f}", (acc, yy), xytext=(7, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=8.5)
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", color=INK_2)
    ax.set_xlabel("Test accuracy (%)")
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(ys, [row[0] for row in rows])
axes[0].set_ylim(ys[-1] - 0.7, ys[0] + 0.7)
axes[1].xaxis.set_major_locator(MultipleLocator(5))
fig.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=7, color=ORANGE,
                           label="Linear (LR, Naive Bayes)"),
                    Line2D([], [], marker="o", linestyle="", markersize=7, color=BLUE, label="MLP"),
                    Line2D([], [], marker="o", linestyle="", markersize=6.5, markerfacecolor=SURFACE,
                           markeredgecolor=BLUE, markeredgewidth=1.6,
                           label="MLP with RoBERTa MLP hyperparameters (ie. not tuned for these features)")],
           loc="upper center", bbox_to_anchor=(0.55, 0.0), ncol=3, frameon=False)

(HERE / "figures").mkdir(exist_ok=True)
fig.savefig(HERE / "figures" / "frozen_results.png", dpi=200, bbox_inches="tight")
print("wrote", HERE / "figures" / "frozen_results.png")
