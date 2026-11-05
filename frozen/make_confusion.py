"""Draw figures/confusion_matrix.png for FrozenResults.md: where the best model's 20NG test errors go.

    ~/venvs/cs423/bin/python make_confusion.py <predictions.npy> "<model name for the title>"      (from this folder)

Rows are the true class, columns the predicted one, each row in % of that class's test messages. The diagonal
(per-class accuracy) is gray with its number; off-diagonal cells (the errors) are shaded on one blue scale, left
empty under 1% of the row, and labeled from 4%. Same palette and style as make_frozen_results.py.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DIAGONAL = "#f0efec"  # neutral gray: the diagonal isn't on the error scale
# One-hue sequential ramp (validated palette, blue 100 -> 700)
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                   "#184f95", "#0d366b"])
ERR_MAX = 25  # % at which the error scale saturates
LABEL_MIN = 4  # label off-diagonal cells holding at least this % of their row

# Related classes next to each other, so the confusable blocks sit near the diagonal
GROUPS = [["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
           "comp.windows.x"], ["misc.forsale"],
          ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"],
          ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
          ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc"],
          ["alt.atheism", "soc.religion.christian", "talk.religion.misc"]]


def draw(y_true, y_pred, names, title, out):
    order = [names.index(n) for g in GROUPS for n in g]
    cm = confusion_matrix(y_true, y_pred, labels=order)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    k = len(order)

    rgba = BLUES(Normalize(0, ERR_MAX, clip=True)(pct))
    rgba[pct < 1] = matplotlib.colors.to_rgba(SURFACE)  # under 1% of the row: leave the cell empty
    rgba[np.arange(k), np.arange(k)] = matplotlib.colors.to_rgba(DIAGONAL)
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(rgba, interpolation="nearest")
    # Surface-colored gaps between cells, wider between class groups
    edges = np.cumsum([len(g) for g in GROUPS])[:-1]
    for i in range(k + 1):
        w = 3.5 if i in edges else 1.2
        ax.axhline(i - 0.5, color=SURFACE, linewidth=w)
        ax.axvline(i - 0.5, color=SURFACE, linewidth=w)
    for i in range(k):
        for j in range(k):
            v = pct[i, j]
            if i == j:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color=INK_2)
            elif v >= LABEL_MIN:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if v >= 0.45 * ERR_MAX else INK)
    labels = [names[i] for i in order]
    ax.set_xticks(range(k), labels, rotation=90, fontsize=7.5, color=INK_2)
    ax.set_yticks(range(k), labels, fontsize=7.5, color=INK_2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Predicted class", color=INK_2, fontsize=8.5)
    ax.set_ylabel("True class", color=INK_2, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=9, color=INK_2)
    cbar = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, ERR_MAX), BLUES), ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f"Errors: % of the true class predicted as this class ({ERR_MAX}+ is darkest)",
                   color=INK_2, fontsize=7.5)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, labelsize=7, length=0)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    s = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    y_test = np.array([t for d, t in zip(s.data, s.target) if d])  # same empty-post removal as the notebooks
    pred = np.load(sys.argv[1])
    assert len(pred) == len(y_test)
    acc = (pred == y_test).mean() * 100
    draw(y_test, pred, list(s.target_names),
         f"{sys.argv[2]}: full 20 Newsgroups test, {acc:.2f}% correct\n"
         "Each row is a true class, in % of its test messages; the gray diagonal is the share predicted correctly",
         HERE / "figures" / "confusion_matrix.png")
