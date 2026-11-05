"""Draw figures/piazza_results.png for README-piazza.md: the main rows of make_all_results.py, plus all of A2's models.

    ~/venvs/cs423/bin/python make_piazza_results.py      (from this folder)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent

# Colors from a validated palette, light chart surface (the same one as make_all_results.py)
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLOR = "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlesize": 9.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
})

BEST_PUBLISHED = 77.75
HEADLINE = "Qwen, LoRA"
PICKED_USING_TEST = "Qwen, LoRA, lr 2e-4 (picked using test)"
# (section title, rows). A row is (label, 20NG: one run or mean of seeds, 20NG: ensemble, baseball vs. hockey)
SECTIONS = [
    ("Assignment models", [
        ("BoW counts + LR", 64.45, None, 89.06),
        ("BoW counts + Naive Bayes", 64.08, None, 91.12),
        ("CBoW (average word vector) + LR", 64.55, None, 91.51),
        ("Frozen RoBERTa + LR, first 128 tokens", 64.75, None, 91.25)]),
    ("Frozen models", [
        ("RoBERTa, 512 tokens + tuned MLP", 68.33, None, 92.92),
        ("ModernBERT, mean + MLP", 66.28, None, 87.64),
        ("Qwen3-Embedding-4B + LR", 77.39, None, 95.24),
        ("Qwen, tuned input + MLP", 78.80, 78.97, None)]),
    ("Fine-tuned", [
        ("RoBERTa, all weights", 73.13, 73.93, 93.18),
        ("RoBERTa, LoRA", 73.29, 75.07, None),
        (HEADLINE, 78.67, 79.73, None),
        (PICKED_USING_TEST, 79.19, 80.56, None)]),
    ("Distillation", [
        ("RoBERTa, LoRA Qwen as teacher", 73.15, 74.02, None),
        ("RoBERTa, frozen Qwen + LR as teacher", 74.77, 75.29, None)]),
]

# y positions top to bottom: a line for each section's title, a line per row, a small gap between sections
rows, titles, y = [], [], 0.0
for title, section in SECTIONS:
    titles.append((y, title))
    y -= 0.95
    for row in section:
        rows.append((y, *row))
        y -= 1
    y -= 0.35
y_bottom = y + 0.35

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 0.27 * abs(y_bottom) + 1.0), sharey=True,
                              gridspec_kw={"wspace": 0.05, "width_ratios": [2.1, 1]})


def value(a, x, yy, side, bold=False):
    a.annotate(f"{x:.2f}", (x, yy), xytext=(6 if side == "right" else -6, 0), textcoords="offset points", va="center",
               ha="left" if side == "right" else "right", color=INK, fontsize=8.5,
               fontweight="bold" if bold else "normal", bbox=dict(boxstyle="square,pad=0.08", fc=SURFACE, ec="none"),
               zorder=4)


# Everything right of this line beats the best published result on these posts
ax.axvline(BEST_PUBLISHED, color=INK_2, linewidth=0.9, zorder=1)
for yy, label, single, ensemble, sports in rows:
    hollow = label == PICKED_USING_TEST   # a row chosen after seeing test scores
    marks = dict(facecolor=SURFACE if hollow else COLOR, edgecolor=COLOR if hollow else SURFACE,
                 linewidth=1.5 if hollow else 1.0, zorder=3)
    ax.scatter(single, yy, s=48, **marks)
    if ensemble is None:
        value(ax, single, yy, "right")
    else:
        ax.plot([single, ensemble], [yy, yy], color=COLOR, linewidth=1.3, zorder=2, solid_capstyle="butt")
        ax.scatter(ensemble, yy, s=44, marker="D", **marks)
        value(ax, single, yy, "left")
        value(ax, ensemble, yy, "right", bold=label == HEADLINE)
    if sports is not None:
        ax2.scatter(sports, yy, s=48, color=COLOR, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        value(ax2, sports, yy, "right")

ax.set_title("All 20 newsgroups\n20 classes, 7,370 test posts", loc="left", color=INK_2)
ax2.set_title("Baseball vs. hockey (A2's task)\n2 classes, 777 test posts", loc="left", color=INK_2)
ax.set_xlim(62, 83)
ax2.set_xlim(85.5, 98.5)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax2.xaxis.set_major_locator(MultipleLocator(5))
for a in (ax, ax2):
    a.set_xlabel("Test accuracy (%)")
    a.tick_params(axis="y", length=0)
ax.annotate(f"Best published on these posts: {BEST_PUBLISHED:.2f}", (BEST_PUBLISHED, 1.0),
            xycoords=("data", "axes fraction"), xytext=(-5, -2), textcoords="offset points", ha="right", va="top",
            color=INK_2, fontsize=8.5)
ax.set_yticks([r[0] for r in rows], [r[1] for r in rows])
for tick, r in zip(ax.get_yticklabels(), rows):
    tick.set_color(MUTED if r[1] == PICKED_USING_TEST else INK_2)
for yy, title in titles:
    # Section titles start at a fixed spot in the label column
    ax.annotate(title, (0, yy), xycoords=("axes fraction", "data"), xytext=(-195, 0), textcoords="offset points",
                ha="left", va="center", color=INK, fontsize=9.5, fontweight="bold")
ax.set_ylim(y_bottom - 0.1, 0.75)

# The lower part of this panel is empty on purpose
ax2.text(0.05, rows[9][0], "LoRA and distillation were\nrun on 20 classes only", color=INK_2, fontsize=8.5,
         transform=ax2.get_yaxis_transform(), va="top", bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))

fig.legend(handles=[
    Line2D([], [], marker="o", linestyle="", markersize=6.5, color=COLOR, label="One run or mean of seeds"),
    Line2D([], [], marker="D", linestyle="", markersize=5.5, color=COLOR, label="Ensemble of those seeds"),
], loc="upper center", bbox_to_anchor=(0.42, 0.045), ncol=2, frameon=False, fontsize=8.5, columnspacing=1.6,
    handletextpad=0.3)

fig.savefig(HERE / "piazza_results.png", dpi=200, bbox_inches="tight")
print("wrote", HERE / "piazza_results.png")
