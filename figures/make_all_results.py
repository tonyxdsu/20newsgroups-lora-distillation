"""Draw figures/all_results.png for README.md: every test accuracy in the project, in the order I tried things.

    ~/venvs/cs423/bin/python make_all_results.py      (from this folder)

Same style as ../frozen/make_frozen_results.py. The numbers are typed in here so that a row is easy to drop. Where they come from:
- Fine-tuned, LoRA and distilled rows: recomputed from ../sweeps/probs/ (../sweeps/summarize.py --final NAME), except the three earlier
  notebook runs, which are in ../fine-tuning-results.md.
- Frozen rows: ../frozen/hparam_search/final_test.json and ../frozen/hparam_search_mean/{final_test,lr_search,lr_search_threads}.json.
- Assignment rows: the assignment notebook (2 classes) and its PLAYGROUND copy (20 classes).
- Published rows: Sada et al. 2026 and Moreo et al. 2021. They keep the 162 empty test posts, so reported accuracies
  are converted to our non-empty posts by estimated(), unless per-post predictions were available.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent

# Colors from a validated palette, light chart surface (the same one as ../frozen/make_frozen_results.py)
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NO_ENCODER, FROZEN, TRAINED, DISTILLED, PUBLISHED = MUTED, BLUE, ORANGE, AQUA, INK_2
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlesize": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
})


def estimated(reported, total=7532, empty=162, guess_rate=0.05):
    """A reported accuracy, converted to the non-empty posts. A model can only guess an empty post (5% right, 1 in 20)."""
    return (reported / 100 * total - guess_rate * empty) / (total - empty) * 100


# (section title, color of its rows, rows). A row is (label, 20NG: one run or mean of seeds, 20NG: ensemble, baseball vs.
# hockey). Optional: key=True prints the row in bold, untuned=True draws a hollow dot, approx=True prints "≈", and
# color=... overrides the section's color. BoW + LR is the better of unigram and unigram + bigram features.
def row(label, single, ensemble=None, sports=None, **flags):
    return dict(label=label, single=single, ensemble=ensemble, sports=sports, **flags)


SECTIONS = [
    ("1  Assignment models", NO_ENCODER, [
        row("BoW counts + LR", 64.45, sports=89.06),
        row("BoW counts + Naive Bayes", 64.08, sports=91.12),
        row("CBoW (average word vector) + LR", 64.55, sports=91.51),
        row("Frozen RoBERTa <s> + LR, first 128 tokens", 64.75, sports=91.25, key=True, color=FROZEN),
    ]),
    ("2  Frozen RoBERTa-base, first 512 tokens", FROZEN, [
        row("<s> + LR", 67.23, sports=91.63),
        row("<s> + MLP", 68.33, sports=92.92, key=True),
        row("Mean of all tokens + LR", 66.13, sports=89.83),
        row("Mean of all tokens + MLP", 67.23, sports=91.38),
    ]),
    ("3  Other frozen encoders", FROZEN, [
        row("ModernBERT-base [CLS] + LR", 57.26, sports=78.51),
        row("ModernBERT-base [CLS] + MLP", 57.75, sports=76.32, untuned=True),
        row("ModernBERT-base mean + LR", 64.92, sports=88.16),
        row("ModernBERT-base mean + MLP", 66.28, sports=87.64),
        row("Qwen3-Embedding-0.6B + LR", 73.42, sports=92.02),
        row("Qwen3-Embedding-0.6B + MLP", 74.15, sports=91.38, untuned=True),
        row("Qwen3-Embedding-4B + LR", 77.39, sports=95.24, key=True),
        row("Qwen3-Embedding-4B + MLP", 77.54, sports=94.72, untuned=True),
    ]),
    ("4  Frozen Qwen3-Embedding-4B, input tuned", FROZEN, [
        row("2,048 tokens + LR", 77.69),
        row("2,048 tokens + MLP", 78.07, 78.11),
        row("2,048 tokens, new instruction + LR", 78.00),
        row("2,048 tokens, new instruction + MLP", 78.80, 78.97, key=True),
    ]),
    ("5  Fine-tuned RoBERTa-base (all weights)", TRAINED, [
        row("BERT recipe, linear head, 3 runs", 72.28),
        row("BERT recipe, HF head, 1 run", 71.26),
        row("RoBERTa recipe, 1 run (kept epoch 3)", 71.37),
        row("RoBERTa recipe, 3 seeds", 73.13, 73.93, sports=93.18, key=True),
    ]),
    ("6  LoRA", TRAINED, [
        row("RoBERTa-base, rank 4", 72.75, 74.48),
        row("RoBERTa-base, rank 16", 73.29, 75.07, key=True),
        row("RoBERTa-base, rank 64", 73.41, 74.91),
        row("Qwen3-Embedding-4B, lr 5e-5, 1 seed", 78.63),
        row("Qwen3-Embedding-4B, lr 1e-4", 78.67, 79.73, key=True),
        row("Qwen3-Embedding-4B, lr 2e-4 (picked using test)", 79.19, 80.56),
    ]),
    ("7  RoBERTa-base distilled from Qwen", DISTILLED, [
        row("Teacher: LoRA Qwen, T = 2", 73.15, 74.02),
        row("Teacher: LoRA Qwen, T = 4 (run afterwards)", 73.44, 74.30),
        row("Teacher: frozen Qwen + LR, out of fold", 74.77, 75.29, key=True),
        row("Teacher: frozen Qwen + LR, in-sample", 74.65, 75.36),
        row("Teacher: frozen Qwen + MLP, out of fold", 74.32, 75.03),
        row("Control: label smoothing, no teacher", 73.09, 74.19),
    ]),
    ("Published, same setup", PUBLISHED, [
        row("LoRA Qwen3-8B, 4-bit (Sada et al.)", 77.75),                       # rescored from their predictions
        row("Kimi-K2, zero-shot, 494 posts (Sada et al.)", 77.33),              # same
        row("CNN + word-class embeddings (Moreo et al.)", estimated(71.7), approx=True),
        row("Complement NB, TF-IDF bigrams (Sada et al.)", estimated(71.2), approx=True),
        row("Fine-tuned BERT-base (Moreo et al.)", estimated(69.5), approx=True),
        # Their code scores the first 2,000 test posts, 39 of them empty
        row("Fine-tuned DistilBERT, 2,000 posts (Sada et al.)", estimated(67.2, 2000, 39), approx=True),
    ]),
]
BEST_PUBLISHED = 77.75
PANELS = [("Full 20 Newsgroups\n20 classes, 7,370 test posts (1 point ≈ 74)", (54.8, 83)),
          ("Baseball vs. hockey (the assignment's task)\n2 classes, 777 test posts (1 point ≈ 8)", (74, 98.5))]

# y positions top to bottom: a line for each section's title, a line per row, a small gap between sections
items, y = [], 0.0
for title, color, rows in SECTIONS:
    items.append(("title", y, title))
    y -= 0.95
    for r in rows:
        items.append(("row", y, {"color": color} | r))
        y -= 1
    y -= 0.45
rows = [(yy, r) for kind, yy, r in items if kind == "row"]
y_bottom = y + 0.45

ROW_INCHES = 0.2
fig, axes = plt.subplots(1, 2, figsize=(10.8, ROW_INCHES * abs(y_bottom) + 1.0), sharey=True,
                         gridspec_kw={"wspace": 0.05, "width_ratios": [2.15, 1]})
ax, ax2 = axes


def value(a, x, side, approx=False, bold=False):
    text = f"≈{x:.1f}" if approx else f"{x:.2f}"
    a.annotate(text, (x, yy), xytext=(6 if side == "right" else -6, 0), textcoords="offset points", va="center",
               ha="left" if side == "right" else "right", color=INK, fontsize=8, fontweight="bold" if bold else "normal",
               bbox=dict(boxstyle="square,pad=0.08", fc=SURFACE, ec="none"), zorder=4)


def dot(a, x, r):
    if r.get("untuned"):
        a.scatter(x, yy, s=34, facecolor=SURFACE, edgecolor=r["color"], linewidth=1.6, zorder=3)
    else:
        a.scatter(x, yy, s=46, color=r["color"], edgecolor=SURFACE, linewidth=1.0, zorder=3)


# A thin reference line: everything right of it beats the best published result on these posts
ax.axvline(BEST_PUBLISHED, color=INK_2, linewidth=0.9, zorder=1)
for yy, r in rows:
    bold = r.get("key", False)
    dot(ax, r["single"], r)
    if r["ensemble"] is None:
        value(ax, r["single"], "right", r.get("approx", False), bold)
    else:
        ax.plot([r["single"], r["ensemble"]], [yy, yy], color=r["color"], linewidth=1.3, zorder=2, solid_capstyle="butt")
        ax.scatter(r["ensemble"], yy, s=44, marker="D", color=r["color"], edgecolor=SURFACE, linewidth=1.0, zorder=3)
        value(ax, r["single"], "left", bold=bold)
        value(ax, r["ensemble"], "right", bold=bold)
    if r["sports"] is not None:
        dot(ax2, r["sports"], r)
        value(ax2, r["sports"], "right", bold=bold)

for a, (title, xlim) in zip(axes, PANELS):
    a.set_xlim(*xlim)
    a.set_title(title, loc="left", color=INK_2)
    a.set_xlabel("Test accuracy (%)")
    a.tick_params(axis="y", length=0)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax2.xaxis.set_major_locator(MultipleLocator(5))
ax.annotate(f"Best published on these posts: {BEST_PUBLISHED:.2f}", (BEST_PUBLISHED, 1.0),
            xycoords=("data", "axes fraction"), xytext=(-5, -1), textcoords="offset points", ha="right", va="top",
            color=INK_2, fontsize=8)

ax.set_yticks([yy for yy, _ in rows], [r["label"] for _, r in rows])
for tick, (_, r) in zip(ax.get_yticklabels(), rows):
    tick.set_color(INK if r.get("key") else INK_2)
    tick.set_fontweight("bold" if r.get("key") else "normal")
    tick.set_fontsize(8.5)
for kind, yy, title in items:
    if kind == "title":
        # Section titles start at a fixed spot in the label column, and run into the plot if they're long
        ax.annotate(title, (0, yy), xycoords=("axes fraction", "data"), xytext=(-232, 0), textcoords="offset points",
                    ha="left", va="center", color=INK, fontsize=9, fontweight="bold")
ax.set_ylim(y_bottom - 0.2, 0.75)

# The lower part of this panel is empty on purpose
last_sports_y = min(yy for yy, r in rows if r["sports"] is not None)
ax2.text(0.04, last_sports_y - 1.5, "Everything below was run on\n20 classes only", color=INK_2, fontsize=8.5,
         transform=ax2.get_yaxis_transform(), va="top", bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))

fig.legend(handles=[
    Line2D([], [], marker="o", linestyle="", markersize=6.5, color=MUTED, label="One run or mean of seeds"),
    Line2D([], [], marker="D", linestyle="", markersize=5.5, color=MUTED, label="Ensemble of those seeds"),
    Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=SURFACE, markeredgecolor=MUTED,
           markeredgewidth=1.5, label="MLP with RoBERTa's settings (not tuned for these features)"),
    Line2D([], [], marker="o", linestyle="", markersize=6.5, color=FROZEN, label="Frozen encoder"),
    Line2D([], [], marker="o", linestyle="", markersize=6.5, color=TRAINED, label="Encoder trained on the labels"),
    Line2D([], [], marker="o", linestyle="", markersize=6.5, color=DISTILLED, label="Trained on labels + a teacher"),
    Line2D([], [], marker="o", linestyle="", markersize=6.5, color=NO_ENCODER, label="No encoder"),
    Line2D([], [], marker="o", linestyle="", markersize=6.5, color=PUBLISHED, label="Published"),
], loc="upper center", bbox_to_anchor=(0.42, 0.055), ncol=4, frameon=False, fontsize=8.5, columnspacing=1.6,
    handletextpad=0.3)

fig.savefig(HERE / "all_results.png", dpi=200, bbox_inches="tight")
print("wrote", HERE / "all_results.png")
