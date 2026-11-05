"""Draw the figures in MLPHyperparameterSearch.md from trials.jsonl, lr_baselines.jsonl and final_test.json.

    ~/venvs/cs423/bin/python make_figures.py      (from this folder)

Every figure is rebuilt from the result files. The style (colors, fonts, markers) matches the A1 search write-ups.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"

# Colors from a validated palette, light chart surface (the A1 figures' palette)
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"                         # MLPs, linear models
RATE_COLORS = {0.1: "#86b6ef", 0.3: "#3987e5", 0.5: "#184f95"}  # one-hue ordinal ramp for dropout rates
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#184f95", "#104281", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlesize": 10,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False,
})

trials = [json.loads(line) for line in (HERE / "trials.jsonl").read_text().splitlines()]
lr_runs = [json.loads(line) for line in (HERE / "lr_baselines.jsonl").read_text().splitlines()]
final = json.loads((HERE / "final_test.json").read_text())
TUNED_LR = max(r["mean_dev_acc"] for r in lr_runs)  # best LR cross-validation score (C and scaling tuned)

SEEDS = (0, 1, 2)
runs = defaultdict(dict)  # config as JSON -> {seed: CV accuracy}
for t in trials:
    runs[json.dumps(t["cfg"], sort_keys=True)][t["seed"]] = t["mean_dev_acc"]


def score(cfg):
    """Mean CV accuracy over the 3 seeds, or None if the config wasn't run with all of them."""
    seeds = runs.get(json.dumps(cfg, sort_keys=True), {})
    return float(np.mean([seeds[s] for s in SEEDS])) if all(s in seeds for s in SEEDS) else None


def phase_cfgs(phase):
    return [t["cfg"] for t in trials if t["phase"] == phase]


def dot(ax, x, y, color, size=46, z=3, ring=1.5):
    """A marker with a surface-colored ring, so it stays legible where marks overlap."""
    ax.scatter(x, y, s=size, color=color, edgecolor=SURFACE, linewidth=ring, zorder=z)


def ref_line(ax, x, color, label, y):
    """A solid hairline at a reference score, labeled in text ink beside it."""
    ax.axvline(x, color=color, linewidth=1.2, zorder=1)
    ax.text(x + 0.03, y, label, color=INK_2, fontsize=8, va="center")


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG_DIR / name)


# ---------------------------------------------------------------------------------------------------------------
def fig1_final_test():
    labels = {
        "MLP, best 2 hidden layers": "MLP 2048 → 64, tuned",
        "MLP, best 1 hidden layer": "MLP 2048 (1 hidden layer), tuned",
        "LR, best C from phase 0": "Logistic regression, C and scaling tuned",
        "Linear, 0 hidden layers": "Linear layer, trained like the MLPs",
        "LR, notebook setup (C=1, raw)": "Logistic regression, A2 setup (C = 1)",
        "MLP, notebook config": "MLP from the first notebook, untuned",
    }
    rows = [(labels[name], BLUE if name.startswith("MLP") else ORANGE, [r["test_acc"] for r in m["runs"]])
            for name, m in final["models"].items()]
    rows += [(labels[name], ORANGE, [m["test_acc"]]) for name, m in final["sklearn_lr"].items()]
    rows.sort(key=lambda row: np.mean(row[2]))  # best at the top

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    jitter = np.random.default_rng(0)
    for i, (label, color, accs) in enumerate(rows):
        if len(accs) > 1:
            dot(ax, accs, i + jitter.uniform(-0.16, 0.16, len(accs)), color, size=16, z=2, ring=0.8)
        dot(ax, np.mean(accs), i, color, size=80, z=4)
        ax.text(max(accs) + 0.1, i, f"{np.mean(accs):.2f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(64.4, 69.3)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("Test accuracy, 20 Newsgroups: small dots are 10 seeds, big dot is their mean "
                 "(logistic regression is one deterministic run)", loc="left", fontsize=9, color=INK_2)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=7, color=BLUE, label="MLP"),
                       Line2D([], [], marker="o", linestyle="", markersize=7, color=ORANGE, label="Linear model")],
              loc="lower right")
    save(fig, "fig1_final_test.png")


# ---------------------------------------------------------------------------------------------------------------
def fig2_sweeps():
    """Each setting changed on its own from the sweep's starting config: longer rows mattered more."""
    sweep = phase_cfgs("sweep1")
    fields = list(sweep[0])
    center = {f: Counter(c[f] for c in sweep).most_common(1)[0][0] for f in fields}  # unchanged in most runs

    def fmt(field, v):
        if field == "learning_rate":
            return f"{v:.0e}".replace("e-0", "e-")
        if isinstance(v, bool):  # checked first: 0.0 == False and 1.0 == True in Python
            return "standardized" if v else "raw"
        return v if isinstance(v, str) else f"{v:g}"

    names = {"learning_rate": "Learning rate", "standardize": "Input scaling", "dropout1": "Dropout after layer 1",
             "activation": "Activation", "label_smoothing": "Label smoothing", "hidden1": "Width of layer 1",
             "hidden2": "Width of layer 2", "dropout2": "Dropout after layer 2", "dropout_in": "Input dropout",
             "weight_decay": "Weight decay", "batch_size": "Batch size", "lr_schedule": "Learning-rate schedule"}
    rows = []
    for field, name in names.items():
        if field == "standardize":  # raw inputs at their own best learning rate (they need a different range)
            raw = [(score(c), c["learning_rate"]) for c in sweep if not c["standardize"]]
            best_raw = max(s for s in raw if s[0] is not None)
            values = [("standardized", score(center)), (f"raw (lr {fmt('learning_rate', best_raw[1])})", best_raw[0])]
        else:
            same_otherwise = [c for c in sweep if all(c[g] == center[g] for g in fields if g != field)]
            # The starting config itself ran in the confirm phase, so its value is added separately
            values = sorted({c[field] for c in same_otherwise} | {center[field]}, key=str)
            values = [(fmt(field, v), score({**center, field: v})) for v in values]
        values = [(v, s) for v, s in values if s is not None]
        rows.append((name, values))
    rows.sort(key=lambda row: max(s for _, s in row[1]) - min(s for _, s in row[1]))  # biggest spread at the top

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, (name, values) in enumerate(rows):
        (lo_name, lo), (hi_name, hi) = min(values, key=lambda v: v[1]), max(values, key=lambda v: v[1])
        ax.plot([lo, hi], [i, i], color=AXIS, linewidth=2, solid_capstyle="round", zorder=1)
        for _, s in values:
            dot(ax, s, i, MUTED, size=26, z=2)
        dot(ax, hi, i, BLUE, size=46, z=3)
        ax.text(lo - 0.07, i, lo_name, ha="right", va="center", color=MUTED, fontsize=8)
        ax.text(hi + 0.07, i, hi_name, ha="left", va="center", color=INK_2, fontsize=8)
    ref_line(ax, TUNED_LR, ORANGE, f"tuned logistic regression ({TUNED_LR:.2f})", len(rows) - 0.35)
    ax.set_yticks(range(len(rows)), [name for name, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlim(68.3, 73.6)
    ax.set_xlabel("Cross-validation accuracy (%), mean of 3 seeds × 5 folds")
    ax.set_title("One setting changed at a time from the best random-search config: each dot is one value, "
                 "blue is the best, gray text the worst", loc="left", fontsize=9, color=INK_2)
    save(fig, "fig2_one_change_at_a_time.png")


# ---------------------------------------------------------------------------------------------------------------
def fig3_dropout_placement():
    placement = phase_cfgs("placement")
    places = {"in": "dropout_in", "1": "dropout1", "2": "dropout2"}
    base = next(c for c in placement if all(c[f] == 0.0 for f in places.values()))  # "none": no dropout anywhere
    non_dropout = [f for f in base if f not in places.values()]
    # The best dropout rates tried with the same other settings (tuned separately for each place)
    tuned = max(((score(cfg), cfg) for cfg in map(json.loads, runs)
                 if all(cfg[f] == base[f] for f in non_dropout) and score(cfg) is not None), key=lambda sc: sc[0])
    combos = [((), "none"), (("in",), "input only"), (("1",), "after layer 1 only"), (("2",), "after layer 2 only"),
              (("in", "1"), "input + layer 1"), (("in", "2"), "input + layer 2"), (("1", "2"), "layers 1 + 2"),
              (("in", "1", "2"), "all three places")]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    rows = list(reversed(combos))  # "none" at the top
    for i, (combo, label) in enumerate(rows):
        scores = {rate: score({**base, **{f: (rate if p in combo else 0.0) for p, f in places.items()}})
                  for rate in RATE_COLORS}
        if not combo:
            dot(ax, scores[0.1], i, MUTED, size=46)
            continue
        ax.plot([min(scores.values()), max(scores.values())], [i, i], color=AXIS, linewidth=2, zorder=1)
        for rate, s in scores.items():
            dot(ax, s, i, RATE_COLORS[rate], size=46)
    t = tuned[1]
    ref_line(ax, tuned[0], BLUE, f"tuned per place: {t['dropout_in']:g} / {t['dropout1']:g} / {t['dropout2']:g} "
             f"({tuned[0]:.2f})", len(rows) - 0.3)
    ref_line(ax, TUNED_LR, ORANGE, f"tuned logistic regression ({TUNED_LR:.2f})", len(rows) - 0.3)
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.set_xlim(70.4, 73.6)
    ax.set_xlabel("Cross-validation accuracy (%), mean of 3 seeds × 5 folds")
    ax.set_title("Dropout placement: the same rate in each listed place, 0 elsewhere (other settings: the best config)",
                 loc="left", fontsize=9, color=INK_2)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=7, color=c, label=f"rate {r:g}")
                       for r, c in RATE_COLORS.items()], loc="lower right", title="Dropout", title_fontsize=8)
    save(fig, "fig3_dropout_placement.png")


# ---------------------------------------------------------------------------------------------------------------
def fig4_sizes():
    sizes = phase_cfgs("sizes")
    base = sizes[0]
    h1s = sorted({c["hidden1"] for c in sizes})
    h2s = sorted({c["hidden2"] for c in sizes})
    grid = np.array([[score({**base, "hidden1": h1, "hidden2": h2}) for h2 in h2s] for h1 in h1s], dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    norm = Normalize(np.nanmin(grid), np.nanmax(grid))
    mesh = ax.pcolormesh(grid, cmap=cmap, norm=norm, edgecolors=SURFACE, linewidth=2)
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    for (i, j), value in np.ndenumerate(grid):
        dark = norm(value) > 0.55  # white text on the darker cells
        ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if dark else INK, fontweight="bold" if (i, j) == best else "normal")
    ax.set_xticks(np.arange(len(h2s)) + 0.5, h2s)
    ax.set_yticks(np.arange(len(h1s)) + 0.5, h1s)
    ax.invert_yaxis()  # 256 at the top, like a table
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Width of hidden layer 2")
    ax.set_ylabel("Width of hidden layer 1")
    ax.set_title("CV accuracy (%) by layer widths (other settings: the best config)", loc="left", fontsize=9,
                 color=INK_2)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)
    save(fig, "fig4_layer_sizes.png")


if __name__ == "__main__":
    fig1_final_test()
    fig2_sweeps()
    fig3_dropout_placement()
    fig4_sizes()
