"""Generate figures for the AFD-RSIR diagnostic paper (all numbers are the measured
values already reported in the .tex; greedy and temperature are 5-seed means)."""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description="Regenerate the paper's three figures.")
ap.add_argument("--outdir", default=".", help="output directory for the figures")
_args = ap.parse_args()
OUT = os.path.join(_args.outdir, "")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
})

BLUE = "#1f77b4"
RED = "#d62728"
GRAY = "#7f7f7f"
GREEN = "#2ca02c"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Figure 1: single-round collapse (head-20% share: real vs greedy vs temperature)
# ---------------------------------------------------------------------------
labels = ["Real\ntraining", "Greedy\nRSIR", "Temperature\n$T{=}2.0$"]
sports = [0.595, 0.851, 0.603]   # head-20% share (temperature on generated suffix, seq[1:])
yelp = [0.542, 0.927, 0.568]

x = np.arange(3)
w = 0.34

fig, ax = plt.subplots(figsize=(5.4, 3.1))
ax.bar(x - w / 2, sports, w, label="Sports", color=BLUE)
ax.bar(x + w / 2, yelp, w, label="Yelp", color=RED)

for i in range(3):
    ax.text(x[i] - w / 2, sports[i] + 0.015, "%.2f" % sports[i], ha="center", fontsize=8, color=BLUE)
    ax.text(x[i] + w / 2, yelp[i] + 0.015, "%.2f" % yelp[i], ha="center", fontsize=8, color=RED)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Head-20\\% share")
ax.set_ylim(0, 1.06)
ax.axhline(0.20, ls="--", lw=0.8, color="black", alpha=0.4)
ax.text(0.01, 0.215, "uniform (0.20)", fontsize=8, color="black", alpha=0.55)
ax.legend(frameon=False, loc="upper left", ncol=2)
style(ax)
fig.tight_layout()
fig.savefig(OUT + "fig_collapse.pdf", bbox_inches="tight")
fig.savefig(OUT + "fig_collapse.png", bbox_inches="tight", dpi=150)
print("saved fig_collapse")


# ---------------------------------------------------------------------------
# Figure 2: multi-round compounding (head-20% and Gini vs round)
# ---------------------------------------------------------------------------
rounds = np.arange(0, 9)  # 0 = real, 1..8
sport_head = [0.595, 0.658, 0.701, 0.741, 0.738, 0.735, 0.732, 0.729, 0.725]
yelp_head = [0.542, 0.645, 0.688, 0.782, 0.807, 0.826, 0.840, 0.850, 0.860]
sport_gini = [0.533, 0.610, 0.672, 0.737, 0.740, 0.743, 0.746, 0.749, 0.752]
yelp_gini = [0.475, 0.593, 0.643, 0.753, 0.784, 0.808, 0.826, 0.841, 0.854]

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))

ax = axes[0]
ax.plot(rounds, sport_head, "o-", color=BLUE, lw=1.2, ms=3.5, label="Sports")
ax.plot(rounds, yelp_head, "s-", color=RED, lw=1.2, ms=3.5, label="Yelp")
ax.set_xlabel("Self-training round (0 = real)")
ax.set_ylabel("Head-20\\% share (combined)")
ax.set_xticks(rounds)
ax.set_xlim(-0.3, 8.3)
ax.set_ylim(0.50, 0.90)
ax.legend(frameon=False, loc="lower right")
ax.set_title("(a) Head-20\\% share", fontsize=10)
style(ax)

ax = axes[1]
ax.plot(rounds, sport_gini, "o-", color=BLUE, lw=1.2, ms=3.5, label="Sports")
ax.plot(rounds, yelp_gini, "s-", color=RED, lw=1.2, ms=3.5, label="Yelp")
ax.set_xlabel("Self-training round (0 = real)")
ax.set_ylabel("Gini coefficient (combined)")
ax.set_xticks(rounds)
ax.set_xlim(-0.3, 8.3)
ax.set_ylim(0.44, 0.90)
ax.legend(frameon=False, loc="lower right")
ax.set_title("(b) Gini coefficient", fontsize=10)
style(ax)

fig.tight_layout()
fig.savefig(OUT + "fig_recursion.pdf", bbox_inches="tight")
fig.savefig(OUT + "fig_recursion.png", bbox_inches="tight", dpi=150)
print("saved fig_recursion")


# ---------------------------------------------------------------------------
# Figure 3: accuracy--diversity frontier over decoding temperature (five-seed
# mean, fidelity gate ON). Single panel for direct comparison. Sports labels sit
# above its curve, Yelp labels below. Only the anchors greedy / T=1.0 / T=2.0 are
# labeled (the full sweep T in {1.0,...,2.0} is listed in the caption). Open
# diamonds mark greedy decoding with the gate OFF.
# ---------------------------------------------------------------------------
sport = [(0.266, 0.0722), (0.570, 0.0699), (0.630, 0.0689), (0.667, 0.0683),
         (0.691, 0.0674), (0.707, 0.0676), (0.719, 0.0675)]  # greedy -> T=2.0
yelp = [(0.188, 0.0684), (0.697, 0.0663), (0.762, 0.0660), (0.802, 0.0646),
        (0.825, 0.0639), (0.837, 0.0640), (0.840, 0.0642)]
fig, ax = plt.subplots(figsize=(7.2, 3.5))

ax.plot([p[0] for p in sport], [p[1] for p in sport], "o-", color=BLUE, lw=1.2, ms=4, label="Sports")
ax.plot([p[0] for p in yelp], [p[1] for p in yelp], "s-", color=RED, lw=1.2, ms=4, label="Yelp")

# labels, hand-placed to avoid overlapping the curves
ax.annotate("greedy", sport[0], xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=BLUE)
ax.annotate("$T{=}1.0$", sport[1], xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=BLUE)
ax.annotate("$T{=}2.0$", sport[6], xytext=(6, 14), textcoords="offset points", ha="left", fontsize=8, color=BLUE)
ax.annotate("greedy", yelp[0], xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8, color=RED)
ax.annotate("$T{=}1.0$", yelp[1], xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8, color=RED)
ax.annotate("$T{=}2.0$", yelp[6], xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8, color=RED)

# gate-off (greedy decoding, fidelity gate off)
ax.plot(0.983, 0.0748, "D", color=BLUE, mfc="none", mew=1.3, ms=7, zorder=4)
ax.plot(0.986, 0.0695, "D", color=RED, mfc="none", mew=1.3, ms=7, zorder=4)
ax.annotate("gate off", (0.983, 0.0748), xytext=(-5, 3), textcoords="offset points",
            ha="right", fontsize=8, color="black")

ax.set_xlabel("Catalog coverage (diversity)")
ax.set_ylabel("Recall@20 (accuracy)")
ax.set_xlim(0.15, 1.04)
ax.set_ylim(0.061, 0.076)
ax.legend(frameon=False, loc="lower left")
style(ax)
fig.tight_layout()
fig.savefig(OUT + "fig_pareto.pdf", bbox_inches="tight")
fig.savefig(OUT + "fig_pareto.png", bbox_inches="tight", dpi=150)
print("saved fig_pareto")
