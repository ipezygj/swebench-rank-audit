"""Card/thumbnail for the Kaggle writeup, 560 x 280 px: the Claude 3.5 Sonnet and Claude Sonnet 4 rows of fig1.
    python thumb.py -> results/thumb.png
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import HERE

OUT = HERE / "results"
INK, MUTED, GRID, C_H, C_R = "#1f2328", "#59636e", "#e6e8eb", "#2a6fdb", "#d1660f"

I = pd.read_csv(OUT / "a_submissions.csv")
R = pd.read_csv(OUT / "b_runs.csv"); R = R[R.usable]
rows = [("20241022-3-5-claude-sonnet", "Claude 3.5 Sonnet"), ("20250514-4-claude-sonnet", "Claude Sonnet 4")]
fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
fig.text(0.04, 0.86, "What is a harness worth?", fontsize=17, color=INK, weight="bold")
fig.text(0.04, 0.76, "Same model, different harness - SWE-bench Verified, resolved %", fontsize=8.5, color=MUTED)
ax = fig.add_axes([0.27, 0.13, 0.68, 0.42])
for y, (m, lab) in enumerate(rows):
    s = I[(I.model == m) & ~I.flag_any].score * 100
    ax.plot([s.min(), s.max()], [y, y], color=GRID, lw=4, solid_capstyle="round", zorder=1)
    ax.scatter(s, [y] * len(s), s=34, color=C_H, edgecolor="white", lw=1.1, zorder=3,
               label="leaderboard harnesses" if y == 0 else None)
    rb = R[R.model == m]
    if len(rb):
        ax.scatter(rb.score.max() * 100, y, marker="D", s=40, color=C_R, edgecolor="white", lw=1.1, zorder=4,
                   label="reference harness")
ax.set_yticks([0, 1]); ax.set_yticklabels([r[1] for r in rows], color=INK, fontsize=9)
ax.set_xlim(30, 80); ax.set_ylim(-0.6, 1.6)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(MUTED); ax.tick_params(colors=MUTED, labelsize=8, left=False)
ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
fig.savefig(OUT / "thumb.png", dpi=200)
