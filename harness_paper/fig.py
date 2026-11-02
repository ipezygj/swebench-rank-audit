"""Figures for the write-up.
fig1: every leaderboard harness on one exact model (dots) and that model on the fixed reference harness
      (mini-SWE-agent, diamond), SWE-bench Verified resolve rate.
fig2: the reference-harness board over model release date, with the best leaderboard harness on two
      May-2025 models drawn as horizontal reference lines.
    python fig.py -> results/fig1.png, results/fig2.png
"""
from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common import HERE

OUT = HERE / "results"
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"
C_HARNESS, C_REF = "#2a6fdb", "#d1660f"

LABEL = {
    "20250514-4-claude-sonnet": "Claude Sonnet 4", "20250219-3-7-claude-sonnet": "Claude 3.7 Sonnet",
    "20241022-3-5-claude-sonnet": "Claude 3.5 Sonnet (Oct)", "30b-a3b-coder-instruct-qwen3": "Qwen3-Coder-30B-A3B",
    "deepswe-preview": "DeepSWE-Preview", "2-5-qwen": "Qwen 2.5 (unspecified)", "20240513-4o-gpt": "GPT-4o (May)",
    "20240806-4o-gpt": "GPT-4o (Aug)", "1106-4-gpt-preview": "GPT-4 1106", "20240229-3-claude-opus": "Claude 3 Opus",
}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig1():
    I = pd.read_csv(OUT / "a_submissions.csv")
    G = pd.read_csv(OUT / "a_groups.csv").sort_values("best")
    R = pd.read_csv(OUT / "b_runs.csv")
    R = R[R.usable]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    for y, m in enumerate(G.model):
        s = I[I.model == m].score * 100
        ax.plot([s.min(), s.max()], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.scatter(s, [y] * len(s), s=36, color=C_HARNESS, edgecolor="white", lw=1.2, zorder=3,
                   label="leaderboard harness" if y == 0 else None)
        rb = R[R.model == m]
        if len(rb):
            ax.scatter(rb.score.max() * 100, y, marker="D", s=44, color=C_REF, edgecolor="white", lw=1.2,
                       zorder=4, label="reference harness (mini-SWE-agent)")
    ax.set_yticks(range(len(G)))
    ax.set_yticklabels([f"{LABEL.get(m, m)}  (n={n})" for m, n in zip(G.model, G.n)], color=INK, fontsize=9)
    ax.set_xlabel("SWE-bench Verified resolved (%)", color=INK, fontsize=9)
    ax.set_xlim(0, 85)
    style(ax)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Same model, different harness: one row per exact model tag", color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1.png")


def fig2():
    R = pd.read_csv(OUT / "b_runs.csv")
    R = R[R.usable & R.release.notna()].copy()
    R["date"] = pd.to_datetime(R.release.astype(int).astype(str), format="%Y%m%d")
    H2 = pd.read_csv(OUT / "h2.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=160)
    ax.scatter(R.date, R.score * 100, s=30, color=C_REF, marker="D", edgecolor="white", lw=1, zorder=3,
               label="model on the reference harness")
    for m, lab, ls in (("20250514-4-claude-sonnet", "best harness, Claude Sonnet 4 (attempts 2+)", "-"),
                       ("20250514-4-claude-opus", "best harness, Claude Opus 4 (attempts 1)", "--")):
        r = H2[H2.model == m].iloc[0]
        ax.axhline(r.best_score * 100, color=C_HARNESS, lw=1.6, ls=ls, zorder=2, label=lab)
    ax.axvline(pd.Timestamp("2025-05-22"), color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.text(pd.Timestamp("2025-05-15"), 81, "Claude 4 release", ha="right", color=MUTED, fontsize=8)
    ax.set_ylabel("SWE-bench Verified resolved (%)", color=INK, fontsize=9)
    ax.set_ylim(0, 85)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Later models on a fixed harness vs. the best harness on a May-2025 model", color=INK,
                 fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2.png")


if __name__ == "__main__":
    fig1(); fig2()
