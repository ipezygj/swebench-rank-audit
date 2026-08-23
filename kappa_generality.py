"""Does the sharp-pair finding hold at ranks other than the top?

kappa is 0.44-0.94 for the pair a board argues about and 1.00 on average.
Two readings fit that: (a) systems near each other in ABILITY are similar
because similar systems perform similarly - a statement about the whole
board, in which case the effect should be just as strong at rank 50 as at
rank 1; or (b) the top of a board is where entrants pile onto one lineage -
in which case the effect should FADE with depth.

pair_sharpness.py tested frontier pairs against ability-matched pairs and
the frontier still won, which already argues against (a). This measures the
shape directly: kappa of the (r, r+1) pair as a function of r, and kappa of
pairs at a fixed ability distance at different depths.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * kappa of adjacent pairs rises with rank on >= 6 of 9 boards (Spearman
    of kappa against rank positive) - the top is sharper than the middle;
  * the top decile's mean adjacent kappa is below the bottom decile's on
    >= 7 of 9;
  * exploratory: the depth at which adjacent kappa reaches 0.95, reported
    as "the rank below which the board behaves like independent entrants".

SELF-CHECKS
  * on a field where the probability of sharing a lineage decays with rank,
    neighbourhood kappa must rise with rank;
  * on a field with one lineage spread uniformly over ranks, it must be
    flat (|Spearman| < 0.3).

    python kappa_generality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from entropy_law_test import MATRICES
from pair_sharpness import kappa_matrix

SEED = 20260823


def adjacent_kappa(x, window=5):
    """Mean kappa between a system and its `window` nearest ranks.

    The first build used only the (r, r+1) pair. On a planted field where
    the strongest quarter shares a lineage, noise interleaves sharers and
    non-sharers in the observed order, so consecutive values alternate
    between 0.63 and 1.08 and the rank trend washes out (Spearman -0.05,
    although the planting was intact: kappa 0.63 within the quarter, 1.08
    across it). A neighbourhood mean measures the same thing with the
    variance a single pair cannot avoid.
    """
    K = kappa_matrix(x)
    order = np.argsort(-x.mean(axis=1))
    J = len(order)
    out = []
    for r in range(J):
        lo, hi = max(0, r - window), min(J, r + window + 1)
        vals = [K[order[r], order[t]] for t in range(lo, hi) if t != r]
        out.append(float(np.nanmean(vals)))
    return np.array(out)


def planted(top_only, rng, J=80, n=300, load=0.85):
    """Sharing probability decays with rank (top_only) or is uniform."""
    ability = np.sort(rng.normal(0.4, 0.06, J))[::-1]
    if top_only:
        prob = 1.0 - np.arange(J) / J          # certain at the top, none at the bottom
    else:
        prob = np.full(J, 0.5)
    lab = rng.random(J) < prob
    # The shared vector must be centred: its sample mean (~0.026 at n = 300)
    # would otherwise shift every sharer's score by the same amount, bunching
    # the sharers at one end of the ranking and creating the very rank trend
    # the check is supposed to detect. The uniform control read -0.34 before
    # this line existed.
    base = rng.normal(0, 0.45, n)
    base -= base.mean()
    noise = rng.normal(0, 0.45, (J, n))
    noise -= noise.mean(axis=1, keepdims=True)
    resid = np.where(lab[:, None], load * base[None, :] + np.sqrt(1 - load ** 2) * noise, noise)
    return ability[:, None] + resid


def _check_top_only():
    rng = np.random.default_rng(SEED)
    k = adjacent_kappa(planted(True, rng))
    r = spearmanr(np.arange(len(k)), k).statistic
    return r > 0.3, f"lineage only at the top: Spearman(rank, kappa) {r:+.2f}"


def _check_uniform():
    rng = np.random.default_rng(SEED + 1)
    k = adjacent_kappa(planted(False, rng))
    r = spearmanr(np.arange(len(k)), k).statistic
    return abs(r) < 0.3, f"lineage spread over all ranks: Spearman(rank, kappa) {r:+.2f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_top_only(), _check_uniform()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("IS THE SHARPNESS AT THE TOP, OR EVERYWHERE? ADJACENT-PAIR KAPPA BY RANK")
    p("=" * 84)
    p(f"  {'leaderboard':<22} {'J':>4} {'Spearman':>9} {'p':>6} {'top decile':>11} {'bottom decile':>14} "
      f"{'rank at 0.95':>13}")
    rises, lower = 0, 0
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        k = adjacent_kappa(x)
        m = np.isfinite(k)
        r = spearmanr(np.arange(len(k))[m], k[m])
        d = max(2, len(k) // 10)
        top, bot = float(np.nanmean(k[:d])), float(np.nanmean(k[-d:]))
        # first rank where a running mean of five adjacent kappas reaches 0.95
        run = pd.Series(k).rolling(5, min_periods=3).mean().to_numpy()
        hit = np.flatnonzero(run >= 0.95)
        depth = int(hit[0]) + 1 if hit.size else -1
        rises += r.statistic > 0
        lower += top < bot
        p(f"  {name:<22} {x.shape[0]:>4} {r.statistic:>+9.2f} {r.pvalue:>6.3f} {top:>11.3f} {bot:>14.3f} "
          f"{(str(depth) if depth > 0 else 'never'):>13}")
    N = sum(1 for _, pth in MATRICES.items() if Path(pth).exists())
    p("")
    p(f"  adjacent kappa rises with rank: {rises}/{N} (pre-registered >= 6)")
    p(f"  top decile sharper than bottom decile: {lower}/{N} (pre-registered >= 7)")
    p("")
    p("  'rank at 0.95' is the first rank where a five-pair running mean of")
    p("  adjacent kappa reaches 0.95 - below it the board behaves like a field")
    p("  of independent entrants, above it neighbours share item behaviour.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("kappa_generality_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote kappa_generality_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
