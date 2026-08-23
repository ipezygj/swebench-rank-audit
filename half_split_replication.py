"""If the benchmark had used a different half of its items, what would change?

The plainest replication question a leaderboard can be asked. Split the
items at random into halves A and B. Rank on A, rank on B, and compare:

  agreement      Spearman between the two rankings
  contradicted   of the pairs A establishes (simultaneous test on half A),
                 the share B orders the OTHER WAY at all - not merely
                 fails to establish, but reverses
  flipped top    how often the two halves name a different first place

Repeated over 20 random splits. This is not the bootstrap: both halves are
real items the benchmark already owns, so it answers "would our own data,
split differently, have said the same thing".

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * fewer than 5 % of A-established pairs are reversed in B on >= 7 of 9
    boards - the simultaneous test's whole point is that it does not
    establish what a resample would reverse;
  * Spearman between halves above 0.90 on >= 7 of 9;
  * the two halves name a different first place on more than a third of
    splits on >= 4 of 9 boards - the top is where the disagreement lives.

SELF-CHECKS
  * on a matrix with a clear ranking and low noise, contradictions are zero
    and the top never flips;
  * on pure noise, the top flips on nearly every split.

    python half_split_replication.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import rank_sets as rs
from entropy_law_test import MATRICES

SEED = 20260823
SPLITS = 20
DRAWS = 500


def one_split(x, rng, draws=DRAWS):
    J, n = x.shape
    perm = rng.permutation(n)
    A, B = perm[: n // 2], perm[n // 2:]
    ra = rs.rank_sets(x[:, A], draws=draws, seed=int(rng.integers(1 << 31)))
    rb = rs.rank_sets(x[:, B], draws=draws, seed=int(rng.integers(1 << 31)))
    ba, bb = ra["beats"], rb["beats"]
    est_a = int(ba.sum())
    contradicted = int((ba & bb.T).sum())          # A says i>j, B says j>i
    sa, sb = x[:, A].mean(axis=1), x[:, B].mean(axis=1)
    rho = spearmanr(sa, sb).statistic
    flip = int(np.argmax(sa)) != int(np.argmax(sb))
    return est_a, contradicted, rho, flip


def _check_clear():
    rng = np.random.default_rng(3)
    J, n = 20, 300
    x = np.linspace(0.2, 0.8, J)[:, None] + rng.normal(0, 0.05, (J, n))
    tot, con, rho, flips = 0, 0, [], 0
    for s in range(5):
        e, c, r, f = one_split(x, np.random.default_rng(10 + s), 300)
        tot += e; con += c; rho.append(r); flips += f
    return con == 0 and flips == 0, \
        f"clear ranking, low noise: {con} contradictions of {tot}, {flips} top flips, Spearman {np.mean(rho):.3f}"


def _check_noise():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.4, (20, 300))
    flips = sum(one_split(x, np.random.default_rng(20 + s), 300)[3] for s in range(10))
    return flips >= 8, f"pure noise: top flips on {flips} of 10 splits"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_clear(), _check_noise()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WOULD A DIFFERENT HALF OF THE ITEMS HAVE SAID THE SAME THING?")
    p("=" * 86)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>5} {'estab in A':>11} {'reversed in B':>14} "
      f"{'share':>7} {'Spearman':>9} {'top flips':>10}")
    lowc, agree, flippy = 0, 0, 0
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        rng = np.random.default_rng(SEED)
        tot, con, rhos, flips = 0, 0, [], 0
        for s in range(SPLITS):
            e, c, r, f = one_split(x, rng)
            tot += e; con += c; rhos.append(r); flips += f
        share = con / tot if tot else float("nan")
        rho = float(np.mean(rhos))
        fl = flips / SPLITS
        lowc += share < 0.05
        agree += rho > 0.90
        flippy += fl > 1 / 3
        p(f"  {name:<22} {x.shape[0]:>4} {x.shape[1]:>5} {tot / SPLITS:>11.0f} {con / SPLITS:>14.1f} "
          f"{100 * share:>6.2f}% {rho:>9.3f} {100 * fl:>9.0f}%")
    N = sum(1 for _, pth in MATRICES.items() if Path(pth).exists())
    p("")
    p(f"  fewer than 5 % of established pairs reversed: {lowc}/{N} (pre-registered >= 7)")
    p(f"  Spearman between halves above 0.90: {agree}/{N} (pre-registered >= 7)")
    p(f"  the two halves disagree on first place on more than a third of splits: "
      f"{flippy}/{N} (pre-registered >= 4)")
    p("")
    p("  Each half is half the items, so its intervals are wider than the full")
    p("  board's - the reversal share is therefore a conservative reading of how")
    p("  often the full board would contradict itself. 'top flips' asks only who")
    p("  is first, with no interval at all, which is how a headline reads it.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("half_split_replication_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote half_split_replication_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
