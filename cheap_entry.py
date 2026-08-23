"""How many items does a NEW entrant have to run to be placed?

minimal_benchmark.py asked how few items a board needs to reproduce its own
orderings. This asks the question a newcomer faces: the board already has J
systems with full scores; I can afford k items; how precisely can the board
place me?

Procedure: hold out one system, treat it as the newcomer, give it a random
subset of k items, and compute its rank set against the existing field -
using each pair's own overlap, which for the newcomer is only those k
items. Repeat over systems and subsets.

The answer is a curve: rank-set width against k. Its shape decides whether
cheap entry is possible, and the fraction of full-board width at k / n =
0.25 is the number a benchmark owner would quote in a submission policy.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * at 25 % of items the newcomer's rank set is less than twice the width
    it gets with all items, on >= 7 of 9 boards;
  * at 10 % it is more than three times wider on >= 5 of 9;
  * the width scales roughly as 1 / sqrt(k): the ratio between the 25 % and
    100 % widths should be near 2 on most boards, since sqrt(4) = 2.

SELF-CHECKS
  * at k = n the procedure must reproduce the full-board rank set of the
    held-out system exactly;
  * on a board where the newcomer is far above everyone, its rank set is
    [1, 1] at every k.

    python cheap_entry.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from entropy_law_test import MATRICES

SEED = 20260823
DRAWS = 600
FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)
SYSTEMS = 12          # how many hold-out systems to average over
SUBSETS = 4           # random item subsets per system per fraction


def newcomer_width(x, j, cols, draws=DRAWS, seed=SEED):
    """Rank set of system j when it has only `cols`, others have everything."""
    J, n = x.shape
    others = [i for i in range(J) if i != j]
    theta = x.mean(axis=1)
    tj = float(x[j, cols].mean())
    # pairwise SE against each other system uses the newcomer's items only
    rng = np.random.default_rng(seed)
    k = len(cols)
    d = x[np.ix_(others, cols)] - x[j, cols][None, :]
    se = d.std(axis=1, ddof=1) / math.sqrt(k)
    se = np.where(se > 0, se, np.inf)
    gap = theta[others] - tj
    # simultaneous critical value over the J-1 comparisons, multiplier bootstrap
    u = d - d.mean(axis=1, keepdims=True)
    S = u @ rng.standard_normal((k, draws)) / (math.sqrt(k) * se[:, None] * math.sqrt(k))
    crit = float(np.quantile(np.max(np.abs(S), axis=0), 0.95))
    above = int(np.sum(gap - crit * se > 0))
    below = int(np.sum(-gap - crit * se > 0))
    return (J - below) - (1 + above) + 1        # worst - best + 1


def _check_full():
    rng = np.random.default_rng(3)
    x = 0.5 + rng.normal(0, 0.08, 20)[:, None] + rng.normal(0, 0.4, (20, 200))
    r = rs.rank_sets(x, draws=DRAWS)
    j = 5
    full = int(r["worst"][j] - r["best"][j] + 1)
    got = newcomer_width(x, j, np.arange(200))
    return abs(got - full) <= 2, f"k = n reproduces the full rank set: {got} vs {full}"


def _check_dominant():
    rng = np.random.default_rng(5)
    x = 0.4 + rng.normal(0, 0.03, 15)[:, None] + rng.normal(0, 0.2, (15, 200))
    x[0] += 0.6
    widths = [newcomer_width(x, 0, np.random.default_rng(10 + s).choice(200, 40, replace=False)) for s in range(3)]
    return max(widths) == 1, f"dominant newcomer with 40 items: widths {widths}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_full(), _check_dominant()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHAT A NEWCOMER GETS FOR ITS ITEM BUDGET")
    p("=" * 84)
    p(f"  {'leaderboard':<22} {'n':>5} " + " ".join(f"{int(100 * f):>7}%" for f in FRACTIONS)
      + f" {'25 % / full':>12} {'10 % / full':>12}")
    ok25, bad10, near2 = 0, 0, 0
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J, n = x.shape
        rng = np.random.default_rng(SEED)
        picks = rng.choice(J, min(SYSTEMS, J), replace=False)
        means = {}
        for f in FRACTIONS:
            k = max(4, int(round(f * n)))
            ws = []
            for j in picks:
                for s in range(1 if f == 1.0 else SUBSETS):
                    cols = np.arange(n) if f == 1.0 else rng.choice(n, k, replace=False)
                    ws.append(newcomer_width(x, int(j), cols))
            means[f] = float(np.mean(ws))
        r25 = means[0.25] / means[1.00] if means[1.00] else float("nan")
        r10 = means[0.10] / means[1.00] if means[1.00] else float("nan")
        ok25 += r25 < 2.0
        bad10 += r10 > 3.0
        near2 += 1.5 <= r25 <= 2.5
        p(f"  {name:<22} {n:>5} " + " ".join(f"{means[f]:>8.1f}" for f in FRACTIONS)
          + f" {r25:>12.2f} {r10:>12.2f}")
    N = sum(1 for _, pth in MATRICES.items() if Path(pth).exists())
    p("")
    p(f"  25 % of items keeps the width under 2x: {ok25}/{N} (pre-registered >= 7)")
    p(f"  10 % of items costs more than 3x: {bad10}/{N} (pre-registered >= 5)")
    p(f"  25 % ratio between 1.5 and 2.5, as 1/sqrt(k) predicts: {near2}/{N}")
    p("")
    p("  The newcomer's comparisons use only its own items; the incumbents keep")
    p("  their full scores. Width is worst rank minus best rank plus one, averaged")
    p(f"  over {SYSTEMS} held-out systems and {SUBSETS} random subsets each. A board whose")
    p("  ratio at 25 % is near 1 can accept cheap submissions without losing")
    p("  anything; one whose ratio is 4 cannot.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("cheap_entry_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote cheap_entry_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
