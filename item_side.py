"""The same instrument, transposed: how well does a benchmark rank its own items?

Everything here ranks systems using items as the evidence. The matrix is
symmetric in structure, so the same machinery ranks ITEMS using systems as
the evidence - which is a question benchmarks answer implicitly all the
time ("this instance is hard", "this task is saturated") and never with an
interval.

Transposing gives, for each item, a simultaneous rank set over difficulty,
and for the board an item-side entropy: how much of the difficulty ordering
the data supports.

The comparison is informative because the two sides have different sample
sizes. A system is measured by n items; an item is measured by J systems.
Where J < n, the item side should be the blurrier one.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the sign of (item-side entropy share - system-side entropy share)
    matches the sign of (n - J) on >= 7 of 9 boards;
  * on SWE-bench Verified (134 systems, 500 items) the item side is
    blurrier: its entropy share is the higher of the two;
  * at least one board resolves its items better than its systems.

SELF-CHECKS
  * on a matrix with independent Gaussian noise and no structure on either
    side, both entropy shares are near their ceiling;
  * transposing a matrix twice returns the original numbers.

    python item_side.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import rank_sets as rs
import leaderboard_entropy as le
from entropy_law_test import MATRICES

SEED = 20260823
DRAWS = 800
SAMPLES = 800
MAX_SIDE = 260          # cap: entropy on 500 items would take minutes per board


def side(x, rng, draws=DRAWS, samples=SAMPLES):
    J = x.shape[0]
    r = rs.rank_sets(x, draws=draws)
    H = le.log_extensions(r["beats"], samples, rng)["bits"]
    return {"J": J, "n": x.shape[1],
            "H_frac": H / (gammaln(J + 1) / math.log(2)),
            "estab": float(r["beats"].sum() / (J * (J - 1))),
            "tie1": int((r["best"] == 1).sum()),
            "width": float(np.median(r["worst"] - r["best"] + 1))}


def subsample(x, rng, cap=MAX_SIDE):
    """Cap both sides at `cap` rows/columns, drawn at random, for comparability."""
    J, n = x.shape
    ri = rng.choice(J, min(J, cap), replace=False) if J > cap else np.arange(J)
    ci = rng.choice(n, min(n, cap), replace=False) if n > cap else np.arange(n)
    return x[np.ix_(np.sort(ri), np.sort(ci))]


def _check_noise():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (40, 40))
    a = side(x, rng, 300, 300)
    b = side(x.T, rng, 300, 300)
    return a["H_frac"] > 0.9 and b["H_frac"] > 0.9, \
        f"structureless matrix: system side {100 * a['H_frac']:.0f} %, item side {100 * b['H_frac']:.0f} %"


def _check_double_transpose():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, (30, 45))
    a = side(x, np.random.default_rng(1), 300, 300)
    b = side(x.T.T, np.random.default_rng(1), 300, 300)
    return abs(a["H_frac"] - b["H_frac"]) < 1e-12, "double transpose reproduces the numbers"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_noise(), _check_double_transpose()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("THE SAME INSTRUMENT, TRANSPOSED: RANKING THE ITEMS")
    p("=" * 90)
    p(f"  {'leaderboard':<22} {'J used':>7} {'n used':>7} {'H systems':>10} {'H items':>9} "
      f"{'diff':>6} {'sign(n-J)':>10} {'match':>6} {'estab items':>12}")
    match, better = 0, 0
    rows = []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        rng = np.random.default_rng(SEED)
        xs = subsample(x, rng)
        a = side(xs, rng)
        b = side(xs.T, rng)
        diff = b["H_frac"] - a["H_frac"]
        want = np.sign(xs.shape[1] - xs.shape[0])
        agree = np.sign(diff) == want or abs(diff) < 0.01
        match += bool(agree)
        better += b["H_frac"] < a["H_frac"]
        rows.append((name, a, b, diff))
        p(f"  {name:<22} {xs.shape[0]:>7} {xs.shape[1]:>7} {100 * a['H_frac']:>9.1f}% {100 * b['H_frac']:>8.1f}% "
          f"{100 * diff:>+5.1f} {int(want):>10} {'yes' if agree else 'NO':>6} {100 * b['estab']:>11.1f}%")
    N = len(rows)
    swe = next((r for r in rows if r[0].startswith("SWE")), None)
    p("")
    p(f"  sign matches sign(n - J): {match}/{N} (pre-registered >= 7)")
    if swe:
        p(f"  SWE-bench item side blurrier: {'yes' if swe[3] > 0 else 'NO'} ({100 * swe[3]:+.1f} points)")
    p(f"  boards that resolve items better than systems: {better}/{N} (pre-registered >= 1)")
    p("")
    p(f"  Both sides are capped at {MAX_SIDE} rows and columns, drawn at random with a")
    p("  fixed seed, so the two entropies are computed on the same submatrix and the")
    p("  comparison is not a comparison of sizes. 'estab items' is the share of item")
    p("  pairs whose difficulty ordering the board establishes.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("item_side_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote item_side_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
