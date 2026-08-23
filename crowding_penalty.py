"""Is a leaderboard punished for being popular?

A simultaneous rank set controls the error rate over all J(J-1)/2 pairs, so
the critical value grows with J. Two boards with identical data quality
will differ in what they can say if one has more entrants - and entrants
arrive for reasons that have nothing to do with measurement.

This subsamples systems: keep a random J' of them, recompute, and watch
tie@1, the tie@1 share, and whether the top pair separates. The top pair is
kept in every subsample, so its comparison is the same data every time and
only the multiplicity correction changes.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * halving J cuts tie@1 by at least 25 % on >= 6 of 9 boards;
  * the tie@1 SHARE (tie@1 / J) moves by less than 10 points on >= 6 of 9 -
    the crowd grows in proportion, so the penalty is mostly bookkeeping;
  * on at least one board the top pair, which cannot be separated with all
    systems present, separates at J' = 10 - the multiplicity correction
    alone is decisive there.

SELF-CHECKS
  * at J' = J the subsample reproduces the full board's tie@1;
  * on a board with one dominant system, tie@1 is 1 at every J'.

    python crowding_penalty.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from entropy_law_test import MATRICES

SEED = 20260823
DRAWS = 800
REPS = 6
FRACS = (0.25, 0.50, 1.00)
SMALL = 10


def subsample_tie1(x, keep, rng, draws=DRAWS):
    """tie@1 and top-pair separability with `keep` systems, top two always in."""
    J = x.shape[0]
    order = np.argsort(-x.mean(axis=1))
    i1, i2 = int(order[0]), int(order[1])
    others = [int(j) for j in range(J) if j not in (i1, i2)]
    pick = [i1, i2] + list(rng.choice(others, max(keep - 2, 0), replace=False))
    pick = sorted(pick)
    sub = x[pick]
    r = rs.rank_sets(sub, draws=draws)
    loc = {p: k for k, p in enumerate(pick)}
    return int((r["best"] == 1).sum()), bool(r["beats"][loc[i1], loc[i2]]), len(pick)


def _check_full():
    rng = np.random.default_rng(3)
    x = 0.5 + rng.normal(0, 0.07, 30)[:, None] + rng.normal(0, 0.4, (30, 200))
    t_full = int((rs.rank_sets(x, draws=400)["best"] == 1).sum())
    t_sub, _, k = subsample_tie1(x, 30, rng, 400)
    return t_sub == t_full and k == 30, f"J' = J reproduces tie@1: {t_sub} vs {t_full}"


def _check_dominant():
    rng = np.random.default_rng(5)
    x = 0.4 + rng.normal(0, 0.02, 25)[:, None] + rng.normal(0, 0.25, (25, 200))
    x[0] += 0.5
    ts = [subsample_tie1(x, k, np.random.default_rng(10 + k), 400)[0] for k in (5, 10, 25)]
    return set(ts) == {1}, f"dominant system: tie@1 {ts} at J' = 5, 10, 25"


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
    p("IS A LEADERBOARD PUNISHED FOR BEING POPULAR?")
    p("=" * 92)
    p(f"  {'board':<22} {'J':>4} " + " ".join(f"{'tie@1 @' + str(int(100 * f)) + '%':>13}" for f in FRACS)
      + f" {'share 50 %':>11} {'share 100 %':>12} {'top pair at J=10':>17}")
    cut, stable, decisive = 0, 0, 0
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J = x.shape[0]
        vals = {}
        for f in FRACS:
            keep = max(4, int(round(f * J)))
            rng = np.random.default_rng(SEED)
            got = [subsample_tie1(x, keep, rng) for _ in range(1 if f == 1.0 else REPS)]
            vals[f] = (float(np.mean([g[0] for g in got])), keep)
        rng = np.random.default_rng(SEED + 1)
        small = [subsample_tie1(x, min(SMALL, J), rng) for _ in range(REPS)]
        sep_small = sum(s[1] for s in small)
        t50, t100 = vals[0.50][0], vals[1.00][0]
        cut += t50 <= 0.75 * t100
        sh50, sh100 = t50 / vals[0.50][1], t100 / vals[1.00][1]
        stable += abs(sh50 - sh100) < 0.10
        decisive += sep_small > 0
        p(f"  {name:<22} {J:>4} " + " ".join(f"{vals[f][0]:>13.1f}" for f in FRACS)
          + f" {100 * sh50:>10.0f}% {100 * sh100:>11.0f}% {f'{sep_small} of {REPS}':>17}")
    N = sum(1 for _, pth in MATRICES.items() if Path(pth).exists())
    p("")
    p(f"  halving J cuts tie@1 by at least 25 %: {cut}/{N} (pre-registered >= 6)")
    p(f"  the tie@1 share moves less than 10 points: {stable}/{N} (pre-registered >= 6)")
    p(f"  the top pair separates at J' = 10 on: {decisive}/{N} boards (pre-registered >= 1)")
    p("")
    p("  The top two systems are kept in every subsample, so their comparison uses")
    p("  the same items and the same scores throughout; only the number of other")
    p("  systems changes. Where the pair separates at J' = 10 and not at full J,")
    p("  the multiplicity correction - not the data - is what stands between the")
    p("  board and naming a winner.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("crowding_penalty_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote crowding_penalty_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
