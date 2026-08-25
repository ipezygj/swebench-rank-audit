"""Is the targeted six a property of the systems, or of the noise?

top_verdicts.py found that a report card's top six rows are the worst place to
print pairwise verdicts - they recover none of the ordering freedom the bands
invent - while six systems chosen to remove the most orderings recover 24 to
56 %. That choice was made on the same poset it was scored against. For a
benchmark owner that is legitimate: they hold the poset and can run the
selection. It still leaves the question of what was selected. A choice that only
works on the items it was made from has found the noise.

This splits the items. The six systems are chosen on one half of a board's
items and scored on the poset built from the OTHER half - a different relation,
different bands, different slack. Three references on the same second half: the
top six, six at random, and the six chosen on that half, which is the ceiling
nothing out-of-sample can beat.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  the six chosen on half A recover more of half B's slack than six chosen
      at random, on at least 6 of 8 boards.
  P2  they also beat the top six by half B's own ranking, on at least 6 of 8.
      This is the weaker claim of the two - the top six recovered nothing
      in-sample and there is no reason for it to do better out of sample.
  P3  they recover at least half of what the oracle six - chosen on half B
      itself - recover, on at least 5 of 8 boards.

  What a miss on P1 would mean: the selection is fitting the particular set of
  items, the recommendation in top_verdicts.py does not survive contact with a
  second sample, and it should be withdrawn rather than qualified.

SELF-CHECKS (no table if any fails)
  * the halves must be disjoint and together cover every item, counted;
  * the two halves must produce DIFFERENT posets, or nothing is being tested
    out of sample; the number of differing cells is printed per board;
  * half B's slack must be positive, or there is nothing to recover and the
    board cannot contribute - boards where it is zero are named and excluded
    from the counts rather than scored as failures;
  * at least 8 boards measured.

    python targeted_split.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from band_slack import band_matrix, bands_of, permanent01
from entropy_law_test import MATRICES
from exact_extensions import exact_log2
from top_verdicts import count_with, restrict

SEED = 20260825
SUB = 18
K = 6
RAND_REPS = 25


def parts(beats: np.ndarray):
    best, worst = bands_of(beats)
    M = band_matrix(best, worst)
    e = exact_log2(beats)[0]
    b = permanent01(M)
    return M, e, b, math.log2(b) - math.log2(e), best, worst


def greedy(beats, M, e, k):
    """The k systems whose printed verdicts remove the most orderings."""
    chosen: list[int] = []
    for _ in range(k):
        chosen.append(min((v for v in range(beats.shape[0]) if v not in chosen),
                          key=lambda v: count_with(
                              M, restrict(beats, np.sort(np.array(chosen + [v]))))))
    return np.sort(np.array(chosen))


def recovered(beats, M, e, slack, pick):
    if slack <= 0:
        return float("nan")
    c = count_with(M, restrict(beats, np.sort(np.asarray(pick))))
    return (slack - (math.log2(c) - math.log2(e))) / slack


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    rows, skipped, notes = [], [], []
    ok_cover = True
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J, n = x.shape
        if J < SUB or n < 8:
            skipped.append((name, J, n))
            continue
        rng = np.random.default_rng(SEED + J + n)
        perm = rng.permutation(n)
        A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
        if len(set(A.tolist()) & set(B.tolist())) or len(A) + len(B) != n:
            ok_cover = False

        pick = np.sort(np.random.default_rng(SEED + J).choice(J, SUB, replace=False))
        bA = rs.rank_sets(x[:, A])["beats"][np.ix_(pick, pick)]
        bB = rs.rank_sets(x[:, B])["beats"][np.ix_(pick, pick)]
        differ = int((bA != bB).sum())

        MA, eA, _, slA, _, _ = parts(bA)
        MB, eB, _, slB, bestB, worstB = parts(bB)

        if slB <= 0:
            notes.append((name, "half B has no slack to recover"))
            continue

        selA = greedy(bA, MA, eA, K)
        selB = greedy(bB, MB, eB, K)
        topB = np.sort(np.argsort(bestB * 100 + worstB, kind="stable")[:K])
        r = np.random.default_rng(SEED + 5 + J)
        rnd = float(np.mean([recovered(bB, MB, eB, slB,
                                       r.choice(SUB, K, replace=False))
                             for _ in range(RAND_REPS)]))
        rows.append({"name": name, "J": J, "n": n, "differ": differ, "slB": slB,
                     "outA": recovered(bB, MB, eB, slB, selA),
                     "oracle": recovered(bB, MB, eB, slB, selB),
                     "top": recovered(bB, MB, eB, slB, topB),
                     "rand": rnd})
        print(f"  {name:<22} out-of-sample {100 * rows[-1]['outA']:5.0f}%  "
              f"oracle {100 * rows[-1]['oracle']:5.0f}%")

    print("self-checks ...")
    print(f"  [{'ok  ' if ok_cover else 'FAIL'}] the halves are disjoint and cover "
          f"every item on every board")
    ok_diff = all(r["differ"] > 0 for r in rows)
    print(f"  [{'ok  ' if ok_diff else 'FAIL'}] the two halves give different posets "
          f"on {sum(1 for r in rows if r['differ'] > 0)} of {len(rows)}")
    ok_slack = all(r["slB"] > 0 for r in rows)
    print(f"  [{'ok  ' if ok_slack else 'FAIL'}] half B has slack to recover on every "
          f"scored board ({len(notes)} board(s) excluded and named)")
    ok_n = len(rows) >= 8
    print(f"  [{'ok  ' if ok_n else 'FAIL'}] {len(rows)} boards scored (need >= 8)")

    if not (ok_cover and ok_diff and ok_slack and ok_n):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("DOES THE TARGETED SIX SURVIVE A SPLIT OF THE ITEMS?")
    p("=" * 100)
    p(f"  Six systems chosen on half of a board's items, scored on the poset")
    p(f"  built from the other half. {SUB}-system sub-posets throughout.")
    if skipped:
        p("  Not measured: " + ", ".join(f"{a} (J={b}, n={c})" for a, b, c in skipped) + ".")
    for a, b in notes:
        p(f"  Excluded: {a} - {b}.")
    p("")
    p(f"  {'board':<22}{'J':>5}{'n':>7}{'cells differ':>14}{'B slack':>10}"
      f"{'chosen on A':>13}{'oracle on B':>13}{'top 6':>8}{'random':>9}")
    for r in rows:
        p(f"  {r['name']:<22}{r['J']:>5}{r['n']:>7}{r['differ']:>14}{r['slB']:>10.2f}"
          f"{100 * r['outA']:>12.0f}%{100 * r['oracle']:>12.0f}%"
          f"{100 * r['top']:>7.0f}%{100 * r['rand']:>8.0f}%")
    p("")
    n1 = sum(1 for r in rows if r["outA"] > r["rand"])
    n2 = sum(1 for r in rows if r["outA"] > r["top"])
    n3 = sum(1 for r in rows if r["oracle"] > 0 and r["outA"] >= 0.5 * r["oracle"])
    p(f"  P1  beats a random six on {n1} of {len(rows)}                   "
      f"pre-registered >= 6:  {'HIT' if n1 >= 6 else 'MISS'}")
    p(f"  P2  beats the top six on {n2} of {len(rows)}                    "
      f"pre-registered >= 6:  {'HIT' if n2 >= 6 else 'MISS'}")
    p(f"  P3  reaches half the oracle on {n3} of {len(rows)}              "
      f"pre-registered >= 5:  {'HIT' if n3 >= 5 else 'MISS'}")
    p("")
    p("  The oracle column is the same selection run on the half it is scored")
    p("  on. It is not a competitor - nothing chosen out of sample can beat it -")
    p("  it is the ceiling, and the gap between it and the out-of-sample column")
    p("  is what the split costs.")
    p("")
    p("  The cells-differ column is the check that this is a real test. The two")
    p("  halves are built from disjoint items, so they give different relations")
    p("  on the same systems; if they did not, choosing on one and scoring on")
    p("  the other would be choosing and scoring on the same thing.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("targeted_split_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                  newline=chr(10))
    print(chr(10) + "wrote targeted_split_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
