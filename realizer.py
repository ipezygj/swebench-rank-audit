"""If not bands, then what? A report card that is exact.

Twelve iterations established what the band summary costs and never said what to
print instead. The one constructive proposal - choose six systems and print
their verdicts - was withdrawn when it failed a split of the items. This is the
replacement, and it selects no systems at all.

A band is two numbers per system and it RELAXES: it admits orderings the data
excludes, 1.3 to 4.8 bits of them, growing with J. The classical alternative is
a REALIZER. Print k orderings and let the reader deduce

    i beats j  <=>  i is ahead of j in ALL k orderings

The relation that rule implies is the intersection of the k orderings. Every one
of them is a linear extension of the measured poset, so the intersection always
CONTAINS it: a realizer errs by over-claiming, asserting pairs as ordered that
the data leaves open, where a band errs by under-claiming. At k equal to the
poset's dimension the intersection IS the poset and the summary is exact - no
slack in either direction, and nothing selected that a second sample could
unselect.

So the question is what a leaderboard's dimension is. Two numbers per system buy
a band; the same two numbers buy a 2-realizer, and the comparison is fair
because the printed budget is identical.

Both errors are measured in the same unit: the log2 count of orderings the
printed summary admits, against the log2 count the poset admits. The band is
above it. A realizer short of full dimension is below it. Zero is exact.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  a realizer of at most 6 orderings suffices on all 8 boards, so a complete
      and exact report card costs at most 6 numbers per system.
  P2  at k = 2 - the same printed budget as a band - the realizer's error is
      smaller in absolute bits than the band's, on at least 6 of 8 boards.
  P3  the error falls monotonically in k and is exactly 0 at the realizer's own
      size, on all 8 boards.
  P4  a standard example S_3 is present on at least 6 of 8 boards - three
      systems and three others, each of the first below each of the second
      except its own partner - which proves the dimension is at least 3 there
      and that no two numbers per system can be exact.

  What a miss on P1 would mean: exactness costs more printed table than a
  report card can carry, and the honest recommendation is to keep bands and
  state their cost rather than to replace them.

SELF-CHECKS (no table if any fails)
  * every ordering produced must be a linear extension of the poset: each of
    its edges respected, checked, not assumed;
  * each partial intersection must CONTAIN the poset - a realizer that lost an
    edge would be summarising something else;
  * the ordering count must never exceed the poset's own, at any k;
  * at the realizer's full size the intersection must equal the poset cell for
    cell, or it is not a realizer;
  * at least 8 boards.

    python realizer.py
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from band_slack import band_matrix, bands_of, permanent01
from entropy_law_test import MATRICES
from exact_extensions import exact_log2

SEED = 20260825
SUB = 18
TRIES = 400          # candidate orderings considered per realizer step
MAXK = 12


def random_extension(beats: np.ndarray, rng, bias: np.ndarray | None = None):
    """A uniform-ish linear extension, optionally biased toward a target order.

    bias[i] is a preference score; among the currently minimal elements the one
    with the smallest bias is taken. With bias None the choice is uniform, which
    is Knuth's sampler.
    """
    J = beats.shape[0]
    indeg = beats.sum(axis=0).astype(int)
    placed = np.zeros(J, dtype=bool)
    out = []
    for _ in range(J):
        avail = np.flatnonzero((indeg == 0) & ~placed)
        if bias is None:
            k = int(avail[rng.integers(len(avail))])
        else:
            k = int(avail[np.argmin(bias[avail])])
        out.append(k)
        placed[k] = True
        indeg -= beats[k].astype(int)
        indeg[k] = 1 << 30
    return out


def order_to_relation(order) -> np.ndarray:
    J = len(order)
    pos = np.empty(J, dtype=int)
    pos[np.asarray(order)] = np.arange(J)
    return pos[:, None] < pos[None, :]


def build_realizer(beats: np.ndarray, rng, maxk=MAXK, tries=TRIES):
    """Orderings whose intersection shrinks to the poset. Greedy set cover.

    An incomparable pair {i, j} needs one ordering with i first and another with
    j first; until both exist the intersection still asserts an order the data
    does not support. Each step takes the candidate ordering that covers the
    most still-uncovered directions.
    """
    J = beats.shape[0]
    inc = ~(beats | beats.T) & ~np.eye(J, dtype=bool)
    need = inc.copy()                      # need[i, j]: some ordering must put i after j
    chosen = []
    for _ in range(maxk):
        if not need.any():
            break
        best, best_gain = None, -1
        for t in range(tries):
            b = None if t % 2 else rng.random(J)
            o = random_extension(beats, rng, b)
            R = order_to_relation(o)
            gain = int((need & R.T).sum())
            if gain > best_gain:
                best, best_gain = o, gain
        chosen.append(best)
        need &= ~order_to_relation(best).T
        if best_gain == 0:
            break
    return chosen


def intersection(orders, J) -> np.ndarray:
    Q = np.ones((J, J), dtype=bool) & ~np.eye(J, dtype=bool)
    for o in orders:
        Q &= order_to_relation(o)
    return Q


def find_s3(beats: np.ndarray):
    """A standard example S_3: a1..a3, b1..b3 with a_i < b_j exactly when i != j.

    Its presence proves the dimension is at least 3: no two linear orders can
    reverse all six incomparable pairs a_i, b_i at once.
    """
    J = beats.shape[0]
    for A in itertools.combinations(range(J), 3):
        for B in itertools.combinations(range(J), 3):
            if set(A) & set(B):
                continue
            for perm in itertools.permutations(range(3)):
                ok = True
                for i in range(3):
                    for j in range(3):
                        want = (i != j)
                        if bool(beats[A[i], B[perm[j]]]) != want:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return tuple(A), tuple(B[p] for p in perm)
    return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    rows, skipped = [], []
    ok_ext = ok_contain = ok_le = ok_exact = True
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        df = pd.read_csv(path, index_col=0).dropna(axis=0)
        x = df.to_numpy(dtype=float)
        J = x.shape[0]
        if J < SUB:
            skipped.append((name, J))
            continue
        rng = np.random.default_rng(SEED + J)
        beats = rs.rank_sets(x)["beats"]
        pick = np.sort(np.random.default_rng(SEED + J).choice(J, SUB, replace=False))
        s = beats[np.ix_(pick, pick)]
        labels = [list(df.index)[i] for i in pick]

        best, worst = bands_of(s)
        B = permanent01(band_matrix(best, worst))
        e = exact_log2(s)[0]
        band_err = math.log2(B) - math.log2(e)

        orders = build_realizer(s, rng)
        for o in orders:
            R = order_to_relation(o)
            if (s & ~R).any():
                ok_ext = False

        curve = []
        for k in range(1, len(orders) + 1):
            Q = intersection(orders[:k], SUB)
            if (s & ~Q).any():
                ok_contain = False
            cq = exact_log2(Q)[0]
            if cq > e:
                ok_le = False
            curve.append(math.log2(e) - math.log2(cq))
        if len(orders) and not np.array_equal(intersection(orders, SUB), s):
            ok_exact = False

        w = find_s3(s)
        rows.append({"name": name, "J": J, "k": len(orders), "band": band_err,
                     "curve": curve, "s3": None if w is None else
                     ([labels[i] for i in w[0]], [labels[i] for i in w[1]])})
        print(f"  {name:<22} realizer {len(orders)} orderings, band error "
              f"{band_err:5.2f} bits, S_3 {'yes' if w else 'no'}")

    print("self-checks ...")
    print(f"  [{'ok  ' if ok_ext else 'FAIL'}] every ordering is a linear extension "
          f"of its poset")
    print(f"  [{'ok  ' if ok_contain else 'FAIL'}] every partial intersection contains "
          f"the poset")
    print(f"  [{'ok  ' if ok_le else 'FAIL'}] no intersection admits more orderings "
          f"than the poset")
    print(f"  [{'ok  ' if ok_exact else 'FAIL'}] at full size the intersection equals "
          f"the poset cell for cell")
    ok_n = len(rows) >= 8
    print(f"  [{'ok  ' if ok_n else 'FAIL'}] {len(rows)} boards (need >= 8)")

    if not (ok_ext and ok_contain and ok_le and ok_exact and ok_n):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("A REPORT CARD THAT IS EXACT: PRINT ORDERINGS, NOT BANDS")
    p("=" * 100)
    p("  Print k orderings; the reader takes i to beat j when i is ahead in all")
    p("  k. A band is 2 numbers per system and admits orderings the data")
    p("  excludes. A realizer of k orderings is k numbers per system and, short")
    p("  of full dimension, excludes orderings the data admits. Both errors are")
    p(f"  in bits of log2 ordering count. Sub-posets of {SUB} systems.")
    if skipped:
        p("  Not measured: " + ", ".join(f"{a} (J={b})" for a, b in skipped) + ".")
    p("")
    kmax = max(len(r["curve"]) for r in rows)
    p(f"  {'board':<22}{'band (2 nums)':>15}"
      + "".join(f"{'k=' + str(k):>8}" for k in range(1, kmax + 1)) + f"{'exact at':>10}")
    for r in rows:
        cells = "".join(f"{-v:>8.2f}" for v in r["curve"])
        pad = "".join(f"{'':>8}" for _ in range(kmax - len(r["curve"])))
        p(f"  {r['name']:<22}{r['band']:>+15.2f}{cells}{pad}{r['k']:>10}")
    p("")
    p("  Positive is invented freedom, negative is invented precision, zero is")
    p("  exact. The band column is what a report card prints today.")
    p("")
    k6 = sum(1 for r in rows if r["k"] <= 6)
    k2 = sum(1 for r in rows if len(r["curve"]) >= 2
             and abs(r["curve"][1]) < abs(r["band"]))
    mono = sum(1 for r in rows if all(a >= b - 1e-9 for a, b in
                                      zip(r["curve"], r["curve"][1:]))
               and abs(r["curve"][-1]) < 1e-9)
    ns3 = sum(1 for r in rows if r["s3"])
    p(f"  P1  realizer of at most 6 orderings on {k6} of {len(rows)}      "
      f"pre-registered = all:  {'HIT' if k6 == len(rows) else 'MISS'}")
    p(f"  P2  at k=2, same budget as a band, smaller error on {k2} of {len(rows)}  "
      f"pre-registered >= 6:  {'HIT' if k2 >= 6 else 'MISS'}")
    p(f"  P3  error monotone in k and exactly 0 at full size on {mono} of {len(rows)}  "
      f"pre-registered = all:  {'HIT' if mono == len(rows) else 'MISS'}")
    p(f"  P4  a standard example S_3 present on {ns3} of {len(rows)}       "
      f"pre-registered >= 6:  {'HIT' if ns3 >= 6 else 'MISS'}")
    p("")
    p("  S_3 WITNESSES. Three systems and three others, each of the first below")
    p("  each of the second except its own partner. Two orderings cannot reverse")
    p("  all three partner pairs at once, so where this appears the dimension is")
    p("  at least 3 and two numbers per system - a band, or any pair of columns -")
    p("  cannot be exact.")
    for r in rows:
        if r["s3"]:
            a, b = r["s3"]
            p(f"    {r['name']}")
            p(f"      below: {', '.join(str(t)[:28] for t in a)}")
            p(f"      above: {', '.join(str(t)[:28] for t in b)}")
    p("")
    p("  The realizer size printed here is an UPPER bound on the dimension: it")
    p("  is what a greedy cover found, not a proof of minimality. The S_3")
    p("  witnesses are the matching lower bound. Where the two meet the")
    p("  dimension is pinned; where they do not, the honest statement is the")
    p("  interval between them.")
    p("")
    p("  Nothing here selects a system. The failure of the six-system")
    p("  recommendation was that a different sample of items chose different")
    p("  six; a realizer is derived from the whole relation and has no such")
    p("  handle to be brittle at.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("realizer_results.txt").write_text(text + chr(10), encoding="utf-8",
                                            newline=chr(10))
    print(chr(10) + "wrote realizer_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
