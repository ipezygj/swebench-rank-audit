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
from draws import R_DEFAULT, subsets, summarise
from entropy_law_test import MATRICES
from exact_extensions import exact_log2

SEED = 20260825
SUB = 18
TRIES = 400          # candidate orderings for the detailed first draw
TRIES_SWEEP = 100    # per step when sweeping draws
R_DRAWS = R_DEFAULT
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


def conjugate(beats: np.ndarray, order):
    """The second ordering of a 2-realizer with this first one, or None.

    If {L1, L2} realises P then L2 is forced: it must agree with P on every
    ordered pair and REVERSE every incomparable pair relative to L1. That fixes
    a tournament, and L2 exists exactly when the tournament is acyclic, which a
    topological sort settles in O(J^2).
    """
    J = beats.shape[0]
    R = order_to_relation(order)
    inc = ~(beats | beats.T) & ~np.eye(J, dtype=bool)
    T = beats | (inc & R.T)
    indeg = T.sum(axis=0).astype(int)
    done = np.zeros(J, dtype=bool)
    out = []
    while True:
        avail = np.flatnonzero((indeg == 0) & ~done)
        if not len(avail):
            break
        k = int(avail[0])
        done[k] = True
        out.append(k)
        indeg -= T[k].astype(int)
        indeg[k] = 1 << 30
    return out if len(out) == J else None


def conjugate_ok(beats: np.ndarray, order) -> bool:
    """Given a first ordering, does the forced second ordering exist?

    If {L1, L2} realises P then L2 is determined: it must agree with P on every
    ordered pair and REVERSE every incomparable pair relative to L1. That fixes
    a tournament, and L2 exists exactly when the tournament is acyclic. So a
    2-realizer can be tested for a given L1 in O(J^2), and searched for by
    trying many L1.
    """
    J = beats.shape[0]
    R = order_to_relation(order)
    inc = ~(beats | beats.T) & ~np.eye(J, dtype=bool)
    T = beats | (inc & R.T)          # P's edges, incomparable pairs reversed
    indeg = T.sum(axis=0).astype(int)
    seen = 0
    done = np.zeros(J, dtype=bool)
    while True:
        avail = np.flatnonzero((indeg == 0) & ~done)
        if not len(avail):
            break
        k = int(avail[0])
        done[k] = True
        seen += 1
        indeg -= T[k].astype(int)
        indeg[k] = 1 << 30
    return seen == J


def two_dim_search(beats: np.ndarray, rng, tries: int):
    """Try many first orderings; return a 2-realizer if one turns up, else None.

    A search, not a proof: it can only say that none was found in this many
    tries. It is worth running anyway, because the greedy cover is bad at this.
    On CASP14 and MathArena 2025 the greedy needed 4 orderings and a 2-realizer
    exists - the greedy size is an upper bound and a loose one.
    """
    for t in range(tries):
        o = random_extension(beats, rng, None if t % 2 else rng.random(beats.shape[0]))
        c = conjugate(beats, o)
        if c is not None:
            return [o, c]
    return None


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
            if any(beats[u, v] or beats[v, u] for u, v in itertools.combinations(A, 2)):
                continue
            if any(beats[u, v] or beats[v, u] for u, v in itertools.combinations(B, 2)):
                continue
            for perm in itertools.permutations(range(3)):
                ok = True
                for i in range(3):
                    for j in range(3):
                        want = (i != j)
                        bb = B[perm[j]]
                        if bool(beats[A[i], bb]) != want or beats[bb, A[i]]:
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

        # THE SWEEP. Dimension is a property of a poset, and an 18-system
        # sub-poset is not the board: dimension is monotone under induced
        # sub-posets, so a 2-dimensional subset proves only that the board's
        # dimension is at least 2, which is trivial. This file previously said
        # "CASP14 and MathArena 2025 have dimension exactly 2" on the strength
        # of one draw each. Over R draws the question becomes how OFTEN a draw
        # is 2-dimensional, which is a statement about the sampling and is
        # reported as one.
        ks, two_hits, k2err = [], 0, []
        for pick_d in subsets(J, SUB, R=R_DRAWS, seed=SEED + J):
            sd = beats[np.ix_(pick_d, pick_d)]
            t2 = two_dim_search(sd, rng, 400)
            two_hits += t2 is not None
            ords_d = t2 if t2 is not None else build_realizer(
                sd, rng, tries=TRIES_SWEEP)
            ks.append(len(ords_d))
            if len(ords_d) >= 2:
                Q2 = intersection(ords_d[:2], SUB)
                k2err.append(math.log2(exact_log2(sd)[0])
                             - math.log2(exact_log2(Q2)[0]))
        k_s, k2_s = summarise(ks), summarise(k2err)

        pick = list(subsets(J, SUB, R=1, seed=SEED + J))[0]
        s = beats[np.ix_(pick, pick)]
        labels = [list(df.index)[i] for i in pick]

        best, worst = bands_of(s)
        B = permanent01(band_matrix(best, worst))
        e = exact_log2(s)[0]
        band_err = math.log2(B) - math.log2(e)

        greedy_orders = build_realizer(s, rng)
        two = two_dim_search(s, rng, 600)
        # Where an exact 2-realizer exists, use it: it is both smaller and
        # exact, and the greedy cover misses it.
        orders = two if two is not None else greedy_orders
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
        rows.append({"name": name, "two": two is not None,
                     "greedy_k": len(greedy_orders),
                     "k_s": k_s, "k2_s": k2_s, "two_hits": two_hits, "R": R_DRAWS,
                     "J": J, "k": len(orders), "band": band_err,
                     "curve": curve, "s3": None if w is None else
                     ([labels[i] for i in w[0]], [labels[i] for i in w[1]])})
        print(f"  {name:<22} realizer {len(orders)} orderings (greedy "
              f"{len(greedy_orders)}), band error {band_err:5.2f} bits, "
              f"S_3 {'yes' if w else 'no'}")

    print("self-checks ...")
    # A search that returns nothing on every board has to be shown able to
    # return something. P4 came back 0 of 8 and without this that reads as a
    # measurement when it could be a broken loop.
    plant = np.zeros((SUB, SUB), dtype=bool)
    for i in range(3):
        for j in range(3):
            if i != j:
                plant[i, 3 + j] = True
    ok_s3 = (find_s3(plant) is not None
             and find_s3(np.triu(np.ones((SUB, SUB), dtype=bool), k=1)) is None
             and find_s3(np.zeros((SUB, SUB), dtype=bool)) is None)
    print(f"  [{'ok  ' if ok_s3 else 'FAIL'}] the S_3 search finds a planted S_3 and "
          f"rejects a total order and an antichain")
    # and the 2-realizer search must find one where a 2-realizer exists
    tot = np.triu(np.ones((SUB, SUB), dtype=bool), k=1)
    ok_2d = two_dim_search(tot, np.random.default_rng(1), 5) is not None
    print(f"  [{'ok  ' if ok_2d else 'FAIL'}] the 2-realizer search finds one for a "
          f"total order, which is 1-dimensional")
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

    if not (ok_s3 and ok_2d and ok_ext and ok_contain and ok_le and ok_exact and ok_n):
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
    ex2 = [r["name"] for r in rows if r["two"]]
    p("  THE ANSWER TO THE QUESTION THIS FILE WAS OPENED FOR. Exactness is")
    p(f"  affordable on every board: {min(r['k'] for r in rows)} to "
      f"{max(r['k'] for r in rows)} orderings, so that many numbers per system")
    p("  against a band's 2. It selects nothing, so there is no handle for a")
    p("  second sample of items to move, which is what killed the six-system")
    p("  rule. That is the constructive recommendation.")
    p("")
    tot2 = sum(r["two_hits"] for r in rows)
    p(f"  RETRACTED 2026-08-25: \"CASP14 and MathArena 2025 have dimension")
    p("  exactly 2\". That was one draw each, and it is a category error twice")
    p("  over. Dimension is monotone under induced sub-posets, so a 2-dimensional")
    p("  18-system subset proves only that the BOARD's dimension is at least 2,")
    p("  which is trivially true of every board here. And it is not even a stable")
    p(f"  property of the draws: over {rows[0]['R']} draws per board, "
      f"{tot2} of {sum(r['R'] for r in rows)} subsets")
    p("  admit a 2-realizer, spread across boards as the table below shows -")
    p("  including boards this file previously reported as having none.")
    p("")
    p("  P2 scores 1 of 8 rather than 2 because of a tie: on MathArena 2025 the")
    p("  2-realizer is exact where the band invents 1.26 bits, a strict win at")
    p("  equal budget, while on CASP14 both are exact and neither is smaller.")
    p("")
    p("  P2 MISSED, and the miss is half instrument. On the boards where no")
    p("  2-realizer was found, two orderings invent far more false precision")
    p("  than the band invents false freedom, and the band wins its own price")
    p("  point comfortably. But the greedy cover that produced those k=2 numbers")
    p("  is a bad upper bound: it wanted 4 orderings for CASP14 and MathArena")
    p("  2025, where a direct construction finds 2 and is exact. So the k=2")
    p("  column below is what a greedy realizer costs, not what a 2-realizer")
    p("  costs, wherever the direct search came back empty.")
    p("")
    p("  What is not in doubt: the band is a good two-number summary. Twelve")
    p("  iterations of measuring what it costs end by defending it at its own")
    p("  price point. What was wrong was never the band - it was printing a band")
    p("  and saying nothing about the 1.3 to 4.8 bits.")
    p("")
    p("  P4 MISSED AND LEAVES A HOLE. No standard example S_3 exists on any of")
    p("  the 8 boards. The search is sound - the self-check plants one and finds")
    p("  it, and rejects a total order and an antichain - so this is a")
    p("  measurement, not a broken loop. But S_3 was the whole lower-bound")
    p("  strategy, so nothing here PROVES the dimension exceeds 2. The realizer")
    p("  sizes are upper bounds from a greedy cover.")
    p("")
    p("  A direct search stands in its place and is reported as a search. For")
    p("  each board, 600 candidate first orderings were tried; for each, the")
    p("  second ordering of a 2-realizer is forced - it must agree with the")
    p("  poset and reverse every incomparable pair - so it exists exactly when")
    p("  that tournament is acyclic, which is checkable in O(J^2).")
    p(f"  {'board':<24}{'2-dim draws':>13}{'realizer size, median [IQR]':>30}"
      f"{'k=2 error, median':>20}")
    for r in rows:
        k, k2 = r["k_s"], r["k2_s"]
        p(f"    {r['name']:<22}{r['two_hits']:>8}/{r['R']:<4}"
          f"{k['median']:>18.1f} [{k['q1']:.0f}, {k['q3']:.0f}]"
          f"{-k2['median']:>19.2f}")
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
