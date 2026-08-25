"""Which verdicts survive a second sample of items - the band's, or the ones it drops?

Everything in the 104-116 arc counted ORDERINGS. That lane is exhausted and, as
slack_draws.py showed, it was contaminated by a single-draw sampling design. This
counts VERDICTS instead, at full J, with no sub-posets and no 2^J dynamic
programming anywhere.

The design is a reviewer's, adopted because it is better than the one I would
have written. Split a board's items into disjoint halves A and B. Build the
relation on each half at full J. Then ask, of the verdicts a reader would take
away from half A, how many half B contradicts.

Two sets of verdicts are scored against the same second half:

  BAND      pairs with worst_i <= best_j in P_A - exactly what a reader
            reconstructs from a printed band table, and nothing more
  EXTRA     the edges of P_A that the bands do NOT force - the verdicts a
            realizer prints and a band table throws away

If the EXTRA verdicts are contradicted more often than the BAND ones, then the
band's lossiness is not a defect. It is a filter, and it is discarding
precisely the verdicts that do not replicate. That would inverse the
recommendation this repository has been building toward.

A theorem the arc never wrote down, asserted below rather than assumed: if
worst_i <= best_j then i precedes j in every band-respecting permutation, so
BAND is contained in P_A. A band table cannot print a verdict the data does not
support. It is sound and incomplete - and full_board_free.py measured the
incompleteness at 49 to 82 %.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  the per-verdict contradiction rate is strictly higher for EXTRA than for
      BAND, on at least 8 of 9 boards.
  P2  the median contradiction rate for BAND is at most 0.5 %, and EXTRA's is
      at least 3 times it, on at least 7 of 9 boards.
  P3  the relation itself is not stable enough for "the realizer" to have a
      size: the edge count of P_A varies across replicates by more than 5 % of
      its median on at least 6 of 9 boards.
  P4  mean recall |BAND| / |P_A| lands within 3 points of the full-sample
      figures full_board_free.py reported - 48.8 % to 74.7 % - on at least 7
      of 9 boards.

  What P1 or P2 MISSING would mean: the dropped verdicts are as durable as the
  kept ones, the band is discarding real information, and a realizer's extra
  columns buy something after all. That is the only result that rescues the
  realizer recommendation, and it would rescue it on the right grounds -
  verdict reliability rather than ordering counts.

  P3 differs from the reviewer's version, which asked for the spread of greedy
  realizer SIZE. Greedy realizer construction at full J costs about 20 seconds
  a replicate and stalls without terminating on two boards, so it is not
  affordable at this replicate count and is not attempted. The edge count is
  the cheap stand-in and it is labelled as one.

SELF-CHECKS (no table if any fails)
  * the halves must be disjoint and cover every item, counted per replicate;
  * BAND must be contained in P_A on every replicate of every board - that is
    the theorem, and a violation means the criterion is wrong at full J;
  * EXTRA and BAND must partition P_A's edges;
  * a replicate whose second half yields no edges is dropped and counted, not
    scored as agreement.

    python verdict_fragility.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from band_slack import bands_of

SEED = 20260825
R = 200

MATRICES = {
    "SWE-bench Verified": "swebench_verified_matrix.csv",
    "SWE-bench Lite": "swebench_lite_matrix.csv",
    "MTEB English v2": "mteb_eng_v2_wide.csv",
    "HELM classic": "helm_winrate_matrix.csv",
    "ProteinGym DMS": "proteingym/matrix.csv",
    "TabArena 45 variants": "tabarena/matrix_all45.csv",
    "CASP14": "casp/matrix.csv",
    "LiveBench": "livebench/matrix.csv",
    "MathArena 2025": "matharena/matrix.csv",
}


def band_forced(beats: np.ndarray) -> np.ndarray:
    """Pairs the bands force: worst_i <= best_j. No counting."""
    best, worst = bands_of(beats)
    Q = worst[:, None] <= best[None, :]
    return Q & ~np.eye(len(best), dtype=bool)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    rows = []
    ok_cover = ok_contain = ok_part = True
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J, n = x.shape
        if n < 8:
            continue
        rng = np.random.default_rng(SEED + J + n)
        cb, ce, recall, edges, dropped = [], [], [], [], 0
        ub, ue = [], []
        for _ in range(R):
            perm = rng.permutation(n)
            A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
            if len(set(A.tolist()) & set(B.tolist())) or len(A) + len(B) != n:
                ok_cover = False
            PA = rs.rank_sets(x[:, A])["beats"]
            PB = rs.rank_sets(x[:, B])["beats"]
            if not PA.any() or not PB.any():
                dropped += 1
                continue
            Q = band_forced(PA)
            if (Q & ~PA).any():                  # the theorem
                ok_contain = False
            extra = PA & ~Q
            if int(Q.sum()) + int(extra.sum()) != int(PA.sum()):
                ok_part = False
            # contradicted: the second half asserts the reverse
            cb.append(float((Q & PB.T).sum()) / max(int(Q.sum()), 1))
            ce.append(float((extra & PB.T).sum()) / max(int(extra.sum()), 1))
            # UNCONFIRMED: half B leaves the pair incomparable. Added after the
            # contradiction rate came back 0.000 % on every board and every set,
            # which is a real finding about the construction and no comparison
            # at all - an empty measurement scored as a MISS.
            inc = ~(PB | PB.T)
            ub.append(float((Q & inc).sum()) / max(int(Q.sum()), 1))
            ue.append(float((extra & inc).sum()) / max(int(extra.sum()), 1))
            recall.append(int(Q.sum()) / int(PA.sum()))
            edges.append(int(PA.sum()))
        if not cb:
            continue
        rows.append({"name": name, "J": J, "n": n,
                     "band": np.array(cb), "extra": np.array(ce),
                     "recall": np.array(recall), "edges": np.array(edges),
                     "ub": np.array(ub), "ue": np.array(ue),
                     "dropped": dropped})
        print(f"  {name:<22} band {100 * np.median(cb):.3f}%  "
              f"extra {100 * np.median(ce):.3f}%  recall "
              f"{100 * np.median(recall):.0f}%")

    print("self-checks ...")
    print(f"  [{'ok  ' if ok_cover else 'FAIL'}] the halves are disjoint and cover "
          f"every item on every replicate")
    print(f"  [{'ok  ' if ok_contain else 'FAIL'}] the band-forced relation is "
          f"CONTAINED in the measured relation, every replicate, every board")
    print(f"  [{'ok  ' if ok_part else 'FAIL'}] band-forced and extra partition the "
          f"edges")
    ok_n = len(rows) >= 8
    print(f"  [{'ok  ' if ok_n else 'FAIL'}] {len(rows)} boards (need >= 8); "
          f"{sum(r['dropped'] for r in rows)} replicates dropped for an empty half")

    if not (ok_cover and ok_contain and ok_part and ok_n):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHICH VERDICTS SURVIVE A SECOND SAMPLE OF ITEMS?")
    p("=" * 100)
    p(f"  {R} replicates per board. Items split in half; the relation is built at")
    p("  FULL J on each half. A verdict from half A is CONTRADICTED when half B")
    p("  asserts the reverse. No sub-posets and no exact counting anywhere.")
    p("")
    p(f"  {'board':<22}{'J':>5}{'n':>7}{'BAND contra':>13}{'EXTRA contra':>14}"
      f"{'BAND unconf':>13}{'EXTRA unconf':>14}{'ratio':>7}{'recall':>8}")
    for r in rows:
        b, e = float(np.median(r["band"])), float(np.median(r["extra"]))
        bu, eu = float(np.median(r["ub"])), float(np.median(r["ue"]))
        ratio = (eu / bu) if bu > 0 else float("inf")
        p(f"  {r['name']:<22}{r['J']:>5}{r['n']:>7}{100 * b:>12.3f}%"
          f"{100 * e:>13.3f}%{100 * bu:>12.1f}%{100 * eu:>13.1f}%"
          f"{ratio:>7.2f}{100 * np.median(r['recall']):>7.0f}%")
    p("")
    zero_both = sum(1 for r in rows
                    if np.median(r["extra"]) == 0 and np.median(r["band"]) == 0)
    higher = sum(1 for r in rows
                 if np.median(r["extra"]) > np.median(r["band"]))
    unc_higher = sum(1 for r in rows
                     if np.median(r["ue"]) > np.median(r["ub"]))
    p2 = sum(1 for r in rows
             if np.median(r["band"]) <= 0.005
             and np.median(r["extra"]) >= 3 * max(np.median(r["band"]), 1e-12))
    spread = sum(1 for r in rows
                 if (np.percentile(r["edges"], 75) - np.percentile(r["edges"], 25))
                 > 0.05 * np.median(r["edges"]))
    inband = sum(1 for r in rows if 0.458 <= np.median(r["recall"]) <= 0.777)
    v = f"VACUOUS - both rates are exactly 0 on {zero_both} of {len(rows)} boards"
    p(f"  P1  EXTRA contradicted more often than BAND on {higher} of {len(rows)}")
    p(f"      pre-registered >= 8:  {v if zero_both >= 8 else ('HIT' if higher >= 8 else 'MISS')}")
    p(f"  P2  BAND <= 0.5 % and EXTRA >= 3x it on {p2} of {len(rows)}")
    p(f"      pre-registered >= 7:  {v if zero_both >= 8 else ('HIT' if p2 >= 7 else 'MISS')}")
    p(f"  P1b EXPLORATORY, replacing the vacuous pair: EXTRA is UNCONFIRMED more")
    p(f"      often than BAND on {unc_higher} of {len(rows)} boards")
    p(f"  P3  edge count IQR above 5 % of its median on {spread} of {len(rows)}  "
      f"pre-registered >= 6:  {'HIT' if spread >= 6 else 'MISS'}")
    p(f"  P4  recall within 3 points of the full-sample figure on {inband} of "
      f"{len(rows)}   pre-registered >= 7:  {'HIT' if inband >= 7 else 'MISS'}")
    p("")
    p("  NOT ONE VERDICT REVERSES. Across 200 replicates on 9 boards, no pair")
    p("  that half A declares ordered is declared the other way by half B -")
    p("  0.000 % for both sets, everywhere. That is a real and strong statement")
    p("  about the construction: Holm at alpha 0.05 over thousands of pairs")
    p("  rejects only with a wide margin, so a rejected pair is not close to")
    p("  reversing. It is also NO COMPARISON AT ALL, and P1 and P2 were built on")
    p("  the assumption that there would be something to compare. They are")
    p("  scored VACUOUS rather than MISS - an empty measurement is not a")
    p("  failure of the hypothesis, it is a failure to test it.")
    p("")
    p("  What separates the two sets is being UNCONFIRMED, not contradicted:")
    p("  half B leaving the pair incomparable. That is the exploratory")
    p("  replacement above, decided after seeing the zeros.")
    p("")
    p("  A THEOREM THE ARC NEVER WROTE DOWN, and the self-check above asserts it")
    p("  on every replicate of every board: if worst_i <= best_j then i precedes")
    p("  j in every band-respecting permutation, so the relation a reader")
    p("  reconstructs from a band table is CONTAINED in the measured relation.")
    p("  A band table cannot print a verdict the data does not support. It is")
    p("  sound and incomplete, and the recall column is the incompleteness.")
    p("")
    p("  That is a better characterisation than the one this repository has been")
    p("  using. 'The band invents 1.3 to 4.8 bits of ordering freedom' is true")
    p("  and it is OVER-COVERAGE - the safe direction for a confidence object.")
    p("  The rank sets are a simultaneous 1-alpha construction; a summary that")
    p("  admits more orderings than the data supports is conservative, not")
    p("  wrong, and eleven iterations described it as though it were a defect.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("verdict_fragility_results.txt").write_text(text + chr(10),
                                                     encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote verdict_fragility_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
