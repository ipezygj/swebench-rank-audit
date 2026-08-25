"""What would a report card have to print to stop inventing orderings?

band_slack.py counted what the printed bands cost: between 1.3 and 4.8 bits of
ordering freedom that the data excludes and the table admits. This asks the
constructive half. If a report card printed more than bands, what is the
cheapest more?

A band is not an arbitrary summary. best_i = 1 + (systems that beat i) and
worst_i = J - (systems i beats), so the pair IS the in-degree and out-degree of
system i. Printing bands is printing the degree sequence. Everything the slack
measures is what the STRUCTURE adds beyond degrees - which pairs, not how many.

So the natural addition is to print some of the pairs. A report card already
gives its top rows the most space, so the candidate is: bands for everyone, plus
the full pairwise verdicts among the top k. This measures, exactly, how much of
the slack that buys as k grows, and controls it against the same number of
verdicts chosen at random rather than from the top.

One counter does all of it. Orderings consistent with (bands for all) AND (a
precedence relation R) are counted by a DP over subsets; R empty reproduces the
permanent, R the whole poset reproduces the linear-extension count. Both are
asserted against the counters already in the repository, so the new code is
pinned at both ends before any of its middle values are read.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  k = 0 reproduces B(P) and k = 18 reproduces e(P), exactly, as integers,
      on all 8 boards. An implementation guard: a miss here voids the rest.
  P2  printing the verdicts among the top 6 recovers at least half the slack,
      on at least 6 of 8 boards.
  P3  printing the top 12 recovers at least 90 % of it, on at least 6 of 8.
  P4  the top is special: 6 systems chosen at random recover LESS than the top
      6, on at least 6 of 8 boards, averaged over 40 random choices.

  What a miss on P4 would mean: the slack is spread evenly through the board,
  the top rows are not where the ambiguity lives, and a report card cannot fix
  its picture by spending more ink where its readers are looking.

SELF-CHECKS (no table if any fails)
  * the DP must reproduce the permanent at R empty and the extension count at
    R full, as exact integers, on 200 random posets at J = 7, against the two
    counters already in the repository - not against itself;
  * it must also agree with brute-force enumeration of all 7! orderings under
    a partial R, which neither existing counter can check;
  * the counts must be monotone in k: adding verdicts can only remove
    orderings, so the sequence must be non-increasing on every board;
  * at least 8 boards must be measured.

    python top_verdicts.py
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from band_slack import band_matrix, bands_of, permanent01, _random_poset
from entropy_law_test import MATRICES
from exact_extensions import exact_log2

SEED = 20260825
SUB = 18
KS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)
RANDOM_K = 6
RANDOM_REPS = 40


def count_with(bandM: np.ndarray, R: np.ndarray) -> int:
    """Orderings where every row lands inside its band AND R is respected.

    f(S) = ways to fill the first |S| ranks using exactly the systems in S.
    A system x may take rank |S| when its band allows that rank and every
    R-predecessor of x is already placed.
    """
    J = bandM.shape[0]
    pred = [int(sum(1 << t for t in np.flatnonzero(R[:, x]))) for x in range(J)]
    allow = [int(sum(1 << r for r in range(J) if bandM[x, r])) for x in range(J)]
    full = (1 << J) - 1
    f = [0] * (full + 1)
    f[0] = 1
    for S in range(1, full + 1):
        r = bin(S).count("1") - 1          # the rank being filled, 0-based
        tot = 0
        T = S
        while T:
            low = T & -T
            T ^= low
            x = low.bit_length() - 1
            rest = S ^ low
            if (allow[x] >> r) & 1 and (pred[x] & ~rest) == 0:
                tot += f[rest]
        f[S] = tot
    return f[full]


def restrict(beats: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """The relation with only the edges whose BOTH ends are in keep."""
    R = np.zeros_like(beats)
    if len(keep):
        R[np.ix_(keep, keep)] = beats[np.ix_(keep, keep)]
    return R


# --- self-checks ------------------------------------------------------------

def _brute(bandM, R) -> int:
    J = bandM.shape[0]
    c = 0
    for p in itertools.permutations(range(J)):
        pos = {v: i for i, v in enumerate(p)}
        if not all(bandM[v, pos[v]] for v in range(J)):
            continue
        if all(pos[i] < pos[j] for i in range(J) for j in range(J) if R[i, j]):
            c += 1
    return c


def _check_endpoints(rng) -> tuple[bool, str]:
    badp = bade = 0
    for _ in range(200):
        b = _random_poset(rng, 7)
        best, worst = bands_of(b)
        M = band_matrix(best, worst)
        if count_with(M, np.zeros_like(b)) != permanent01(M):
            badp += 1
        if count_with(M, b) != exact_log2(b)[0]:
            bade += 1
    return badp == 0 and bade == 0, (
        f"200 posets at J=7: disagreements with the permanent {badp}, "
        f"with the extension counter {bade}")


def _check_middle(rng) -> tuple[bool, str]:
    bad = 0
    for _ in range(200):
        b = _random_poset(rng, 7)
        best, worst = bands_of(b)
        M = band_matrix(best, worst)
        keep = np.sort(rng.choice(7, int(rng.integers(2, 6)), replace=False))
        R = restrict(b, keep)
        if count_with(M, R) != _brute(M, R):
            bad += 1
    return bad == 0, f"200 posets at J=7 with a partial relation, brute-force disagreements: {bad}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print("self-checks ...")
    ok1, m1 = _check_endpoints(np.random.default_rng(SEED + 1))
    print(f"  [{'ok  ' if ok1 else 'FAIL'}] {m1}")
    ok2, m2 = _check_middle(np.random.default_rng(SEED + 2))
    print(f"  [{'ok  ' if ok2 else 'FAIL'}] {m2}")

    rows, skipped = [], []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J = x.shape[0]
        if J < SUB:
            skipped.append((name, J))
            continue
        b = rs.rank_sets(x)["beats"]
        pick = np.sort(np.random.default_rng(SEED + J).choice(J, SUB, replace=False))
        sub = b[np.ix_(pick, pick)].copy()
        best, worst = bands_of(sub)
        M = band_matrix(best, worst)
        order = np.argsort(best * 100 + worst, kind="stable")   # top rows first

        e_cnt = exact_log2(sub)[0]
        b_cnt = permanent01(M)
        curve = {k: count_with(M, restrict(sub, order[:k])) for k in KS}

        rng = np.random.default_rng(SEED + 3 + J)
        rand = [count_with(M, restrict(sub, np.sort(rng.choice(SUB, RANDOM_K,
                                                               replace=False))))
                for _ in range(RANDOM_REPS)]
        even = order[np.linspace(0, SUB - 1, RANDOM_K).round().astype(int)]
        even_cnt = count_with(M, restrict(sub, np.sort(even)))
        # A benchmark owner holds the poset, so the best choice of which
        # verdicts to print is available to them: add the system that removes
        # the most orderings, six times. Reported as what a targeted choice
        # buys, not as a rule that could be applied without the poset.
        chosen = []
        for _ in range(RANDOM_K):
            chosen.append(min((v for v in range(SUB) if v not in chosen),
                              key=lambda v: count_with(
                                  M, restrict(sub, np.sort(np.array(chosen + [v]))))))
        greedy_cnt = count_with(M, restrict(sub, np.sort(np.array(chosen))))
        top_edges = int(sub[np.ix_(order[:RANDOM_K], order[:RANDOM_K])].sum())
        rows.append({"name": name, "J": J, "e": e_cnt, "b": b_cnt,
                     "curve": curve, "even": even_cnt, "greedy": greedy_cnt,
                     "top_edges": top_edges,
                     "rand": float(np.mean([math.log2(v) for v in rand]))})
        print(f"  {name:<22} slack {math.log2(b_cnt / e_cnt):6.3f} bits")

    ok3 = len(rows) >= 8
    print(f"  [{'ok  ' if ok3 else 'FAIL'}] {len(rows)} boards measured (need >= 8)")
    mono = all(all(r["curve"][a] >= r["curve"][c] for a, c in zip(KS, KS[1:]))
               for r in rows)
    print(f"  [{'ok  ' if mono else 'FAIL'}] the count is non-increasing in k on every board")

    if not (ok1 and ok2 and ok3 and mono):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    for r in rows:
        r["slack"] = math.log2(r["b"] / r["e"])
        r["rec"] = {k: (r["slack"] - (math.log2(r["curve"][k]) - math.log2(r["e"])))
                    / r["slack"] if r["slack"] > 0 else 1.0 for k in KS}
        def _rec(v):
            return 1.0 if r["slack"] <= 0 else (
                (r["slack"] - (v - math.log2(r["e"]))) / r["slack"])
        r["rand_rec"] = _rec(r["rand"])
        r["even_rec"] = _rec(math.log2(r["even"]))
        r["greedy_rec"] = _rec(math.log2(r["greedy"]))

    L = []
    p = L.append
    p("WHAT WOULD A REPORT CARD HAVE TO PRINT?")
    p("=" * 96)
    p("  A band is the in-degree and out-degree of a system: best = 1 + systems")
    p("  that beat it, worst = J - systems it beats. Printing bands is printing")
    p("  the degree sequence. The slack is everything the structure adds.")
    p(f"  Induced sub-posets of {SUB} systems; k = how many top rows get their")
    p("  full pairwise verdicts printed alongside.")
    if skipped:
        p("  Not measured, fewer systems than the sub-poset size: "
          + ", ".join(f"{n} (J={j})" for n, j in skipped) + ".")
    p("")
    p(f"  {'board':<22}{'slack':>8}" + "".join(f"{'k=' + str(k):>7}" for k in KS[1:])
      + f"{'rand 6':>9}")
    for r in rows:
        p(f"  {r['name']:<22}{r['slack']:>8.3f}"
          + "".join(f"{100 * r['rec'][k]:>6.0f}%" for k in KS[1:])
          + f"{100 * r['rand_rec']:>8.0f}%")
    p("")
    p("  SIX VERDICTS, CHOSEN FOUR WAYS. Same budget, same boards.")
    p(f"  {'board':<22}{'top 6':>9}{'evenly spaced':>15}{'random':>9}"
      f"{'targeted':>10}{'edges in top 6':>16}")
    for r in rows:
        p(f"  {r['name']:<22}{100 * r['rec'][RANDOM_K]:>8.0f}%"
          f"{100 * r['even_rec']:>14.0f}%{100 * r['rand_rec']:>8.0f}%"
          f"{100 * r['greedy_rec']:>9.0f}%{r['top_edges']:>16}")
    p("")
    p("  Each cell is the share of the slack removed by printing that many")
    p("  verdicts. 100 % means the printed table admits exactly the orderings")
    p("  the data admits, and nothing more.")
    p("")
    ok_ends = sum(1 for r in rows
                  if r["curve"][0] == r["b"] and r["curve"][SUB] == r["e"])
    half6 = sum(1 for r in rows if r["rec"][6] >= 0.5)
    p90 = sum(1 for r in rows if r["rec"][12] >= 0.9)
    beat = sum(1 for r in rows if r["rec"][RANDOM_K] > r["rand_rec"])
    p(f"  P1  endpoints exact on {ok_ends} of {len(rows)}                 "
      f"pre-registered = all:  {'HIT' if ok_ends == len(rows) else 'MISS'}")
    p(f"  P2  top 6 recovers >= half the slack on {half6} of {len(rows)}  "
      f"pre-registered >= 6:   {'HIT' if half6 >= 6 else 'MISS'}")
    p(f"  P3  top 12 recovers >= 90 % on {p90} of {len(rows)}            "
      f"pre-registered >= 6:   {'HIT' if p90 >= 6 else 'MISS'}")
    p(f"  P4  top 6 beats a random 6 on {beat} of {len(rows)}            "
      f"pre-registered >= 6:   {'HIT' if beat >= 6 else 'MISS'}")
    p("")
    p("  The random column is the control, averaged over 40 draws of 6 systems.")
    p("  Without it, a large number under k=6 would say only that printing")
    p("  verdicts helps, which is true of any 6 systems and says nothing about")
    p("  where a report card should spend its space.")
    p("")
    p("  P4 did not merely fail. It failed the other way round: the top 6")
    p("  recovers NOTHING on 7 of 8 boards while 6 systems drawn at random")
    p("  recover 8 to 14 %. The worst place to spend the ink is where every")
    p("  report card spends it.")
    p("")
    p("  The reason is not that the top rows are unrelated to each other. The")
    p("  last column counts the relations among them and several boards have")
    p("  five to eight. Those verdicts are already IMPLIED by the bands: a")
    p("  system whose band starts at 1 and a system it beats cannot be ordered")
    p("  any other way inside the band constraints, so printing the verdict")
    p("  adds no information that the degree sequence did not already carry.")
    p("")
    p("  WITHDRAWN 2026-08-25. This file used to close by recommending that a")
    p("  benchmark owner choose which verdicts to print by adding the system")
    p("  that removes the most orderings, six times, on the grounds that it")
    p("  recovers 24 to 56 % against 0 % for the top six. Those numbers stand -")
    p("  they are arithmetic on this poset and the summary printed beside it -")
    p("  but targeted_split.py tested whether the choice is a property of the")
    p("  SYSTEMS by making it on half a board's items and scoring it on the")
    p("  poset from the other half. It beat a random six on only 5 of 8 boards,")
    p("  against a pre-registered 6, and reached half the in-sample ceiling on")
    p("  4 of 8 against a pre-registered 5. HELM classic is the clearest case:")
    p("  its two half-item posets differ in 4 cells out of 153, and the six")
    p("  chosen on one half recover 100 % scored on themselves and 0 % scored on")
    p("  the other. So the recommendation is withdrawn rather than qualified,")
    p("  which is what that file's pre-registration said a miss would mean.")
    p("")
    p("  What is left is the negative half, and it is the robust half: the top")
    p("  six recovers nothing, in sample and out, on 7 of 8 boards. Whatever a")
    p("  report card should print beyond bands, it is not more detail about the")
    p("  rows at the top.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("top_verdicts_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                newline=chr(10))
    print(chr(10) + "wrote top_verdicts_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
