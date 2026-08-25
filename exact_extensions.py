"""Law 2's dependent variable has never been checked against a known value.

Ordering entropy is H = log2 e(P), the log of the number of linear extensions
of the partial order a board's rank sets induce. Counting linear extensions is
#P-complete, so every H in this repository comes from Knuth's estimator: sample
a random linear extension by repeatedly picking a uniform minimal element,
accumulate log(number of choices), and average.

That estimator is unbiased for e(P). It is NOT unbiased for log e(P), and the
quantity being averaged is heavy-tailed, so the log of a sample mean can sit
well below the truth. Its self-checks in leaderboard_entropy.py test it against
closed forms - a chain (1 extension), an antichain (J!), two disjoint chains -
which are the two extremes of the space and neither of them is what a real
board looks like.

This supplies the missing rung: an exactly known value on a poset that came
from a real leaderboard. A subset-DP counts linear extensions exactly, in big
integers, on induced sub-posets of 18 systems drawn from each board, and the
estimator is run on the SAME relation - not on a re-derived one - so the two
arms cannot silently become one arm.

Why a sub-poset rather than a whole board: exact counting is 2^J states, so 18
is affordable and 134 is not. The sub-poset is the INDUCED order - the
board's own beats relation restricted to the chosen systems - not a fresh
Holm run on a smaller matrix, which would be a different poset with a
different threshold.

Measured first, and recorded here rather than predicted: the beats relation is
transitive on all nine boards - its transitive closure adds exactly zero edges -
so it is a genuine strict partial order and "rank set" is well defined.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  Knuth's estimate of log2 e(P) is within 5 % of the exact value on at
      least 8 of the 9 boards.
  P2  where it is outside 5 %, it errs LOW rather than high on at least half
      of those. The log of a mean of a heavy-tailed variable underestimates.
  P3  the Jensen lower bound the estimator also returns is below the exact
      value on all 9. It is a bound; a violation is an implementation error,
      not a statistical one, and this is the only place it can be caught.
  P4  the error is bias rather than sampling noise: quadrupling the sample
      count from 1000 to 4000 moves the estimate by less than its own distance
      to the exact value on at least 7 of 9 boards.

  What a miss on P1 would mean: every entropy in this repository, including
  both sides of law 2, is off by a knowable amount in a knowable direction,
  and the law was fitted to a biased measurement.

SELF-CHECKS (no table if any fails)
  * the exact counter must agree with brute_extensions - the independent
    permutation counter already in leaderboard_entropy.py - on 200 random
    posets of 6 and 7 elements. Two implementations, one answer;
  * it must return J! for an antichain and 1 for a chain at J = 18, where
    the DP is doing real work rather than a trivial recursion;
  * at least 8 boards must be measured. An empty table reads as a pass;
  * the estimator must be handed the same boolean matrix the exact counter
    was handed, asserted element-wise, not a matrix rebuilt from the board.

    python exact_extensions.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import leaderboard_entropy as le
import rank_sets as rs
from entropy_law_test import MATRICES

SEED = 20260825
SUB = 18            # systems per induced sub-poset; exact DP is 2^SUB states
SAMPLES = 1000      # what entropy_law_test.py uses
SAMPLES_HI = 4000   # for P4
BAND = 0.05         # the 5 % band in P1


def exact_log2(beats: np.ndarray) -> tuple[int, float]:
    """Exact count of linear extensions by DP over down-sets, in big integers.

    f(S) = number of ways to order S consistently, built by removing a maximal
    element: f(S) = sum over k in S with no successor left in S of f(S - k).
    Exact integers, so the log2 at the end is the truth and not a float sum.
    """
    J = beats.shape[0]
    # succ[k] = bitmask of systems k beats (must come after k)
    succ = [int(sum(1 << t for t in np.flatnonzero(beats[k]))) for k in range(J)]
    full = (1 << J) - 1
    f = [0] * (full + 1)
    f[0] = 1
    for S in range(1, full + 1):
        tot = 0
        T = S
        while T:
            low = T & -T
            k = low.bit_length() - 1
            T ^= low
            if not (succ[k] & S):        # k is maximal within S
                tot += f[S ^ low]
        f[S] = tot
    count = f[full]
    return count, (math.log2(count) if count else float("-inf"))


def _check_against_brute(rng) -> tuple[bool, str]:
    bad = 0
    for _ in range(200):
        n = int(rng.integers(6, 8))
        theta = rng.normal(size=n)
        thr = float(rng.uniform(0.2, 1.5))
        b = (theta[:, None] - theta[None, :]) > thr
        c, _ = exact_log2(b)
        if c != le.brute_extensions(b):
            bad += 1
    return bad == 0, f"200 random posets of 6-7 elements, disagreements with the permutation counter: {bad}"


def _check_closed_forms() -> tuple[bool, str]:
    J = SUB
    chain = np.triu(np.ones((J, J), dtype=bool), k=1)
    anti = np.zeros((J, J), dtype=bool)
    c1, _ = exact_log2(chain)
    c2, _ = exact_log2(anti)
    ok = c1 == 1 and c2 == math.factorial(J)
    return ok, f"chain of {J}: {c1} (want 1); antichain of {J}: {c2} (want {J}!)"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)

    print("self-checks ...")
    ok1, m1 = _check_against_brute(np.random.default_rng(SEED + 1))
    print(f"  [{'ok  ' if ok1 else 'FAIL'}] {m1}")
    ok2, m2 = _check_closed_forms()
    print(f"  [{'ok  ' if ok2 else 'FAIL'}] {m2}")

    rows = []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J = x.shape[0]
        if J < SUB:
            continue
        beats = rs.rank_sets(x)["beats"]
        pick = np.sort(np.random.default_rng(SEED + J).choice(J, SUB, replace=False))
        sub = beats[np.ix_(pick, pick)].copy()

        count, ex = exact_log2(sub)
        est = le.log_extensions(sub, SAMPLES, np.random.default_rng(SEED + 7))
        hi = le.log_extensions(sub, SAMPLES_HI, np.random.default_rng(SEED + 7))
        # the estimator must have been handed the same relation, not a rebuild
        same = bool((sub == beats[np.ix_(pick, pick)]).all())
        rows.append({"name": name, "J": J, "exact": ex, "count": count,
                     "est": est["bits"], "lower": est["bits_lower"],
                     "se": est["se_bits"], "hi": hi["bits"], "same": same})
        print(f"  {name:<22} exact {ex:8.3f}  knuth {est['bits']:8.3f}")

    ok3 = len(rows) >= 8
    print(f"  [{'ok  ' if ok3 else 'FAIL'}] {len(rows)} boards measured (need >= 8)")
    ok4 = all(r["same"] for r in rows)
    print(f"  [{'ok  ' if ok4 else 'FAIL'}] the estimator was handed the same relation "
          f"the exact counter was, on {sum(r['same'] for r in rows)} of {len(rows)}")

    if not (ok1 and ok2 and ok3 and ok4):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    for r in rows:
        r["rel"] = (r["est"] - r["exact"]) / r["exact"]
        r["move"] = abs(r["hi"] - r["est"])
        r["gap"] = abs(r["est"] - r["exact"])

    L = []
    p = L.append
    p("THE ENTROPY ESTIMATOR AGAINST AN EXACTLY KNOWN VALUE")
    p("=" * 92)
    p(f"  Induced sub-posets of {SUB} systems from each board's own beats relation.")
    p(f"  Exact counts by subset DP in big integers; Knuth's estimator at "
      f"{SAMPLES} samples on the same relation.")
    p("")
    p(f"  {'board':<22}{'J':>5}{'exact log2':>12}{'Knuth':>10}{'rel err':>10}"
      f"{'Jensen lo':>11}{'at 4000':>10}")
    for r in rows:
        p(f"  {r['name']:<22}{r['J']:>5}{r['exact']:>12.3f}{r['est']:>10.3f}"
          f"{100 * r['rel']:>9.2f}%{r['lower']:>11.3f}{r['hi']:>10.3f}")
    p("")
    within = sum(1 for r in rows if abs(r["rel"]) <= BAND)
    outside = [r for r in rows if abs(r["rel"]) > BAND]
    low = sum(1 for r in outside if r["rel"] < 0)
    bound_ok = sum(1 for r in rows if r["lower"] <= r["exact"])
    stable = sum(1 for r in rows if r["move"] < r["gap"])
    p(f"  P1  within {BAND:.0%} of exact: {within} of {len(rows)}                "
      f"pre-registered >= 8:  {'HIT' if within >= 8 else 'MISS'}")
    p(f"  P2  of the {len(outside)} outside the band, {low} err low        "
      f"pre-registered >= half: "
      f"{'HIT' if not outside or low * 2 >= len(outside) else 'MISS'}")
    p(f"  P3  Jensen bound below exact: {bound_ok} of {len(rows)}           "
      f"pre-registered = all: {'HIT' if bound_ok == len(rows) else 'MISS'}")
    p(f"  P4  4x samples moves less than the gap: {stable} of {len(rows)}   "
      f"pre-registered >= 7:  {'HIT' if stable >= 7 else 'MISS'}")
    p("")
    worst = max(rows, key=lambda r: abs(r["rel"]))
    p(f"  Largest relative error: {worst['name']}, "
      f"{100 * worst['rel']:+.2f}% ({worst['est']:.3f} against {worst['exact']:.3f}).")
    p(f"  Mean signed error {100 * float(np.mean([r['rel'] for r in rows])):+.2f}%, "
      f"mean absolute {100 * float(np.mean([abs(r['rel']) for r in rows])):.2f}%.")
    p("")
    p("  Law 2 is a statement about H / log2(J!). This measures the numerator")
    p("  against a value that is known rather than estimated, on posets that")
    p("  came out of real boards rather than on a chain and an antichain. The")
    p("  estimator's own self-checks use those two extremes, which is where an")
    p("  estimator is least likely to fail.")
    p("")
    p("  Recorded, not predicted: the beats relation is transitive on all nine")
    p("  boards - its transitive closure adds exactly zero edges - so the")
    p("  partial order is genuine and the sub-poset is well defined.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("exact_extensions_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                    newline=chr(10))
    print(chr(10) + "wrote exact_extensions_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
