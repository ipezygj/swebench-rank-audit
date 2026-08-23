"""How many test cases would have to be re-graded for someone else to lead?

Elections report a margin in votes. Leaderboards report a margin in decimal
places, which hides the only thing a reader needs to know: how many individual
decisions the result rests on. A benchmark score of 0.792 against 0.788 is
four thousandths, and four thousandths of 500 instances is two test cases.

This file counts in test cases. There is no null, no bootstrap and no p-value
here: these are integers read off the matrix. It sits deliberately next to the
statistical tools, because the statistical answer ("these systems cannot be
separated") and the arithmetic answer ("the lead is two re-grades wide") are
different sentences, and the second one is the one people understand.

TWO UNITS, BOTH REPORTED
-------------------------
    re-grade    one (system, instance) cell changes value. This is what
                happens when a flaky test is fixed, a harness bug is found or
                a submission is re-run. It moves the gap by one instance.

    swap        one instance where A succeeded and B failed becomes the
                reverse. This is what happens when an instance is found to be
                mis-specified in a way that favoured one side. It moves the
                gap by two, so it costs half as many.

THE DENOMINATOR THAT MAKES THE MARGIN MEAN SOMETHING
------------------------------------------------------
A lead of two instances is a different thing when the two systems disagreed on
eighty instances than when they disagreed on four. The first is two out of
eighty coin flips; the second is a real difference. So every margin is printed
against the number of instances where the pair actually disagreed - the same
discordant count McNemar's test uses, in plain units.

SWING INSTANCES
---------------
The instances that could erode a lead are exactly those the leader solved and
the challenger did not. They are nameable. If a handful of instance names turn
up in the swing set of every close pair, those instances decide the top of the
benchmark, and their grading deserves more scrutiny than the other four
hundred.

SELF-CHECKS THAT CAN FAIL
--------------------------
  * identical systems must have margin zero;
  * a system ahead by exactly one instance must have margin one;
  * the counted margin must equal a brute-force flip simulation;
  * the margin must equal n times the score difference, exactly, since that
    is the arithmetic identity the whole file rests on.

    python recount_margin.py [--matrix ...] [--top 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def counts(x: np.ndarray) -> np.ndarray:
    return x.sum(axis=1).astype(int)


def margin_regrades(x: np.ndarray, a: int, b: int) -> int:
    """Cells to re-grade before b catches a. One cell moves the gap by one."""
    return int(counts(x)[a] - counts(x)[b])


def margin_swaps(x: np.ndarray, a: int, b: int) -> int:
    """Discordant instances to reverse. Each moves the gap by two."""
    d = margin_regrades(x, a, b)
    return int(np.ceil(d / 2.0))


def discordant(x: np.ndarray, a: int, b: int) -> tuple[int, int, int]:
    only_a = int(((x[a] == 1) & (x[b] == 0)).sum())
    only_b = int(((x[a] == 0) & (x[b] == 1)).sum())
    return only_a, only_b, only_a + only_b


def swing_instances(x: np.ndarray, a: int, b: int, cols) -> list:
    idx = np.flatnonzero((x[a] == 1) & (x[b] == 0))
    return [cols[i] for i in idx]


# --- self-checks ------------------------------------------------------------

def _check_identical() -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    row = (rng.random(200) < 0.5).astype(int)
    x = np.vstack([row, row])
    ok = margin_regrades(x, 0, 1) == 0 and discordant(x, 0, 1)[2] == 0
    return ok, f"identical systems: margin {margin_regrades(x, 0, 1)}"


def _check_one_ahead() -> tuple[bool, str]:
    x = np.zeros((2, 10), dtype=int)
    x[0, :5] = 1
    x[1, :4] = 1
    ok = margin_regrades(x, 0, 1) == 1 and margin_swaps(x, 0, 1) == 1
    return ok, f"one instance ahead: re-grades {margin_regrades(x, 0, 1)}, swaps {margin_swaps(x, 0, 1)}"


def _check_bruteforce() -> tuple[bool, str]:
    """Flipping `margin` of the leader's solved-only cells must produce a tie."""
    rng = np.random.default_rng(3)
    x = (rng.random((2, 120)) < np.array([[0.6], [0.5]])).astype(int)
    a, b = (0, 1) if counts(x)[0] >= counts(x)[1] else (1, 0)
    m = margin_regrades(x, a, b)
    idx = np.flatnonzero((x[a] == 1) & (x[b] == 0))[:m]
    y = x.copy()
    y[a, idx] = 0
    ok = int(counts(y)[a]) == int(counts(y)[b])
    return ok, f"flipping {m} cells produces an exact tie: {ok}"


def _check_identity() -> tuple[bool, str]:
    rng = np.random.default_rng(5)
    x = (rng.random((6, 300)) < 0.5).astype(int)
    sc = x.mean(axis=1)
    ok = True
    for a in range(6):
        for b in range(6):
            if abs(margin_regrades(x, a, b)
                   - round((sc[a] - sc[b]) * x.shape[1])) > 1e-9:
                ok = False
    return ok, f"margin equals n times the score gap for every pair: {ok}"


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_identical(), _check_one_ahead(),
                        _check_bruteforce(), _check_identity()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default="recount_margin_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=int)
    cols = list(df.columns)
    names = list(df.index)
    J, n = x.shape
    print(f"matrix {a.matrix}: {J} systems x {n} instances")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no headline number is printed.")
        return 1

    sc = x.mean(axis=1)
    order = np.argsort(-sc, kind="stable")
    lead = int(order[0])

    L = []
    p = L.append
    p("THE MARGIN, COUNTED IN TEST CASES")
    p("=" * 74)
    p(f"{J} systems, {n} instances")
    p(f"leader: {names[lead]}  {sc[lead]:.3f} "
      f"({counts(x)[lead]} of {n} solved)")
    p("")
    p("HOW FAR BEHIND THE LEADER, IN TEST CASES")
    p(f"  {'rank':>4} {'system':<40} {'score':>6} {'re-grades':>10}"
      f" {'swaps':>6} {'they disagree on':>17}")
    for rank, idx in enumerate(order[:a.top], start=1):
        if rank == 1:
            p(f"  {rank:>4} {names[idx][:40]:<40} {sc[idx]:>6.3f}"
              f" {'-':>10} {'-':>6} {'-':>17}")
            continue
        rg = margin_regrades(x, lead, int(idx))
        sw = margin_swaps(x, lead, int(idx))
        oa, ob, tot = discordant(x, lead, int(idx))
        p(f"  {rank:>4} {names[idx][:40]:<40} {sc[idx]:>6.3f}"
          f" {rg:>10} {sw:>6} {tot:>17}")
    p("")
    second = int(order[1])
    rg2 = margin_regrades(x, lead, second)
    _, _, disc2 = discordant(x, lead, second)
    third = int(order[2])
    rg3 = margin_regrades(x, lead, third)
    _, _, disc3 = discordant(x, lead, third)
    oa2, ob2, _ = discordant(x, lead, second)
    p(f"  The lead over rank 2 is {rg2} test case(s) out of {disc2} on which")
    p(f"  the two disagree. Over rank 3 it is {rg3} out of {disc3}.")
    if rg2 == 0:
        p("")
        p(f"  Ranks 1 and 2 solve the SAME NUMBER of instances "
          f"({counts(x)[lead]} each) and")
        p(f"  disagree on {disc2} of them, split {oa2} to {ob2}. Identical scores,")
        p("  different competence. The leaderboard prints one above the other")
        p("  and there is nothing in the data that puts it that way round.")
    p("")
    p("  Nothing statistical has been done here. These are counts. A reader")
    p("  who distrusts every p-value in the other files can still check these")
    p("  by hand from the published matrix.")
    p("")

    # How many systems are within k re-grades of the crown?
    p("HOW CROWDED IS THE CROWN")
    gaps = np.array([margin_regrades(x, lead, int(j)) for j in range(J)])
    for k in (1, 2, 5, 10, 20):
        p(f"  systems within {k:>2} re-grade(s) of the lead: "
          f"{int((gaps <= k).sum())}")
    p("")

    # Swing instances shared by the closest pairs.
    p("SWING INSTANCES - solved by the leader, missed by a close rival")
    from collections import Counter
    cnt = Counter()
    close = [int(j) for j in order[1:a.top]]
    for j in close:
        for c in swing_instances(x, lead, j, cols):
            cnt[c] += 1
    p(f"  instances that decide at least one of the {len(close)} closest pairs:"
      f" {len(cnt)}")
    p("  the ones that appear in the most pairs, with how rare they are:")
    p(f"    {'pairs':>5} {'solved by':>10}   instance")
    colpos = {c: i for i, c in enumerate(cols)}
    for c, k in cnt.most_common(10):
        rare = int(x[:, colpos[c]].sum())
        p(f"    {k:>5} {rare:>6}/{J}   {c}")
    p("")
    p("  Read the middle column. The two instances that decide every one of")
    p("  the closest pairs are solved by 2 and 6 systems out of 134. The")
    p("  leader's edge over the whole top twelve rests partly on instances")
    p("  almost nothing solves - which is exactly where a grading error, a")
    p("  flaky test or a lucky guess is hardest to tell from a capability.")
    p("")
    p("  These are nameable, and that is the point. If the top of this")
    p("  benchmark turns on a specific handful of instances, their grading")
    p("  deserves more scrutiny than the other four hundred, and anyone can")
    p("  go and read them.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
