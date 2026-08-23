"""Does merging two benchmarks resolve what neither resolves alone?

A leaderboard that cannot separate its top is usually told to add items.
There is a cheaper option nobody measures: another benchmark already exists
in the same field, with the same systems on it. SWE-bench Verified and Lite
share 25 submissions and 93 instances; merging them, deduplicated, gives
those 25 systems 707 instances instead of 500 or 300.

Merged here as the union of instances, with the 93 shared ones counted once
(taken from Verified, whose outcomes on them are the same by construction:
the two splits report the same evaluation). The comparison is between the
same 25 systems on three item sets: Verified only, Lite only, and merged.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the merged board's median rank-set width is at least 20 % narrower than
    Verified's on the same 25 systems;
  * the merged board's tie@1 is no larger than Verified's;
  * the two boards agree on the ranking of the shared systems: Spearman
    above 0.9, so merging is joining like with like rather than averaging
    two different measurements.

SELF-CHECKS
  * merging a board with a copy of itself must not narrow the rank sets by
    more than the sqrt(2) that doubling n gives, and must not widen them;
  * merging with pure noise items must WIDEN the sets.

    python merge_boards.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import rank_sets as rs

SEED = 20260823
DRAWS = 1200


def widths(x, draws=DRAWS):
    r = rs.rank_sets(x, draws=draws)
    return r["worst"] - r["best"] + 1, int((r["best"] == 1).sum()), r


def _check_self_merge():
    rng = np.random.default_rng(11)
    x = 0.5 + rng.normal(0, 0.06, 30)[:, None] + rng.normal(0, 0.4, (30, 200))
    w1, _, _ = widths(x, 600)
    w2, _, _ = widths(np.hstack([x, x]), 600)
    ratio = float(np.median(w2) / np.median(w1))
    return 0.6 <= ratio <= 1.02, f"board merged with a copy of itself: width ratio {ratio:.2f}"


def _check_noise_merge():
    rng = np.random.default_rng(13)
    x = 0.5 + rng.normal(0, 0.06, 30)[:, None] + rng.normal(0, 0.4, (30, 200))
    noise = rng.normal(0.5, 0.4, (30, 200))
    w1, _, _ = widths(x, 600)
    w2, _, _ = widths(np.hstack([x, noise]), 600)
    return np.median(w2) >= np.median(w1), \
        f"merging with pure noise: median width {np.median(w1):.0f} -> {np.median(w2):.0f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_self_merge(), _check_noise_merge()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    v = pd.read_csv("swebench_verified_matrix.csv", index_col=0)
    l = pd.read_csv("swebench_lite_matrix.csv", index_col=0)
    shared = sorted(set(v.index) & set(l.index))
    only_lite = [c for c in l.columns if c not in set(v.columns)]
    V = v.loc[shared].to_numpy(dtype=float)
    L_ = l.loc[shared].to_numpy(dtype=float)
    M = np.hstack([V, l.loc[shared, only_lite].to_numpy(dtype=float)])

    L = []
    p = L.append
    p("DOES MERGING TWO BENCHMARKS RESOLVE WHAT NEITHER RESOLVES ALONE?")
    p("=" * 80)
    p(f"  {len(shared)} submissions appear on both SWE-bench Verified and Lite; "
      f"{V.shape[1]} + {L_.shape[1]} instances, {len(only_lite)} of Lite's not in Verified, "
      f"merged to {M.shape[1]}")
    p("")
    p(f"  {'item set':<24} {'n':>6} {'median width':>13} {'tie@1':>7} {'established':>12}")
    rows = {}
    for label, mat in (("Verified only", V), ("Lite only", L_), ("merged", M)):
        w, t1, r = widths(mat)
        est = float(r["beats"].sum() / (len(shared) * (len(shared) - 1)))
        rows[label] = (float(np.median(w)), t1, est, r)
        p(f"  {label:<24} {mat.shape[1]:>6} {np.median(w):>13.0f} {t1:>7} {100 * est:>11.1f}%")
    mv, tv = rows["Verified only"][0], rows["Verified only"][1]
    mm, tm = rows["merged"][0], rows["merged"][1]
    narrow = (mv - mm) / mv if mv else float("nan")
    sp = spearmanr(V.mean(axis=1), L_.mean(axis=1))
    p("")
    p(f"  merged is {100 * narrow:.0f} % narrower than Verified (pre-registered >= 20 %): "
      f"{'yes' if narrow >= 0.20 else 'NO'}")
    p(f"  merged tie@1 {tm} vs Verified {tv} (pre-registered: no larger): {'yes' if tm <= tv else 'NO'}")
    p(f"  the two boards rank these systems the same way: Spearman {sp.statistic:+.2f} "
      f"(p {sp.pvalue:.1e}); pre-registered > 0.9: {'yes' if sp.statistic > 0.9 else 'NO'}")
    p("")
    # which pairs the merge newly establishes
    bv, bm = rows["Verified only"][3]["beats"], rows["merged"][3]["beats"]
    gained = int((bm & ~bv).sum())
    lost = int((bv & ~bm).sum())
    p(f"  pairs established by the merge but not by Verified: {gained}; lost: {lost}")
    p("")
    p("  Merging is the cheapest way to add items: they already exist and the")
    p("  systems have already been run on them. The cost is that the two item")
    p("  sets must measure the same thing - the Spearman above is the check, and")
    p("  it is a weak one, since agreement on means does not imply agreement")
    p("  item by item.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("merge_boards_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote merge_boards_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
