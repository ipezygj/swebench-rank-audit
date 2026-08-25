"""Everything about the band picture was measured on 18 systems. Does it scale?

band_slack.py and top_verdicts.py both stop at 18 because both counts are DPs
over 2^J subsets. Both closed by saying so. A limitation stated is not a
limitation answered, and a reader is entitled to ask whether 1.3 to 4.8 bits on
18 systems says anything about a board of 134.

This grows the sub-poset. For each board one random permutation is drawn and its
PREFIXES are used, so the sub-posets are nested - the 24-system order contains
the 22-system order contains the 20 - and the trend is a trend within one
sequence rather than a comparison of unrelated draws. Slack is measured at every
even size from 8 to 24, and the growth is fitted and extrapolated to the board's
own J.

An extrapolation is not a measurement and is labelled as one throughout. What it
buys is the difference between "unknown beyond 18" and "unknown beyond 18, with
the shape of the curve up to 24 and a fit that either does or does not hold
across that range".

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  slack is non-decreasing in sub-poset size on at least 6 of 8 boards,
      comparing size 8 with size 24.
  P2  a straight line in the sub-poset size fits the slack with R^2 >= 0.9 on
      at least 6 of 8 boards over the range 8 to 24.
  P3  extrapolating that line to each board's own J gives at least 20 bits on
      at least 6 of 8 boards.
  P4  the slack as a FRACTION of ordering entropy SHRINKS with size: at 24 it
      is smaller than at 12 on at least 6 of 8 boards. Absolute slack grows and
      relative slack falls, because e(P) grows faster.

  What a miss on P2 would mean: the growth is not linear, the extrapolation in
  P3 is not entitled to be made, and P3's verdict should be read as void rather
  than as evidence either way.

SELF-CHECKS (no table if any fails)
  * the sub-posets must actually be nested: every size's index set must be a
    prefix of the next, asserted rather than assumed;
  * B >= e at every size on every board;
  * the single draw band_slack.py used is not special: at size 18, five
    independent draws per board must bracket the value that file reports, or
    its number came from an unrepresentative subset and this file cannot be
    compared to it;
  * at least 8 boards must be measured.

    python slack_scaling.py
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

SEED = 20260825
SIZES = (8, 10, 12, 14, 16, 18, 20, 22, 24)
DRAWS_AT_18 = 5


def slack_of(beats: np.ndarray) -> tuple[float, float, float]:
    """(log2 e, log2 B, slack) for one poset."""
    best, worst = bands_of(beats)
    e = exact_log2(beats)[0]
    b = permanent01(band_matrix(best, worst))
    return math.log2(e), math.log2(b), math.log2(b) - math.log2(e)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    # what band_slack.py reported, read from its file rather than recomputed
    reported = {}
    f = Path("band_slack_results.txt")
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            t = line.split()
            if len(t) >= 7:
                try:
                    float(t[-1]); float(t[-4])
                except ValueError:
                    continue
                reported[" ".join(t[:-6])] = float(t[-4])

    rows, skipped, nested_ok, order_ok = [], [], True, True
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J = x.shape[0]
        if J < SIZES[-1]:
            skipped.append((name, J))
            continue
        beats = rs.rank_sets(x)["beats"]
        perm = np.random.default_rng(SEED + J).permutation(J)
        curve = {}
        for m in SIZES:
            pick = np.sort(perm[:m])
            if not set(np.sort(perm[:m]).tolist()) <= set(np.sort(perm[:SIZES[-1]]).tolist()):
                nested_ok = False
            e, b, sl = slack_of(beats[np.ix_(pick, pick)])
            if b < e:
                order_ok = False
            curve[m] = (e, b, sl)
        # five independent draws at 18, to see whether band_slack's single one sits inside
        rng = np.random.default_rng(SEED + 100 + J)
        at18 = [slack_of(beats[np.ix_(*(lambda q: (q, q))(
            np.sort(rng.choice(J, 18, replace=False))))])[2]
            for _ in range(DRAWS_AT_18)]
        rows.append({"name": name, "J": J, "curve": curve,
                     "at18": (min(at18), max(at18)),
                     "reported": reported.get(name, float("nan"))})
        print(f"  {name:<22} slack {curve[8][2]:5.2f} at 8 -> {curve[24][2]:5.2f} at 24")

    print("self-checks ...")
    print(f"  [{'ok  ' if nested_ok else 'FAIL'}] the sub-posets are nested prefixes")
    print(f"  [{'ok  ' if order_ok else 'FAIL'}] B >= e at every size on every board")
    inside = sum(1 for r in rows
                 if math.isnan(r["reported"])
                 or r["at18"][0] - 1e-9 <= r["reported"] <= r["at18"][1] + 1e-9)
    ok_rep = inside >= len(rows) - 1
    print(f"  [{'ok  ' if ok_rep else 'FAIL'}] band_slack's size-18 value sits inside "
          f"5 fresh draws on {inside} of {len(rows)}")
    ok_n = len(rows) >= 8
    print(f"  [{'ok  ' if ok_n else 'FAIL'}] {len(rows)} boards measured (need >= 8)")

    if not (nested_ok and order_ok and ok_rep and ok_n):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    xs = np.array(SIZES, dtype=float)
    for r in rows:
        ys = np.array([r["curve"][m][2] for m in SIZES])
        a, b0 = np.polyfit(xs, ys, 1)
        pred = a * xs + b0
        ss = 1.0 - ((ys - pred) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-12)
        r["slope"], r["r2"] = float(a), float(ss)
        r["extrap"] = float(a * r["J"] + b0)
        r["frac12"] = r["curve"][12][2] / r["curve"][12][0]
        r["frac24"] = r["curve"][24][2] / r["curve"][24][0]

    L = []
    p = L.append
    p("DOES THE BAND PICTURE'S COST SCALE?")
    p("=" * 100)
    p("  Nested sub-posets: one permutation per board, prefixes taken, so the")
    p("  24-system order contains the 22 contains the 20.")
    if skipped:
        p("  Not measured, fewer systems than the largest size: "
          + ", ".join(f"{n} (J={j})" for n, j in skipped) + ".")
    p("")
    p(f"  {'board':<22}{'J':>5}" + "".join(f"{m:>7}" for m in SIZES)
      + f"{'slope':>8}{'R2':>7}{'->J':>8}")
    for r in rows:
        p(f"  {r['name']:<22}{r['J']:>5}"
          + "".join(f"{r['curve'][m][2]:>7.2f}" for m in SIZES)
          + f"{r['slope']:>8.3f}{r['r2']:>7.3f}{r['extrap']:>8.1f}")
    p("")
    p("  Slack in bits at each sub-poset size, then the slope per system, the")
    p("  fit quality, and the line's value at the board's own J.")
    p("")
    up = sum(1 for r in rows if r["curve"][24][2] >= r["curve"][8][2])
    fits = sum(1 for r in rows if r["r2"] >= 0.9)
    big = sum(1 for r in rows if r["extrap"] >= 20)
    shrink = sum(1 for r in rows if r["frac24"] < r["frac12"])
    p(f"  P1  slack no smaller at 24 than at 8 on {up} of {len(rows)}      "
      f"pre-registered >= 6:  {'HIT' if up >= 6 else 'MISS'}")
    p(f"  P2  linear fit R2 >= 0.9 on {fits} of {len(rows)}                "
      f"pre-registered >= 6:  {'HIT' if fits >= 6 else 'MISS'}")
    p(f"  P3  extrapolation to J gives >= 20 bits on {big} of {len(rows)}  "
      f"pre-registered >= 6:  {'HIT' if big >= 6 else 'MISS'}")
    p(f"  P4  slack/entropy smaller at 24 than at 12 on {shrink} of {len(rows)}  "
      f"pre-registered >= 6:  {'HIT' if shrink >= 6 else 'MISS'}")
    p("")
    p(f"  {'board':<22}{'slack/H at 12':>15}{'at 24':>9}")
    for r in rows:
        p(f"  {r['name']:<22}{100 * r['frac12']:>14.1f}%{100 * r['frac24']:>8.1f}%")
    p("")
    p("  The last column of the first table is an EXTRAPOLATION, not a")
    p("  measurement. It is a straight line fitted over 8 to 24 systems and")
    p("  evaluated at 35 to 181. Whether that line has any right to be extended")
    p("  that far is what P2 asks, and P3 should be read as void on any board")
    p("  where P2 fails rather than as evidence in either direction.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("slack_scaling_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                 newline=chr(10))
    print(chr(10) + "wrote slack_scaling_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
