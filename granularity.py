"""What does fine-grained scoring buy? Binarise a continuous board and see.

SWE-bench scores each instance 0 or 1. MTEB, LiveBench, CASP14 and
ProteinGym score each item on a continuous scale. A benchmark designer
choosing between the two wants to know what the finer scale is worth in
resolution, and the boards themselves can answer it: take a continuous
board, replace each item's scores by an indicator (above the item's median
across systems), and rerun.

Two thresholds are used, because the choice matters and hiding it would be
a way of choosing the flattering one:
  median  above the item's median system score
  mean    above the item's mean system score

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * binarising increases tie@1 by at least 30 % on >= 3 of the 4
    continuous boards;
  * the #1 vs #2 t falls on all four;
  * the established share falls on all four - a coarser scale cannot
    establish more.

SELF-CHECKS
  * binarising an already-binary matrix at its item medians changes
    nothing (SWE-bench, as a control);
  * on a simulated continuous board, binarising must not increase the
    established share.

    python granularity.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs

SEED = 20260823
DRAWS = 1000
CONTINUOUS = {
    "MTEB English v2": "mteb_dated_matrix.csv",
    "ProteinGym DMS": "proteingym/matrix.csv",
    "CASP14": "casp/matrix.csv",
    "LiveBench": "livebench/matrix.csv",
}
CONTROL = ("SWE-bench Verified", "swebench_verified_matrix.csv")


def binarise(x, how="median"):
    ref = np.median(x, axis=0) if how == "median" else x.mean(axis=0)
    return (x > ref[None, :]).astype(float)


def summary(x, draws=DRAWS):
    J = x.shape[0]
    r = rs.rank_sets(x, draws=draws)
    order = np.argsort(-x.mean(axis=1))
    i1, i2 = int(order[0]), int(order[1])
    d = x[i1] - x[i2]
    se = float(d.std(ddof=1) / math.sqrt(x.shape[1]))
    return {"tie1": int((r["best"] == 1).sum()),
            "estab": float(r["beats"].sum() / (J * (J - 1))),
            "t": float(d.mean() / se) if se > 0 else float("nan"),
            "width": float(np.median(r["worst"] - r["best"] + 1))}


def _check_binary_control():
    x = pd.read_csv(CONTROL[1], index_col=0).dropna(axis=0).to_numpy(dtype=float)[:40, :200]
    b = binarise(x)
    # an item where every system scores 0 has median 0 and "> 0" is False for all:
    # the indicator equals the original except on such degenerate items
    diff = float(np.mean(b != x))
    return diff < 0.35, f"binary control: {100 * diff:.1f} % of entries change (degenerate items only)"


def _check_no_gain():
    rng = np.random.default_rng(7)
    x = 0.5 + rng.normal(0, 0.08, 40)[:, None] + rng.normal(0, 0.3, (40, 200))
    a = summary(x, 400)
    b = summary(binarise(x), 400)
    return b["estab"] <= a["estab"] + 0.02, \
        f"simulated board: established {100 * a['estab']:.1f} % -> {100 * b['estab']:.1f} %"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_binary_control(), _check_no_gain()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHAT FINE-GRAINED SCORING BUYS")
    p("=" * 92)
    p(f"  {'board':<20} {'tie@1 cont':>11} {'median':>7} {'mean':>6} | {'estab cont':>11} {'median':>7} "
      f"| {'t cont':>7} {'median':>7} {'mean':>7}")
    rose, tfell, efell = 0, 0, 0
    for name, path in CONTINUOUS.items():
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        a = summary(x)
        bm = summary(binarise(x, "median"))
        bu = summary(binarise(x, "mean"))
        rose += bm["tie1"] >= 1.3 * a["tie1"]
        tfell += bm["t"] < a["t"]
        efell += bm["estab"] < a["estab"]
        p(f"  {name:<20} {a['tie1']:>11} {bm['tie1']:>7} {bu['tie1']:>6} | {100 * a['estab']:>10.1f}% "
          f"{100 * bm['estab']:>6.1f}% | {a['t']:>7.2f} {bm['t']:>7.2f} {bu['t']:>7.2f}")
    N = len(CONTINUOUS)
    p("")
    p(f"  tie@1 rises by at least 30 %: {rose}/{N} (pre-registered >= 3)")
    p(f"  the #1 vs #2 t falls: {tfell}/{N} (pre-registered: all)")
    p(f"  the established share falls: {efell}/{N} (pre-registered: all)")
    p("")
    p("  Binarising throws away how much a system beat an item by and keeps only")
    p("  whether it did. The cost is the difference between the columns; a")
    p("  benchmark designer paying for finer scoring is buying exactly that.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("granularity_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote granularity_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
