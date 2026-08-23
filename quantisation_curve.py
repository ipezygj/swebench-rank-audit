"""How many score levels does an item need?

granularity.py showed that binarising a continuous board costs a great
deal - CASP14's decisive top pair (t = 9.89) falls to t = 1.78 when each
target is scored above/below its median. That leaves the practical
question a benchmark designer actually faces: not binary versus
continuous, but how many levels are enough. A three-point rubric is cheap
to write and cheap to grade; a continuous metric is neither.

Quantise each item's scores to k equally spaced levels between that item's
minimum and maximum across systems, for k = 2, 3, 4, 6, 8, 16, and compare
with the unquantised board.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * tie@1 is within 2 of its continuous value by k = 8 on >= 3 of 4 boards;
  * the largest single improvement is from k = 2 to k = 3 or 4;
  * CASP14's top-pair t recovers to above 5 by k = 8.

SELF-CHECKS
  * k = 2 leaves exactly two distinct values per item;
  * quantising with k larger than the number of distinct values leaves the
    matrix unchanged.

    python quantisation_curve.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs

SEED = 20260823
DRAWS = 800
LEVELS = (2, 3, 4, 6, 8, 16)
BOARDS = {
    "MTEB English v2": "mteb_dated_matrix.csv",
    "ProteinGym DMS": "proteingym/matrix.csv",
    "CASP14": "casp/matrix.csv",
    "LiveBench": "livebench/matrix.csv",
}


def quantise(x, k):
    lo = x.min(axis=0, keepdims=True)
    hi = x.max(axis=0, keepdims=True)
    rng = np.where(hi > lo, hi - lo, 1.0)
    z = (x - lo) / rng
    return np.round(z * (k - 1)) / (k - 1)


def summary(x, draws=DRAWS):
    J = x.shape[0]
    r = rs.rank_sets(x, draws=draws)
    order = np.argsort(-x.mean(axis=1))
    i1, i2 = int(order[0]), int(order[1])
    d = x[i1] - x[i2]
    se = float(d.std(ddof=1) / math.sqrt(x.shape[1]))
    return int((r["best"] == 1).sum()), (float(d.mean() / se) if se > 0 else float("nan"))


def _check_two_levels():
    """k = 2 must leave exactly two distinct values per item.

    The first version compared k = 2 against the median split of
    granularity.py and allowed a difference of 6 in tie@1; it came out 8.
    That was a comparison of two different threshold RULES (item midpoint
    versus item median), which can legitimately differ - a weak check
    dressed as a strong one. This checks the quantiser instead.
    """
    x = pd.read_csv(BOARDS["CASP14"], index_col=0).dropna(axis=0).to_numpy(dtype=float)
    q = quantise(x, 2)
    counts = [len(np.unique(q[:, i])) for i in range(q.shape[1])]
    return max(counts) <= 2, f"k = 2 leaves at most {max(counts)} distinct values per item"


def _check_identity():
    """Quantising a matrix that already sits on k levels changes nothing.

    The first version used a matrix whose columns did not span 0 to 1, so
    the per-item rescaling moved every value and the check failed for a
    reason that had nothing to do with the quantiser. The columns are now
    constructed to span the full range.
    """
    rng = np.random.default_rng(3)
    x = np.round(rng.random((20, 50)) * 3) / 3
    x[0, :] = 0.0
    x[1, :] = 1.0
    q = quantise(x, 4)
    return np.allclose(x, q, atol=1e-9), "quantising a 4-level matrix to 4 levels changes nothing"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_two_levels(), _check_identity()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("HOW MANY SCORE LEVELS DOES AN ITEM NEED?")
    p("=" * 90)
    p(f"  {'board':<20} " + " ".join(f"{'k=' + str(k):>7}" for k in LEVELS)
      + f" {'cont':>7} | {'t k=2':>7} {'t k=8':>7} {'t cont':>7}")
    by8, jump, casp8 = 0, 0, None
    for name, path in BOARDS.items():
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        ties, ts = {}, {}
        for k in LEVELS:
            ties[k], ts[k] = summary(quantise(x, k))
        t_cont, tt_cont = summary(x)
        by8 += abs(ties[8] - t_cont) <= 2
        gains = {k: ties[prev] - ties[k] for prev, k in zip(LEVELS, LEVELS[1:])}
        best_k = max(gains, key=gains.get) if gains else None
        jump += best_k in (3, 4)
        if name == "CASP14":
            casp8 = ts[8]
        p(f"  {name:<20} " + " ".join(f"{ties[k]:>7}" for k in LEVELS)
          + f" {t_cont:>7} | {ts[2]:>7.2f} {ts[8]:>7.2f} {tt_cont:>7.2f}"
          + (f"   biggest gain at k={best_k}" if best_k else ""))
    N = len(BOARDS)
    p("")
    p(f"  tie@1 within 2 of continuous by k = 8: {by8}/{N} (pre-registered >= 3)")
    p(f"  the biggest single gain is at k = 3 or 4: {jump}/{N} (pre-registered: most)")
    p(f"  CASP14 top-pair t above 5 at k = 8: {'yes' if casp8 and casp8 > 5 else 'NO'} ({casp8:.2f})")
    p("")
    p("  Levels are equally spaced between each item's own minimum and maximum")
    p("  across systems, so the quantisation is per item and uses no information")
    p("  the grader would not have. A board that reaches its continuous tie@1 at")
    p("  k = 4 can be graded on a four-point rubric without losing resolution.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("quantisation_curve_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote quantisation_curve_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
