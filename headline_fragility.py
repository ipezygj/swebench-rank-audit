"""How many items decide the headline?

A leaderboard's headline is one sign: does the printed first beat the
printed second. This asks how many items that sign rests on - the smallest
number of items whose removal reverses it, and the smallest number whose
removal makes the pair significant in the other direction.

The removal is adversarial but not exotic: drop the items where the leader
gains most. That is exactly what a different sample of the benchmark, or a
slightly different collection policy, would do by accident.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * on >= 6 of 9 boards, removing five items or fewer flips the sign of the
    top pair's mean difference;
  * on CASP14 it takes at least 20 % of the items;
  * the number of items needed correlates with the pair's t (Spearman above
    0.7) - which would make this a restatement of t rather than new
    information, and is worth knowing either way.

SELF-CHECKS
  * a pair with a huge, uniform difference needs nearly all items removed;
  * a pair with zero difference needs one item.

    python headline_fragility.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from entropy_law_test import MATRICES

SEED = 20260823


def flip_count(d):
    """Fewest items to remove (largest first) to make the mean non-positive."""
    if d.mean() <= 0:
        return 0
    order = np.argsort(-d)
    total = d.sum()
    for k, i in enumerate(order, start=1):
        total -= d[i]
        if total <= 0:
            return k
    return len(d)


def significant_other_way(d):
    """Fewest removals to make the reversed difference significant at 5 %."""
    order = np.argsort(-d)
    keep = list(range(len(d)))
    for k, i in enumerate(order, start=1):
        keep.remove(int(i))
        if len(keep) < 5:
            return len(d)
        v = d[keep]
        sd = float(v.std(ddof=1))
        if sd > 0 and v.mean() / (sd / math.sqrt(len(v))) < -1.96:
            return k
    return len(d)


def _check_huge():
    d = np.full(200, 0.5)
    return flip_count(d) == 200, f"a uniform half-point lead needs {flip_count(d)} of 200 removed"


def _check_zero():
    rng = np.random.default_rng(3)
    d = rng.normal(0, 0.3, 200)
    d = d - d.mean() + 1e-9
    return flip_count(d) <= 1, f"a zero-difference pair needs {flip_count(d)} removed"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_huge(), _check_zero()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("HOW MANY ITEMS DECIDE THE HEADLINE?")
    p("=" * 84)
    p(f"  {'board':<22} {'n':>5} {'top t':>7} {'to flip':>8} {'share':>7} {'to reverse sig.':>16} {'share':>7}")
    few, ts, ks = 0, [], []
    casp = None
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        n = x.shape[1]
        order = np.argsort(-x.mean(axis=1))
        d = x[int(order[0])] - x[int(order[1])]
        sd = float(d.std(ddof=1))
        t = float(d.mean() / (sd / math.sqrt(n))) if sd > 0 else 0.0
        k = flip_count(d)
        k2 = significant_other_way(d)
        few += k <= 5
        ts.append(t); ks.append(k)
        if name == "CASP14":
            casp = k / n
        p(f"  {name:<22} {n:>5} {t:>7.2f} {k:>8} {100 * k / n:>6.1f}% {k2:>16} {100 * k2 / n:>6.1f}%")
    N = len(ts)
    r = spearmanr(ts, ks)
    p("")
    p(f"  five items or fewer flip the sign: {few}/{N} (pre-registered >= 6)")
    p(f"  CASP14 needs at least 20 % of its items: {'yes' if casp and casp >= 0.20 else 'NO'} "
      f"({100 * casp:.1f} %)")
    p(f"  Spearman(top t, items needed) = {r.statistic:+.2f} (p {r.pvalue:.3f}); pre-registered above 0.7")
    p("")
    p("  'to flip' removes the items where the leader gains most, until the mean")
    p("  difference is no longer positive. 'to reverse sig.' keeps going until the")
    p("  runner-up leads significantly. Both are what a differently drawn benchmark")
    p("  could have produced without anybody doing anything unusual.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("headline_fragility_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote headline_fragility_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
