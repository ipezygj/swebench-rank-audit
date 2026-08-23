"""Is the top unresolvable because the items are too easy for it?

Every board here has items its best systems all solve. Those items cost
evaluation budget and contribute nothing to separating the top - the
established-pair machinery already knows this, but nobody has measured how
much of each benchmark is in that state FOR THE TOP GROUP specifically,
as opposed to the field as a whole.

For each board: take the top ten systems, and classify every item by how
many of them get it right (binary boards) or by the spread of their scores
on it (continuous boards):

  live      the top systems disagree on it (their range exceeds 5 % of the
            field's range on that item) - the only items that can separate them
  saturated they agree, at or above the field median
  dead      they agree, below it

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * on every board the live share for the top ten is below 60 %;
  * SWE-bench Verified has the lowest live share of the binary boards;
  * the live count, not the total item count, tracks tie@1 across boards
    (Spearman between live count and tie@1 is negative).

SELF-CHECKS
  * on a matrix where every system solves everything, the live share is 0;
  * on a matrix of coin flips, the live share is near 1.

    python ceiling_effect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import rank_sets as rs
from entropy_law_test import MATRICES

SEED = 20260823
TOP = 10


def classify(x, top_idx, flat_frac=0.05):
    """dead / saturated / live counts for the top group.

    The first version called an item dead when no top system scored above
    the item's field MEDIAN. On binary data that is wrong: a column whose
    median is 1 can never be exceeded, so 79 of 200 coin-flip columns were
    recorded as dead and the self-check caught it (live share 60 % where
    the right answer is nearly 100 %).

    The definition that works on both kinds of data looks only at the top
    group's own spread. An item is LIVE when the top systems disagree on
    it - their range exceeds a small fraction of the field's range on that
    item. When they agree, it is SATURATED if they agree above the field
    median and DEAD if below. Binary data reduces to the obvious thing:
    all top systems solve it, or none does.
    """
    sub = x[top_idx]
    rng_field = x.max(axis=0) - x.min(axis=0)
    rng_top = sub.max(axis=0) - sub.min(axis=0)
    live_mask = rng_top > flat_frac * np.where(rng_field > 0, rng_field, 1.0)
    med = np.median(x, axis=0)
    above = sub.mean(axis=0) >= med
    live = int(live_mask.sum())
    sat = int((~live_mask & above).sum())
    dead = int((~live_mask & ~above).sum())
    return dead, sat, live


def _check_all_solved():
    x = np.ones((20, 50))
    return classify(x, list(range(10)))[2] == 0, "a matrix everyone solves has no live items"


def _check_coinflips():
    rng = np.random.default_rng(3)
    x = (rng.random((20, 200)) < 0.5).astype(float)
    d, s, l = classify(x, list(range(10)))
    return l / 200 > 0.9, f"coin flips: live share {100 * l / 200:.0f} %"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_all_solved(), _check_coinflips()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("HOW MANY ITEMS CAN STILL SEPARATE THE TOP TEN?")
    p("=" * 84)
    p(f"  {'board':<22} {'n':>5} {'dead':>6} {'saturated':>10} {'live':>6} {'live share':>11} {'tie@1':>6}")
    below, lives, ties, rows = 0, [], [], []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        order = np.argsort(-x.mean(axis=1))
        top = [int(i) for i in order[: min(TOP, x.shape[0])]]
        d, s, l = classify(x, top)
        share = l / x.shape[1]
        t1 = int((rs.rank_sets(x, draws=800)["best"] == 1).sum())
        below += share < 0.60
        lives.append(l); ties.append(t1); rows.append((name, share, l, t1))
        p(f"  {name:<22} {x.shape[1]:>5} {d:>6} {s:>10} {l:>6} {100 * share:>10.0f}% {t1:>6}")
    N = len(rows)
    r = spearmanr(lives, ties)
    binary_lowest = min((rw for rw in rows if rw[0].startswith("SWE") or rw[0].startswith("MathArena")),
                        key=lambda rw: rw[1], default=None)
    p("")
    p(f"  live share below 60 %: {below}/{N} (pre-registered: all)")
    p(f"  lowest live share among the binary boards: {binary_lowest[0] if binary_lowest else '-'} "
      f"({100 * binary_lowest[1]:.0f} %)" if binary_lowest else "")
    p(f"  Spearman(live count, tie@1) = {r.statistic:+.2f} (p {r.pvalue:.2f}); pre-registered negative")
    p("")
    p("  An item is live for the top group when the top ten disagree on it by")
    p("  more than 5 % of the field's range; otherwise they agree, above the")
    p("  field median (saturated) or below it (dead). Only live items carry")
    p("  information about which of the ten is best - and the benchmark pays to")
    p("  run all of them. A high live share is not automatically good: on a")
    p("  continuous board almost every item is technically live because exact")
    p("  ties are rare, which is why the binary boards are the informative rows.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("ceiling_effect_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote ceiling_effect_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
