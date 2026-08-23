"""How often does the crown change hands if the items are resampled?

Rank sets say which ranks are compatible with the data. A blunter question,
and the one a reader actually asks: if this benchmark had drawn a different
sample of items from the same pool, would the same system be on top?

Item bootstrap: resample the n items with replacement, recompute the mean
scores, see who is first. Repeated 2000 times per board. Two numbers:
the share of resamples the printed leader keeps, and how many distinct
systems take the crown at least once.

This is not the same as the rank set. A rank set is a simultaneous
statement about all systems at once; crown stability is a marginal
statement about one position, and it is the one a headline makes.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the printed leader keeps the crown in fewer than 60 % of resamples on
    >= 7 of 9 boards;
  * at least 5 distinct systems take the crown on >= 6 of 9;
  * crown stability is higher than tie@1 would suggest - a rank set
    contains 1 whenever separation fails, but the bootstrap weights by how
    often it actually happens, so the number of distinct crown-takers is
    SMALLER than tie@1 on >= 7 of 9.

SELF-CHECKS
  * a field with one system far above the rest keeps the crown ~100 % of
    the time;
  * a field of identical systems spreads the crown roughly uniformly.

    python crown_stability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from entropy_law_test import MATRICES

SEED = 20260823
BOOT = 2000


def crown(x, boot=BOOT, seed=SEED):
    J, n = x.shape
    rng = np.random.default_rng(seed)
    lead = int(np.argmax(x.mean(axis=1)))
    counts = np.zeros(J, dtype=int)
    # resample items in blocks to keep memory bounded
    step = 200
    done = 0
    while done < boot:
        k = min(step, boot - done)
        idx = rng.integers(0, n, size=(k, n))
        means = x[:, idx].mean(axis=2)            # J x k
        winners = np.argmax(means, axis=0)
        np.add.at(counts, winners, 1)
        done += k
    return lead, counts


def _check_clear():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.4, (20, 300))
    x[0] += 1.5
    lead, c = crown(x, boot=400, seed=1)
    return c[lead] / 400 > 0.95, f"one system far above: keeps the crown {100 * c[lead] / 400:.0f} %"


def _check_identical():
    rng = np.random.default_rng(5)
    base = rng.normal(0, 0.4, (10, 300))
    x = base - base.mean(axis=1, keepdims=True)     # identical abilities
    lead, c = crown(x, boot=1000, seed=2)
    share = c.max() / 1000
    return share < 0.35, f"identical systems: the most frequent winner takes {100 * share:.0f} %"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_clear(), _check_identical()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("IF THE ITEMS WERE RESAMPLED, WHO WOULD BE FIRST?")
    p("=" * 84)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>5} {'leader keeps':>13} {'runner-up':>10} "
      f"{'distinct winners':>17} {'tie@1':>6}")
    weak, many, fewer = 0, 0, 0
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J, n = x.shape
        lead, c = crown(x)
        share = c[lead] / BOOT
        order = np.argsort(-c)
        second = c[order[1]] / BOOT if order[0] == lead else c[order[0]] / BOOT
        distinct = int((c > 0).sum())
        tie1 = int((rs.rank_sets(x, draws=800)["best"] == 1).sum())
        weak += share < 0.60
        many += distinct >= 5
        fewer += distinct < tie1
        p(f"  {name:<22} {J:>4} {n:>5} {100 * share:>12.1f}% {100 * second:>9.1f}% "
          f"{distinct:>17} {tie1:>6}")
    N = sum(1 for _, pth in MATRICES.items() if Path(pth).exists())
    p("")
    p(f"  leader keeps the crown in under 60 % of resamples: {weak}/{N} (pre-registered >= 7)")
    p(f"  at least 5 distinct crown-takers: {many}/{N} (pre-registered >= 6)")
    p(f"  distinct crown-takers fewer than tie@1: {fewer}/{N} (pre-registered >= 7)")
    p("")
    p("  Item bootstrap, 2000 resamples with replacement. 'leader keeps' is the")
    p("  share of resamples in which the printed first-place system is still")
    p("  first. tie@1 is the simultaneous statement - how many systems have a")
    p("  rank set containing 1 - and it is the more conservative of the two.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("crown_stability_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote crown_stability_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
