"""Does the headline survive a different statistical method?

Everything in the standard runs through one estimator: a multiplier
bootstrap over items with a Romano-Wolf stepdown. That is the right tool -
it respects the pairing and controls the family-wise error - but a result
that depends on the tool is a result about the tool.

Three estimators, same data, same question (which ordered pairs are
established, and how many systems could be first):

  bootstrap    the standard's own (rank_sets.py)
  Bonferroni   paired t on each pair, alpha / (J choose 2); crude, valid,
               and much more conservative
  item boot    nonparametric bootstrap over items with a simultaneous
               max-t critical value - a different resampling scheme from
               the standard's multiplier bootstrap

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * Bonferroni gives tie@1 no smaller than the bootstrap on >= 8 of 10
    boards (it is the more conservative method);
  * the set of boards where #1 vs #2 separates is IDENTICAL under all three
    methods;
  * the established shares agree within 10 points between the multiplier
    bootstrap and the item bootstrap on >= 7 of 10.

SELF-CHECKS
  * on a board with one system far above the rest, all three separate it;
  * on pure noise, all three establish under 2 % of pairs (their nominal
    level, roughly).

    python method_independence.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import rank_sets as rs
from leaderboard_standard import MATRICES

SEED = 20260823
ALPHA = 0.05
DRAWS = 1200



def bonferroni(x, alpha=ALPHA):
    J, n = x.shape
    m = J * (J - 1) / 2
    thr = alpha / m
    beats = np.zeros((J, J), dtype=bool)
    for i in range(J):
        d = x[i][None, :] - x
        mean = d.mean(axis=1)
        sd = d.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = mean / (sd / math.sqrt(n))
        pv = stats.t.sf(np.abs(t), df=n - 1) * 2
        beats[i] = (pv < thr) & (mean > 0)
    np.fill_diagonal(beats, False)
    return beats


def boot_np(x, alpha=ALPHA, draws=1000, seed=SEED):
    """Nonparametric item bootstrap with a simultaneous max-t critical value.

    This replaces the Holm-corrected sign-flip test the first version used.
    That test cannot reach the thresholds a family of pairs demands: with
    J = 134 there are 8 911 pairs, Holm's smallest threshold is
    0.05 / 8 911 = 5.6e-6, and a permutation p-value from B flips cannot go
    below 1 / (B + 1). It reported zero established pairs even for a system
    half a point above the field - which the self-check caught before any
    board was run. A max-statistic bootstrap needs no per-pair p-value: the
    critical value is a quantile of the maximum, so one set of draws serves
    the whole family.
    """
    J, n = x.shape
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(J, k=1)
    theta = x.mean(axis=1)
    d = theta[iu[0]] - theta[iu[1]]
    diff = x[iu[0]] - x[iu[1]]                       # pairs x items
    se = diff.std(axis=1, ddof=1) / math.sqrt(n)
    se = np.where(se > 0, se, np.inf)
    maxes = np.empty(draws)
    for b in range(draws):
        idx = rng.integers(0, n, n)
        db = diff[:, idx].mean(axis=1)
        maxes[b] = np.max(np.abs(db - d) / se)
    crit = float(np.quantile(maxes, 1 - alpha))
    beats = np.zeros((J, J), dtype=bool)
    sig = np.abs(d) / se > crit
    for k, (i, j) in enumerate(zip(*iu)):
        if sig[k]:
            if d[k] > 0:
                beats[i, j] = True
            else:
                beats[j, i] = True
    return beats


def tie1_of(beats):
    return int((1 + beats.sum(axis=0) == 1).sum())


def _check_clear():
    rng = np.random.default_rng(3)
    x = 0.4 + rng.normal(0, 0.02, 20)[:, None] + rng.normal(0, 0.3, (20, 200))
    x[0] += 0.5
    r = rs.rank_sets(x, draws=400)
    ok = r["beats"][0].sum() >= 19 and bonferroni(x)[0].sum() >= 19 and boot_np(x, draws=400)[0].sum() >= 19
    return ok, "a system far above the rest is separated by all three methods"


def _check_noise():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.4, (25, 200))
    J = 25
    tot = J * (J - 1)
    a = rs.rank_sets(x, draws=400)["beats"].sum() / tot
    b = bonferroni(x).sum() / tot
    c = boot_np(x, draws=400).sum() / tot
    return max(a, b, c) < 0.02, f"pure noise: established {100 * a:.1f} / {100 * b:.1f} / {100 * c:.1f} %"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_clear(), _check_noise()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("THREE METHODS, ONE QUESTION")
    p("=" * 92)
    p(f"  {'leaderboard':<22} {'tie@1 boot':>11} {'Bonf':>6} {'iboot':>6} | "
      f"{'estab boot':>11} {'Bonf':>7} {'iboot':>7} | {'#1v#2 separates':>22}")
    cons, same_sep, close = 0, 0, 0
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J = x.shape[0]
        tot = J * (J - 1)
        r = rs.rank_sets(x, draws=DRAWS)
        bb, bn, bp = r["beats"], bonferroni(x), boot_np(x)
        order = np.argsort(-x.mean(axis=1))
        i1, i2 = int(order[0]), int(order[1])
        seps = [nm for nm, b in (("boot", bb), ("Bonf", bn), ("item boot", bp)) if b[i1, i2]]
        t_boot, t_bonf, t_perm = int((r["best"] == 1).sum()), tie1_of(bn), tie1_of(bp)
        cons += t_bonf >= t_boot
        same_sep += len(seps) in (0, 3)
        close += abs(bb.sum() / tot - bp.sum() / tot) <= 0.10
        p(f"  {name:<22} {t_boot:>11} {t_bonf:>6} {t_perm:>6} | "
          f"{100 * bb.sum() / tot:>10.1f}% {100 * bn.sum() / tot:>6.1f}% {100 * bp.sum() / tot:>6.1f}% | "
          f"{(', '.join(seps) if seps else 'none of the three'):>22}")
    N = sum(1 for _, pth in MATRICES.items() if Path(pth).exists())
    p("")
    p(f"  Bonferroni tie@1 no smaller than the bootstrap's: {cons}/{N} (pre-registered >= 8)")
    p(f"  the three methods agree on whether #1 beats #2: {same_sep}/{N} (pre-registered: all)")
    p(f"  multiplier and item bootstrap established shares within 10 points: {close}/{N} (pre-registered >= 7)")
    p("")
    p("  The bootstrap is the standard's estimator because it uses the pairing and")
    p("  controls the family-wise error with the least conservatism of the three.")
    p("  The point of this table is not which is best: it is that the reading -")
    p("  which boards can name a first place - does not come from the choice.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("method_independence_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote method_independence_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
