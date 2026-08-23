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
  permutation  sign-flip on the paired differences, Holm-corrected across
               pairs; distribution-free

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * Bonferroni gives tie@1 no smaller than the bootstrap on >= 8 of 10
    boards (it is the more conservative method);
  * the set of boards where #1 vs #2 separates is IDENTICAL under all three
    methods;
  * the established shares agree within 10 points between bootstrap and
    permutation on >= 7 of 10.

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
FLIPS = 4000


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


def permutation(x, alpha=ALPHA, flips=FLIPS, seed=SEED):
    """Sign-flip test on every pair, Holm-corrected over the pair family."""
    J, n = x.shape
    rng = np.random.default_rng(seed)
    F = rng.choice([-1.0, 1.0], size=(flips, n))
    iu = np.triu_indices(J, k=1)
    pvals = np.ones(len(iu[0]))
    means = np.empty(len(iu[0]))
    for k, (i, j) in enumerate(zip(*iu)):
        d = x[i] - x[j]
        obs = abs(d.mean())
        means[k] = d.mean()
        null = np.abs(F @ d) / n
        pvals[k] = (np.sum(null >= obs - 1e-15) + 1) / (flips + 1)
    order = np.argsort(pvals)
    m = len(pvals)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    beats = np.zeros((J, J), dtype=bool)
    for k, (i, j) in enumerate(zip(*iu)):
        if reject[k]:
            if means[k] > 0:
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
    ok = r["beats"][0].sum() >= 19 and bonferroni(x)[0].sum() >= 19 and permutation(x, flips=1000)[0].sum() >= 19
    return ok, "a system far above the rest is separated by all three methods"


def _check_noise():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.4, (25, 200))
    J = 25
    tot = J * (J - 1)
    a = rs.rank_sets(x, draws=400)["beats"].sum() / tot
    b = bonferroni(x).sum() / tot
    c = permutation(x, flips=1000).sum() / tot
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
    p(f"  {'leaderboard':<22} {'tie@1 boot':>11} {'Bonf':>6} {'perm':>6} | "
      f"{'estab boot':>11} {'Bonf':>7} {'perm':>7} | {'#1v#2 separates':>22}")
    cons, same_sep, close = 0, 0, 0
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J = x.shape[0]
        tot = J * (J - 1)
        r = rs.rank_sets(x, draws=DRAWS)
        bb, bn, bp = r["beats"], bonferroni(x), permutation(x)
        order = np.argsort(-x.mean(axis=1))
        i1, i2 = int(order[0]), int(order[1])
        seps = [nm for nm, b in (("boot", bb), ("Bonf", bn), ("perm", bp)) if b[i1, i2]]
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
    p(f"  bootstrap and permutation established shares within 10 points: {close}/{N} (pre-registered >= 7)")
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
