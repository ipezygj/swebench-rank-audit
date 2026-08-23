"""What does pairing buy? Rank sets with and without the pair covariance.

R2 requires simultaneous rank sets computed from the paired bootstrap, and
the standard's justification says a per-system error bar "overstates
precision 8x because it ignores pairing". That figure came from one board.
Here it is measured on all nine, in the currency that matters: the width of
the rank set each system gets.

Two computations, identical except for one line. The paired version uses
    sigma_jk^2 = Var(u_j) + Var(u_k) - 2 Cov(u_j, u_k)
The independence version drops the covariance term, which is exactly what a
board does when it publishes each system's own standard error and compares
them by eye. Everything else - the multiplier bootstrap, the simultaneous
critical value, the stepdown - is the same code path.

The ratio of median rank-set widths is the pairing dividend: how much
resolution a leaderboard already has and does not use.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the independence version gives wider rank sets on >= 8 of 9 boards;
  * the median width ratio exceeds 1.3 on >= 6 of 9;
  * the dividend is largest where kappa is smallest, i.e. it tracks
    1 / mean(kappa) across boards (Spearman < -0.4 between mean kappa and
    the ratio).

SELF-CHECKS
  * on an iid field the two versions agree within 10 % (there is no
    covariance to exploit);
  * on a field with a planted shared component the paired version must be
    strictly narrower.

    python pairing_dividend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import rank_sets as rs
from entropy_law_test import MATRICES
from pair_sharpness import kappa_matrix

SEED = 20260823
DRAWS = 1500


def widths(x, paired: bool, draws=DRAWS):
    """Median rank-set width; paired=False drops the covariance term."""
    if paired:
        r = rs.rank_sets(x, draws=draws)
        return r["worst"] - r["best"] + 1, r

    # Same estimator with independent rows: shuffle each row's items so the
    # cross-system covariance is zero while every row keeps its own variance.
    rng = np.random.default_rng(SEED)
    y = np.empty_like(x)
    for j in range(x.shape[0]):
        y[j] = x[j, rng.permutation(x.shape[1])]
    # the scores must not move - only the co-movement is destroyed
    y = y - y.mean(axis=1, keepdims=True) + x.mean(axis=1, keepdims=True)
    r = rs.rank_sets(y, draws=draws)
    return r["worst"] - r["best"] + 1, r


def _check_iid():
    rng = np.random.default_rng(11)
    x = 0.5 + rng.normal(0, 0.06, 40)[:, None] + rng.normal(0, 0.3, 200)[None, :] + rng.normal(0, 0.45, (40, 200))
    a, _ = widths(x, True, 600)
    b, _ = widths(x, False, 600)
    ratio = float(np.median(b) / np.median(a))
    return abs(ratio - 1) < 0.10, f"iid field: width ratio {ratio:.2f}"


def _check_planted():
    rng = np.random.default_rng(13)
    J, n = 40, 200
    base = rng.normal(0, 0.45, n)
    base -= base.mean()
    x = 0.5 + rng.normal(0, 0.06, J)[:, None] + 0.85 * base[None, :] + np.sqrt(1 - 0.85 ** 2) * rng.normal(0, 0.45, (J, n))
    a, _ = widths(x, True, 600)
    b, _ = widths(x, False, 600)
    return np.median(a) < np.median(b), f"planted shared component: paired {np.median(a):.0f} vs independent {np.median(b):.0f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_iid(), _check_planted()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("THE PAIRING DIVIDEND: RANK-SET WIDTH WITH AND WITHOUT CO-MOVEMENT")
    p("=" * 88)
    p(f"  {'leaderboard':<22} {'J':>4} {'paired':>8} {'independent':>12} {'ratio':>6} "
      f"{'tie@1 paired':>13} {'indep':>7} {'mean kappa':>11}")
    wider, big, ratios, kaps = 0, 0, [], []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        wa, ra = widths(x, True)
        wb, rb = widths(x, False)
        ma, mb = float(np.median(wa)), float(np.median(wb))
        ratio = mb / ma if ma > 0 else float("nan")
        K = kappa_matrix(x)
        iu = np.triu_indices(x.shape[0], k=1)
        mk = float(np.nanmean(K[iu]))
        wider += mb > ma
        big += ratio > 1.3
        ratios.append(ratio); kaps.append(mk)
        p(f"  {name:<22} {x.shape[0]:>4} {ma:>8.0f} {mb:>12.0f} {ratio:>6.2f} "
          f"{int((ra['best'] == 1).sum()):>13} {int((rb['best'] == 1).sum()):>7} {mk:>11.3f}")
    N = len(ratios)
    r = spearmanr(kaps, ratios)
    p("")
    p(f"  independent version wider: {wider}/{N} (pre-registered >= 8)")
    p(f"  ratio above 1.3: {big}/{N} (pre-registered >= 6)")
    p(f"  Spearman(mean kappa, ratio) = {r.statistic:+.2f} (p {r.pvalue:.2f}); pre-registered < -0.4")
    p("")
    p("  'independent' repeats the identical estimator on a board whose rows have")
    p("  been permuted item-wise: every system keeps its score and its own")
    p("  variance, only the co-movement between systems is destroyed. The ratio")
    p("  is the resolution a leaderboard already owns and throws away when it")
    p("  publishes per-system error bars instead of paired comparisons.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("pairing_dividend_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote pairing_dividend_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
