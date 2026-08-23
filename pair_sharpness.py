"""Pair sharpness: resolution is a property of the PAIR, not the benchmark.

broad_or_deep.py located the separability gap: at the same step size, a real
frontier pair has a larger paired statistic than a simulated one, and not
because the new leader wins on more items (w barely moves). It is the SD of
the difference vector that is smaller.

Define, for a pair (j, k),

    kappa(j,k) = sd(x_j - x_k) / sqrt(sd(x_j)^2 + sd(x_k)^2)

the observed difference SD over what independence would give. kappa = 1 is
independence; kappa < 1 means the two systems' item-level behaviour moves
together and the comparison between them is SHARPER than the benchmark's
global resolution suggests; kappa > 1 means the opposite.

Every twin in this repo collapses kappa to a single value. If frontier
pairs sit systematically below the board's typical kappa, then the
resolution that matters for a SOTA claim is not the benchmark's, and a
standard that quotes one resolution for the whole board understates the
evidence for exactly the comparisons people care about.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * median kappa of frontier pairs is below the median kappa of all pairs
    on >= 4 of 5 boards;
  * it is also below the median kappa of ABILITY-MATCHED pairs (pairs whose
    score gap is within +-25 % of the frontier pair's) on >= 4 of 5 - so it
    is not just that close systems are similar;
  * exploratory: across boards, the frontier kappa deficit against the P
    gap (real P minus chase-model P from chase_model_results).

SELF-CHECKS
  * kappa is affine-invariant and equals 1 within 0.05 on an iid field;
  * on a field with a planted shared component, kappa of the sharing pairs
    is clearly below 1.

    python pair_sharpness.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from evidence_trajectory import load
from sota_audit import advances
from chase_model import BOARDS

# real P minus chase-model P, from chase_model_results.txt (46ef91c)
P_GAP = {"SWE-bench Verified": 0.11, "SWE-bench Lite": 0.01,
         "MTEB English v2": 0.24, "LiveBench": 0.19, "ProteinGym DMS": 0.27}


def kappa_matrix(x):
    """Item difficulty is removed first.

    The first build used the raw row SDs in the denominator and its iid
    self-check gave 0.85, not 1: a system's score SD across items is mostly
    ITEM DIFFICULTY, which is common to both systems and cancels in the
    difference, so independence does not imply sd(d)^2 = sd_j^2 + sd_k^2 on
    raw scores. Removing the item means leaves each system's own deviation
    from the item's difficulty, and there independence does imply it.
    """
    x = x - x.mean(axis=0, keepdims=True)
    J = x.shape[0]
    sd = x.std(axis=1, ddof=1)
    K = np.ones((J, J))
    for j in range(J):
        d = x[j][None, :] - x
        s = d.std(axis=1, ddof=1)
        indep = np.sqrt(sd[j] ** 2 + sd ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            K[j] = np.where(indep > 0, s / indep, 1.0)
    np.fill_diagonal(K, np.nan)
    return K


def _check_iid():
    rng = np.random.default_rng(11)
    x = rng.normal(0, 0.06, 60)[:, None] + rng.normal(0, 0.3, 200)[None, :] + rng.normal(0, 0.45, (60, 200))
    K = kappa_matrix(x)
    m = float(np.nanmedian(K))
    K2 = kappa_matrix(3 * x + 5)
    return abs(m - 1) < 0.05 and np.allclose(np.nan_to_num(K), np.nan_to_num(K2)), \
        f"iid field: median kappa {m:.3f}; affine invariant"


def _check_planted():
    rng = np.random.default_rng(13)
    J, n = 40, 200
    f = rng.normal(0, 0.45, n)
    noise = rng.normal(0, 0.45, (J, n))
    load_ = 0.9
    x = rng.normal(0, 0.06, J)[:, None] + np.vstack([np.tile(f, (20, 1)), np.zeros((20, n))]) * load_ \
        + math.sqrt(1 - load_ ** 2) * noise
    K = kappa_matrix(x)
    inside = np.nanmedian(K[:20, :20])
    across = np.nanmedian(K[:20, 20:])
    return inside < 0.7 and across > 0.9, f"planted family: kappa inside {inside:.2f}, across {across:.2f}"


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
    p("PAIR SHARPNESS: IS THE FRONTIER COMPARISON SHARPER THAN THE BOARD?")
    p("=" * 86)
    p(f"  {'board':<20} {'adv':>4} {'kappa frontier':>15} {'all pairs':>10} {'gap-matched':>12} "
      f"{'vs all':>7} {'vs matched':>11}")
    okAll, okMatch, defs, gaps = 0, 0, [], []
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        K = kappa_matrix(x)
        sc = x.mean(axis=1)
        J = x.shape[0]
        iu = np.triu_indices(J, k=1)
        allk = float(np.nanmedian(K[iu]))
        fr, matchk = [], []
        for a in advances(x, dates):
            i, j = a["new"], a["old"]
            fr.append(K[i, j])
            g = abs(sc[i] - sc[j])
            gapm = np.abs(sc[:, None] - sc[None, :])
            sel = (np.abs(gapm - g) <= 0.25 * g) & ~np.eye(J, dtype=bool)
            vals = K[sel]
            if vals.size:
                matchk.append(float(np.nanmedian(vals)))
        f_med = float(np.nanmedian(fr))
        m_med = float(np.nanmedian(matchk)) if matchk else float("nan")
        okAll += f_med < allk
        okMatch += f_med < m_med
        defs.append(allk - f_med)
        gaps.append(P_GAP.get(name, float("nan")))
        p(f"  {name:<20} {len(fr):>4} {f_med:>15.3f} {allk:>10.3f} {m_med:>12.3f} "
          f"{'yes' if f_med < allk else 'NO':>7} {'yes' if f_med < m_med else 'NO':>11}")
    N = len(BOARDS)
    p("")
    p(f"  frontier kappa below all-pair kappa: {okAll}/{N} (pre-registered >= 4)")
    p(f"  frontier kappa below gap-matched kappa: {okMatch}/{N} (pre-registered >= 4)")
    r = spearmanr(defs, gaps)
    p(f"  exploratory: Spearman(kappa deficit, P gap) = {r.statistic:+.2f} (p {r.pvalue:.2f})")
    p("")
    p("  kappa = sd(x_j - x_k) / sqrt(sd_j^2 + sd_k^2): 1 under independence,")
    p("  below 1 when two systems move together and their comparison is sharper")
    p("  than the board's global resolution. Every twin in this repo uses one")
    p("  kappa for all pairs; the frontier is where that assumption is tested.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("pair_sharpness_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote pair_sharpness_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
