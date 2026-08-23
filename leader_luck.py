"""The winner's curse, seen in the residuals.

isotonic_families.py found, on all four dated boards, that the new leader
is LESS correlated with the old leader than with the old runner-up. One
explanation needs no families at all: the leader is the system whose
observed score was pushed up most by its item-level luck. That luck is a
residual vector that is, by construction, orthogonal to everybody else's.
A system selected for being on top is therefore a system whose residuals
correlate with the field less than an unselected system's do - and the
runner-up, less selected, less so.

If that is right, it is a property of the selection, not of any one pair:
every system present at a date should correlate less with the leader than
with the runner-up, and the effect should appear on an iid field with no
families, and it should VANISH when the leader is chosen on an independent
half of the items (split-half selection) and the correlation measured on
the other half.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * median corr(system, #1) < median corr(system, #2), isotonic residuals,
    pooled over checkpoint dates, on >= 3 of 4 dated boards;
  * the same inequality on iid Gaussian fields (no families) in >= 80 % of
    100 simulated fields;
  * split-half: when #1 and #2 are chosen on half A of the items and the
    correlations measured on half B, the difference shrinks to within
    +-0.01 on the iid fields (mean over fields).

SELF-CHECKS
  * isotonic residuals have zero item means;
  * on a field where the leader is FIXED in advance (not selected), the
    difference is within +-0.01 over 100 fields - the effect must need
    selection to exist.

    python leader_luck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from isotonic_families import isotonic_residuals
from evidence_trajectory import load, checkpoints
from sota_families import BOARDS


def corr_with(resid, j, others):
    c = np.corrcoef(resid[[j] + list(others)])[0, 1:]
    return c


def leader_gap(x, resid):
    """median corr with #1 minus median corr with #2, over the other systems."""
    sc = x.mean(axis=1)
    order = np.argsort(-sc)
    i1, i2 = order[0], order[1]
    others = [k for k in range(x.shape[0]) if k not in (i1, i2)]
    if len(others) < 3:
        return np.nan
    return float(np.median(corr_with(resid, i1, others)) - np.median(corr_with(resid, i2, others)))


def split_half_gap(x, rng):
    """#1, #2 chosen on half A; correlations from isotonic residuals on half B."""
    n = x.shape[1]
    perm = rng.permutation(n)
    A, B = perm[: n // 2], perm[n // 2:]
    sc = x[:, A].mean(axis=1)
    order = np.argsort(-sc)
    i1, i2 = order[0], order[1]
    rB = isotonic_residuals(x[:, B])
    others = [k for k in range(x.shape[0]) if k not in (i1, i2)]
    return float(np.median(corr_with(rB, i1, others)) - np.median(corr_with(rB, i2, others)))


def iid_field(rng, J=60, n=150):
    return 0.5 + rng.normal(0, 0.08, J)[:, None] + rng.normal(0, 0.45, (J, n))


def _check_zero_mean():
    r = isotonic_residuals(np.random.default_rng(1).random((30, 40)))
    return float(np.abs(r.mean(axis=0)).max()) < 1e-9, "isotonic residuals zero item mean"


def _check_fixed_leader():
    gaps = []
    for s in range(100):
        rng = np.random.default_rng(500 + s)
        x = iid_field(rng)
        r = isotonic_residuals(x)
        # 'leader' and 'runner-up' fixed as rows 0 and 1, regardless of score
        others = list(range(2, x.shape[0]))
        gaps.append(np.median(corr_with(r, 0, others)) - np.median(corr_with(r, 1, others)))
    m = float(np.mean(gaps))
    return abs(m) < 0.01, f"fixed (unselected) leader: mean gap {m:+.4f} (must be within 0.01)"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_zero_mean(), _check_fixed_leader()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("THE WINNER'S CURSE IN THE RESIDUALS: DOES THE FIELD CORRELATE LESS WITH #1 THAN #2?")
    p("=" * 86)
    # iid fields
    gaps, shg = [], []
    for s in range(100):
        rng = np.random.default_rng(900 + s)
        x = iid_field(rng)
        gaps.append(leader_gap(x, isotonic_residuals(x)))
        shg.append(split_half_gap(x, rng))
    gaps, shg = np.array(gaps), np.array(shg)
    p(f"  iid fields (100): gap < 0 in {100 * np.mean(gaps < 0):.0f} % (pre-registered >= 80 %), mean gap {gaps.mean():+.4f}")
    p(f"  split-half on the same fields: mean gap {shg.mean():+.4f} (pre-registered within +-0.01)")
    p("")
    p(f"  {'dated board':<20} {'checkpoints':>11} {'median gap':>10} {'gap<0 share':>11}")
    neg = 0
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        g = []
        for d in checkpoints(dates, k=10, jmin=10):
            xs = x[dates <= d]
            g.append(leader_gap(xs, isotonic_residuals(xs)))
        g = np.array([v for v in g if not np.isnan(v)])
        neg += np.median(g) < 0
        p(f"  {name:<20} {len(g):>11} {np.median(g):>+10.3f} {100 * np.mean(g < 0):>10.0f}%")
    p("")
    p(f"  median gap < 0 on {neg}/4 dated boards (pre-registered >= 3)")
    p("")
    p("  gap = median corr(system, #1) - median corr(system, #2) over the other")
    p("  systems present, isotonic residuals. Negative = the field resembles the")
    p("  leader less than the runner-up. On an iid field that can only be the")
    p("  selection: the leader is the system whose luck was largest, and luck is")
    p("  orthogonal to everyone. Split-half removes the selection and the gap.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("leader_luck_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote leader_luck_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
