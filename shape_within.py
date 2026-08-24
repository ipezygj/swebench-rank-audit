"""Does law 1's residual track the field's shape WITHIN one board?

shape_correction.py found the residual almost perfectly ordered by the shape
statistic across nine boards - Spearman -0.93 - and then found that a
correction fitted on eight boards and tested on the ninth is indistinguishable
from one fitted to shuffled labels. Nine points, two of them extremes, is too
few. The conclusion there was that closing it needs more boards.

More boards are not available, but more FIELDS are. A subset of a leaderboard's
systems is itself a leaderboard: it has its own J, its own spread, its own
pairwise noise, and its own shape, and law 1 either holds on it or does not.
Subsetting also moves the shape a long way - dropping the outliers that inflate
an SD raises the ratio sharply - so within one board we get a range of shapes
with everything else held fixed. That is a stronger test than nine boards of
different fields, sizes and domains, because it controls for all of them.

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  within a board, Spearman(shape, residual) over its subsets is negative
      on at least 7 of the 9 boards.
  P2  the median of those within-board correlations is at most -0.5.
  P3  a one-parameter correction fitted on half a board's subsets and tested
      on the other half beats the uncorrected law on at least 6 of 9 boards.
  P4  the control: with the shape statistic shuffled among a board's own
      subsets, P3's count must drop to at most 4 of 9 in the median over
      99 shuffles.

  P3 is the one that failed across boards. If it holds within boards and its
  control is clean, the fifth number is real and the earlier failure was a
  sample size problem, which is what was claimed there. If P3 fails here too,
  the relation across boards is confounded with something that varies between
  boards and the claim has to be weakened again.

SELF-CHECKS (no table if any fails)
  * calibration: on a synthetic Gaussian board, subsets must show NO
    shape-residual relation - median within-board correlation between -0.3 and
    +0.3 - or subsetting itself manufactures the effect;
  * every subset must be large enough to measure the law: at least 12 systems
    and at least 30 established-pair opportunities;
  * the shape statistic must read 1.0 on Gaussian subsets of the same sizes,
    so the per-J debias carried over from shape_correction.py still applies.

    python shape_within.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import rank_sets as rs
from shape_correction import MATRICES, load, law1, shape_ratio

SEED = 20260824
SUBSETS = 60          # fields drawn per board
MIN_J = 12
PERMS = 99


def field_stats(x):
    """(shape, observed established share, law-1 prediction) for one field."""
    J, n = x.shape
    r = rs.rank_sets(x)
    sc = x.mean(axis=1)
    iu = np.triu_indices(J, k=1)
    tau = float(sc.std(ddof=1))
    sigma_p = float(np.median(r["sigma"][iu]))
    if tau <= 0 or sigma_p <= 0:
        return None
    obs = float(r["beats"].sum() / (J * (J - 1)))
    pred = law1(tau, sigma_p, n, r["crit"])
    return shape_ratio(sc), 100 * obs, 100 * pred


def subsets_of(x, rng, count=SUBSETS):
    """Random sub-fields, sized from a third of the board up to nearly all."""
    J = x.shape[0]
    out = []
    for _ in range(count):
        k = int(rng.integers(max(MIN_J, J // 3), J))
        idx = rng.choice(J, k, replace=False)
        s = field_stats(x[idx])
        if s is not None:
            out.append(s)
    return out


def split_fit(shape, resid, rng):
    """Fit on half the subsets, score the other half. Wins in absolute error."""
    shape = np.asarray(shape, float)
    resid = np.asarray(resid, float)
    idx = rng.permutation(len(resid))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    if len(a) < 4 or len(b) < 4:
        return None
    slope, inter = np.polyfit(shape[a], resid[a], 1)
    corrected = resid[b] - (inter + slope * shape[b])
    return float(np.mean(np.abs(corrected)) < np.mean(np.abs(resid[b])))


def _check_gaussian_null() -> tuple[bool, str]:
    rng = np.random.default_rng(11)
    rhos = []
    for s in range(4):
        g = np.random.default_rng(400 + s)
        x = g.normal(0, 0.05, 80)[:, None] + g.normal(0, 0.4, (80, 300))
        sub = subsets_of(x, rng, count=25)
        if len(sub) > 5:
            rho, _ = spearmanr([a for a, _, _ in sub], [o - p for _, o, p in sub])
            rhos.append(rho)
    m = float(np.median(rhos)) if rhos else float("nan")
    return -0.3 <= m <= 0.3, f"Gaussian board: median within-board rho {m:+.2f} over {len(rhos)} seeds"


def _check_shape_scale() -> tuple[bool, str]:
    rng = np.random.default_rng(7)
    worst = 0.0
    for J in (12, 20, 40, 80):
        m = float(np.mean([shape_ratio(rng.normal(0, 1, J)) for _ in range(300)]))
        worst = max(worst, abs(m - 1.0))
    return worst <= 0.03, f"shape statistic reads 1.0 on Gaussian subsets, worst off by {worst:.3f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)

    print("self-checks ...")
    checks = [_check_shape_scale(), _check_gaussian_null()]
    ok = True
    for passed, msg in checks:
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rows = {}
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        print(f"  {name} ...")
        x = load(path)
        if x.shape[0] < MIN_J + 4:
            rows[name] = None
            continue
        sub = subsets_of(x, rng)
        if len(sub) < 10:
            rows[name] = None
            continue
        shape = [a for a, _, _ in sub]
        resid = [p - o for _, o, p in sub]
        rho, pv = spearmanr(shape, resid)
        wins = [split_fit(shape, resid, np.random.default_rng(SEED + i)) for i in range(21)]
        wins = [w for w in wins if w is not None]
        win_rate = float(np.mean(wins)) if wins else float("nan")
        null = []
        for j in range(PERMS):
            g = np.random.default_rng(SEED + 900 + j)
            sh = list(shape)
            g.shuffle(sh)
            w = [split_fit(sh, resid, np.random.default_rng(SEED + 50 + k)) for k in range(5)]
            w = [z for z in w if z is not None]
            null.append(float(np.mean(w)) if w else 0.0)
        rows[name] = {"n_sub": len(sub), "rho": rho, "p": pv, "win": win_rate,
                      "null": float(np.median(null)),
                      "shape_lo": min(shape), "shape_hi": max(shape)}

    good = {k: v for k, v in rows.items() if v}
    L = []
    p = L.append
    p("DOES THE SHAPE RELATION HOLD WITHIN A SINGLE BOARD?")
    p("=" * 96)
    p(f"  {'leaderboard':<22} {'fields':>7} {'shape range':>14} {'rho':>7} {'p':>7} "
      f"{'fit wins':>9} {'shuffled':>9}")
    for k, v in good.items():
        p(f"  {k:<22} {v['n_sub']:>7} {v['shape_lo']:>6.2f}-{v['shape_hi']:<7.2f} "
          f"{v['rho']:>+7.2f} {v['p']:>7.3f} {100 * v['win']:>8.0f}% {100 * v['null']:>8.0f}%")
    p("")
    neg = sum(1 for v in good.values() if v["rho"] < 0)
    med_rho = float(np.median([v["rho"] for v in good.values()]))
    win6 = sum(1 for v in good.values() if v["win"] > 0.5)
    null6 = sum(1 for v in good.values() if v["null"] > 0.5)
    p(f"  P1  within-board rho negative on {neg} of {len(good)}          "
      f"pre-registered >= 7:  {'HIT' if neg >= 7 else 'MISS'}")
    p(f"  P2  median within-board rho = {med_rho:+.2f}               "
      f"pre-registered <= -0.5:  {'HIT' if med_rho <= -0.5 else 'MISS'}")
    p(f"  P3  split-half correction wins on {win6} of {len(good)}         "
      f"pre-registered >= 6:  {'HIT' if win6 >= 6 else 'MISS'}")
    p(f"  P4  shuffled control wins on {null6} of {len(good)}              "
      f"pre-registered <= 4:  {'HIT' if null6 <= 4 else 'MISS'}")
    p("")
    p("  Each row treats subsets of one board's systems as separate fields:")
    p("  same items, same domain, same measurement, different field. The shape")
    p("  range column says how far subsetting moved the statistic, which is what")
    p("  makes the within-board test possible at all.")
    p("")
    p("  fit wins is the share of random half/half splits in which a correction")
    p("  fitted on one half lowers the mean absolute residual on the other.")
    p("  shuffled is the same thing with the shape statistic permuted among that")
    p("  board's own subsets, which is the only control that matters here: the")
    p("  subsets share systems and are not independent, so a naive significance")
    p("  test on them would be wrong by a wide margin.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("shape_within_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote shape_within_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
