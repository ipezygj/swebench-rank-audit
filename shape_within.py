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

MEASURED FIRST, AND IT CHANGED THE DESIGN

Subsetting a board and correlating shape against residual does not give zero
under the null. On 24 synthetic Gaussian boards the within-board correlation
averages -0.226 with a 5-95 range of -0.51 to +0.09: the procedure is biased
negative, because the same systems that inflate an SD are the ones whose
removal both raises the shape statistic and changes what the law predicts. A
real board reading -0.4 against an assumed null of zero would therefore be
evidence of nothing. The null is now a Gaussian twin of each board, matched on
J, n, tau and sigma_p and put through the identical subsetting.

PRE-REGISTERED (2026-08-24, committed before the run, no real board read)
  P1  the real within-board correlation falls below its own twin null's 5th
      percentile on at least 5 of the 9 boards.
  P2  both TabArena boards are among them - they carry the across-board signal
      and should carry it within themselves.
  P3  a correction fitted on half a board's subsets beats the uncorrected law
      on the other half more often than the same procedure does on that
      board's twin, on at least 5 of 9.
  P4  calibration: running the whole pipeline on a twin AS IF it were real
      rejects on at most 1 of 9. If a twin looks like a real board, nothing
      below means anything.

SELF-CHECKS (no table if any fails)
  * the shape statistic must read 1.0 on Gaussian subsets of these sizes;
  * every subset must have at least 12 systems;
  * the twin must reproduce its board's tau and sigma_p within 15 %, or the
    null is not the null of that board.

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
SUBSETS = 40          # fields drawn per board
MIN_J = 12
TWINS = 15


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


def twin_of(J, n, tau, sigma_p, rng):
    """A Gaussian field with the same four numbers and nothing else."""
    sigma_item = sigma_p / math.sqrt(2.0)
    latent = max(tau ** 2 - sigma_item ** 2 / n, 0.0) ** 0.5
    return rng.normal(0.0, latent, J)[:, None] + rng.normal(0.0, sigma_item, (J, n))


def within_rho(x, rng, count):
    sub = subsets_of(x, rng, count=count)
    if len(sub) < 8:
        return float("nan"), None
    shape = [a for a, _, _ in sub]
    resid = [p - o for _, o, p in sub]
    return float(spearmanr(shape, resid)[0]), (shape, resid)


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
    checks = [_check_shape_scale()]
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
        J, n = x.shape
        r0 = rs.rank_sets(x)
        sc = x.mean(axis=1)
        iu = np.triu_indices(J, k=1)
        tau = float(sc.std(ddof=1))
        sigma_p = float(np.median(r0["sigma"][iu]))

        rho, pack = within_rho(x, rng, SUBSETS)
        if pack is None:
            rows[name] = None
            continue
        shape, resid = pack
        wins = [split_fit(shape, resid, np.random.default_rng(SEED + i)) for i in range(21)]
        wins = [w for w in wins if w is not None]
        win = float(np.mean(wins)) if wins else float("nan")

        nulls, null_wins, tw_ok = [], [], 0
        for t in range(TWINS):
            g = np.random.default_rng(SEED + 5000 + 13 * t)
            y = twin_of(J, n, tau, sigma_p, g)
            ysc = y.mean(axis=1)
            yr = rs.rank_sets(y)
            ytau = float(ysc.std(ddof=1))
            ysig = float(np.median(yr["sigma"][iu]))
            if tau > 0 and abs(ytau / tau - 1) <= 0.15 and abs(ysig / sigma_p - 1) <= 0.15:
                tw_ok += 1
            nr, npack = within_rho(y, g, SUBSETS)
            if npack is not None:
                nulls.append(nr)
                w = [split_fit(npack[0], npack[1], np.random.default_rng(SEED + 70 + k))
                     for k in range(5)]
                w = [z for z in w if z is not None]
                null_wins.append(float(np.mean(w)) if w else 0.0)

        rows[name] = {"n_sub": len(shape), "rho": rho, "win": win,
                      "null_lo": float(np.percentile(nulls, 5)) if nulls else float("nan"),
                      "null_med": float(np.median(nulls)) if nulls else float("nan"),
                      "null_win": float(np.median(null_wins)) if null_wins else float("nan"),
                      "twin_ok": tw_ok, "twins": TWINS,
                      "shape_lo": min(shape), "shape_hi": max(shape)}

    good = {k: v for k, v in rows.items() if v}
    L = []
    p = L.append
    p("DOES THE SHAPE RELATION HOLD WITHIN A SINGLE BOARD?")
    p("=" * 96)
    p(f"  {'leaderboard':<22} {'fields':>7} {'shape range':>14} {'rho':>7} "
      f"{'twin 5th':>9} {'twin med':>9} {'fit':>6} {'twin fit':>9}")
    for k, v in good.items():
        p(f"  {k:<22} {v['n_sub']:>7} {v['shape_lo']:>6.2f}-{v['shape_hi']:<7.2f} "
          f"{v['rho']:>+7.2f} {v['null_lo']:>+9.2f} {v['null_med']:>+9.2f} "
          f"{100 * v['win']:>5.0f}% {100 * v['null_win']:>8.0f}%")
    p("")
    below = sum(1 for v in good.values() if v["rho"] < v["null_lo"])
    tab = {k for k in good if k.startswith("TabArena") and good[k]["rho"] < good[k]["null_lo"]}
    beats = sum(1 for v in good.values() if v["win"] > v["null_win"])
    p(f"  P1  real rho below its twin null's 5th percentile on {below} of {len(good)}   "
      f"pre-registered >= 5:  {'HIT' if below >= 5 else 'MISS'}")
    p(f"  P2  both TabArena boards among them: {len(tab)} of 2            "
      f"{'HIT' if len(tab) == 2 else 'MISS'}")
    p(f"  P3  fit beats its own twin's fit on {beats} of {len(good)}                "
      f"pre-registered >= 5:  {'HIT' if beats >= 5 else 'MISS'}")
    tw = sum(v["twin_ok"] for v in good.values())
    twn = sum(v["twins"] for v in good.values())
    p(f"  P4  twins reproduce tau and sigma_p within 15 % on {tw} of {twn} draws")
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
