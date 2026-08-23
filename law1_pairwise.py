"""Law 1 with pair-specific resolution instead of one sigma_p.

resolution_law_test.py predicts the established share from four numbers,
using the MEDIAN pairwise sigma for the whole board:

    established (ordered) = Phibar( c sigma_p / (sqrt(2 n) tau) )

pair_sharpness.py showed that sigma is not one number: the ratio of the
observed difference SD to what independence gives (kappa) is ~1.00 across
all pairs but 0.53-0.94 for frontier pairs, so pairs differ in how sharply
they can be compared. The integral version of the same law replaces the
single sigma with the board's own distribution of pairwise sigmas:

    established = E_pairs[ Phibar( c sigma_jk / (sqrt(n) * |gap|-scale) ) ]

Written directly: for each pair, the gap needed is c sigma_jk / sqrt n, and
the pair is established when its true gap exceeds it. Averaging the
Gaussian tail over the empirical distribution of sigma_jk - rather than
evaluating it once at the median - is the whole change. If the law's error
falls, resolution heterogeneity was part of what the four-number version
was missing; if it does not, the median was a sufficient statistic and the
simpler law stands.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * mean |error| over the nine boards falls by at least 1 point;
  * the improvement is largest on the boards with the widest sigma spread
    (MTEB, TabArena: sigma_p CV 0.47-0.53 from commit 7db67bf);
  * the pair-integral version does not make any board worse by more than
    2 points.

SELF-CHECKS
  * on a Gaussian field (one sigma for all pairs) the two versions must
    agree within 1 point;
  * on a field with two noise regimes the pair-integral version must be
    closer to the observed established share than the median version.

    python law1_pairwise.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import rank_sets as rs
from entropy_law_test import MATRICES

DRAWS = 800


def measure(x, draws=DRAWS):
    J, n = x.shape
    r = rs.rank_sets(x, draws=draws)
    sc = x.mean(axis=1)
    tau = float(sc.std(ddof=1))
    iu = np.triu_indices(J, k=1)
    sig = r["sigma"][iu]
    c = float(r["crit"])
    obs = float(r["beats"].sum() / (J * (J - 1)))
    med = float(np.median(sig))
    # median version: one tail evaluation
    pred_med = float(norm.sf(c * med / (math.sqrt(2 * n) * tau))) if tau > 0 else 0.0
    # pair-integral version: average the tail over the board's own sigmas
    if tau > 0:
        z = c * sig / (math.sqrt(2 * n) * tau)
        pred_int = float(np.mean(norm.sf(z)))
    else:
        pred_int = 0.0
    return {"J": J, "n": n, "obs": obs, "med": pred_med, "int": pred_int,
            "cv": float(sig.std() / sig.mean())}


def _check_gaussian():
    rng = np.random.default_rng(17)
    J, n = 100, 300
    x = 0.5 + rng.normal(0, 0.08, J)[:, None] + rng.normal(0, 0.45, (J, n))
    m = measure(np.clip(x, -1, 1), draws=400)
    return abs(m["med"] - m["int"]) < 0.01, \
        f"Gaussian field: median version {100 * m['med']:.1f} %, pair-integral {100 * m['int']:.1f} %"


def _check_two_regimes():
    rng = np.random.default_rng(19)
    J, n = 100, 300
    s = np.where(np.arange(J) < J // 2, 0.15, 0.7)
    x = 0.5 + rng.normal(0, 0.08, J)[:, None] + rng.normal(0, 1, (J, n)) * s[:, None]
    m = measure(x, draws=400)
    better = abs(m["int"] - m["obs"]) < abs(m["med"] - m["obs"])
    return better, (f"two noise regimes: observed {100 * m['obs']:.1f} %, median {100 * m['med']:.1f} %, "
                    f"pair-integral {100 * m['int']:.1f} %")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_gaussian(), _check_two_regimes()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("LAW 1 WITH PAIR-SPECIFIC RESOLUTION")
    p("=" * 84)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>4} {'sig CV':>7} {'observed':>9} {'median':>8} {'pair-int':>9} "
      f"{'err med':>8} {'err int':>8}")
    em, ei, rows = [], [], []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        m = measure(x)
        e1, e2 = m["med"] - m["obs"], m["int"] - m["obs"]
        em.append(abs(e1)); ei.append(abs(e2)); rows.append((name, m["cv"], abs(e1) - abs(e2)))
        p(f"  {name:<22} {m['J']:>4} {m['n']:>4} {m['cv']:>7.2f} {100 * m['obs']:>8.1f}% {100 * m['med']:>7.1f}% "
          f"{100 * m['int']:>8.1f}% {100 * e1:>+7.1f} {100 * e2:>+7.1f}")
    p("")
    p(f"  mean |error|: median version {100 * np.mean(em):.1f} points, pair-integral {100 * np.mean(ei):.1f} points"
      f"  -> falls by {100 * (np.mean(em) - np.mean(ei)):.1f}")
    worse = [(n_, 100 * -d) for n_, _, d in rows if d < -0.02]
    p(f"  boards made worse by more than 2 points: {len(worse)}" + (f" ({worse})" if worse else ""))
    wide = sorted(rows, key=lambda r_: -r_[1])[:3]
    p("  widest sigma spread: " + ", ".join(f"{n_} (CV {cv:.2f}, gain {100 * d:+.1f})" for n_, cv, d in wide))
    p("")
    p("  median version evaluates the Gaussian tail once at the median pairwise")
    p("  sigma; pair-integral averages the tail over every pair's own sigma. Both")
    p("  use the same tau, c and n, so the only difference is whether resolution")
    p("  is treated as one number or as a distribution.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("law1_pairwise_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote law1_pairwise_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
