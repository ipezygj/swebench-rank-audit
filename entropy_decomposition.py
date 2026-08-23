"""Where does the entropy law's residual live? A two-term decomposition.

entropy_law_test.py: H/ceiling of a real board matches a Gaussian twin (same
J, n, tau, sigma_p) on 6 of 9 boards. entropy_law_twin2.py: the misses are
NOT explained by unequal per-system noise. What is left is the SHAPE of the
ability distribution (the twin draws abilities from a Gaussian; real fields
have tails and clumps) and whatever is not captured by "ability + iid noise"
at all (item structure: heterogeneous items, block correlations).

Twin 3 keeps the REAL ability vector - each system's observed mean score -
and adds homogeneous Gaussian item noise sigma_item = sigma_p / sqrt 2 with
the real item difficulties. It is the parametric bootstrap of the board
under "ability + iid noise". Then

    H_real - H_gauss = (H_shape - H_gauss)  +  (H_real - H_shape)
                     =      SHAPE term       +     RESIDUAL term

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the SHAPE term closes TabArena 16, TabArena 45 and CASP14: their
    residual term is within 5 points (the misses were all real > twin and
    all on skewed fields);
  * the SHAPE term does NOT close MTEB: its residual stays below -3 points.
    MTEB's 41 "items" are whole tasks of different kinds (retrieval,
    classification, clustering, STS); a system's scores on them are not
    iid around its ability. If so, MTEB's anomaly is item structure;
  * SWE-bench, LiveBench, HELM, ProteinGym, MathArena: both terms within
    3 points (they already fit).

SELF-CHECKS
  * twin 3 of a Gaussian field with Gaussian abilities must have a shape
    term within 3 points of zero;
  * twin 3 of a field with far-below OUTLIERS must have a shape term of
    the SAME SIGN as entropy_law_test's real-minus-twin misses (positive),
    or the construction cannot explain what it is meant to explain. (The
    first version of this check used a smooth exponential lower tail and
    failed: smooth skew does not move H, outliers and clusters do. Kept in
    git; the pre-registered expectations about the real boards stand.)

    python entropy_decomposition.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from entropy_law_test import MATRICES, gaussian_twin, stats_of

SEED = 20260823
DRAWS, SAMPLES = 800, 1000


def shape_twin(x: np.ndarray, sigma_p: float, rng) -> np.ndarray:
    """Real abilities (observed means), real item difficulties, iid Gaussian noise."""
    J, n = x.shape
    sc = x.mean(axis=1)
    item = x.mean(axis=0) - x.mean()
    sigma_item = sigma_p / math.sqrt(2)
    # observed means already contain one draw of noise; shrink toward the
    # field mean by the reliability so the twin's tau matches the real tau
    tau2 = sc.var(ddof=1)
    rel = max(tau2 - sigma_item ** 2 / n, 0.0) / tau2 if tau2 > 0 else 0.0
    ability = sc.mean() + math.sqrt(rel) * (sc - sc.mean())
    return ability[:, None] + item[None, :] + rng.normal(0.0, sigma_item, (J, n))


def _check_gaussian_zero():
    rng = np.random.default_rng(31)
    x = gaussian_twin(90, 200, 0.06, 0.5, rng)
    a = stats_of(x, 500, 600, rng)
    g = np.mean([stats_of(gaussian_twin(90, 200, a["tau"], a["sigma_p"], np.random.default_rng(40 + s)), 500, 600, rng)["H_frac"] for s in range(2)])
    sh = np.mean([stats_of(shape_twin(x, a["sigma_p"], np.random.default_rng(50 + s)), 500, 600, rng)["H_frac"] for s in range(2)])
    return abs(sh - g) < 0.03, f"Gaussian field: shape term {100 * (sh - g):+.1f} points (must be ~0)"


def _check_skewed_sign():
    # First version used an exponential lower tail and FAILED (-0.9): a smooth
    # skew does not move H. What does is OUTLIERS - a few systems far below the
    # bulk inflate tau, the Gaussian twin spreads the whole field by that tau,
    # establishes more pairs inside the bulk and loses entropy (+13.5 with 10 %
    # at -0.5; +4.2 for two clusters; +0.3 for the exponential tail). The
    # check now uses the outlier field, which is also what TabArena looks like.
    rng = np.random.default_rng(33)
    J, n = 60, 150
    ability = np.concatenate([rng.normal(0, 0.05, J - 6), np.full(6, -0.5)])
    x = 0.6 + ability[:, None] + rng.normal(0, 0.45, (J, n))
    a = stats_of(x, 500, 600, rng)
    g = np.mean([stats_of(gaussian_twin(J, n, a["tau"], a["sigma_p"], np.random.default_rng(60 + s)), 500, 600, rng)["H_frac"] for s in range(2)])
    sh = np.mean([stats_of(shape_twin(x, a["sigma_p"], np.random.default_rng(70 + s)), 500, 600, rng)["H_frac"] for s in range(2)])
    return (sh - g) > 0.02, f"skewed field: shape term {100 * (sh - g):+.1f} points (must be clearly positive)"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_gaussian_zero(), _check_skewed_sign()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rng = np.random.default_rng(SEED)
    L = []
    p = L.append
    p("ENTROPY LAW RESIDUAL: SHAPE TERM + RESIDUAL TERM")
    p("=" * 78)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>4} {'real':>7} {'gauss':>7} {'shape':>7} {'SHAPE':>7} {'RESID':>7}")
    rows = {}
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        a = stats_of(x, DRAWS, SAMPLES, rng)
        J, n = x.shape
        g = float(np.mean([stats_of(gaussian_twin(J, n, a["tau"], a["sigma_p"], np.random.default_rng(SEED + 10 * s + 1)), DRAWS, SAMPLES, rng)["H_frac"] for s in range(2)]))
        sh = float(np.mean([stats_of(shape_twin(x, a["sigma_p"], np.random.default_rng(SEED + 300 + s)), DRAWS, SAMPLES, rng)["H_frac"] for s in range(2)]))
        rows[name] = (a["H_frac"], g, sh)
        p(f"  {name:<22} {J:>4} {n:>4} {100 * a['H_frac']:>6.1f}% {100 * g:>6.1f}% {100 * sh:>6.1f}% {100 * (sh - g):>+6.1f} {100 * (a['H_frac'] - sh):>+6.1f}")
    p("")
    p("  pre-registered:")
    def res(k): return rows[k][0] - rows[k][2]
    def shp(k): return rows[k][2] - rows[k][1]
    for k in ("TabArena 16 models", "TabArena 45 variants", "CASP14"):
        if k in rows:
            p(f"    {k:<22} residual within 5: {'yes' if abs(res(k)) <= 0.05 else 'NO'} ({100 * res(k):+.1f}; shape {100 * shp(k):+.1f})")
    if "MTEB English v2" in rows:
        k = "MTEB English v2"
        p(f"    {k:<22} residual stays below -3: {'yes' if res(k) < -0.03 else 'NO'} ({100 * res(k):+.1f}; shape {100 * shp(k):+.1f})")
    for k in ("SWE-bench Verified", "LiveBench", "HELM classic", "ProteinGym DMS", "MathArena 2025"):
        if k in rows:
            p(f"    {k:<22} both terms within 3: {'yes' if abs(res(k)) <= 0.03 and abs(shp(k)) <= 0.03 else 'NO'} (shape {100 * shp(k):+.1f}, resid {100 * res(k):+.1f})")
    p("")
    p("  SHAPE = H(shape twin) - H(Gaussian twin): what the ability distribution's")
    p("  form adds beyond (J, n, tau, sigma_p). RESID = H(real) - H(shape twin):")
    p("  what 'ability + iid item noise' cannot produce at all - item structure.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("entropy_decomposition_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote entropy_decomposition_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
