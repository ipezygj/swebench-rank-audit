"""Is the established share of a leaderboard determined by one number?

resolution_law.py derived, for one pair, the gap needed to separate: delta >=
(z_a + z_b) sqrt(d/n). Across a whole field that becomes a prediction for the
SHARE of pairs established, and the prediction needs only quantities that are
measurable from the matrix:

    tau        the spread of observed scores across systems
    sigma_p    the typical per-item standard deviation of a pairwise difference
    n          items
    c          the simultaneous critical value (rank_sets)

Under a Gaussian field, the gap between two random systems is N(0, 2 tau^2),
the half-width of the simultaneous interval is c sigma_p / sqrt(n), so

    established  =  2 * Phibar( c sigma_p / (sqrt(2 n) tau) )

One dimensionless argument: the signal-to-noise ratio of the field,
tau sqrt(2n) / (c sigma_p). If this predicts the observed established share
across nine leaderboards from seven fields, then the amount of ranking a
benchmark supports is a function of that one number - and a benchmark owner
can compute it before running anything.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * within 5 points on most matrices;
  * the Gaussian prediction will OVER-state established share on fields with
    heavy lower tails (SWE-bench, whose 2023 RAG baselines sit far below a
    dense top): tau is inflated by outliers that establish pairs with
    everyone but do not make the dense part separable;
  * CASP14 similarly (AlphaFold2 inflates tau).
  A clean way to test the second: recompute with tau from the interquartile
  range instead of the SD. If IQR-tau fits better, the law holds and the
  deviation was the Gaussian assumption, not the law.

SELF-CHECKS
  * on a simulated Gaussian field the prediction must match the simultaneous
    established share within 3 points at the real shape;
  * the formula must give 0 when tau = 0 and -> 1 as tau -> infinity.

    python resolution_law_test.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import rank_sets as rs

SEED = 20260823
MATRICES = {
    "SWE-bench Verified": "swebench_verified_matrix.csv",
    "MTEB English v2": "mteb_eng_v2_wide.csv",
    "HELM classic": "helm_winrate_matrix.csv",
    "ProteinGym DMS": "proteingym/matrix.csv",
    "TabArena 16 models": "tabarena/matrix_one_per_model.csv",
    "TabArena 45 variants": "tabarena/matrix_all45.csv",
    "CASP14": "casp/matrix.csv",
    "LiveBench": "livebench/matrix.csv",
    "MathArena 2025": "matharena/matrix.csv",
}


def predict(tau: float, sigma_p: float, n: int, c: float) -> float:
    if tau <= 0:
        return 0.0
    z = c * sigma_p / (math.sqrt(2 * n) * tau)
    return float(2 * norm.sf(z))


def measure(x: np.ndarray, draws: int = 800):
    J, n = x.shape
    r = rs.rank_sets(x, draws=draws)
    sc = x.mean(axis=1)
    tau_sd = float(sc.std(ddof=1))
    q75, q25 = np.percentile(sc, [75, 25])
    tau_iqr = float((q75 - q25) / 1.349)            # Gaussian-equivalent SD
    iu = np.triu_indices(J, k=1)
    sigma_p = float(np.median(r["sigma"][iu]))
    obs = float(r["beats"].sum() / (J * (J - 1)))
    return {"J": J, "n": n, "tau_sd": tau_sd, "tau_iqr": tau_iqr,
            "sigma_p": sigma_p, "crit": float(r["crit"]), "obs": obs,
            "pred_sd": predict(tau_sd, sigma_p, n, r["crit"]),
            "pred_iqr": predict(tau_iqr, sigma_p, n, r["crit"])}


def _check_gaussian_field() -> tuple[bool, str]:
    rng = np.random.default_rng(5)
    J, n = 120, 300
    ability = rng.normal(0, 0.08, J)
    x = 0.5 + ability[:, None] + rng.normal(0, 0.45, (J, n))
    x = np.clip(x, -1, 1)
    m = measure(x, draws=500)
    gap = abs(m["pred_sd"] - m["obs"])
    return gap < 0.03, (f"Gaussian field: predicted {100 * m['pred_sd']:.1f} % "
                        f"vs simultaneous observed {100 * m['obs']:.1f} %")


def _check_limits() -> tuple[bool, str]:
    lo = predict(0.0, 0.3, 200, 4.0)
    hi = predict(1e6, 0.3, 200, 4.0)
    return lo == 0.0 and hi > 0.999, f"tau=0 -> {lo:.3f}, tau->inf -> {hi:.3f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_limits(), _check_gaussian_field()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print("\nA CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("IS THE ESTABLISHED SHARE DETERMINED BY ONE NUMBER?")
    p("=" * 78)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>4} {'SNR':>6} {'observed':>9}"
      f" {'pred (sd)':>10} {'pred (iqr)':>11} {'err sd':>7} {'err iqr':>8}")
    errs_sd, errs_iqr = [], []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        m = measure(x)
        snr = m["tau_sd"] * math.sqrt(2 * m["n"]) / (m["crit"] * m["sigma_p"])
        e1, e2 = m["pred_sd"] - m["obs"], m["pred_iqr"] - m["obs"]
        errs_sd.append(abs(e1)); errs_iqr.append(abs(e2))
        p(f"  {name:<22} {m['J']:>4} {m['n']:>4} {snr:>6.2f} "
          f"{100 * m['obs']:>8.1f}% {100 * m['pred_sd']:>9.1f}% "
          f"{100 * m['pred_iqr']:>10.1f}% {100 * e1:>+6.1f} {100 * e2:>+7.1f}")
    p("")
    p(f"  mean |error|: Gaussian-SD {100 * np.mean(errs_sd):.1f} points, "
      f"IQR-robust {100 * np.mean(errs_iqr):.1f} points")
    p("")
    p("  SNR = tau * sqrt(2n) / (c * sigma_p): the field's spread in units of")
    p("  one pair's simultaneous resolution. established = 2 * Phibar(1 / SNR).")
    p("  'pred (sd)' uses the SD of scores as tau; 'pred (iqr)' uses the IQR")
    p("  scaled to a Gaussian SD, which ignores outliers at the bottom or top")
    p("  of the field. The pre-registered expectation was that the SD version")
    p("  over-predicts on heavy-tailed fields (SWE-bench, CASP14) and the IQR")
    p("  version fits better there. Whether it did is in the last two columns,")
    p("  and the law is as good as the smaller of them.")
    text = "\n".join(L)
    print("\n" + text)
    Path("resolution_law_test_results.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    print("\nwrote resolution_law_test_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
