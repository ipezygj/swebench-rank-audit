"""Is a leaderboard's entropy also determined by (J, n, SNR)?

resolution_law_test.py showed the established share follows Phibar(1/SNR)
on seven of nine leaderboards. Entropy is a harder target: H counts linear
extensions of the whole established poset, which depends on WHICH pairs are
established, not just how many. No closed form. But the question is still
well-posed: take a Gaussian field with the same J, n, score spread tau and
per-item pair noise sigma_p as the real leaderboard, run the identical
machinery on it, and see whether H/ceiling comes out the same.

If it does, then a leaderboard's evidential entropy is a function of four
measurable numbers and nothing else about the field - which would make it
computable before any benchmark run, from the planned size and the expected
spread of entrants.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * |H/ceiling(real) - H/ceiling(Gaussian twin)| <= 5 points on the seven
    leaderboards where the established-share law held;
  * TabArena (both) off by more, for the same skew reason;
  * direction of the TabArena error: the Gaussian twin has a symmetric
    field, so it spreads established pairs evenly; the real skewed field
    concentrates them at the bottom. Concentrated pairs constrain fewer
    extensions, so the REAL H/ceiling should be HIGHER than the twin's.

SELF-CHECKS
  * a Gaussian twin of a Gaussian field must reproduce its own H/ceiling
    within 3 points (two independent seeds);
  * the twin must reproduce tau and sigma_p of its target within 10 %.

    python entropy_law_test.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import rank_sets as rs
import leaderboard_entropy as le

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


def stats_of(x: np.ndarray, draws: int, samples: int, rng) -> dict:
    J, n = x.shape
    r = rs.rank_sets(x, draws=draws)
    sc = x.mean(axis=1)
    iu = np.triu_indices(J, k=1)
    H = le.log_extensions(r["beats"], samples, rng)
    return {"J": J, "n": n, "tau": float(sc.std(ddof=1)),
            "sigma_p": float(np.median(r["sigma"][iu])),
            "H_frac": H["bits"] / (gammaln(J + 1) / math.log(2)),
            "estab": float(r["beats"].sum() / (J * (J - 1)))}


def gaussian_twin(J: int, n: int, tau: float, sigma_p: float, rng) -> np.ndarray:
    """A field with the same J, n, score spread and per-item pair noise.

    Observed tau already includes measurement noise, so the latent spread is
    tau^2 - sigma_item^2 / n, with sigma_item = sigma_p / sqrt(2). Clipped at
    zero: a field whose spread is all noise has no latent spread.
    """
    sigma_item = sigma_p / math.sqrt(2)
    latent = max(tau ** 2 - sigma_item ** 2 / n, 0.0) ** 0.5
    ability = rng.normal(0.0, latent, J)
    return ability[:, None] + rng.normal(0.0, sigma_item, (J, n))


def _check_twin_of_gaussian() -> tuple[bool, str]:
    rng = np.random.default_rng(3)
    x = gaussian_twin(80, 200, 0.06, 0.5, rng)
    a = stats_of(x, 500, 600, rng)
    y = gaussian_twin(a["J"], a["n"], a["tau"], a["sigma_p"], np.random.default_rng(4))
    b = stats_of(y, 500, 600, rng)
    gap = abs(a["H_frac"] - b["H_frac"])
    ok = gap < 0.03 and abs(b["tau"] / a["tau"] - 1) < 0.1 and abs(b["sigma_p"] / a["sigma_p"] - 1) < 0.1
    return ok, (f"twin of a Gaussian field: H/ceiling {100 * a['H_frac']:.1f} vs "
                f"{100 * b['H_frac']:.1f}; tau ratio {b['tau'] / a['tau']:.2f}, "
                f"sigma_p ratio {b['sigma_p'] / a['sigma_p']:.2f}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok, msg = _check_twin_of_gaussian()
    print(f"  [{'ok  ' if ok else 'FAIL'}] {msg}")
    if not ok:
        print("\nA CHECK FAILED - no table is printed.")
        return 1

    rng = np.random.default_rng(SEED)
    L = []
    p = L.append
    p("IS ENTROPY DETERMINED BY (J, n, tau, sigma_p)? REAL vs GAUSSIAN TWIN")
    p("=" * 78)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>4} {'H/ceil real':>12} {'twin':>7}"
      f" {'diff':>6} {'estab real':>11} {'twin':>7}")
    diffs = {}
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        a = stats_of(x, 800, 1000, rng)
        twins = []
        for s in range(2):
            y = gaussian_twin(a["J"], a["n"], a["tau"], a["sigma_p"],
                              np.random.default_rng(SEED + 10 * s + 1))
            twins.append(stats_of(y, 800, 1000, rng))
        tH = float(np.mean([t["H_frac"] for t in twins]))
        tE = float(np.mean([t["estab"] for t in twins]))
        diffs[name] = a["H_frac"] - tH
        p(f"  {name:<22} {a['J']:>4} {a['n']:>4} {100 * a['H_frac']:>11.1f}% "
          f"{100 * tH:>6.1f}% {100 * (a['H_frac'] - tH):>+5.1f} "
          f"{100 * a['estab']:>10.1f}% {100 * tE:>6.1f}%")
    p("")
    within = [k for k, v in diffs.items() if abs(v) <= 0.05]
    p(f"  within 5 points: {len(within)} of {len(diffs)}")
    tab = [k for k in diffs if k.startswith("TabArena")]
    p("  TabArena direction (pre-registered: real HIGHER than twin): "
      + ", ".join(f"{k.split()[1]} {100 * diffs[k]:+.1f}" for k in tab))
    p("")
    p("  The twin has the same J, n, score spread and per-item pair noise and")
    p("  NOTHING else of the real field - no skew, no clusters, no outliers.")
    p("  Where it matches, entropy is a function of those four numbers and a")
    p("  benchmark owner can compute it from a planned size and an expected")
    p("  spread before a single system is run. Where it does not, the field's")
    p("  shape carries information the four numbers do not, and the")
    p("  difference is the size of that information.")
    text = "\n".join(L)
    print("\n" + text)
    Path("entropy_law_test_results.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    print("\nwrote entropy_law_test_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
