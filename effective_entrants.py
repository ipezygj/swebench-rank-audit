"""How many INDEPENDENT entrants does a leaderboard really have?

residual_correlation.py showed that what makes every real board more
ordered than an independent-noise model is correlation among entrants'
residuals: systems that share a base model, a scaffold, a method, fail the
same items. That correlation has a natural summary - the number of
effectively independent entrants, J_eff.

Take the two-way-centred residual matrix, its J x J correlation matrix C,
eigenvalues lambda, and the participation ratio PR = (sum lambda)^2 /
sum lambda^2. A sample correlation matrix of iid residuals has PR < J
already (finite n), so PR is reported against its own iid null: the mean
PR of the board with every system's residuals permuted independently
(twin 4 of residual_correlation.py). Then

    F = PR_real / PR_null          share of independent entrants, (0, 1]
    J_eff = F * J

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * calibration on KNOWN structure: TabArena 45 variants are configurations
    of the 16 TabArena models, so J_eff(45) < 1.5 * J_eff(16) although
    J is 2.8x larger;
  * F < 1 on all nine boards (every board has families);
  * exploratory, reported not judged: Spearman of (1 - F) against the size
    of the residual term from entropy_decomposition (expected positive:
    more family structure, more hidden order).

SELF-CHECKS
  * an iid field must give F within 0.1 of 1;
  * a field of 80 systems in two planted families (loading 0.9) must give
    J_eff below 8.

    python effective_entrants.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from entropy_law_test import MATRICES
from residual_correlation import decompose, twin4, DECOMP_RESID

SEED = 20260823
NULLS = 20


def participation_ratio(x: np.ndarray) -> float:
    _, _, resid = decompose(x)
    c = np.corrcoef(resid)
    c = np.nan_to_num(c)
    lam = np.linalg.eigvalsh(c)
    lam = lam[lam > 1e-12]
    return float(lam.sum() ** 2 / np.sum(lam ** 2))


def f_share(x: np.ndarray, rng) -> tuple[float, float, float]:
    pr = participation_ratio(x)
    null = np.mean([participation_ratio(twin4(x, np.random.default_rng(int(rng.integers(1 << 31))))) for _ in range(NULLS)])
    return pr, float(null), pr / null


def _check_iid():
    rng = np.random.default_rng(51)
    x = rng.normal(0, 0.06, 80)[:, None] + rng.normal(0, 0.3, 200)[None, :] + rng.normal(0, 0.45, (80, 200))
    pr, null, F = f_share(x, rng)
    return abs(F - 1) < 0.1, f"iid field: PR {pr:.1f} vs null {null:.1f}, F {F:.2f}"


def _check_families():
    rng = np.random.default_rng(53)
    J, n = 80, 200
    f1, f2 = rng.normal(0, 0.45, n), rng.normal(0, 0.45, n)
    noise = rng.normal(0, 0.45, (J, n))
    load = 0.9
    shared = np.vstack([np.tile(f1, (J // 2, 1)), np.tile(f2, (J - J // 2, 1))])
    x = rng.normal(0, 0.06, J)[:, None] + load * shared + math.sqrt(1 - load ** 2) * noise
    pr, null, F = f_share(x, rng)
    return F * J < 8, f"two planted families of 40: J_eff {F * J:.1f} (must be < 8)"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_iid(), _check_families()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rng = np.random.default_rng(SEED)
    L = []
    p = L.append
    p("EFFECTIVE NUMBER OF INDEPENDENT ENTRANTS")
    p("=" * 72)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>4} {'PR':>7} {'PR null':>8} {'F':>6} {'J_eff':>7} {'resid':>6}")
    rows = {}
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        pr, null, F = f_share(x, rng)
        rows[name] = (x.shape[0], F, DECOMP_RESID.get(name, float("nan")))
        p(f"  {name:<22} {x.shape[0]:>4} {x.shape[1]:>4} {pr:>7.1f} {null:>8.1f} {F:>6.2f} {F * x.shape[0]:>7.1f} {DECOMP_RESID.get(name, float('nan')):>+6.1f}")
    p("")
    j16 = rows["TabArena 16 models"][0] * rows["TabArena 16 models"][1]
    j45 = rows["TabArena 45 variants"][0] * rows["TabArena 45 variants"][1]
    p(f"  calibration: J_eff(TabArena 45) {j45:.1f} vs 1.5 x J_eff(TabArena 16) {1.5 * j16:.1f}  -> {'yes' if j45 < 1.5 * j16 else 'NO'}")
    p(f"  F < 1 on all boards: {'yes' if all(v[1] < 1 for v in rows.values()) else 'NO'} (max F {max(v[1] for v in rows.values()):.2f})")
    r = spearmanr([1 - v[1] for v in rows.values()], [abs(v[2]) for v in rows.values()])
    p(f"  exploratory: Spearman(1 - F, |residual term|) = {r.statistic:+.2f} (p {r.pvalue:.2f}); expected positive")
    p("")
    p("  PR = participation ratio of the residual correlation spectrum; null =")
    p("  the same board with each system's residuals permuted (no families).")
    p("  F = PR / null is the share of entrants that are effectively independent;")
    p("  J_eff = F * J. A leaderboard of J entrants carries the evidence of J_eff.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("effective_entrants_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote effective_entrants_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
