"""Families, conditioned on ability: isotonic residuals.

sota_families.py died on its own control: after an additive two-way fit,
two strong systems share positive residuals on the hard items whatever
their lineage, because the additive model misfits at the top (a binary
item cannot be 'solved 1.3 times'). The same misfit inflates the residual
correlation spectrum behind J_eff in effective_entrants.py.

Fix the residual, not the question. For each ITEM, fit a monotone
(isotonic) function of the systems' mean scores and take the residual from
that. Any item x ability interaction of monotone shape - logistic ceilings,
floors, the lot - is absorbed. What remains correlated across systems is
family (or a non-monotone interaction, which is rarer).

Then re-run both measurements with the isotonic residual:
  * J_eff (participation ratio against the permuted null);
  * frontier-pair percentile vs the ability-matched control.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * on a Rasch field WITHOUT families the additive method gives F < 0.9
    (spurious structure) while the isotonic method gives F within 0.1 of 1;
  * on the real boards F rises under the isotonic residual on >= 7 of 9
    (less misfit counted as family), and the TabArena calibration
    J_eff(45) < 1.5 x J_eff(16) still holds (families are real);
  * the family question: if frontier percentile minus control percentile is
    >= 10 points on >= 3 of 4 dated boards, the family reading is restored
    under the better residual; otherwise it stays dead. No prior either way.

SELF-CHECKS
  * a planted two-family field must still give J_eff < 8 under isotonic
    residuals (the fix must not erase real families);
  * isotonic residuals must have zero mean per item (by construction).

    python isotonic_families.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from entropy_law_test import MATRICES
from sota_audit import advances
from evidence_trajectory import load
from sota_twin import synth_dates
import effective_entrants as ee
import residual_correlation as rc
from sota_families import BOARDS

SEED = 20260823
NULLS = 20


def isotonic_residuals(x: np.ndarray) -> np.ndarray:
    sc = x.mean(axis=1)
    out = np.empty_like(x, dtype=float)
    for i in range(x.shape[1]):
        fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(sc, x[:, i])
        out[:, i] = x[:, i] - fit.predict(sc)
    return out


def pr_of(resid: np.ndarray) -> float:
    c = np.nan_to_num(np.corrcoef(resid))
    lam = np.linalg.eigvalsh(c)
    lam = lam[lam > 1e-12]
    return float(lam.sum() ** 2 / np.sum(lam ** 2))


def permute_rows(resid: np.ndarray, rng) -> np.ndarray:
    out = np.empty_like(resid)
    for j in range(resid.shape[0]):
        out[j] = resid[j, rng.permutation(resid.shape[1])]
    return out


def f_iso(x: np.ndarray, rng) -> float:
    r = isotonic_residuals(x)
    pr = pr_of(r)
    null = np.mean([pr_of(permute_rows(r, np.random.default_rng(int(rng.integers(1 << 31))))) for _ in range(NULLS)])
    return pr / null


def f_add(x: np.ndarray, rng) -> float:
    return ee.f_share(x, rng)[2]


def percentiles(x, dates, resid, control):
    c = np.nan_to_num(np.corrcoef(resid))
    J = x.shape[0]
    iu = np.triu_indices(J, k=1)
    allc = np.sort(c[iu])
    sc = x.mean(axis=1)
    out = []
    for a in advances(x, dates):
        other = a["old"]
        if control:
            present = np.flatnonzero(dates <= a["date"])
            present = present[(present != a["new"]) & (present != a["old"])]
            if len(present) == 0:
                continue
            other = int(present[np.argmax(sc[present])])
        out.append(100.0 * np.searchsorted(allc, c[a["new"], other]) / len(allc))
    return np.array(out)


def rasch_field(J, n, rng):
    theta = rng.normal(0, 1.2, J)
    b = rng.normal(0, 1.5, n)
    pr = 1 / (1 + np.exp(-(theta[:, None] - b[None, :])))
    return (rng.random((J, n)) < pr).astype(float)


def _check_planted():
    rng = np.random.default_rng(53)
    J, n = 80, 200
    f1, f2 = rng.normal(0, 0.45, n), rng.normal(0, 0.45, n)
    noise = rng.normal(0, 0.45, (J, n))
    load = 0.9
    shared = np.vstack([np.tile(f1, (J // 2, 1)), np.tile(f2, (J - J // 2, 1))])
    x = rng.normal(0, 0.06, J)[:, None] + load * shared + math.sqrt(1 - load ** 2) * noise
    F = f_iso(x, rng)
    return F * J < 8, f"planted two families: isotonic J_eff {F * J:.1f} (must be < 8)"


def _check_zero_mean():
    rng = np.random.default_rng(5)
    x = rng.random((30, 50))
    r = isotonic_residuals(x)
    m = float(np.abs(r.mean(axis=0)).max())
    return m < 1e-9, f"isotonic residuals: max |item mean| {m:.1e}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_planted(), _check_zero_mean()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rng = np.random.default_rng(SEED)
    L = []
    p = L.append
    p("FAMILIES CONDITIONED ON ABILITY: ISOTONIC RESIDUALS")
    p("=" * 78)
    # 1. Rasch field without families
    x = rasch_field(120, 400, np.random.default_rng(77))
    fa, fi = f_add(x, rng), f_iso(x, rng)
    p(f"  Rasch field, no families (120 x 400): F additive {fa:.2f}   F isotonic {fi:.2f}")
    p(f"    pre-registered: additive < 0.9 {'yes' if fa < 0.9 else 'NO'}; isotonic within 0.1 of 1 {'yes' if abs(fi - 1) < 0.1 else 'NO'}")
    p("")
    p(f"  {'leaderboard':<22} {'J':>4} {'F add':>6} {'F iso':>6} {'J_eff add':>9} {'J_eff iso':>9}")
    rows = {}
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        fa, fi = f_add(x, rng), f_iso(x, rng)
        rows[name] = (x.shape[0], fa, fi)
        p(f"  {name:<22} {x.shape[0]:>4} {fa:>6.2f} {fi:>6.2f} {fa * x.shape[0]:>9.1f} {fi * x.shape[0]:>9.1f}")
    rises = sum(v[2] > v[1] for v in rows.values())
    j16 = rows["TabArena 16 models"][0] * rows["TabArena 16 models"][2]
    j45 = rows["TabArena 45 variants"][0] * rows["TabArena 45 variants"][2]
    p("")
    p(f"  F rises under isotonic residual: {rises}/{len(rows)} (pre-registered >= 7)")
    p(f"  TabArena calibration (iso): J_eff(45) {j45:.1f} < 1.5 x J_eff(16) {1.5 * j16:.1f} -> {'yes' if j45 < 1.5 * j16 else 'NO'}")
    p("")
    p(f"  {'dated board':<20} {'advances':>8} {'frontier pct':>12} {'control pct':>11} {'diff':>6}")
    diffs = []
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        r = isotonic_residuals(x)
        fr = percentiles(x, dates, r, False)
        ct = percentiles(x, dates, r, True)
        d = float(np.median(fr) - np.median(ct))
        diffs.append(d)
        p(f"  {name:<20} {len(fr):>8} {np.median(fr):>12.0f} {np.median(ct):>11.0f} {d:>+6.0f}")
    p("")
    n_ok = sum(d >= 10 for d in diffs)
    p(f"  frontier - control >= 10 points: {n_ok}/4  -> family reading {'RESTORED' if n_ok >= 3 else 'stays dead'}")
    p("")
    p("  isotonic residual = item score minus a monotone fit on the systems' mean")
    p("  scores; absorbs any monotone item x ability interaction. F add / F iso =")
    p("  share of independent entrants under the additive / isotonic residual.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("isotonic_families_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote isotonic_families_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
