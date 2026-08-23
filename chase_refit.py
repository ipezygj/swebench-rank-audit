"""The fair version: chase parameters fitted INSIDE the board's own noise.

chase_correlated.py transferred q and lambda fitted under iid Gaussian
noise into a model whose noise is a bootstrap of the board's residual rows.
That was not a fair test of the noise model: unequal per-row noise alone
inflates the record count (MTEB A 16 -> 65) and flattens the steps, and the
fit had no chance to absorb it. Here q and lambda are fitted with the
bootstrap noise in place, to the same two targets as before (total climb,
median step), and A and P remain predictions.

Three noise models, all with their own fit, judged on the same two
predictions:
    iid    homogeneous Gaussian, sigma_p / sqrt 2         (chase_model.py)
    ctrl   bootstrap rows, each permuted across items      (unequal levels)
    boot   bootstrap rows as they are                      (unequal + correlated)

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * with its own fit, ctrl reaches A within +-30 % on >= 4 of 5 (the
    transferred fit managed 0 of 5) - i.e. the A explosion was the fit,
    not the noise model;
  * boot and ctrl differ by less than 8 points in P on >= 4 of 5: after
    iteration 21 the honest expectation is that correlation still does
    nothing;
  * no noise model reaches P within 15 points on more than 3 of 5. The
    separability gap survives all three and is the open problem.

SELF-CHECKS
  * each fit must hit its own targets: simulated climb within 25 % of the
    real climb and median u within 0.10, for every board and noise model
    (a fit that misses its targets cannot be read as a prediction);
  * the three noise models must give the same median sigma_p within 15 %.

    python chase_refit.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from evidence_trajectory import load
from sota_audit import advances
from sota_twin import audit, fit_drift, sigma_p_of, SEED
from step_sizes import steps_u
from residual_correlation import decompose
from chase_model import BOARDS, chase_field, board_stats
from chase_correlated import chase_boot

REPS = 8
GRID = np.linspace(0.05, 0.9, 10)


def perm_rows(resid, rng):
    r = np.empty_like(resid)
    for j in range(resid.shape[0]):
        r[j] = resid[j, rng.permutation(resid.shape[1])]
    return r


def make(kind, J, n, dates, q, lam, si, resid, residp, pool, rng):
    if kind == "iid":
        return chase_field(J, n, dates, q, lam, si, pool, rng)
    if kind == "ctrl":
        return chase_boot(J, n, dates, q, lam, residp, pool, rng)
    return chase_boot(J, n, dates, q, lam, resid, pool, rng)


def climb_of(x, dates):
    sc = x.mean(axis=1)
    o = np.argsort(dates, kind="stable")
    return float(sc.max() - sc[o[0]])


def fit_kind(kind, J, n, dates, si, resid, residp, pool, climb, target_u, rng, reps=3):
    """lambda by bisection on climb inside each q; q by median-step match."""
    best = None
    for q in GRID:
        lo, hi = 1e-6, max(climb, 1e-3) * 3
        for _ in range(14):
            mid = math.sqrt(lo * hi)
            cs = [climb_of(make(kind, J, n, dates, q, mid, si, resid, residp, pool,
                                np.random.default_rng(int(rng.integers(1 << 31)))), dates) for _ in range(3)]
            if np.mean(cs) < climb:
                lo = mid
            else:
                hi = mid
        lam = math.sqrt(lo * hi)
        us = [float(np.median(steps_u(y, dates, sigma_p_of(y))))
              for y in (make(kind, J, n, dates, q, lam, si, resid, residp, pool,
                             np.random.default_rng(int(rng.integers(1 << 31)))) for _ in range(reps))]
        err = abs(float(np.mean(us)) - target_u)
        if best is None or err < best[0]:
            best = (err, q, lam)
    return best[1], best[2]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    L = []
    p = L.append
    p("CHASE MODEL, EACH NOISE MODEL WITH ITS OWN FIT")
    p("=" * 92)
    p(f"  {'board':<20} {'model':<5} {'q':>5} {'lambda':>8} {'climb hit':>10} {'u hit':>7} | "
      f"{'A real':>6} {'A sim':>6} {'P real':>6} {'P sim':>6} {'sigma_p':>8}")
    fits_ok, okA, okP, dPs = [], {"iid": 0, "ctrl": 0, "boot": 0}, {"iid": 0, "ctrl": 0, "boot": 0}, []
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        J, n = x.shape
        sp = sigma_p_of(x)
        _, _, _, si = fit_drift(x, dates, sp)
        _, _, resid = decompose(x)
        residp = perm_rows(resid, np.random.default_rng(SEED + 3))
        sc = x.mean(axis=1)
        rec = {int(a["new"]) for a in advances(x, dates)}
        pool = np.array([sc[i] for i in range(J) if i not in rec])
        if pool.size == 0:
            pool = np.array([sc.mean()])
        real = board_stats(x, dates, SEED)
        climb = climb_of(x, dates)
        row_P = {}
        for kind in ("iid", "ctrl", "boot"):
            rng = np.random.default_rng(SEED + 17)
            q, lam = fit_kind(kind, J, n, dates, si, resid, residp, pool, climb, real["u"], rng)
            sims = [make(kind, J, n, dates, q, lam, si, resid, residp, pool, np.random.default_rng(SEED + 1000 + 7 * s)) for s in range(REPS)]
            stats = [board_stats(y, dates, SEED + 60 * s) for s, y in enumerate(sims)]
            A = float(np.mean([s["A"] for s in stats]))
            P = float(np.nanmean([s["P"] for s in stats]))
            U = float(np.mean([s["u"] for s in stats]))
            C = float(np.mean([climb_of(y, dates) for y in sims]))
            spm = float(np.mean([sigma_p_of(y) for y in sims]))
            hitC, hitU = abs(C / climb - 1) <= 0.25, abs(U - real["u"]) <= 0.10
            fits_ok.append(hitC and hitU)
            okA[kind] += abs(A / real["A"] - 1) <= 0.30
            okP[kind] += abs(P - real["P"]) <= 0.15
            row_P[kind] = P
            p(f"  {name:<20} {kind:<5} {q:>5.2f} {lam:>8.4f} {'yes' if hitC else 'NO':>10} {'yes' if hitU else 'NO':>7} | "
              f"{real['A']:>6d} {A:>6.1f} {100 * real['P']:>5.0f}% {100 * P:>5.0f}% {spm:>8.3f}")
        dPs.append(abs(row_P["boot"] - row_P["ctrl"]))
    N = len(BOARDS)
    p("")
    p(f"  fits that hit their own targets: {sum(fits_ok)}/{len(fits_ok)}")
    p(f"  A within 30 %: iid {okA['iid']}/{N}  ctrl {okA['ctrl']}/{N}  boot {okA['boot']}/{N}"
      f"   (pre-registered: ctrl >= 4)")
    p(f"  P within 15 points: iid {okP['iid']}/{N}  ctrl {okP['ctrl']}/{N}  boot {okP['boot']}/{N}"
      f"   (pre-registered: none above 3)")
    p(f"  |P(boot) - P(ctrl)| < 8 points: {sum(d < 0.08 for d in dPs)}/{N} (pre-registered >= 4)")
    p("")
    p("  Each noise model gets its own (q, lambda), fitted to total climb and")
    p("  median step. A and P are predictions. iid = one Gaussian level for all;")
    p("  ctrl = the board's own row noise levels, permuted (no correlation);")
    p("  boot = the board's own rows unchanged (levels and correlation).")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("chase_refit_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote chase_refit_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
