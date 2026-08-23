"""Chase model, final form: the chaser inherits the record-holder's noise.

Three attempts failed to reproduce P, the share of frontier advances a
paired test can separate, and each failure narrowed the target:
  iteration 21  bootstrap noise from random rows        - no effect
  iteration 22  the same with its own fit               - no effect
  iteration 24  the reason: the FRONTIER PAIR is sharp (kappa 0.53-0.94)
                while the board's average pair is not (kappa ~ 1.00)
A bootstrap that hands a chaser a random real row cannot produce that: the
sharpness is a property of the pair, and a chaser is sharp with the system
it descends from, not with an arbitrary one.

Sibling chase: when an entrant is a chaser, its residual vector is
    rho * (record-holder's residual) + sqrt(1 - rho^2) * (fresh noise)
so it inherits the record-holder's item-level behaviour to degree rho and
its comparison with the record-holder has kappa = sqrt(1 - rho) exactly.
rho is not fitted to P: it is set from the board's own measured frontier
kappa, rho = 1 - kappa_frontier^2 ... see identity check below. q and
lambda are fitted as before to climb and median step.

IDENTITY  For unit-variance residuals, sd(d)^2 = 2(1 - rho), and the
independence denominator is sqrt(2), so kappa^2 = 1 - rho. The self-check
verifies this on generated data before anything else runs.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * P within +-15 points on >= 4 of 5 boards (iid chase: 2 of 5; bootstrap
    chase: 1 of 5);
  * the simulated frontier kappa lands within 0.10 of the real frontier
    kappa on >= 4 of 5 (the mechanism is installed, not just the outcome);
  * A stays within +-30 % on >= 3 of 5, as the iid version managed.

SELF-CHECKS
  * the identity kappa^2 = 1 - rho holds within 0.02 on generated data;
  * with rho = 0 the model reproduces the iid chase model's P within 5
    points (it is the same model).

    python sibling_chase.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from evidence_trajectory import load
from sota_audit import advances
from sota_twin import audit, fit_drift, sigma_p_of, synth_dates, SEED
from step_sizes import steps_u
from chase_model import BOARDS, chase_field, fit_q_lambda, board_stats
from pair_sharpness import kappa_matrix

REPS = 8


def sibling_field(J, n, dates, q, lam, sigma_item, base_pool, rho, rng):
    order = np.argsort(dates, kind="stable")
    ability = np.empty(J)
    resid = np.empty((J, n))
    record, rec_idx = -np.inf, None
    for k, idx in enumerate(order):
        fresh = rng.normal(0, sigma_item, n)
        if k == 0 or rng.random() >= q or rec_idx is None:
            ability[idx] = rng.choice(base_pool)
            resid[idx] = fresh
        else:
            ability[idx] = record + rng.exponential(lam)
            resid[idx] = rho * resid[rec_idx] + math.sqrt(max(1 - rho ** 2, 0.0)) * fresh
        if ability[idx] > record:
            record, rec_idx = ability[idx], idx
    return ability[:, None] + resid


def frontier_kappa(x, dates):
    K = kappa_matrix(x)
    return float(np.nanmedian([K[a["new"], a["old"]] for a in advances(x, dates)]))


def _check_identity():
    rng = np.random.default_rng(31)
    n = 4000
    out = []
    for rho in (0.0, 0.3, 0.6, 0.9):
        a = rng.normal(0, 1, n)
        b = rho * a + math.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)
        k = (a - b).std(ddof=1) / math.sqrt(a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2)
        out.append(abs(k ** 2 - (1 - rho)))
    return max(out) < 0.02, f"identity kappa^2 = 1 - rho: max deviation {max(out):.3f}"


def _check_rho_zero():
    rng = np.random.default_rng(33)
    J, n = 70, 200
    dates = synth_dates("2023-01-01", np.sort(rng.integers(0, 700, J)))
    pool = rng.normal(0.4, 0.06, 200)
    Pa = np.nanmean([audit(sibling_field(J, n, dates, 0.3, 0.02, 0.45, pool, 0.0, np.random.default_rng(40 + s)), dates, 50 + s)["P"] for s in range(4)])
    Pb = np.nanmean([audit(chase_field(J, n, dates, 0.3, 0.02, 0.45, pool, np.random.default_rng(60 + s)), dates, 70 + s)["P"] for s in range(4)])
    return abs(Pa - Pb) < 0.05, f"rho = 0 reproduces the iid chase model: P {100 * Pa:.0f} % vs {100 * Pb:.0f} %"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_identity(), _check_rho_zero()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("SIBLING CHASE: THE CHASER INHERITS THE RECORD-HOLDER'S ITEM BEHAVIOUR")
    p("=" * 92)
    p(f"  {'board':<20} {'kap real':>8} {'rho':>5} {'q':>5} {'lambda':>7} | {'A real':>6} {'A sim':>6} | "
      f"{'P real':>6} {'P sib':>6} {'P iid':>6} | {'kap sim':>7}")
    okP, okK, okA = 0, 0, 0
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        J, n = x.shape
        sp = sigma_p_of(x)
        _, _, _, si = fit_drift(x, dates, sp)
        sc = x.mean(axis=1)
        rec = {int(a["new"]) for a in advances(x, dates)}
        pool = np.array([sc[i] for i in range(J) if i not in rec])
        if pool.size == 0:
            pool = np.array([sc.mean()])
        o = np.argsort(dates, kind="stable")
        climb = float(sc.max() - sc[o[0]])
        real = board_stats(x, dates, SEED)
        kap_real = frontier_kappa(x, dates)
        rho = float(min(max(1 - kap_real ** 2, 0.0), 0.98))
        q, lam = fit_q_lambda(J, n, dates, si, pool, climb, real["u"], np.random.default_rng(SEED + 11))
        sims = [sibling_field(J, n, dates, q, lam, si, pool, rho, np.random.default_rng(SEED + 1300 + s)) for s in range(REPS)]
        st = [board_stats(y, dates, SEED + 70 * s) for s, y in enumerate(sims)]
        A = float(np.mean([s["A"] for s in st])); P = float(np.nanmean([s["P"] for s in st]))
        kap_sim = float(np.nanmedian([frontier_kappa(y, dates) for y in sims]))
        iid = [board_stats(chase_field(J, n, dates, q, lam, si, pool, np.random.default_rng(SEED + 900 + s)), dates, SEED + 30 * s) for s in range(REPS)]
        Pi = float(np.nanmean([s["P"] for s in iid]))
        okP += abs(P - real["P"]) <= 0.15
        okK += abs(kap_sim - kap_real) <= 0.10
        okA += abs(A / real["A"] - 1) <= 0.30
        p(f"  {name:<20} {kap_real:>8.2f} {rho:>5.2f} {q:>5.2f} {lam:>7.4f} | {real['A']:>6d} {A:>6.1f} | "
          f"{100 * real['P']:>5.0f}% {100 * P:>5.0f}% {100 * Pi:>5.0f}% | {kap_sim:>7.2f}")
    N = len(BOARDS)
    p("")
    p(f"  P within 15 points: {okP}/{N} (pre-registered >= 4; iid 2/5, bootstrap 1/5)")
    p(f"  simulated frontier kappa within 0.10 of real: {okK}/{N} (pre-registered >= 4)")
    p(f"  A within 30 %: {okA}/{N} (pre-registered >= 3)")
    p("")
    p("  rho is set from the board's own frontier kappa through kappa^2 = 1 - rho,")
    p("  not fitted to P. q and lambda are the same climb / median-step fits as")
    p("  chase_model.py. P iid repeats the same fit with independent noise.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("sibling_chase_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote sibling_chase_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
