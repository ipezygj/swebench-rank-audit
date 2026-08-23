"""Chase model with the board's own noise: does correlation fix P?

chase_model.py reproduced how MANY frontier advances a board has and how
BIG they are (A within 30 % on 3 of 5, beating the drift twin 4 of 5), but
under-predicted how often an advance is statistically separable on every
board (P 14 % vs 25 %, 14 vs 38, 14 vs 33, 28 vs 55). Iteration 12 found
why that might be: real entrants' residuals correlate, so a paired
comparison between two real systems is sharper than independent noise
allows, and a step of the same size is separable more often.

This replaces the iid Gaussian item noise of the chase model with real
residual vectors: the synthetic field's noise is a row bootstrap of the
board's own two-way-centred residual matrix. Sampling whole rows keeps each
system's own noise level and, for rows drawn from real pairs, their
correlation. Nothing else changes: same q, same lambda, same dates.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * P within +-15 points on >= 4 of 5 boards (iid version: 2 of 5);
  * P rises relative to the row-permuted control on >= 4 of 5
    (restated after the control change; the iid comparison is still
    printed for continuity with chase_model.py);
  * A stays within +-30 % on the same 3 boards it held for, or better;
  * if P still falls short everywhere, correlation is NOT the explanation
    and the gap is something else - recorded as such, no third model
    tonight.

CONTROL CHANGED AFTER THE FIRST SELF-CHECK (ae13d3d). The check said: on a
board whose residual correlation has been destroyed, the bootstrap version
must give the same P as the iid Gaussian version within 8 points, "the only
difference being correlation". It failed, 8 % vs 19 % - and the premise was
wrong. Bootstrap rows also carry UNEQUAL noise levels (row SD 0.042 to 0.115
on MTEB) while the iid version gives everyone the same. Heteroscedastic
noise changes P on its own: a noisy row is likelier to set a record by luck
and its pair is then harder to separate. So the iid model is not the right
control for correlation. The control used below is the SAME bootstrap with
each row's residuals permuted across items: identical per-row noise levels,
correlation destroyed. That isolates the one thing being tested.

SELF-CHECKS
  * the bootstrap noise must reproduce the board's median sigma_p within
    15 % (it is the board's own noise, so this is a wiring check);
  * the row-permuted control must keep each row's SD exactly and drive the
    mean off-diagonal residual correlation to the -1/(J-1) floor.

    python chase_correlated.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from evidence_trajectory import load
from sota_audit import advances
from sota_twin import audit, fit_drift, sigma_p_of, SEED
from step_sizes import steps_u
from residual_correlation import decompose, twin4
from chase_model import BOARDS, chase_field, fit_q_lambda, board_stats, REPS


def chase_boot(J, n, dates, q, lam, resid, base_pool, rng):
    """Chase abilities; noise = bootstrap of real residual ROWS."""
    order = np.argsort(dates, kind="stable")
    ability = np.empty(J)
    record = -np.inf
    for k, idx in enumerate(order):
        a = rng.choice(base_pool) if (k == 0 or rng.random() >= q) else record + rng.exponential(lam)
        ability[idx] = a
        record = max(record, a)
    rows = rng.integers(0, resid.shape[0], J)
    return ability[:, None] + resid[rows]


def chase_boot_perm(J, n, dates, q, lam, resid, base_pool, rng):
    """Control: same rows, each permuted across items - no correlation."""
    r = np.empty_like(resid)
    for j in range(resid.shape[0]):
        r[j] = resid[j, rng.permutation(resid.shape[1])]
    return chase_boot(J, n, dates, q, lam, r, base_pool, rng)


def _check_sigma(rng):
    x, dates = load(*BOARDS["SWE-bench Verified"])
    _, _, resid = decompose(x)
    y = chase_boot(x.shape[0], x.shape[1], dates, 0.3, 0.01, resid, x.mean(axis=1), rng)
    r = sigma_p_of(y) / sigma_p_of(x)
    return abs(r - 1) < 0.15, f"bootstrap noise reproduces sigma_p: ratio {r:.2f}"


def _check_control(rng):
    x, _ = load(*BOARDS["MTEB English v2"])
    _, _, resid = decompose(x)
    r = np.empty_like(resid)
    rr = np.random.default_rng(9)
    for j in range(resid.shape[0]):
        r[j] = resid[j, rr.permutation(resid.shape[1])]
    same_sd = float(np.max(np.abs(r.std(axis=1, ddof=1) - resid.std(axis=1, ddof=1))))
    J = resid.shape[0]
    iu = np.triu_indices(J, k=1)
    rho = float(np.nanmean(np.corrcoef(r)[iu]))
    floor = -1.0 / (J - 1)
    return same_sd < 1e-12 and abs(rho - floor) < 0.01, (
        f"row-permuted control: row SDs unchanged (max diff {same_sd:.1e}), "
        f"mean residual correlation {rho:+.4f} vs floor {floor:+.4f}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    print("self-checks")
    ok = True
    for passed, msg in (_check_sigma(rng), _check_control(rng)):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("CHASE MODEL WITH THE BOARD'S OWN CORRELATED NOISE")
    p("=" * 84)
    p(f"  {'board':<20} {'q':>5} {'lambda':>7} | {'A real':>6} {'boot':>6} | {'P real':>6} {'boot':>6} {'ctrl':>6} {'iid':>6} | {'u real':>6} {'boot':>6}")
    okP, rises, okA = 0, 0, 0
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        J, n = x.shape
        sp = sigma_p_of(x)
        _, _, _, si = fit_drift(x, dates, sp)
        _, _, resid = decompose(x)
        real = board_stats(x, dates, SEED)
        sc = x.mean(axis=1)
        o = np.argsort(dates, kind="stable")
        climb = float(sc.max() - sc[o[0]])
        rec = {int(a["new"]) for a in advances(x, dates)}
        pool = np.array([sc[i] for i in range(J) if i not in rec])
        if pool.size == 0:
            pool = np.array([sc.mean()])
        q, lam = fit_q_lambda(J, n, dates, si, pool, climb, real["u"], np.random.default_rng(SEED + 11))
        boot = [board_stats(chase_boot(J, n, dates, q, lam, resid, pool, np.random.default_rng(SEED + 700 + s)), dates, SEED + 40 * s) for s in range(REPS)]
        ctrl = [board_stats(chase_boot_perm(J, n, dates, q, lam, resid, pool, np.random.default_rng(SEED + 800 + s)), dates, SEED + 50 * s) for s in range(REPS)]
        iid = [board_stats(chase_field(J, n, dates, q, lam, si, pool, np.random.default_rng(SEED + 900 + s)), dates, SEED + 30 * s) for s in range(REPS)]
        bA = float(np.mean([s["A"] for s in boot])); bP = float(np.nanmean([s["P"] for s in boot])); bU = float(np.mean([s["u"] for s in boot]))
        cP = float(np.nanmean([s["P"] for s in ctrl]))
        iP = float(np.nanmean([s["P"] for s in iid]))
        okP += abs(bP - real["P"]) <= 0.15
        rises += bP > cP
        okA += abs(bA / real["A"] - 1) <= 0.30
        p(f"  {name:<20} {q:>5.2f} {lam:>7.4f} | {real['A']:>6d} {bA:>6.1f} | {100 * real['P']:>5.0f}% {100 * bP:>5.0f}% {100 * cP:>5.0f}% {100 * iP:>5.0f}% | {real['u']:>6.2f} {bU:>6.2f}")
    N = len(BOARDS)
    p("")
    p(f"  P within 15 points: {okP}/{N} (pre-registered >= 4; iid version was 2/5)")
    p(f"  P higher than the row-permuted control: {rises}/{N} (pre-registered >= 4)")
    p(f"  A within 30 %:      {okA}/{N} (iid version was 3/5)")
    p("")
    p("  Noise is a row bootstrap of the board's own two-way-centred residuals,")
    p("  so each synthetic entrant carries a real system's noise level, and pairs")
    p("  drawn from real pairs carry their correlation. q and lambda are the same")
    p("  fits as chase_model.py (climb and median step). ctrl = same bootstrap")
    p("  with each row permuted across items: same noise levels, no correlation.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("chase_correlated_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote chase_correlated_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
