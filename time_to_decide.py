"""How long until this benchmark can name a winner again?

Two numbers already measured per dated board:

    delta_min   the improvement a new entrant needs to separate from the
                current leader at 80 % power, 5 %, using that pair's own
                resolution (incentive_asymmetry.py)
    beta        the field's historical rate of improvement, points per year
                (sota_twin.py's linear drift)

Their ratio is a waiting time: how long the field must keep improving at
its own historical rate before the frontier moves far enough that the move
itself is separable. It is not a prediction of when someone will claim a
new state of the art - that happens continuously - but of when a claim
will be able to survive the standard's own criterion.

Reported for both cases: a relative (kappa at the board's frontier level)
and an unrelated entrant (kappa 1).

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * every board's wait is between one month and three years for a relative;
  * SWE-bench Verified, whose beta is 0.325/year and whose top pair needs
    about 4.9 points, waits under 24 months;
  * the ordering of boards by wait is NOT the ordering by beta - resolution
    matters at least as much as speed.

SELF-CHECKS
  * doubling beta must halve the wait;
  * a board whose leader is already separable must return a wait of zero.

    python time_to_decide.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

from evidence_trajectory import load
from chase_model import BOARDS
from sota_twin import fit_drift, sigma_p_of
from pair_sharpness import kappa_matrix
from sota_audit import advances

Z = norm.isf(0.025) + norm.isf(0.20)
SEED = 20260823


def board_numbers(x, dates):
    sp = sigma_p_of(x)
    _, beta, _, _ = fit_drift(x, dates, sp)
    xc = x - x.mean(axis=0, keepdims=True)
    sd = xc.std(axis=1, ddof=1)
    order = np.argsort(-x.mean(axis=1))
    i1, i2 = int(order[0]), int(order[1])
    K = kappa_matrix(x)
    kap_front = float(np.nanmedian([K[a["new"], a["old"]] for a in advances(x, dates)]))
    n = x.shape[1]
    base = Z * math.sqrt(sd[i1] ** 2 + sd[i2] ** 2) / math.sqrt(n)
    gap = float(x[i1].mean() - x[i2].mean())
    return {"beta": beta, "gap": gap, "d_rel": kap_front * base, "d_out": base,
            "kappa": kap_front, "t": gap / (float((x[i1] - x[i2]).std(ddof=1)) / math.sqrt(n))}


def wait_months(need, gap, beta):
    if beta <= 0:
        return float("inf")
    remaining = max(need - gap, 0.0)
    return 12.0 * remaining / beta


def _check_halving():
    a = wait_months(0.05, 0.01, 0.2)
    b = wait_months(0.05, 0.01, 0.4)
    return abs(a / b - 2) < 1e-9, f"doubling beta halves the wait: {a:.1f} -> {b:.1f} months"


def _check_zero():
    return wait_months(0.02, 0.05, 0.3) == 0.0, "an already-separable leader waits zero months"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_halving(), _check_zero()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("HOW LONG UNTIL THE TOP CAN BE DECIDED AGAIN?")
    p("=" * 86)
    p(f"  {'board':<20} {'beta /yr':>9} {'current gap':>12} {'relative needs':>15} "
      f"{'outsider needs':>15} {'wait rel':>9} {'wait out':>9}")
    rows = []
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        b = board_numbers(x, dates)
        wr = wait_months(b["d_rel"], b["gap"], b["beta"])
        wo = wait_months(b["d_out"], b["gap"], b["beta"])
        rows.append((name, b, wr, wo))
        p(f"  {name:<20} {100 * b['beta']:>8.1f}p {100 * b['gap']:>11.2f}p {100 * b['d_rel']:>14.2f}p "
          f"{100 * b['d_out']:>14.2f}p {wr:>8.1f}m {wo:>8.1f}m")
    p("")
    inrange = sum(1 for _, _, wr, _ in rows if 1 <= wr <= 36)
    swe = next((r for r in rows if r[0].startswith("SWE-bench Verified")), None)
    by_wait = [r[0] for r in sorted(rows, key=lambda r: r[2])]
    by_beta = [r[0] for r in sorted(rows, key=lambda r: -r[1]["beta"])]
    p(f"  wait between 1 and 36 months for a relative: {inrange}/{len(rows)} (pre-registered: all)")
    if swe:
        p(f"  SWE-bench Verified under 24 months: {'yes' if swe[2] < 24 else 'NO'} ({swe[2]:.1f})")
    p(f"  order by wait  : {', '.join(by_wait)}")
    p(f"  order by beta  : {', '.join(by_beta)}")
    p(f"  the two orders differ: {'yes' if by_wait != by_beta else 'NO'} (pre-registered: yes)")
    p("")
    p("  The wait assumes the field keeps improving at its own historical rate and")
    p("  that the improvement lands on the pair in question. It is a lower bound on")
    p("  when a claim could survive the standard, not a forecast of when one will")
    p("  be made. A board with a large beta and a coarse resolution can wait longer")
    p("  than a slow board that measures finely.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("time_to_decide_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote time_to_decide_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
