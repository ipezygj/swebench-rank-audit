"""How large is a SOTA step compared with luck among equals? (corrected)

FIRST VERSION (fbfb7b6) claimed a parameter-free half-normal: if the new
leader equals the old one, the step is N(0, sigma_p/sqrt n) conditioned
positive, median u = 0.344. The self-check on equal-ability fields gave
median u = 0.171 and KS p = 0.00. The derivation forgot that the old leader
is the MAXIMUM of the k entrants so far: a new equal must exceed a record,
and record increments are far smaller than a random pair's positive
difference (and shrink with k). The half-normal is the wrong null. Kept in
git; the correct null is simulated.

FLAT TWIN  Same J, n, sigma_p and ENTRY DATES as the board, every entrant
of identical ability (beta = 0, tau_res = 0). Its frontier steps are what
luck among equals produces on that exact board. The drift twin of
sota_twin.py (linear rise) is the other reference. Where a board's real
median u sits between the two is its PROGRESS FRACTION:

    pf = (median u_real - median u_flat) / (median u_drift - median u_flat)

pf = 0: SOTA steps are what equal systems and luck give. pf = 1: steps are
what a steadily improving Gaussian field gives. The 'genuine share' is
redefined against the flat twin: share of real steps with u above the flat
twin's 95th percentile, minus 5 %.

EXPECTATIONS (written before this corrected run, after the medians of the
five boards and the failed self-check were seen - consistency checks, not
discoveries):
  * real median u > flat-twin median u on all five boards;
  * pf between 0 and 1 on the four unselected boards; SWE-bench test (24
    pre-selected entrants) may exceed 1.

SELF-CHECKS
  * flat twin of a flat twin: medians within 0.05 (pooled over fields);
  * drifting field: median u above the flat twin's.

    python sota_luck_law.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kstest

from evidence_trajectory import load
from sota_twin import sigma_p_of, synth_dates
from step_sizes import steps_u
from sota_families import BOARDS as DATED4

from sota_twin import fit_drift, dated_twin, SEED, TWINS

BOARDS = dict(DATED4)
BOARDS["SWE-bench test (J=24, pre-selected)"] = ("swebench_test_matrix.csv", None)


def flat_twin(J, n, dates, sigma_item, rng):
    return 0.5 + rng.normal(0, sigma_item, (J, n))


def twin_us(maker, x, dates, k=TWINS):
    J, n = x.shape
    sp = sigma_p_of(x)
    a, beta, tau_res, si = fit_drift(x, dates, sp)
    out = []
    for s in range(k):
        rng = np.random.default_rng(SEED + 50 + s)
        y = maker(J, n, dates, a, beta, tau_res, si, rng)
        out.append(steps_u(y, dates, sigma_p_of(y)))
    return np.concatenate(out)


def _flat(J, n, dates, a, beta, tau_res, si, rng):
    return flat_twin(J, n, dates, si, rng)


def _check_flat_of_flat():
    meds = []
    # 30 fields x 5 twins: a flat field of 60 has only ~5 records, so the
    # medians are noisy; the first run (10 x 3) gave 0.197 vs 0.131 - power.
    for s in range(30):
        rng = np.random.default_rng(600 + s)
        J, n = 60, 200
        dates = synth_dates("2023-01-01", rng.permutation(J) * 5)
        x = flat_twin(J, n, dates, 0.45, rng)
        u = steps_u(x, dates, sigma_p_of(x))
        v = twin_us(_flat, x, dates, k=5)
        meds.append((np.median(u), np.median(v)))
    a = np.array(meds)
    gap = abs(a[:, 0].mean() - a[:, 1].mean())
    return gap < 0.05, f"flat twin of flat fields: median u {a[:, 0].mean():.3f} vs twin {a[:, 1].mean():.3f}"


def _check_drift():
    rng = np.random.default_rng(801)
    J, n = 60, 200
    t = np.sort(rng.uniform(0, 2, J))
    x = 0.3 + 0.15 * t[:, None] + rng.normal(0, 0.45, (J, n))
    dates = synth_dates("2023-01-01", np.round(t * 365).astype(int))
    u = steps_u(x, dates, sigma_p_of(x))
    v = twin_us(_flat, x, dates, k=5)
    return np.median(u) > np.median(v), f"drifting field: median u {np.median(u):.2f} vs flat twin {np.median(v):.2f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_flat_of_flat(), _check_drift()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("SOTA STEPS BETWEEN LUCK AMONG EQUALS AND STEADY PROGRESS")
    p("=" * 90)
    p(f"  {'board':<34} {'steps':>5} {'real med':>8} {'flat med':>8} {'drift med':>9} {'pf':>6} {'genuine':>8} {'KS vs flat':>10}")
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        u = steps_u(x, dates, sigma_p_of(x))
        uf = twin_us(_flat, x, dates)
        ud = twin_us(dated_twin, x, dates)
        mr, mf, md = np.median(u), np.median(uf), np.median(ud)
        pf = (mr - mf) / (md - mf) if md != mf else float("nan")
        thr = np.percentile(uf, 95)
        genuine = max(0.0, float(np.mean(u > thr)) - 0.05)
        pv = float(kstest(u, uf).pvalue)
        p(f"  {name:<34} {len(u):>5} {mr:>8.2f} {mf:>8.2f} {md:>9.2f} {pf:>6.2f} {100 * genuine:>7.0f}% {pv:>10.3f}")
    p("")
    p("  flat = same J, n, sigma_p, entry dates, every entrant equal; drift = the")
    p("  linear-rise twin. pf = where the real median step sits between them.")
    p("  genuine = share of real steps above the flat twin's 95th percentile,")
    p("  minus the 5 % luck gives. KS vs flat: p small = real steps are not")
    p("  luck among equals. First version's half-normal null was wrong (see head).")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("sota_luck_law_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote sota_luck_law_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
