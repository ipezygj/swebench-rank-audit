"""A parameter-free prediction for the size of a SOTA step: the half-normal.

Suppose the new leader is exactly as good as the old one and the observed
step is luck alone. The paired difference of their mean scores is
N(0, sigma_p / sqrt n) - sigma_p is by definition the per-item SD of the
pairwise difference - and a frontier advance is that difference conditioned
to be positive: a HALF-NORMAL with scale sigma_p / sqrt n. In resolution
units u = step / (1.96 sigma_p / sqrt n):

    median u = 0.6745 / 1.96 = 0.344     q25 = 0.163     q75 = 0.587
    share of steps with u > 1 (p < 0.05):  5.0 %

Every part of that is fixed before any data. If the median SOTA step on a
board is near 0.34, the median new leader was NOT better than the old one;
the observed improvement is what equal ability plus luck produces. Steps
with u > 1 beyond the 5 % that luck gives are the genuine advances.

HONESTY NOTE  This derivation was written AFTER the medians of five boards
had been seen (step_sizes 0.40 / 0.27 / 0.41, fourth_board 0.57 / 1.93) and
after their quartiles were printed. It is therefore not a pre-registered
test on those boards. What is new here and not previously looked at:
  * the full-shape comparison of the u < 1 part with the truncated
    half-normal (KS, p reported);
  * the derived 'genuine share' pi = (obs share u>1 - 0.05) / 0.95;
  * the prediction for ANY future dated board: median u in [0.25, 0.45]
    unless the board's entrants are pre-selected (as SWE-bench full test's
    24 are), and u>1 share above 5 %.

SELF-CHECKS
  * on a simulated field where every new entrant has EXACTLY the leader's
    ability, the frontier u's must be half-normal: KS p > 0.05 pooled over
    fields and median within 0.05 of 0.344;
  * on a field where entrants are drifting upward, median u must exceed 0.40.

    python sota_luck_law.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import halfnorm, kstest

from evidence_trajectory import load
from sota_twin import sigma_p_of, synth_dates
from step_sizes import steps_u
from sota_families import BOARDS as DATED4

SCALE = 1 / 1.96
MED, Q25, Q75 = halfnorm.ppf(0.5, scale=SCALE), halfnorm.ppf(0.25, scale=SCALE), halfnorm.ppf(0.75, scale=SCALE)
TAIL = float(halfnorm.sf(1.0, scale=SCALE))

BOARDS = dict(DATED4)
BOARDS["SWE-bench test (J=24, pre-selected)"] = ("swebench_test_matrix.csv", None)


def truncated_ks(u):
    """KS of the u < 1 part against the half-normal truncated at 1."""
    v = u[u < 1]
    if len(v) < 5:
        return float("nan"), len(v)
    F1 = halfnorm.cdf(1.0, scale=SCALE)
    cdf = lambda t: halfnorm.cdf(t, scale=SCALE) / F1
    return float(kstest(v, cdf).pvalue), len(v)


def _check_equal_ability():
    us = []
    for s in range(40):
        rng = np.random.default_rng(700 + s)
        J, n = 60, 200
        # every entrant has the same ability; order of arrival random
        x = 0.5 + rng.normal(0, 0.45, (J, n))
        dates = synth_dates("2023-01-01", rng.permutation(J) * 5)
        us.append(steps_u(x, dates, sigma_p_of(x)))
    u = np.concatenate(us)
    pv = float(kstest(u, lambda t: halfnorm.cdf(t, scale=SCALE)).pvalue)
    med = float(np.median(u))
    return pv > 0.05 and abs(med - MED) < 0.05, f"equal-ability fields: {len(u)} steps, median u {med:.3f} (theory {MED:.3f}), KS p {pv:.2f}"


def _check_drift():
    us = []
    for s in range(20):
        rng = np.random.default_rng(800 + s)
        J, n = 60, 200
        t = np.sort(rng.uniform(0, 2, J))
        x = 0.3 + 0.15 * t[:, None] + rng.normal(0, 0.45, (J, n))
        dates = synth_dates("2023-01-01", np.round(t * 365).astype(int))
        us.append(steps_u(x, dates, sigma_p_of(x)))
    med = float(np.median(np.concatenate(us)))
    return med > 0.40, f"drifting fields: median u {med:.2f} (must exceed 0.40)"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_equal_ability(), _check_drift()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("THE HALF-NORMAL SOTA STEP: EQUAL ABILITY PLUS LUCK")
    p("=" * 84)
    p(f"  theory (no parameters): median u {MED:.3f}, q25 {Q25:.3f}, q75 {Q75:.3f}, share u>1 {100 * TAIL:.1f} %")
    p("")
    p(f"  {'board':<34} {'steps':>5} {'median':>7} {'q25':>6} {'q75':>6} {'u>1':>6} {'genuine':>8} {'KS p (u<1)':>11}")
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        u = steps_u(x, dates, sigma_p_of(x))
        tail = float(np.mean(u > 1))
        genuine = max(0.0, (tail - TAIL) / (1 - TAIL))
        pv, m = truncated_ks(u)
        p(f"  {name:<34} {len(u):>5} {np.median(u):>7.2f} {np.percentile(u, 25):>6.2f} {np.percentile(u, 75):>6.2f} "
          f"{100 * tail:>5.0f}% {100 * genuine:>7.0f}% {pv:>8.2f} ({m})")
    p("")
    p("  'genuine' = (share u>1 - 5 %) / 95 %: the share of frontier advances that")
    p("  luck alone cannot account for. KS p compares the u<1 part with the")
    p("  half-normal truncated at 1 - small p means even the small steps are not")
    p("  shaped like luck. This derivation came AFTER the medians had been seen;")
    p("  its honest test is the next dated board: median u in [0.25, 0.45] unless")
    p("  entrants are pre-selected, and u>1 share above 5 %.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("sota_luck_law_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote sota_luck_law_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
