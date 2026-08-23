"""How big is a SOTA step, in units of what the benchmark can resolve?

sota_twin.py found that real boards produce 1.4-2.4x more frontier advances
than a Gaussian field drifting at the same rate, and that fewer of them are
separable. The direct measurement of that reading is the step-size
distribution: each frontier advance's gain divided by the pairwise
resolution of the benchmark at that moment,

    u = gain / (1.96 * sigma_p / sqrt(n))

u < 1 is a step the benchmark cannot tell from zero at the pairwise level.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the REAL median u is below 1 on all three boards;
  * the TWIN median u (linear-drift field, 10 twins pooled) is above 1 on
    all three;
  * at least 60 % of real steps have u < 1 on each board;
  * exploratory, no threshold: whether the three real u-distributions look
    alike once normalised (two-sample KS, p reported, not judged).

SELF-CHECKS
  * on the twin, the share of steps with u > 1.96/1.96 = 1 must agree with
    the twin's pairwise-separable share from sota_twin (within 15 points) -
    u is the same quantity as the test statistic up to the sign-flip null;
  * u must be invariant to an affine rescaling of the matrix.

    python step_sizes.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from sota_audit import advances
from evidence_trajectory import load, DATED
from sota_twin import fit_drift, dated_twin, sigma_p_of, SEED, TWINS

Z = 1.96


def steps_u(x, dates, sigma_p):
    n = x.shape[1]
    sc = x.mean(axis=1)
    thr = Z * sigma_p / math.sqrt(n)
    return np.array([(sc[a["new"]] - sc[a["old"]]) / thr for a in advances(x, dates)])


def _check_affine():
    rng = np.random.default_rng(3)
    x = rng.random((40, 100))
    dates = 20230101 + np.arange(40)   # advances() only sorts these; any ints do
    u1 = steps_u(x, dates, sigma_p_of(x))
    y = 3.0 * x + 7.0
    u2 = steps_u(y, dates, sigma_p_of(y))
    ok = np.allclose(u1, u2, rtol=0.05)
    return ok, f"affine invariance: max rel diff {np.max(np.abs(u1 / u2 - 1)):.3f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok, msg = _check_affine()
    print(f"  [{'ok  ' if ok else 'FAIL'}] {msg}")
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("SOTA STEP SIZES IN UNITS OF THE BENCHMARK'S PAIRWISE RESOLUTION")
    p("=" * 84)
    p(f"  {'leaderboard':<20} {'steps':>6} {'median u':>9} {'u<1':>6} {'q25':>6} {'q75':>6} {'max':>6}"
      f"   {'twin med':>8} {'twin u<1':>8} {'twin steps':>10}")
    real_u = {}
    verdict = []
    twin_sep_check = []
    for name, (path, dc) in DATED.items():
        x, dates = load(path, dc)
        J, n = x.shape
        sp = sigma_p_of(x)
        u = steps_u(x, dates, sp)
        real_u[name] = u
        a, beta, tau_res, si = fit_drift(x, dates, sp)
        tu = []
        for s in range(TWINS):
            y = dated_twin(J, n, dates, a, beta, tau_res, si, np.random.default_rng(SEED + 50 + s))
            tu.append(steps_u(y, dates, sigma_p_of(y)))
        tu = np.concatenate(tu)
        twin_sep_check.append((name, float(np.mean(tu > 1))))
        verdict.append((name, float(np.median(u)) < 1, float(np.median(tu)) > 1, float(np.mean(u < 1)) >= 0.6))
        p(f"  {name:<20} {len(u):>6} {np.median(u):>9.2f} {100 * np.mean(u < 1):>5.0f}% {np.percentile(u, 25):>6.2f} "
          f"{np.percentile(u, 75):>6.2f} {u.max():>6.2f}   {np.median(tu):>8.2f} {100 * np.mean(tu < 1):>7.0f}% {len(tu):>10}")
    p("")
    p("  pre-registered: real median u < 1 on all three; twin median u > 1 on all three;")
    p("  >= 60 % of real steps below 1 on each board.")
    for name, a1, a2, a3 in verdict:
        p(f"    {name:<20} real median<1 {'yes' if a1 else 'NO'}   twin median>1 {'yes' if a2 else 'NO'}   real share<1 >= 60 % {'yes' if a3 else 'NO'}")
    p("")
    p("  twin consistency (share of twin steps with u > 1 vs sota_twin's twin pairwise share 55/60/50 %):")
    for name, sh in twin_sep_check:
        p(f"    {name:<20} u>1 share {100 * sh:.0f} %")
    p("")
    names = list(real_u)
    p("  exploratory: are the three real u-distributions alike? (two-sample KS p, not judged)")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ks = ks_2samp(real_u[names[i]], real_u[names[j]])
            p(f"    {names[i].split()[0]:<10} vs {names[j].split()[0]:<10} KS p = {ks.pvalue:.2f}")
    p("")
    p("  u = (new leader - old leader) / (1.96 sigma_p / sqrt(n)). A step with")
    p("  u < 1 cannot be told from zero by a pairwise test on that benchmark.")
    p("  The twin is the linear-drift Gaussian field of sota_twin.py.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("step_sizes_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote step_sizes_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
