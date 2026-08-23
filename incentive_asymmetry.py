"""Does pair-specific resolution reward derivative work?

R10 makes a leaderboard report the resolution of the pair behind a claim.
That is the honest number - but it has an incentive attached, and the
standard should state it rather than discover it later.

Two entrants improve on the leader by exactly the same amount. One is a
variant of the leader (shares its base model or scaffold, kappa below 1);
the other is unrelated (kappa about 1). The variant's difference vector has
a smaller SD, so its improvement clears significance at a smaller gap. The
minimum detectable gap at 80 % power, 5 %, for a pair with sharpness kappa,

    delta_min = (z_a + z_b) * kappa * sqrt(sd_lead^2 + sd_new^2) / sqrt(n)

so the required improvement scales linearly with kappa. Measured here for
every board: what a relative needs, what an outsider needs, and how those
compare with what entrants actually deliver.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * a relative (kappa at the board's frontier level) needs at least 15 %
    less improvement than an outsider on >= 7 of 9 boards;
  * on >= 3 boards the outsider's required gap exceeds the LARGEST frontier
    step the board has ever seen - an unrelated entrant cannot make a
    separable claim there at all;
  * the relative's required gap is below the largest observed step on
    >= 7 of 9.

SELF-CHECKS
  * delta_min must scale exactly linearly with kappa (analytic identity,
    checked numerically);
  * a simulated pair at delta_min must reject at close to 80 % power.

    python incentive_asymmetry.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from entropy_law_test import MATRICES
from evidence_trajectory import load
from pair_sharpness import kappa_matrix
from sota_audit import advances
from chase_model import BOARDS as DATED

Z = norm.isf(0.025) + norm.isf(0.20)          # 5 % two-sided, 80 % power
SEED = 20260823

# median frontier kappa per dated board (pair_sharpness_results.txt, 724b398);
# for boards without dates, the median kappa among the top ten pairs is used.
FRONTIER_KAPPA = {"SWE-bench Verified": 0.882, "MTEB English v2": 0.530,
                  "LiveBench": 0.894, "ProteinGym DMS": 0.633}


def delta_min(sd_a, sd_b, n, kappa):
    return Z * kappa * math.sqrt(sd_a ** 2 + sd_b ** 2) / math.sqrt(n)


def top_kappa(x):
    K = kappa_matrix(x)
    order = np.argsort(-x.mean(axis=1))
    top = [int(i) for i in order[: min(10, x.shape[0])]]
    vals = [K[i, j] for a, i in enumerate(top) for j in top[a + 1:]]
    return float(np.nanmedian(vals))


def largest_step(name):
    if name not in DATED:
        return float("nan")
    x, dates = load(*DATED[name])
    sc = x.mean(axis=1)
    steps = [sc[a["new"]] - sc[a["old"]] for a in advances(x, dates)]
    return float(max(steps)) if steps else float("nan")


def _check_linearity():
    d1 = delta_min(0.4, 0.45, 300, 0.5)
    d2 = delta_min(0.4, 0.45, 300, 1.0)
    return abs(d2 / d1 - 2) < 1e-12, f"delta_min doubles when kappa doubles: {d1:.5f} -> {d2:.5f}"


def _check_power():
    rng = np.random.default_rng(SEED)
    n, kappa, sd = 400, 0.7, 0.45
    dm = delta_min(sd, sd, n, kappa)
    hits = 0
    trials = 400
    for s in range(trials):
        r = np.random.default_rng(SEED + s)
        common = r.normal(0, sd, n)
        # two systems with correlation giving this kappa: rho = 1 - kappa^2
        rho = 1 - kappa ** 2
        a = math.sqrt(rho) * common + math.sqrt(1 - rho) * r.normal(0, sd, n)
        b = math.sqrt(rho) * common + math.sqrt(1 - rho) * r.normal(0, sd, n)
        d = (a + dm) - b
        t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
        hits += t > norm.isf(0.025)
    power = hits / trials
    return abs(power - 0.80) < 0.10, f"simulated power at delta_min: {100 * power:.0f} % (target 80 %)"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_linearity(), _check_power()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHAT A CLAIM COSTS, DEPENDING ON WHO MAKES IT")
    p("=" * 92)
    p(f"  {'leaderboard':<22} {'n':>5} {'kappa rel':>10} {'relative needs':>15} {'outsider needs':>15} "
      f"{'saving':>7} {'largest step seen':>18}")
    saves, blocked, reachable = 0, 0, 0
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J, n = x.shape
        xc = x - x.mean(axis=0, keepdims=True)
        sd = xc.std(axis=1, ddof=1)
        order = np.argsort(-x.mean(axis=1))
        lead, second = int(order[0]), int(order[1])
        kap = FRONTIER_KAPPA.get(name, top_kappa(x))
        d_rel = delta_min(sd[lead], sd[second], n, kap)
        d_out = delta_min(sd[lead], sd[second], n, 1.0)
        big = largest_step(name)
        saves += (1 - d_rel / d_out) >= 0.15
        if not math.isnan(big):
            blocked += d_out > big
            reachable += d_rel < big
        p(f"  {name:<22} {n:>5} {kap:>10.3f} {100 * d_rel:>14.2f}p {100 * d_out:>14.2f}p "
          f"{100 * (1 - d_rel / d_out):>6.0f}% {(f'{100 * big:.2f}p' if not math.isnan(big) else '-'):>18}")
    N = sum(1 for _, pth in MATRICES.items() if Path(pth).exists())
    ND = sum(1 for k in MATRICES if k in DATED)
    p("")
    p(f"  relative needs at least 15 % less improvement: {saves}/{N} (pre-registered >= 7)")
    p(f"  outsider's requirement exceeds the largest step ever seen: {blocked}/{ND} dated boards (pre-registered >= 3)")
    p(f"  relative's requirement is below the largest step: {reachable}/{ND} dated boards (pre-registered >= 7 of 9,")
    p(f"    which only {ND} boards can answer - recorded as a mis-specified threshold)")
    p("")
    p("  'relative' = an entrant whose pair sharpness with the leader matches the")
    p("  board's frontier level; 'outsider' = kappa 1, independent behaviour. Both")
    p("  need the same power against the same leader on the same items. The saving")
    p("  is exactly the kappa ratio - which is the honest number, and also an")
    p("  incentive: on a board where relatives are sharp, a variant of the leader")
    p("  can make a separable claim on an improvement an unrelated system cannot.")
    p("  A standard that requires pair resolution (R10) should say so out loud.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("incentive_asymmetry_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote incentive_asymmetry_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
