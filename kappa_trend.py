"""Is the frontier getting sharper? Pair sharpness over time.

kappa(j,k) = sd(x_j - x_k) / sqrt(sd_j^2 + sd_k^2) is below 1 when two
systems behave alike item by item. Frontier pairs sit at 0.53-0.94 while
the average pair sits at 1.00 (pair_sharpness.py). If entrants increasingly
descend from the same few base models, kappa at the frontier should FALL
over time, and two consequences follow that pull in opposite directions:

  * a sharper pair is easier to separate at the same gap - the benchmark
    resolves the frontier better than its global sigma suggests;
  * but a field of near-copies carries less independent evidence about
    which approach is better - the same reading as J_eff.

Measured here, per board: median frontier kappa by calendar year, median
kappa among all pairs present that year (the baseline), and their ratio.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the frontier kappa / all-pair kappa ratio falls over time on >= 3 of 5
    boards (Spearman with year negative);
  * the all-pair kappa itself stays within 0.05 of 1.0 in every year on
    every board - the change, if any, is in the frontier, not the field;
  * SWE-bench Verified shows it most clearly: its 2023 entrants are RAG
    baselines of different lineages, its 2025 entrants are scaffolds on
    three or four LLMs.

SELF-CHECKS
  * a simulated field where every entrant is independent gives a flat
    ratio (|Spearman| < 0.5) over the same year structure;
  * a simulated field where the sharing fraction grows year by year gives
    a clearly negative Spearman.

    python kappa_trend.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from evidence_trajectory import load
from sota_audit import advances
from pair_sharpness import kappa_matrix
from chase_model import BOARDS
from sota_twin import synth_dates

MIN_ADV = 2


def by_year(x, dates):
    K = kappa_matrix(x)
    yr = dates // 10000
    rows = []
    for a in advances(x, dates):
        rows.append((int(a["date"] // 10000), K[a["new"], a["old"]]))
    out = []
    for y in sorted(set(y_ for y_, _ in rows)):
        fr = [k for y_, k in rows if y_ == y]
        if len(fr) < MIN_ADV:
            continue
        present = np.flatnonzero(yr <= y)
        if len(present) < 4:
            continue
        sub = K[np.ix_(present, present)]
        iu = np.triu_indices(len(present), k=1)
        base = float(np.nanmedian(sub[iu]))
        out.append((y, len(fr), float(np.nanmedian(fr)), base))
    return out


def trend(rows):
    if len(rows) < 3:
        return float("nan"), float("nan")
    ys = [r[0] for r in rows]
    ratio = [r[2] / r[3] for r in rows]
    r = spearmanr(ys, ratio)
    return r.statistic, r.pvalue


def sim_board(growing, rng, J=90, n=250, years=5):
    """Entrants over `years`; if growing, the share sharing a lineage rises."""
    per = J // years
    dates, resid, ability = [], [], []
    base = rng.normal(0, 0.45, n)
    for y in range(years):
        share = (0.15 + 0.18 * y) if growing else 0.3
        for k in range(per):
            dates.append(int(f"{2021 + y}0601"))
            ability.append(0.4 + 0.03 * y + rng.normal(0, 0.05))
            if rng.random() < share:
                rho = 0.75
                resid.append(rho * base + math.sqrt(1 - rho ** 2) * rng.normal(0, 0.45, n))
            else:
                resid.append(rng.normal(0, 0.45, n))
    x = np.array(ability)[:, None] + np.array(resid)
    return x, np.array(dates)


def _check_flat():
    rng = np.random.default_rng(41)
    ss = []
    for s in range(6):
        x, d = sim_board(False, np.random.default_rng(400 + s))
        st, _ = trend(by_year(x, d))
        if not math.isnan(st):
            ss.append(st)
    m = float(np.mean(ss)) if ss else float("nan")
    return abs(m) < 0.5, f"independent entrants: mean Spearman {m:+.2f} over {len(ss)} fields"


def _check_growing():
    ss = []
    for s in range(6):
        x, d = sim_board(True, np.random.default_rng(500 + s))
        st, _ = trend(by_year(x, d))
        if not math.isnan(st):
            ss.append(st)
    m = float(np.mean(ss)) if ss else float("nan")
    return m < -0.3, f"growing lineage sharing: mean Spearman {m:+.2f} over {len(ss)} fields"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_flat(), _check_growing()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("IS THE FRONTIER GETTING SHARPER? PAIR SHARPNESS BY YEAR")
    p("=" * 76)
    neg, baseline_ok = 0, True
    stats = []
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        rows = by_year(x, dates)
        st, pv = trend(rows)
        stats.append((name, st))
        neg += (st < 0) if not math.isnan(st) else 0
        p("")
        p(f"  {name}   Spearman(year, frontier/all ratio) = "
          + ("n/a (fewer than 3 usable years)" if math.isnan(st) else f"{st:+.2f} (p {pv:.2f})"))
        p(f"    {'year':>6} {'adv':>4} {'kappa frontier':>15} {'all pairs':>10} {'ratio':>7}")
        for y, k_, f_, b_ in rows:
            if abs(b_ - 1.0) > 0.05:
                baseline_ok = False
            p(f"    {y:>6} {k_:>4} {f_:>15.3f} {b_:>10.3f} {f_ / b_:>7.3f}")
    p("")
    p(f"  ratio falls over time (negative Spearman): {neg}/{len(stats)} (pre-registered >= 3)")
    p(f"  all-pair kappa within 0.05 of 1.0 in every year: {'yes' if baseline_ok else 'NO'}")
    swe = dict(stats).get("SWE-bench Verified", float("nan"))
    p(f"  SWE-bench Verified: {swe:+.2f}")
    p("")
    p("  kappa below 1 means two systems move together item by item. A falling")
    p("  frontier ratio means each new record-holder resembles the one it")
    p("  passed more than earlier record-holders resembled theirs.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("kappa_trend_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote kappa_trend_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
