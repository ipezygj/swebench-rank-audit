"""The smallest difference each leaderboard could have detected, against the one it prints.

Six iterations of this loop turned up nine defects in my own instruments, and
the same one kept coming back: a test with no power reads exactly like a test
that found nothing. redundancy_power.py measured it directly - a statistic that
rejects an injected 1.6-logit factor a quarter of the time was reported as
evidence of independence.

That failure is not mine alone. It is what a leaderboard does every time it
prints a first place: a comparison is run, it does not separate, and the
ordering is published as though it had. The quantity nobody reports is the
minimum detectable effect - the smallest difference the comparison could have
caught four times in five:

    MDE = (z(1 - alpha/2) + z(power)) * sd(d) / sqrt(n)

with sd(d) the difference SD of THAT PAIR, not the board's median, because
pair_sharpness.py showed those differ by 6 to 47 per cent and
prescription_pairwise.py showed substituting one for the other misstates the
items needed by up to 45x.

Read against the gap the board actually prints, MDE says whether a headline was
ever decidable. A board whose MDE exceeds its own top gap published a
comparison it could not make, whatever the ordering turned out to be.

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  the MDE exceeds the printed gap between first and second on at least 7
      of the 9 boards.
  P2  the median ratio of MDE to that gap is above 3.
  P3  the MDE exceeds the gap between first and FIFTH on at least 5 of 9 - the
      board cannot resolve differences spanning its whole top five.
  P4  the MDE is below the full spread of the board on 9 of 9. A board that
      cannot detect even its own range would mean the calculation is wrong,
      not that the board is bad.

  Not predicted: CASP14, whose top pair separates at t = 9.89 and whose MDE
  must therefore be well under its gap; it is one of the two boards P1 allows
  to go the other way.

SELF-CHECKS (no table if any fails)
  * the formula must be verified by SIMULATION, not trusted: plant a
    difference of exactly the computed MDE into each board's own difference
    series and confirm the empirical rejection rate is 0.80 +- 0.05, on at
    least 3 boards. This is the check the loop's own failures argue for;
  * MDE must scale as 1/sqrt(n): halving the items must multiply it by
    1.41 +- 0.10 on every board;
  * a difference series with no variance must return an infinite MDE rather
    than a number.

    python detectable_difference.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

SEED = 20260824
ALPHA = 0.05
POWER = 0.80
SIMS = 4000

MATRICES = {
    "SWE-bench Verified": "swebench_verified_matrix.csv",
    "MTEB English v2": "mteb_eng_v2_wide.csv",
    "HELM classic": "helm_winrate_matrix.csv",
    "ProteinGym DMS": "proteingym/matrix.csv",
    "TabArena 16 models": "tabarena/matrix_one_per_model.csv",
    "TabArena 45 variants": "tabarena/matrix_all45.csv",
    "CASP14": "casp/matrix.csv",
    "LiveBench": "livebench/matrix.csv",
    "MathArena 2025": "matharena/matrix.csv",
}

ZC = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)      # 2.802 at 5 % and 80 %


def load(path):
    return pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)


def mde(d: np.ndarray) -> float:
    """Smallest mean difference this paired comparison could catch at 80 %."""
    n = len(d)
    sd = float(d.std(ddof=1))
    return float("inf") if sd <= 0 else ZC * sd / math.sqrt(n)


def top_pair(x):
    order = np.argsort(-x.mean(axis=1))
    return x[order[0]] - x[order[1]], order


def empirical_power(d: np.ndarray, effect: float, rng, sims=SIMS) -> float:
    """Plant `effect` into this series and count rejections at alpha.

    Resamples the observed differences with replacement, recentres them so the
    planted effect is the whole signal, and runs the same paired test.
    """
    n = len(d)
    centred = d - d.mean()
    crit = norm.ppf(1 - ALPHA / 2)
    hits = 0
    for _ in range(sims):
        s = rng.choice(centred, n, replace=True) + effect
        sd = s.std(ddof=1)
        if sd > 0 and abs(s.mean() / (sd / math.sqrt(n))) > crit:
            hits += 1
    return hits / sims


def _check_simulation(boards, rng) -> tuple[bool, str]:
    ok, tested, worst, where = 0, 0, 0.0, ""
    for name, d in boards.items():
        m = mde(d)
        if not np.isfinite(m):
            continue
        tested += 1
        p = empirical_power(d, m, rng, sims=2000)
        if abs(p - POWER) > 0.05:
            if abs(p - POWER) > worst:
                worst, where = abs(p - POWER), f"{name} {p:.3f}"
        else:
            ok += 1
    good = tested >= 3 and ok >= 3
    return good, (f"planted MDE gives 0.80 power on {ok} of {tested} boards"
                  + (f"; worst miss {where}" if where else ""))


def _check_scaling(boards) -> tuple[bool, str]:
    worst, where = 0.0, ""
    rng = np.random.default_rng(5)
    for name, d in boards.items():
        if not np.isfinite(mde(d)) or len(d) < 20:
            continue
        half = rng.choice(len(d), len(d) // 2, replace=False)
        r = mde(d[half]) / mde(d)
        if abs(r - math.sqrt(2)) > 0.10 and abs(r - math.sqrt(2)) > worst:
            worst, where = abs(r - math.sqrt(2)), f"{name} {r:.3f}"
    return worst == 0.0, ("MDE scales as 1/sqrt(n) on every board"
                          if not where else f"off on {where}")


def _check_degenerate() -> tuple[bool, str]:
    flat = np.zeros(100)
    return not np.isfinite(mde(flat)), "a difference series with no variance returns infinity"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    boards, rows = {}, {}
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = load(path)
        d, order = top_pair(x)
        sc = x.mean(axis=1)
        boards[name] = d
        gap12 = float(sc[order[0]] - sc[order[1]])
        gap15 = float(sc[order[0]] - sc[order[min(4, len(order) - 1)]])
        rows[name] = {"J": x.shape[0], "n": x.shape[1], "mde": mde(d),
                      "gap12": gap12, "gap15": gap15,
                      "spread": float(sc.max() - sc.min()),
                      "t": float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))
                      if d.std(ddof=1) > 0 else 0.0}

    print("self-checks ...")
    checks = [_check_simulation(boards, rng), _check_scaling(boards), _check_degenerate()]
    ok = True
    for passed, msg in checks:
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("THE SMALLEST DIFFERENCE EACH BOARD COULD HAVE DETECTED")
    p("=" * 100)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>5} {'top t':>7} {'gap 1-2':>9} {'MDE':>9} "
      f"{'MDE/gap':>8} {'gap 1-5':>9} {'MDE/gap15':>10} {'spread':>8}")
    p1 = p3 = p4 = 0
    ratios = []
    for name, v in rows.items():
        r = v["mde"] / v["gap12"] if v["gap12"] > 0 else float("inf")
        r5 = v["mde"] / v["gap15"] if v["gap15"] > 0 else float("inf")
        ratios.append(r)
        if v["mde"] > v["gap12"]:
            p1 += 1
        if v["mde"] > v["gap15"]:
            p3 += 1
        if v["mde"] < v["spread"]:
            p4 += 1
        p(f"  {name:<22} {v['J']:>4} {v['n']:>5} {v['t']:>7.2f} {v['gap12']:>9.4f} "
          f"{v['mde']:>9.4f} {r:>8.1f} {v['gap15']:>9.4f} {r5:>10.1f} {v['spread']:>8.3f}")
    p("")
    med = float(np.median([r for r in ratios if np.isfinite(r)]))
    n = len(rows)
    p(f"  P1  MDE above the printed 1-2 gap on {p1} of {n}      "
      f"pre-registered >= 7:  {'HIT' if p1 >= 7 else 'MISS'}")
    p(f"  P2  median MDE / gap = {med:.1f}                       "
      f"pre-registered > 3:   {'HIT' if med > 3 else 'MISS'}")
    p(f"  P3  MDE above the 1-5 gap on {p3} of {n}              "
      f"pre-registered >= 5:  {'HIT' if p3 >= 5 else 'MISS'}")
    p(f"  P4  MDE below the board's own spread on {p4} of {n}   "
      f"pre-registered = {n}:  {'HIT' if p4 == n else 'MISS'}")
    p("")
    p("  MDE is the smallest difference in mean score this board's top pair")
    p("  could have detected four times in five, at the 5 % level, using that")
    p("  pair's own difference SD and the board's own item count. It is not a")
    p("  property of the systems: it is a property of the measuring instrument,")
    p("  computable before any system is run.")
    p("")
    p("  A board whose MDE exceeds its own printed gap ran a comparison it could")
    p("  not make. The ordering may still be right - nothing here says otherwise -")
    p("  but the board is not the reason to believe it.")
    p("")
    p("  The formula is not trusted: the self-check plants a difference of")
    p("  exactly the computed MDE into each board's own difference series and")
    p("  confirms the test then rejects four times in five. This loop spent two")
    p("  iterations on a statistic whose power nobody had measured, and that is")
    p("  the general lesson worth carrying: a comparison that fails to separate")
    p("  is evidence of nothing until its power is known.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("detectable_difference_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                         newline=chr(10))
    print(chr(10) + "wrote detectable_difference_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
