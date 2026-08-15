#!/usr/bin/env python3
"""What is this leaderboard's resolution, and which of its rows are ordered?

Give it a matrix of per-item outcomes — one row per system, one column per
evaluation item — and it answers three questions:

    1. how small a gap can this benchmark actually resolve?
    2. which adjacent pairs on the board are not ordered by the data?
    3. how many items would it take to resolve the gap you care about?

Both outcome types are handled by the same report, because the question is the
same and only the test changes:

    binary     (solved / not solved)  -> exact McNemar on discordant items
    continuous (a score per item)     -> paired bootstrap over items

Both are *paired*: the same items, in the same order, for every system. That is
what makes leaderboard comparisons far sharper than their marginal error bars
suggest, and it is why "the confidence intervals overlap" is the wrong way to
read one.

Usage:
    python leaderboard_resolution.py matrix.csv            # auto-detect type
    python leaderboard_resolution.py matrix.csv --binary
    python leaderboard_resolution.py --selftest

CSV layout: first column = system name, remaining columns = items.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
POWER = 0.80
CONST = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)
N_BOOT = 2000


# ---------------------------------------------------------------- tests

def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((b == 1) & (a == 0)))
    n = a_only + b_only
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * stats.binom.cdf(min(a_only, b_only), n, 0.5)))


def paired_bootstrap_p(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    """Share of paired item-resamples in which the sign of the difference flips.

    One item draw is applied to both systems, so the correlation between them is
    preserved rather than being resampled away.
    """
    diff = a - b
    n = len(diff)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT)
    means = counts @ diff / n
    share = float(np.mean(means <= 0)) if diff.mean() > 0 else float(np.mean(means >= 0))
    return min(1.0, 2 * share)


def resolution_binary(matrix: np.ndarray, gaps_close: float = 0.02) -> float:
    """Smallest gap this benchmark can resolve, from equation delta >= C*sqrt(d/n).

    Uses the discordance of *close* pairs: those are the ones whose ordering is
    in question, and they disagree on fewer items than distant pairs do, so the
    all-pairs median gives an answer that is too pessimistic.
    """
    n = matrix.shape[1]
    rates = matrix.mean(axis=1)
    disc = [
        float(np.sum(matrix[i] != matrix[j])) / n
        for i, j in combinations(range(len(rates)), 2)
        if abs(rates[i] - rates[j]) < gaps_close
    ]
    if not disc:
        disc = [
            float(np.sum(matrix[i] != matrix[j])) / n
            for i, j in combinations(range(len(rates)), 2)
        ]
    return CONST * np.sqrt(float(np.median(disc)) / n)


def items_needed(discordance: float, target_gap: float) -> float:
    return discordance * (CONST / target_gap) ** 2


# ---------------------------------------------------------------- report

def report(df: pd.DataFrame, binary: bool) -> dict:
    values = df.to_numpy(dtype=float)
    n_sys, n_items = values.shape
    score = values.mean(axis=1)
    order = np.argsort(-score)
    names = list(df.index[order])
    values, score = values[order], score[order]

    p_of = (lambda i, j: mcnemar_exact(values[i], values[j])) if binary else (
        lambda i, j: paired_bootstrap_p(values[i], values[j]))

    undecided_adj = [
        i for i in range(n_sys - 1) if p_of(i, i + 1) >= ALPHA
    ]
    sep = sum(p_of(i, j) < ALPHA for i, j in combinations(range(n_sys), 2))
    total = n_sys * (n_sys - 1) // 2
    tied_top = sum(p_of(0, j) >= ALPHA for j in range(n_sys))

    out = {
        "systems": n_sys, "items": n_items, "type": "binary" if binary else "continuous",
        "adjacent_undecided": len(undecided_adj), "adjacent_total": n_sys - 1,
        "pairs_separated": sep, "pairs_total": total,
        "tied_with_top": tied_top,
    }
    if binary:
        res = resolution_binary(values)
        out["resolution_pp"] = 100 * res
        out["items_for_1pp"] = items_needed(
            float(np.median([
                float(np.sum(values[i] != values[j])) / n_items
                for i, j in combinations(range(n_sys), 2)
                if abs(score[i] - score[j]) < 0.02
            ] or [0.2])), 0.01)

    print(f"\n{n_sys} systems x {n_items} items ({out['type']})")
    print(f"  adjacent pairs NOT ordered by the data : "
          f"{out['adjacent_undecided']}/{out['adjacent_total']}")
    print(f"  all pairs separated                    : "
          f"{sep}/{total} ({100*sep/total:.1f}%)")
    print(f"  systems indistinguishable from the top : {tied_top}")
    if binary:
        print(f"  resolution of this benchmark           : {out['resolution_pp']:.2f} pp")
        print(f"  items needed to resolve 1 pp          : {out['items_for_1pp']:,.0f}"
              f"  (has {n_items})")
    if undecided_adj[:8]:
        print("  undecided at the top: " + ", ".join(
            f"#{i+1}v#{i+2}" for i in undecided_adj[:8]))
    return out


# ---------------------------------------------------------------- selftest

def selftest() -> int:
    rng = np.random.default_rng(0)
    failures = []

    # Negative control: two systems with identical outcomes are never ordered.
    a = (rng.random(400) < 0.5).astype(float)
    if mcnemar_exact(a, a.copy()) != 1.0:
        failures.append("identical binary systems were separated")
    if paired_bootstrap_p(a, a.copy()) < ALPHA:
        failures.append("identical continuous systems were separated")

    # Planted signal: the control above must not pass for lack of power.
    worse = a.copy()
    ones = np.flatnonzero(worse == 1)
    worse[rng.choice(ones, size=60, replace=False)] = 0
    if mcnemar_exact(a, worse) >= 1e-6:
        failures.append("60 planted binary losses not detected")
    shifted = a + 0.5
    if paired_bootstrap_p(shifted, a) >= ALPHA:
        failures.append("a uniform +0.5 shift not detected")

    # The law must be conservative: on random data no pair whose gap clears the
    # threshold may be unresolvable by the test itself.
    m = (rng.random((12, 600)) < rng.uniform(0.3, 0.7, (12, 1))).astype(float)
    n = m.shape[1]
    for i, j in combinations(range(12), 2):
        d = float(np.sum(m[i] != m[j])) / n
        gap = abs(m[i].mean() - m[j].mean())
        if gap >= CONST * np.sqrt(d / n) and mcnemar_exact(m[i], m[j]) >= ALPHA:
            failures.append(f"law claimed pair {i},{j} resolvable but the test could not")
            break

    if failures:
        print("SELFTEST FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("selftest passed: identical systems never ordered (both types); planted "
          "effects detected; the resolution bound stayed conservative on random data")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest or not args.csv:
        return selftest()

    df = pd.read_csv(args.csv, index_col=0).dropna(axis="index", how="any")
    unique = pd.unique(df.to_numpy().ravel())
    auto_binary = set(np.unique(unique.astype(float))) <= {0.0, 1.0}
    binary = args.binary or (auto_binary and not args.continuous)
    report(df, binary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
