"""How much of the SWE-bench Verified ranking survives its own sampling error?

Every system is scored on the same 500 instances, and the per-instance outcomes
are public, so two systems can be compared the way paired binary data should be:
McNemar's exact test on the instances where they disagree. The instances they
both solve and both miss carry no information about which is better, and a
comparison that ignores that pairing throws away most of the power.

Self-checks run before any headline number is printed:
  - a system against itself must never be called different (p = 1);
  - a system against a copy with k outcomes flipped must be detected at large k
    and not at k = 0;
  - the Wilson interval must reproduce a published reference value.
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

Z = 1.959963984540054
ALPHA = 0.05


def wilson(successes: int, total: int, z: float = Z) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return centre - half, centre + half


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar. Returns (b_wins, a_wins, two-sided p)."""
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((b == 1) & (a == 0)))
    n = a_only + b_only
    if n == 0:
        return b_only, a_only, 1.0
    # Under H0 each discordant instance is a fair coin.
    k = min(a_only, b_only)
    p = min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5))
    return b_only, a_only, p


def selfcheck(matrix: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    # Use a system with plenty of resolved instances: the planted-flip check
    # needs room to plant. The first row alphabetically is an early baseline
    # with almost none, and the check would silently test nothing.
    row = matrix.loc[matrix.sum(axis=1).idxmax()].to_numpy()

    _, _, p_same = mcnemar_exact(row, row.copy())
    assert p_same == 1.0, f"a system compared with itself gave p={p_same}"

    flipped = row.copy()
    ones = np.flatnonzero(flipped == 1)
    flipped[rng.choice(ones, size=60, replace=False)] = 0
    _, _, p_flip = mcnemar_exact(row, flipped)
    assert p_flip < 1e-6, f"60 planted losses not detected (p={p_flip})"

    lo, hi = wilson(5, 10)
    assert abs(lo - 0.2366) < 5e-4 and abs(hi - 0.7634) < 5e-4, (lo, hi)

    print("self-checks passed: identity p=1, planted 60-flip detected, "
          "Wilson 5/10 matches the published [0.2366, 0.7634]")


def main() -> None:
    here = Path(__file__).parent
    matrix = pd.read_csv(here / "swebench_verified_matrix.csv", index_col=0)
    selfcheck(matrix)

    n_inst = matrix.shape[1]
    rate = matrix.mean(axis=1).sort_values(ascending=False)
    order = list(rate.index)
    values = {s: matrix.loc[s].to_numpy() for s in order}

    print(f"\n{len(order)} systems, {n_inst} instances\n")
    print("rank  resolve%   95% Wilson CI      system")
    for i, s in enumerate(order[:12], 1):
        k = int(matrix.loc[s].sum())
        lo, hi = wilson(k, n_inst)
        print(f"{i:>4}  {100 * k / n_inst:6.2f}   [{100 * lo:5.2f}, {100 * hi:5.2f}]   {s[:52]}")

    # --- adjacent pairs -------------------------------------------------
    print("\nadjacent-rank pairs, McNemar exact:")
    undecided_adjacent = 0
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        wins_b, wins_a, p = mcnemar_exact(values[a], values[b])
        if p >= ALPHA:
            undecided_adjacent += 1
        if i < 10:
            d = 100 * (rate[a] - rate[b])
            print(f"  {i+1:>3} vs {i+2:<3} diff {d:+5.2f}pp  discordant {wins_a}/{wins_b}  "
                  f"p={p:.3f}  {'SEPARATED' if p < ALPHA else 'not separated'}")
    print(f"\n  adjacent pairs not separated at 5%: "
          f"{undecided_adjacent}/{len(order) - 1}")

    # --- all pairs ------------------------------------------------------
    total = sep = 0
    for a, b in combinations(order, 2):
        _, _, p = mcnemar_exact(values[a], values[b])
        total += 1
        sep += p < ALPHA
    print(f"  all pairs separated at 5%: {sep}/{total} ({100 * sep / total:.1f}%)")

    # --- how far apart must two systems be? ------------------------------
    print("\nminimum gap that actually separates, measured over real pairs:")
    gaps_sep, gaps_not = [], []
    for a, b in combinations(order, 2):
        _, _, p = mcnemar_exact(values[a], values[b])
        gap = abs(rate[a] - rate[b]) * 100
        (gaps_sep if p < ALPHA else gaps_not).append(gap)
    gaps_sep, gaps_not = np.array(gaps_sep), np.array(gaps_not)
    print(f"  largest gap that was NOT separated : {gaps_not.max():5.2f} pp")
    print(f"  smallest gap that WAS separated    : {gaps_sep.min():5.2f} pp")
    for q in (0.5, 0.9, 0.95):
        print(f"  {int(q*100)}% of pairs below {np.quantile(gaps_not, q):5.2f} pp are undecided")

    # --- rank stability under a paired instance bootstrap ---------------
    print("\nrank interval under a paired bootstrap over instances (2000 draws):")
    arr = np.vstack([values[s] for s in order]).astype(float)
    rng = np.random.default_rng(0)
    counts = rng.multinomial(n_inst, np.full(n_inst, 1 / n_inst), size=2000)
    boot = arr @ counts.T / n_inst                     # systems x draws
    ranks = np.empty_like(boot, dtype=np.int64)
    for i in range(boot.shape[0]):
        ranks[i] = 1 + (boot > boot[i]).sum(axis=0)
    lo = np.quantile(ranks, 0.025, axis=1, method="inverted_cdf")
    hi = np.quantile(ranks, 0.975, axis=1, method="inverted_cdf")
    for i, s in enumerate(order[:10]):
        print(f"  shown #{i+1:<3} occupies ranks {int(lo[i]):>3}-{int(hi[i]):<3}  {s[:46]}")
    width = hi - lo
    print(f"\n  median rank interval width: {np.median(width):.0f} places "
          f"of {len(order)}")
    print(f"  systems whose interval spans >= 10 places: "
          f"{int((width >= 10).sum())}/{len(order)}")


if __name__ == "__main__":
    main()
