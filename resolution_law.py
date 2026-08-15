"""How many instances does a benchmark need before its ranking means anything?

Derivation, done before looking at whether it fits.

Two systems are compared on n paired instances. Let
    b = instances only A solves, c = instances only B solves,
    D = b + c discordant, d = D / n the discordance rate,
    delta = (b - c) / n the difference in resolve rate (what a leaderboard shows).

McNemar tests b/D against 1/2. Writing p = b/D:

    b = n(d + delta)/2,  c = n(d - delta)/2,  so  p = (d + delta) / (2d)
    p - 1/2 = delta / (2d)

A binomial test on D trials detects a shift from 1/2 at significance alpha and
power 1-beta when

    |p - 1/2| >= (z_{alpha/2} + z_beta) * 0.5 / sqrt(D)

Substituting p - 1/2 = delta/(2d) and D = dn:

    delta >= (z_{alpha/2} + z_beta) * sqrt(d / n)                        (*)

So the smallest resolvable gap scales as sqrt(d/n), and the constant is 2.80 at
alpha = 0.05, power 0.8. Two consequences worth stating separately:

  - resolution improves only as sqrt(n): to halve the gap you can resolve, you
    need four times the instances;
  - it degrades as sqrt(d): the more two systems disagree instance by instance,
    the *harder* they are to separate at a given headline gap. Systems that
    differ in style rather than strength are the expensive case.

`d` is not a modelling choice. It is measured from the published per-instance
outcomes, and nobody reports it.

This script checks (*) against four SWE-bench splits that were measured before
the formula existed. The formula is derived, not fitted: no constant in it was
chosen by looking at the data.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from all_splits import build_matrix
from swebench_rank_noise import ALPHA, mcnemar_exact

HERE = Path(__file__).parent
POWER = 0.80
CONST = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)  # 2.80


def mdd(discordance: float, n: int) -> float:
    """Minimum detectable difference, in rate units, from equation (*)."""
    return CONST * np.sqrt(discordance / n)


def split_stats(split: str) -> dict:
    matrix, _ = build_matrix(split)
    n = matrix.shape[1]
    rate = matrix.mean(axis=1).sort_values(ascending=False)
    order = list(rate.index)
    values = {s: matrix.loc[s].to_numpy() for s in order}

    disc, decided, undecided = [], [], []
    for a, b in combinations(order, 2):
        va, vb = values[a], values[b]
        d = int(np.sum(va != vb))
        disc.append(d / n)
        gap = abs(rate[a] - rate[b])
        (decided if mcnemar_exact(va, vb)[2] < ALPHA else undecided).append(gap)

    median_d = float(np.median(disc))
    return {
        "split": split, "n": n, "systems": len(order),
        "median_discordance": median_d,
        "predicted_mdd_pp": 100 * mdd(median_d, n),
        "observed_largest_undecided_pp": 100 * max(undecided) if undecided else float("nan"),
        "observed_smallest_decided_pp": 100 * min(decided) if decided else float("nan"),
    }


def main() -> None:
    print(f"constant in (*) at alpha={ALPHA}, power={POWER}: {CONST:.3f}\n")
    rows = [split_stats(s) for s in ("verified", "lite", "test", "multimodal")]

    print(f"{'split':12s} {'n':>5} {'sys':>4} {'median d':>9} {'predicted MDD':>14} "
          f"{'observed decision boundary':>28}")
    inside = 0
    for r in rows:
        lo = r["observed_smallest_decided_pp"]
        hi = r["observed_largest_undecided_pp"]
        band = f"{lo:.2f} - {hi:.2f} pp"
        ok = lo <= r["predicted_mdd_pp"] <= hi
        inside += ok
        print(f"{r['split']:12s} {r['n']:>5} {r['systems']:>4} "
              f"{r['median_discordance']:>9.3f} {r['predicted_mdd_pp']:>12.2f}pp "
              f"{band:>26}  {'INSIDE' if ok else 'outside'}")

    print(f"\n  predicted MDD falls inside the observed decision boundary "
          f"in {inside}/{len(rows)} splits")

    print("\n--- design table: instances needed to resolve a given gap ---")
    print("(using each split's own measured discordance rate)")
    targets = [0.005, 0.01, 0.02, 0.03]
    header = "  " + "target gap".ljust(12) + "".join(f"{r['split']:>13}" for r in rows)
    print(header)
    for t in targets:
        cells = ""
        for r in rows:
            need = r["median_discordance"] * (CONST / t) ** 2
            cells += f"{need:>13,.0f}"
        print(f"  {100*t:>4.1f} pp     {cells}")

    run_per_pair()

    with open(HERE / "resolution_law_results.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)



def per_pair_test(split: str) -> dict:
    """Sharper test: predict each pair's outcome from its OWN discordance.

    The aggregate check used the median discordance over all pairs, which is the
    wrong quantity -- the pairs that sit near the decision boundary are similar
    systems, and similar systems disagree on fewer instances than distant ones.
    Here every pair is predicted from its own (gap, d, n) with no free parameter,
    and scored against what McNemar actually said.
    """
    matrix, _ = build_matrix(split)
    n = matrix.shape[1]
    rate = matrix.mean(axis=1).sort_values(ascending=False)
    order = list(rate.index)
    values = {s: matrix.loc[s].to_numpy() for s in order}

    tp = tn = fp = fn = 0
    d_close, d_far = [], []
    for a, b in combinations(order, 2):
        va, vb = values[a], values[b]
        d = float(np.sum(va != vb)) / n
        gap = abs(rate[a] - rate[b])
        predicted_decided = gap >= mdd(d, n)
        actually_decided = mcnemar_exact(va, vb)[2] < ALPHA
        if predicted_decided and actually_decided:
            tp += 1
        elif not predicted_decided and not actually_decided:
            tn += 1
        elif predicted_decided and not actually_decided:
            fp += 1
        else:
            fn += 1
        (d_close if gap < 0.02 else d_far).append(d)
    total = tp + tn + fp + fn
    return {
        "split": split, "n": n, "agreement": (tp + tn) / total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "median_d_close_pairs": float(np.median(d_close)) if d_close else float("nan"),
        "median_d_far_pairs": float(np.median(d_far)) if d_far else float("nan"),
    }


def run_per_pair() -> None:
    print("\n--- per-pair test of (*), no free parameters ---")
    print(f"{'split':12s} {'agreement':>10} {'TP':>6} {'TN':>6} {'FP':>5} {'FN':>5} "
          f"{'median d close':>15} {'median d far':>13}")
    for split in ("verified", "lite", "test", "multimodal"):
        r = per_pair_test(split)
        print(f"{r['split']:12s} {100*r['agreement']:>9.1f}% {r['tp']:>6} {r['tn']:>6} "
              f"{r['fp']:>5} {r['fn']:>5} {r['median_d_close_pairs']:>15.3f} "
              f"{r['median_d_far_pairs']:>13.3f}")

if __name__ == "__main__":
    main()
