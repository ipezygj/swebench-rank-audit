"""Run the Verified analysis on every SWE-bench split that publishes per-instance
outcomes, and score the result against PREREGISTERED.md.

The prediction under test: the undecided band is sampling error on a finite
instance set, so the largest-undecided-gap should scale as 1/sqrt(n).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from swebench_rank_noise import ALPHA, mcnemar_exact, wilson

HERE = Path(__file__).parent

# Where the union of instance ids appearing in the submissions undercounts the
# benchmark. SWE-bench Multimodal has 517 instances; only 248 distinct ids appear
# across the 12 submissions that publish lists, because the rest were solved by
# nobody and `generated` is empty in these files. Using the union as the
# denominator halved every rate and broke parity on all 12 systems. One number,
# checked against 12 independent published rates -- not fitted per system.
KNOWN_INSTANCE_COUNT = {"multimodal": 517}
EXP = HERE / "swebench_exp" / "evaluation"
BOARD = HERE / "lb_swebench.json"

# From PREREGISTERED.md, anchored on Verified (n=500 -> 4.60 pp).
PREDICTIONS = {
    "lite": (5.94, 4.5, 7.5),
    "test": (2.15, 1.5, 3.0),
    "multimodal": (4.52, 3.4, 5.9),
}


def build_matrix(split: str) -> tuple[pd.DataFrame, int]:
    """Matrix plus the number of submissions dropped for lacking per-instance data.

    Some submissions publish `resolved` as a COUNT rather than a list of instance
    ids (all 10 such cases are in multimodal). A count cannot be paired against
    another system, so those submissions are dropped and counted, never coerced.
    """
    rows, universe, dropped = {}, set(), 0
    for path in sorted((EXP / split).glob("*/results/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        resolved = data.get("resolved")
        if not isinstance(resolved, list):
            dropped += 1
            continue
        rows[path.parents[1].name] = set(resolved)
        universe |= set(resolved) | set(data.get("generated") or [])
    instances = sorted(universe)
    matrix = pd.DataFrame(
        {s: [1 if i in v else 0 for i in instances] for s, v in rows.items()},
        index=instances,
    ).T
    known = KNOWN_INSTANCE_COUNT.get(split)
    if known and known > matrix.shape[1]:
        # Pad with instances nobody resolved. They are concordant zeros, so they
        # change no McNemar result, but they restore the correct denominator and
        # therefore the correct percentage-point gaps.
        pad = known - matrix.shape[1]
        for j in range(pad):
            matrix[f"__unsolved_{j}"] = 0
    return matrix, dropped


def parity(split: str, matrix: pd.DataFrame) -> tuple[int, int, float]:
    """Recomputed rates vs the published board.

    The board publishes percentages rounded to two decimals, so agreement can
    only ever be exact to 0.005. An earlier version of this check used a 0.001
    tolerance and reported 55 "mismatches" on Lite where every single value in
    fact agreed -- k/300 has a repeating decimal that the board rounds and this
    does not. The gate is therefore 0.006, and the MAXIMUM observed deviation is
    returned so a loosened tolerance cannot hide a real disagreement: if that
    number is at or below 0.005 the two sides agree exactly.

    Returns (compared, mismatches, max_abs_deviation).
    """
    boards = json.loads(BOARD.read_text(encoding="utf-8"))["leaderboards"]
    wanted = {"lite": "Lite", "test": "Test", "multimodal": "Multimodal",
              "verified": "Verified"}[split]
    entries = next(b for b in boards if b["name"] == wanted)["results"]
    n = matrix.shape[1]
    compared = mismatches = 0
    worst = 0.0
    for entry in entries:
        folder, published = entry.get("folder"), entry.get("resolved")
        if folder not in matrix.index or published is None:
            continue
        compared += 1
        deviation = abs(100 * matrix.loc[folder].sum() / n - float(published))
        worst = max(worst, deviation)
        if deviation > 0.006:
            mismatches += 1
    return compared, mismatches, worst


def analyse(split: str) -> dict:
    matrix, dropped = build_matrix(split)
    n_sys, n_inst = matrix.shape
    compared, mismatches, worst = parity(split, matrix)

    rate = matrix.mean(axis=1).sort_values(ascending=False)
    order = list(rate.index)
    values = {s: matrix.loc[s].to_numpy() for s in order}

    undecided_adj = sum(
        mcnemar_exact(values[order[i]], values[order[i + 1]])[2] >= ALPHA
        for i in range(len(order) - 1)
    )
    gaps_not, sep, total = [], 0, 0
    for a, b in combinations(order, 2):
        p = mcnemar_exact(values[a], values[b])[2]
        total += 1
        if p < ALPHA:
            sep += 1
        else:
            gaps_not.append(abs(rate[a] - rate[b]) * 100)

    tied_with_top = sum(
        mcnemar_exact(values[order[0]], values[s])[2] >= ALPHA for s in order
    )
    return {
        "split": split, "systems": n_sys, "instances": n_inst,
        "dropped_no_instance_list": dropped,
        "parity_compared": compared, "parity_mismatches": mismatches,
        "parity_max_deviation_pp": round(float(worst), 5),
        "adjacent_undecided": int(undecided_adj), "adjacent_total": len(order) - 1,
        "pairs_separated": int(sep), "pairs_total": int(total),
        "largest_undecided_gap": float(max(gaps_not)) if gaps_not else float("nan"),
        "tied_with_top": int(tied_with_top),
        "top_rate": float(100 * rate.iloc[0]),
    }


def main() -> None:
    # Verified is the anchor; recomputed here so every row comes from one code path.
    results = [analyse(s) for s in ("verified", "lite", "test", "multimodal")]

    print(f"{'split':12s} {'sys':>4} {'inst':>5} {'parity':>24} "
          f"{'adj undecided':>14} {'pairs sep':>12} {'max undecided gap':>18} {'tied w/#1':>10}")
    for r in results:
        par = f"{r['parity_compared']}/{r['parity_mismatches']}mm(max{r['parity_max_deviation_pp']:.4f})"
        adj = f"{r['adjacent_undecided']}/{r['adjacent_total']}"
        pr = f"{100*r['pairs_separated']/r['pairs_total']:.1f}%"
        drop = f" (-{r['dropped_no_instance_list']} no per-instance data)" if r['dropped_no_instance_list'] else ""
        print(f"{r['split']:12s} {r['systems']:>4} {r['instances']:>5} {par:>24} "
              f"{adj:>14} {pr:>12} {r['largest_undecided_gap']:>16.2f}pp "
              f"{r['tied_with_top']:>10}{drop}")

    print("\n--- scored against PREREGISTERED.md ---")
    anchor = next(r for r in results if r["split"] == "verified")
    print(f"anchor: verified n={anchor['instances']} gap={anchor['largest_undecided_gap']:.2f}pp")
    hits = 0
    for r in results:
        if r["split"] not in PREDICTIONS:
            continue
        pred, lo, hi = PREDICTIONS[r["split"]]
        got = r["largest_undecided_gap"]
        scaled = anchor["largest_undecided_gap"] * np.sqrt(
            anchor["instances"] / r["instances"])
        inside = lo <= got <= hi
        hits += inside
        print(f"  {r['split']:11s} n={r['instances']:>5}  predicted {pred:5.2f}pp "
              f"[{lo}-{hi}]  actual {got:5.2f}pp  "
              f"{'HIT' if inside else 'MISS'}   (1/sqrt(n) from anchor: {scaled:.2f})")
    print(f"\n  {hits}/{len(PREDICTIONS)} predictions inside their pre-registered envelope")

    # Emit the DERIVED totals too, not just the per-split rows. The pair count was
    # first written into prose as 11 807 when the parts sum to 12 739: every component
    # had been computed and saved, only the sum was done by hand. A total that no file
    # contains is a total nobody can check.
    totals = {
        "pairs_total_all_splits": sum(r["pairs_total"] for r in results),
        "systems_total_all_splits": sum(r["systems"] for r in results),
        "adjacent_undecided_all_splits": sum(r["adjacent_undecided"] for r in results),
        "adjacent_total_all_splits": sum(r["adjacent_total"] for r in results),
        "pairs_separated_all_splits": sum(r["pairs_separated"] for r in results),
    }
    print("")
    print("derived totals (written to JSON so the write-up can cite a source):")
    for k, v in totals.items():
        print(f"  {k:34s} {v:>7,}")

    with open(HERE / "all_splits_results.json", "w", encoding="utf-8") as fh:
        json.dump({"splits": results, "totals": totals}, fh, indent=1)


if __name__ == "__main__":
    main()
