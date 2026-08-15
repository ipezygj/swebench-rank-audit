"""Build the SWE-bench Verified system x instance outcome matrix.

Every submission is evaluated on the SAME 500 instances, and each publishes the
list of instance IDs it resolved. That makes the comparison between two systems
genuinely paired: the same problems, the same order, so the noise that makes a
hard instance hard cancels instead of accumulating.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent / "swebench_exp" / "evaluation" / "verified"


def main() -> None:
    rows: dict[str, set[str]] = {}
    universe: set[str] = set()
    skipped: list[tuple[str, str]] = []

    for results in sorted(ROOT.glob("*/results/results.json")):
        system = results.parents[1].name
        data = json.loads(results.read_text(encoding="utf-8"))
        resolved = data.get("resolved")
        generated = data.get("generated") or []
        if resolved is None:
            skipped.append((system, "no 'resolved' key"))
            continue
        rows[system] = set(resolved)
        universe |= set(generated) | set(resolved)

    instances = sorted(universe)
    print(f"submissions: {len(rows)}   union of instance ids: {len(instances)}")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")

    # A system's non-membership in `resolved` only means "not resolved" for
    # instances it was actually run on. Report the spread rather than assume.
    sizes = {s: len(v) for s, v in rows.items()}
    print(f"resolved counts: min {min(sizes.values())}, max {max(sizes.values())}")

    matrix = pd.DataFrame(
        {s: [1 if i in v else 0 for i in instances] for s, v in rows.items()},
        index=instances,
    ).T
    matrix.to_csv(Path(__file__).parent / "swebench_verified_matrix.csv")
    print(f"matrix: {matrix.shape[0]} systems x {matrix.shape[1]} instances")

    rate = matrix.mean(axis=1).sort_values(ascending=False)
    print("\ntop 10 by resolve rate:")
    for name, value in rate.head(10).items():
        print(f"  {value * 100:5.2f}%  {name}")
    print("\nhardest instances (resolved by fewest systems):")
    per_instance = matrix.mean(axis=0).sort_values()
    print(f"  {int((per_instance == 0).sum())} instances resolved by NO system")
    print(f"  {int((per_instance == 1).sum())} instances resolved by EVERY system")


if __name__ == "__main__":
    main()
