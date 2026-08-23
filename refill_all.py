"""How many new items would each leaderboard need to settle its top pair?

refill_prescription.py answered this for SWE-bench: about 13 600 instances at
the right difficulty, twenty-seven times the benchmark, to separate rank 1
from rank 3 with 80 % power. Was that a property of SWE-bench or of
leaderboards? This runs the same calculation on every matrix in the
validation set, in one table, with the same resolution law:

    n_required = d * ((z_alpha + z_beta) / delta)^2

where delta is the score gap between the two systems and d their discordance
rate. For continuous matrices, delta and d are taken from the observed
per-item differences: d becomes the variance-equivalent E[(x_a - x_b)^2]
which reduces to the discordant fraction on binary data.

The top pair is rank 1 vs rank 2 when they differ, else rank 1 vs the first
system with a lower score - on SWE-bench ranks 1 and 2 are an exact tie and
no number of items separates identical scores.

SELF-CHECKS
  * on binary data the variance route must equal the discordance route;
  * a pair ten points apart on 500 items must need fewer items than the
    benchmark already has;
  * an identical pair must return infinity.

    python refill_all.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Z_A = 1.959963984540054
Z_B = 0.8416212335729143

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


def required_items(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(delta, d, n_required) for separating a from b at 80 % power, 5 %."""
    diff = a - b
    delta = float(diff.mean())
    d = float((diff ** 2).mean())          # = discordant fraction when binary
    if delta <= 0:
        return delta, d, float("inf")
    return delta, d, d * ((Z_A + Z_B) / delta) ** 2


def _check_binary_equivalence() -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    a = (rng.random(400) < 0.6).astype(float)
    b = (rng.random(400) < 0.5).astype(float)
    _, d_var, _ = required_items(a, b)
    d_disc = float(np.mean(a != b))
    ok = abs(d_var - d_disc) < 1e-12
    return ok, f"binary: variance route {d_var:.4f} == discordance {d_disc:.4f}"


def _check_wide_gap_small_n() -> tuple[bool, str]:
    rng = np.random.default_rng(2)
    b = (rng.random(500) < 0.6).astype(float)
    a = b.copy()
    a[np.flatnonzero(a == 0)[:50]] = 1.0      # +10 points
    _, _, n_req = required_items(a, b)
    return n_req < 500, f"ten-point gap on 500 items needs {n_req:.0f} items"


def _check_identical_inf() -> tuple[bool, str]:
    a = np.array([1, 0, 1, 0.0])
    _, _, n_req = required_items(a, a.copy())
    return math.isinf(n_req), f"identical pair needs {n_req}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_binary_equivalence(), _check_wide_gap_small_n(),
                        _check_identical_inf()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print("\nA CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("HOW MANY NEW ITEMS WOULD SETTLE THE TOP PAIR? EVERY BENCHMARK, ONE LAW")
    p("=" * 78)
    p(f"  {'leaderboard':<22} {'n':>5} {'pair':>9} {'gap':>7} {'discord':>8}"
      f" {'items needed':>13} {'x benchmark':>12}")
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        df = pd.read_csv(path, index_col=0).dropna(axis=0)
        x = df.to_numpy(dtype=float)
        n = x.shape[1]
        sc = x.mean(axis=1)
        order = np.argsort(-sc, kind="stable")
        lead = order[0]
        # first lower-scoring system
        chal = next((k for k in order[1:] if sc[k] < sc[lead] - 1e-12), None)
        pair = "1 vs 2"
        if chal is None:
            p(f"  {name:<22} {n:>5} {'-':>9}  all tied")
            continue
        rank_c = int(np.sum(sc > sc[chal])) + 1
        pair = f"1 vs {rank_c}"
        delta, d, n_req = required_items(x[lead], x[chal])
        mult = n_req / n
        p(f"  {name:<22} {n:>5} {pair:>9} {100 * delta:>+6.1f}% {d:>8.3f}"
          f" {n_req:>13,.0f} {mult:>11.1f}x")
    p("")
    p("  'items needed' is for 80 % power at 5 %, two-sided, to separate the")
    p("  leader from the first system below it, at the difficulty those two")
    p("  already disagree at. 'x benchmark' is that number over the items the")
    p("  benchmark already has: the factor by which it would have to grow to")
    p("  settle its own top. A benchmark that needs more than itself to decide")
    p("  its own first place is printing a coin flip with a rank number on it.")
    p("")
    p("  Where the pair is an exact tie no number of items helps, and that is")
    p("  reported as such rather than as a large number.")
    text = "\n".join(L)
    print("\n" + text)
    Path("refill_all_results.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    print("\nwrote refill_all_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
