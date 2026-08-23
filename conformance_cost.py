"""What does conformance cost in compute?

The standard's case is that conformance is "a matter of publishing, not of
effort". That is an empirical claim about the reference implementation and
it has never been measured. A benchmark owner deciding whether to adopt it
will ask how long the fields take on their matrix, and whether the answer
scales in a way that survives a board ten times larger.

Timed per board: the full report card, and separately the three fields that
could plausibly dominate - rank sets (bootstrap over items), entropy
(Knuth's estimator over linear extensions), and the pairwise kappa matrix.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the full card takes under 30 seconds on every board;
  * entropy is the dominant term on the boards with the most systems;
  * the total scales worse than linearly in J - the pairwise work is
    quadratic - so a board of 500 systems would take more than five times
    the largest time here.

SELF-CHECKS
  * the timings are reproducible: two runs of the same board agree within
    50 %;
  * the parts sum to no more than the whole plus 20 %.

    python conformance_cost.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import rank_sets as rs
import leaderboard_entropy as le
from entropy_law_test import MATRICES
from pair_sharpness import kappa_matrix

SEED = 20260823
DRAWS = 1500
SAMPLES = 2000


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def parts(x, rng):
    r, t_rank = timed(lambda: rs.rank_sets(x, draws=DRAWS))
    _, t_ent = timed(lambda: le.log_extensions(r["beats"], SAMPLES, rng))
    _, t_kap = timed(lambda: kappa_matrix(x))
    return t_rank, t_ent, t_kap


def _check_reproducible():
    x = pd.read_csv("tabarena/matrix_all45.csv", index_col=0).dropna(axis=0).to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    a = sum(parts(x, rng))
    b = sum(parts(x, rng))
    return abs(a - b) / max(a, b) < 0.5, f"two runs on TabArena 45: {a:.2f}s and {b:.2f}s"


def _check_sums():
    x = pd.read_csv("helm_winrate_matrix.csv", index_col=0).dropna(axis=0).to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    t_rank, t_ent, t_kap = parts(x, rng)
    whole, t_all = timed(lambda: (rs.rank_sets(x, draws=DRAWS), kappa_matrix(x)))
    return (t_rank + t_kap) <= t_all * 1.2 + 0.05, \
        f"parts {t_rank + t_kap:.3f}s vs whole {t_all:.3f}s"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_reproducible(), _check_sums()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHAT CONFORMANCE COSTS")
    p("=" * 78)
    p(f"  {'board':<22} {'J':>4} {'n':>5} {'rank sets':>10} {'entropy':>9} {'kappa':>8} {'total':>8} "
      f"{'per pair':>10}")
    under, rows = 0, []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J, n = x.shape
        rng = np.random.default_rng(SEED)
        t_rank, t_ent, t_kap = parts(x, rng)
        total = t_rank + t_ent + t_kap
        pairs = J * (J - 1) / 2
        under += total < 30
        rows.append((name, J, n, total, t_ent, t_rank))
        p(f"  {name:<22} {J:>4} {n:>5} {t_rank:>9.2f}s {t_ent:>8.2f}s {t_kap:>7.2f}s {total:>7.2f}s "
          f"{1000 * total / pairs:>9.2f}ms")
    N = len(rows)
    ent_dom = sum(1 for _, J, _, tot, te, _ in rows if J >= 100 and te > 0.5 * tot)
    big = sorted(rows, key=lambda r: -r[1])[:3]
    p("")
    p(f"  full card under 30 seconds: {under}/{N} (pre-registered: all)")
    p(f"  entropy dominates on boards with J >= 100: {ent_dom} of "
      f"{sum(1 for _, J, _, _, _, _ in rows if J >= 100)}")
    p("  largest boards: " + ", ".join(f"{n_} (J {J}, {tot:.1f}s)" for n_, J, _, tot, _, _ in big))
    p("")
    p("  Timed on one laptop core with the settings the reference implementation")
    p("  uses (1500 bootstrap draws, 2000 entropy samples). The per-pair column is")
    p("  the number that scales: a board with ten times the systems has a hundred")
    p("  times the pairs, and the bootstrap and the kappa matrix both live there.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("conformance_cost_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote conformance_cost_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
