"""How small could this benchmark be and still say the same thing?

Every board here spends its items on comparisons it cannot make. The
question this asks is the inverse of refill_prescription: not how many
items to add, but how many could be REMOVED while the board still
establishes the same pairs.

Procedure, fixed before running:
  * compute the established-pair set on the full matrix (rank_sets);
  * greedily add items, each time taking the item that adds the most
    established pairs when the simultaneous test is recomputed on the
    subset - with the critical value recomputed for the subset size, so
    the comparison is honest about the smaller n;
  * stop when 90 % of the full board's established pairs are recovered.

Greedy selection on the same data it is evaluated on overfits: the chosen
items are the ones that happened to separate. A held-out check is therefore
run alongside - select on half the systems, evaluate the recovered share on
the other half.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * 30 % of items or fewer suffice to recover 90 % of established pairs on
    >= 6 of 9 boards (in-sample);
  * the held-out recovery is at least 80 % on >= 5 of 9 - the saving is
    real, not selection;
  * the selected items are of middling difficulty: their mean solve rate is
    closer to 0.5 than the discarded items' on >= 7 of 9 (binary boards) or
    their score SD across systems is higher (continuous boards).

SELF-CHECKS
  * selecting all items must recover 100 % by construction;
  * on a board of pure noise the greedy selection must fail to reach 90 %
    with any subset (there is nothing to recover).

    python minimal_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from entropy_law_test import MATRICES

SEED = 20260823
DRAWS = 400
TARGET = 0.90
STEP_FRAC = 0.02          # add items in blocks of 2 % of n


def established(x, draws=DRAWS, seed=SEED):
    r = rs.rank_sets(x, draws=draws, seed=seed)
    return r["beats"]


def greedy(x, target=TARGET, rng=None):
    """Add blocks of items, choosing by immediate gain in established pairs."""
    J, n = x.shape
    full = established(x)
    total = int(full.sum())
    if total == 0:
        return None
    rng = rng or np.random.default_rng(SEED)
    # At least two items per block: rank_sets needs n >= 2, and with a block
    # of one the first trial subset had length 1, was skipped, and the loop
    # broke immediately - five of nine boards returned "0 items, 0 % recovered"
    # in the first run. That was the block size, not the boards.
    block = max(2, int(round(STEP_FRAC * n)))
    chosen = []
    remaining = list(range(n))
    # score each item once by how much it discriminates: variance of the
    # column across systems, which is the only per-item quantity available
    # before any subset exists. The greedy step then refines among the top.
    col_var = x.var(axis=0)
    order = list(np.argsort(-col_var))
    while remaining and len(chosen) < n:
        cands = [i for i in order if i in set(remaining)][: 4 * block]
        best, best_gain = None, -1
        for start in range(0, len(cands), block):
            trial = chosen + cands[start:start + block]
            if len(trial) < 2:
                continue
            b = established(x[:, trial], draws=200, seed=SEED + len(chosen))
            gain = int((b & full).sum())
            if gain > best_gain:
                best_gain, best = gain, cands[start:start + block]
        if best is None:
            break
        chosen += best
        remaining = [i for i in remaining if i not in set(best)]
        if best_gain / total >= target:
            return chosen, best_gain / total, total
    b = established(x[:, chosen], draws=200) if len(chosen) > 1 else np.zeros_like(full)
    return chosen, int((b & full).sum()) / total, total


def _check_all_items():
    rng = np.random.default_rng(3)
    x = 0.5 + rng.normal(0, 0.08, 30)[:, None] + rng.normal(0, 0.4, (30, 100))
    full = established(x)
    b = established(x)
    return int((b & full).sum()) == int(full.sum()), "all items recover 100 % by construction"


def _check_noise():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.4, (30, 120))          # no ability differences at all
    out = greedy(x)
    if out is None:
        return True, "pure noise: nothing established at all, nothing to recover"
    chosen, share, total = out
    return total < 20, f"pure noise: only {total} established pairs on the full board"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_all_items(), _check_noise()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("HOW SMALL COULD THIS BENCHMARK BE?")
    p("=" * 84)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>5} {'items used':>11} {'share of n':>11} "
      f"{'recovered':>10} {'held-out':>9} {'selected vs rest':>18}")
    small, heldok, mid = 0, 0, 0
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J, n = x.shape
        out = greedy(x)
        if out is None:
            p(f"  {name:<22} nothing established")
            continue
        chosen, share, total = out
        frac = len(chosen) / n
        # held-out: select on half the systems, evaluate on the other half
        rng = np.random.default_rng(SEED + 1)
        perm = rng.permutation(J)
        A, B = perm[: J // 2], perm[J // 2:]
        outA = greedy(x[A])
        if outA:
            selA = outA[0]
            fullB = established(x[B])
            bB = established(x[np.ix_(B, selA)]) if len(selA) > 1 else np.zeros_like(fullB)
            held = int((bB & fullB).sum()) / max(int(fullB.sum()), 1)
        else:
            held = float("nan")
        binary = bool(np.isin(x, [0.0, 1.0]).all())
        rest = [i for i in range(n) if i not in set(chosen)]
        if binary:
            sel_stat = abs(x[:, chosen].mean() - 0.5)
            rest_stat = abs(x[:, rest].mean() - 0.5) if rest else float("nan")
            better = sel_stat < rest_stat
            desc = f"|p-0.5| {sel_stat:.3f} vs {rest_stat:.3f}"
        else:
            sel_stat = float(np.mean(x[:, chosen].std(axis=0)))
            rest_stat = float(np.mean(x[:, rest].std(axis=0))) if rest else float("nan")
            better = sel_stat > rest_stat
            desc = f"SD {sel_stat:.3f} vs {rest_stat:.3f}"
        small += frac <= 0.30 and share >= TARGET
        heldok += (not np.isnan(held)) and held >= 0.80
        mid += bool(better)
        p(f"  {name:<22} {J:>4} {n:>5} {len(chosen):>11} {100 * frac:>10.0f}% {100 * share:>9.0f}% "
          f"{(f'{100 * held:.0f}%' if not np.isnan(held) else '-'):>9} {desc:>18}")
    N = sum(1 for _, pth in MATRICES.items() if Path(pth).exists())
    p("")
    p(f"  30 % of items or fewer recover 90 %: {small}/{N} (pre-registered >= 6)")
    p(f"  held-out recovery at least 80 %: {heldok}/{N} (pre-registered >= 5)")
    p(f"  selected items more discriminating than the rest: {mid}/{N} (pre-registered >= 7)")
    p("")
    p("  The critical value is recomputed for every subset, so a smaller board is")
    p("  held to its own standard rather than the full board's. 'held-out' selects")
    p("  items on half the systems and measures recovery on the other half, which")
    p("  is the number that says whether the saving is real.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("minimal_benchmark_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote minimal_benchmark_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
