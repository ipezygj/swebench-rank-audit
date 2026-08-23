"""If you pick the leader, how much do you lose?

Everything so far is about what a leaderboard can prove. A reader wants
something narrower: I am going to use the system at the top; how much worse
is it than the best one available?

Out-of-sample regret. Split the items at random. Pick the leader on half A.
Measure, on half B, the gap between the best system on B and the system A
chose. That is what following the leaderboard costs, in the board's own
score units, on items the choice did not see. Repeated over 40 splits.

Reported beside it: the regret of picking at random from the top ten, and
from the whole board, so the number has a scale.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the median regret of following the leader is below 1 point on >= 6 of 9
    boards - the top is unresolvable but the top systems are close, which
    is the same fact seen from the reader's side;
  * following the leader beats picking randomly from the top ten on every
    board;
  * regret is largest where the crown is least stable (Spearman between
    crown stability and regret below -0.4).

SELF-CHECKS
  * with one system far above the rest, the regret is zero;
  * with all systems identical, the regret of the leader equals the regret
    of a random pick within noise.

    python selection_regret.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from entropy_law_test import MATRICES

SEED = 20260823
SPLITS = 40
CROWN = {"SWE-bench Verified": 32.8, "MTEB English v2": 44.8, "HELM classic": 66.4,
         "ProteinGym DMS": 86.0, "TabArena 16 models": 86.5, "TabArena 45 variants": 72.7,
         "CASP14": 100.0, "LiveBench": 99.8, "MathArena 2025": 41.3}


def regrets(x, rng, splits=SPLITS, topk=10):
    J, n = x.shape
    lead, rnd_top, rnd_all = [], [], []
    for _ in range(splits):
        perm = rng.permutation(n)
        A, B = perm[: n // 2], perm[n // 2:]
        sa, sb = x[:, A].mean(axis=1), x[:, B].mean(axis=1)
        best_b = sb.max()
        pick = int(np.argmax(sa))
        lead.append(best_b - sb[pick])
        top = np.argsort(-sa)[: min(topk, J)]
        rnd_top.append(best_b - sb[int(rng.choice(top))])
        rnd_all.append(best_b - sb[int(rng.integers(0, J))])
    return float(np.median(lead)), float(np.median(rnd_top)), float(np.median(rnd_all))


def _check_dominant():
    rng = np.random.default_rng(3)
    x = 0.4 + rng.normal(0, 0.02, 20)[:, None] + rng.normal(0, 0.2, (20, 200))
    x[0] += 0.5
    lead, _, _ = regrets(x, rng, 10)
    return lead < 0.01, f"one system far above: leader regret {lead:.4f}"


def _check_identical():
    rng = np.random.default_rng(5)
    x = 0.5 + rng.normal(0, 0.3, (20, 200))
    lead, _, rnd_all = regrets(x, rng, 20)
    return abs(lead - rnd_all) < 0.5 * max(rnd_all, 1e-9) + 0.02, \
        f"identical systems: leader {lead:.4f} vs random {rnd_all:.4f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_dominant(), _check_identical()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHAT FOLLOWING THE LEADERBOARD COSTS")
    p("=" * 84)
    p(f"  {'board':<22} {'follow the leader':>18} {'random from top 10':>19} {'random overall':>15} "
      f"{'crown':>7}")
    small, better, regs, crowns = 0, 0, [], []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        rng = np.random.default_rng(SEED)
        lead, rtop, rall = regrets(x, rng)
        small += (100 * lead) < 1.0
        better += lead <= rtop
        regs.append(lead); crowns.append(CROWN.get(name, float("nan")))
        p(f"  {name:<22} {100 * lead:>17.2f}p {100 * rtop:>18.2f}p {100 * rall:>14.2f}p "
          f"{CROWN.get(name, float('nan')):>6.0f}%")
    N = len(regs)
    r = spearmanr(crowns, regs)
    p("")
    p(f"  median regret below 1 point: {small}/{N} (pre-registered >= 6)")
    p(f"  following the leader beats a random top-ten pick: {better}/{N} (pre-registered: all)")
    p(f"  Spearman(crown stability, regret) = {r.statistic:+.2f} (p {r.pvalue:.2f}); "
      f"pre-registered below -0.4")
    p("")
    p("  Regret is measured on the half of the items the choice did not see, in the")
    p("  board's own units, as the gap to the best system on that half. It is the")
    p("  reader's version of every other number here: a board can be unable to")
    p("  prove who is first and still be worth following, if the systems it cannot")
    p("  separate are ones you would be equally happy with.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("selection_regret_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote selection_regret_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
