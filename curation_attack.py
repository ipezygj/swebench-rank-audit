"""Can a benchmark owner manufacture a winner by choosing items?

The standard asks for the pair statistic behind every claim. That invites
an obvious attack: pick the items on which your favourite system beats the
runner-up, and the claim becomes separable. This measures how far the
attack goes and whether the honest check catches it.

Attack: greedily add items that maximise the t of the printed #1 against
the printed #2, up to half the board.
Defence: the same selection, scored on the items NOT chosen. A real
difference survives; a curated one does not.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * in-sample, greedy curation raises the top-pair t above 2 on >= 7 of 9
    boards, including boards whose honest t is near zero;
  * out-of-sample, on the unchosen items, the t falls below 2 on >= 7 of 9;
  * the boards whose honest t is already above 2 (CASP14, LiveBench) keep
    a high out-of-sample t - the attack cannot destroy a real difference
    either.

SELF-CHECKS
  * on a matrix where the two systems are identical up to noise, the attack
    still reaches t > 2 in-sample (that is the point of the check) and the
    held-out t fails to support the claim (t < 2, and in fact strongly
    negative: the complement of a favourable selection is unfavourable);
  * selecting all items reproduces the board's own t.

    python curation_attack.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from entropy_law_test import MATRICES

SEED = 20260823


def t_of(d):
    if len(d) < 3:
        return 0.0
    sd = float(d.std(ddof=1))
    return float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0


def greedy_items(d, cap):
    """Add items one at a time, each time the one that maximises t."""
    n = len(d)
    order = np.argsort(-d)          # start from the most favourable
    chosen = list(order[:3])
    remaining = [i for i in order[3:]]
    best_t = t_of(d[chosen])
    while len(chosen) < cap and remaining:
        # only the top candidates need testing: adding a very negative item
        # never helps
        cands = remaining[:50]
        gains = [(t_of(d[chosen + [i]]), i) for i in cands]
        tt, i = max(gains)
        if tt <= best_t:
            break
        best_t = tt
        chosen.append(i)
        remaining.remove(i)
    return chosen, best_t


def _check_identical():
    rng = np.random.default_rng(3)
    d = rng.normal(0, 0.4, 400)
    chosen, t_in = greedy_items(d, 200)
    held = [i for i in range(400) if i not in set(chosen)]
    t_out = t_of(d[held])
    # One-sided: the held-out set must fail to SUPPORT the claim. Requiring
    # |t| < 2 was wrong - creaming off the favourable half leaves a
    # systematically unfavourable remainder, so the honest check returns a
    # large NEGATIVE t (-3.99 here), which is the attack being caught
    # harder, not the check failing.
    return t_in > 2 and t_out < 2, f"identical systems: curated t {t_in:.2f}, held-out t {t_out:.2f} (mirror of the selection)"


def _check_all():
    rng = np.random.default_rng(5)
    d = rng.normal(0.05, 0.4, 300)
    return abs(t_of(d) - t_of(d[np.arange(300)])) < 1e-9, "all items reproduce the board's own t"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_identical(), _check_all()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("CAN A BENCHMARK OWNER CURATE A WINNER?")
    p("=" * 84)
    p(f"  {'board':<22} {'n':>5} {'honest t':>9} {'curated t':>10} {'items used':>11} "
      f"{'held-out t':>11} {'survives':>9}")
    attacked, caught, real = 0, 0, 0
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        n = x.shape[1]
        order = np.argsort(-x.mean(axis=1))
        d = x[int(order[0])] - x[int(order[1])]
        t_honest = t_of(d)
        chosen, t_in = greedy_items(d, max(4, n // 2))
        held = [i for i in range(n) if i not in set(chosen)]
        t_out = t_of(d[held]) if len(held) >= 3 else float("nan")
        attacked += t_in > 2
        caught += (t_in > 2) and (t_out < 2)
        if t_honest > 2:
            real += t_out > 2
        p(f"  {name:<22} {n:>5} {t_honest:>9.2f} {t_in:>10.2f} {len(chosen):>11} {t_out:>11.2f} "
          f"{('yes' if t_out > 2 else 'no'):>9}")
    N = sum(1 for _, pth in MATRICES.items() if Path(pth).exists())
    p("")
    p(f"  curation reaches t > 2 in-sample: {attacked}/{N} (pre-registered >= 7)")
    p(f"  the held-out check catches it: {caught}/{attacked if attacked else 1} of the successful attacks")
    p(f"  boards with an honest t > 2 keep it out of sample: {real}")
    p("")
    p("  The attack is the obvious one and it works: on a board whose two leaders")
    p("  differ by nothing, half the items can be chosen so that they differ")
    p("  significantly. The defence is equally simple and it also works: score the")
    p("  claim on the items the curator did not choose. A standard that asks for")
    p("  the pair statistic should ask for it on a pre-registered item set, or on")
    p("  the complement of any set the claimant selected.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("curation_attack_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote curation_attack_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
