"""What does pair-specific resolution cost, or save, in items?

R10 asks a leaderboard to report the resolution of the PAIR it makes a claim
about. The practical question for a benchmark owner is what that changes in
the only currency they spend: items.

refill_all.py already prescribes from the pair's own difference vector,

    n_required = E[(x_a - x_b)^2] * ((z_alpha + z_beta) / delta)^2

The naive alternative - the one a board-wide "resolution" number invites -
replaces the pair's variance with the board's typical one, 2 * sigma_item^2.
Since E[(x_a-x_b)^2] = kappa^2 * (sd_a^2 + sd_b^2), the two prescriptions
differ by exactly the pair's kappa^2 (plus the difference between that
pair's own SDs and the board's typical SD).

Reported here for the pair every board actually argues about (#1 vs #2) and
for the median frontier advance on the five dated boards: items required
under the pair-specific rule, under the board-wide rule, and the ratio.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the ratio naive / pair-specific equals 1 / kappa^2 within 15 % on >= 8
    of 9 boards (an identity check on real data, not a discovery);
  * on every board the board-wide rule asks for MORE items than needed for
    the top pair (because kappa < 1 there), and the excess exceeds 20 % on
    >= 6 of 9;
  * the ordering of boards by items required is unchanged between the two
    rules (Spearman > 0.9).

SELF-CHECKS
  * on an iid field the two rules agree within 10 %;
  * doubling the gap at fixed noise must quarter the prescription.

    python prescription_pairwise.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

from entropy_law_test import MATRICES
from evidence_trajectory import load
from sota_audit import advances
from chase_model import BOARDS as DATED

Z_A, Z_B = norm.isf(0.025), norm.isf(0.20)     # two-sided 5 %, 80 % power


def pair_required(x, i, j):
    d = x[i] - x[j]
    delta = float(d.mean())
    if delta <= 0:
        return float("inf"), float("nan")
    var = float((d ** 2).mean())
    return var * ((Z_A + Z_B) / delta) ** 2, delta


def board_required(x, i, j):
    """Same power calculation with the board's typical pair variance."""
    xc = x - x.mean(axis=0, keepdims=True)
    sd = xc.std(axis=1, ddof=1)
    typical = 2.0 * float(np.median(sd)) ** 2
    d = x[i] - x[j]
    delta = float(d.mean())
    if delta <= 0:
        return float("inf")
    return typical * ((Z_A + Z_B) / delta) ** 2


def kappa_of(x, i, j):
    xc = x - x.mean(axis=0, keepdims=True)
    sd = xc.std(axis=1, ddof=1)
    den = math.sqrt(sd[i] ** 2 + sd[j] ** 2)
    return float((xc[i] - xc[j]).std(ddof=1) / den) if den > 0 else float("nan")


def _check_iid():
    rng = np.random.default_rng(23)
    x = 0.5 + rng.normal(0, 0.05, 60)[:, None] + rng.normal(0, 0.3, 300)[None, :] + rng.normal(0, 0.4, (60, 300))
    order = np.argsort(-x.mean(axis=1))
    i, j = int(order[0]), int(order[1])
    a, _ = pair_required(x, i, j)
    b = board_required(x, i, j)
    return abs(b / a - 1) < 0.10, f"iid field: pair rule {a:.0f} items, board rule {b:.0f} (ratio {b / a:.2f})"


def _check_scaling():
    """Doubling the GAP quarters the prescription when noise dominates.

    The first version doubled the whole difference vector and expected a
    quarter; that was arithmetic error in the check, not in the code:
    scaling d by 2 scales both E[d^2] and delta^2 by 4, so n is invariant -
    correctly so, since a benchmark whose scores are all doubled has not
    become more informative. What does help is a larger gap at the same
    noise, and that is what is checked now.
    """
    rng = np.random.default_rng(25)
    n = 4000
    noise = rng.normal(0, 0.4, n)
    noise -= noise.mean()      # a sample mean of 0.006 shifted the effective
                               # gap and made the first run read 0.43, not 0.25
    for delta in (0.01, 0.02):
        d = delta + noise
        req = float((d ** 2).mean()) * ((Z_A + Z_B) / float(d.mean())) ** 2
        if delta == 0.01:
            a = req
        else:
            b = req
    return abs(b / a - 0.25) < 0.05, f"doubling the gap at fixed noise: {a:.0f} -> {b:.0f} items (ratio {b / a:.2f})"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_iid(), _check_scaling()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHAT PAIR-SPECIFIC RESOLUTION COSTS OR SAVES, IN ITEMS")
    p("=" * 90)
    p(f"  {'leaderboard':<22} {'n now':>6} {'kappa':>6} {'pair rule':>11} {'board rule':>11} "
      f"{'ratio':>6} {'1/kappa^2':>9} {'excess':>7}")
    ident, excess, pr_list, br_list = 0, 0, [], []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        order = np.argsort(-x.mean(axis=1))
        i, j = int(order[0]), int(order[1])
        pr, delta = pair_required(x, i, j)
        br = board_required(x, i, j)
        k = kappa_of(x, i, j)
        ratio = br / pr if pr > 0 and math.isfinite(pr) else float("nan")
        pred = 1 / k ** 2 if k > 0 else float("nan")
        ident += abs(ratio / pred - 1) <= 0.15 if math.isfinite(ratio) and math.isfinite(pred) else 0
        excess += ratio > 1.20 if math.isfinite(ratio) else 0
        pr_list.append(pr); br_list.append(br)
        p(f"  {name:<22} {x.shape[1]:>6} {k:>6.2f} {pr:>11.0f} {br:>11.0f} {ratio:>6.2f} {pred:>9.2f} "
          f"{100 * (ratio - 1):>+6.0f}%")
    N = len(pr_list)
    finite = [(a, b) for a, b in zip(pr_list, br_list) if math.isfinite(a) and math.isfinite(b)]
    sp = spearmanr([a for a, _ in finite], [b for _, b in finite]).statistic if len(finite) > 2 else float("nan")
    p("")
    p(f"  ratio equals 1/kappa^2 within 15 %: {ident}/{N} (pre-registered >= 8)")
    p(f"  board rule asks more than 20 % extra: {excess}/{N} (pre-registered >= 6)")
    p(f"  ordering of boards unchanged: Spearman {sp:+.2f} (pre-registered > 0.9)")
    p("")
    p(f"  {'dated board':<20} {'median frontier advance':>24} {'pair rule':>11} {'board rule':>11} {'ratio':>6}")
    for name, (path, dc) in DATED.items():
        x, dates = load(path, dc)
        rows = []
        for a in advances(x, dates):
            pr, delta = pair_required(x, a["new"], a["old"])
            br = board_required(x, a["new"], a["old"])
            if math.isfinite(pr) and math.isfinite(br):
                rows.append((pr, br))
        if not rows:
            continue
        pr_m = float(np.median([r[0] for r in rows]))
        br_m = float(np.median([r[1] for r in rows]))
        p(f"  {name:<20} {len(rows):>24} {pr_m:>11.0f} {br_m:>11.0f} {br_m / pr_m:>6.2f}")
    p("")
    p("  'pair rule' = items needed to separate that pair at 80 % power, 5 %,")
    p("  using the pair's own difference vector (refill_all.py). 'board rule'")
    p("  substitutes the board's typical pair variance, which is what a single")
    p("  published resolution number invites. The gap between them is the cost")
    p("  of not reporting R10.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("prescription_pairwise_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote prescription_pairwise_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
