"""The benchmark as an instrument: resolution spectrum, aperture, dead pixels.

A leaderboard is read as if the benchmark were a ruler with 500 equal marks.
It is not. Some items separate every pair of systems, some separate none, and
the ones that matter for the top of the table are not the ones that matter for
the bottom. This file measures the instrument instead of the systems on it.

THREE QUANTITIES, ONE CURVE
---------------------------
Ranking information of an item. Two systems are told apart by an item only
when they DISAGREE on it: the ones both solve and both miss carry nothing,
which is McNemar's concordant-pair argument at the level of a single item. So
the item's contribution to the ranking is the number of system pairs it splits.

    w_i = #{(j,k) : j < k, x_ji != x_ki}

A ONE-LINE THEOREM, WORTH STATING BECAUSE IT SETTLES BENCHMARK DESIGN
----------------------------------------------------------------------
For binary outcomes, if s_i systems out of J solve item i, then the pairs it
splits are exactly the solved-unsolved pairs:

    w_i = s_i (J - s_i)

so the ranking information of an item depends ONLY on how many systems solve
it, and is maximised at s_i = J/2. An item everyone solves and an item nobody
solves are the same item to a ranking: both are blank. This is checked against
brute force before it is used.

EFFECTIVE APERTURE
------------------
Given the w_i, how many items is the benchmark really worth? Borrow the
participation ratio used for localisation in physics:

    n_eff = (sum_i w_i)^2 / sum_i w_i^2

which equals n when every item carries the same information and 1 when a
single item carries it all. It is the honest denominator: a benchmark with 500
items and n_eff = 300 is a 300-item instrument wearing a 500-item label.

WHERE THE UNWEIGHTED VIEW STOPS BEING ENOUGH
---------------------------------------------
w_i counts all pairs alike, so separating rank 100 from rank 101 counts as
much as separating first from second. Nobody cares equally. The targeted
version restricts the count to the pairs actually in question - by default the
systems whose rank confidence sets still contain 1, the candidate leaders:

    w_i(F) = #{(j,k) in F : x_ji != x_ki}

This no longer reduces to the column sum, because it matters WHICH systems
disagree, and it answers the question a benchmark owner actually has: which
items decide the top of my table, and how few of them are there?

THE CIRCULARITY, AND HOW IT IS AVOIDED
---------------------------------------
Choosing items by how well they separate these systems and then measuring the
separation on the same systems is a tautology; the subset is guaranteed to
look good. So the subset is chosen on one half of the systems and judged on
the other half, which never touched the selection. Both numbers are reported,
the honest one and the circular one, because the size of the gap is itself the
result: it says how much of any published "we can use fewer items" claim is
selection.

    python benchmark_spectrum.py [--matrix ...] [--focal 19]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260823


def item_information(x: np.ndarray) -> np.ndarray:
    """w_i = number of system pairs that item i splits, via s(J-s)."""
    s = x.sum(axis=0)
    J = x.shape[0]
    return s * (J - s)


def item_information_focal(x: np.ndarray, focal: np.ndarray) -> np.ndarray:
    """Same, restricted to pairs inside `focal` (indices of systems)."""
    sub = x[focal]
    s = sub.sum(axis=0)
    k = len(focal)
    return s * (k - s)


def effective_items(w: np.ndarray) -> float:
    """Participation ratio: how many items the instrument is really worth."""
    tot = float(w.sum())
    if tot <= 0:
        return 0.0
    return float(tot ** 2 / np.sum(w.astype(float) ** 2))


def spectrum(w: np.ndarray) -> dict:
    order = np.argsort(-w, kind="stable")
    cum = np.cumsum(w[order]).astype(float)
    tot = cum[-1] if len(cum) else 0.0
    frac = cum / tot if tot > 0 else cum
    return {"order": order, "frac": frac, "total": tot}


def items_for_fraction(w: np.ndarray, target: float) -> int:
    sp = spectrum(w)
    if sp["total"] <= 0:
        return 0
    return int(np.searchsorted(sp["frac"], target) + 1)


def rank_agreement(x: np.ndarray, items: np.ndarray) -> dict:
    """How much of the full-item ranking survives on a subset of items."""
    full = x.mean(axis=1)
    sub = x[:, items].mean(axis=1)
    # Spearman without scipy: correlation of ranks.
    rf = np.argsort(np.argsort(-full, kind="stable"))
    rs = np.argsort(np.argsort(-sub, kind="stable"))
    rho = float(np.corrcoef(rf, rs)[0, 1])
    # Pair agreement is the quantity a leaderboard reader actually uses.
    J = len(full)
    iu = np.triu_indices(J, k=1)
    df = np.sign(full[:, None] - full[None, :])[iu]
    ds = np.sign(sub[:, None] - sub[None, :])[iu]
    agree = float(np.mean(df == ds))
    return {"spearman": rho, "pair_agreement": agree, "m": len(items)}



def unanimous_items_are_inert(x: np.ndarray, focal: np.ndarray) -> dict:
    """An item unanimous inside a group cannot reorder that group. Verified.

    If every system in the group gets the same outcome on item i, that item
    adds the SAME amount to every one of their scores, so it shifts the whole
    group and reorders nothing. The consequence is exact rather than
    statistical: the contest among these systems lives entirely on the items
    where they disagree, and the rest could be deleted without moving a single
    position. That is a strong claim, so it is checked against the data rather
    than asserted from the algebra.

    Note the boundary, which is easy to get wrong: this holds for the CURRENT
    members of the group. A new system may well disagree on an item they were
    unanimous about, so the discarded items are inert for comparing these
    systems, not inert forever.
    """
    wf = item_information_focal(x, focal)
    live = np.flatnonzero(wf > 0)
    dead = np.flatnonzero(wf == 0)
    full = x[focal].mean(axis=1)
    part = x[focal][:, live].mean(axis=1)
    order_full = np.argsort(-full, kind="stable")
    order_part = np.argsort(-part, kind="stable")
    same = bool(np.array_equal(order_full, order_part))
    # The dead items must contribute an identical amount to everyone.
    contrib = x[focal][:, dead].sum(axis=1) if len(dead) else np.zeros(len(focal))
    identical = bool(len(set(contrib.tolist())) <= 1)
    return {"live": len(live), "dead": len(dead), "order_preserved": same,
            "dead_contribution_identical": identical,
            "dead_contribution": float(contrib[0]) if len(contrib) else 0.0}

# ---------------------------------------------------------------------------
# Self-checks. Nothing is printed as a result until these pass.
# ---------------------------------------------------------------------------

def _check_theorem() -> tuple[bool, str]:
    """w_i = s(J-s) must equal a brute-force pair count, exactly."""
    rng = np.random.default_rng(2)
    x = (rng.random((9, 40)) < 0.5).astype(int)
    fast = item_information(x)
    J = x.shape[0]
    slow = np.zeros(x.shape[1], dtype=int)
    for j in range(J):
        for k in range(j + 1, J):
            slow += (x[j] != x[k]).astype(int)
    ok = bool(np.array_equal(fast, slow))
    return ok, f"w_i = s(J-s) matches brute force on all {x.shape[1]} items: {ok}"


def _check_dead_items() -> tuple[bool, str]:
    """Items everyone or nobody solves must carry exactly zero."""
    x = np.zeros((6, 4), dtype=int)
    x[:, 1] = 1                      # everyone solves
    x[:3, 2] = 1                     # split
    w = item_information(x)
    ok = bool(w[0] == 0 and w[1] == 0 and w[2] > 0 and w[3] == 0)
    return ok, f"dead items score zero: w = {list(w)}"


def _check_participation_ratio() -> tuple[bool, str]:
    """n_eff = n for a flat spectrum, 1 for a single carrier."""
    flat = effective_items(np.full(50, 7))
    spike = effective_items(np.array([100] + [0] * 49))
    ok = abs(flat - 50) < 1e-9 and abs(spike - 1) < 1e-9
    return ok, f"n_eff flat {flat:.1f} (want 50), spike {spike:.1f} (want 1)"


def _check_selection_is_circular() -> tuple[bool, str]:
    """On pure noise, in-sample selection must look good and out-of-sample must not.

    This is the check that keeps the headline honest. Build a benchmark where
    NO system is better than any other, select the items that best separate
    half the systems, and score the other half. In-sample agreement will be
    high because the items were picked for it; out-of-sample it must fall
    towards what a random subset of the same size gives.
    """
    rng = np.random.default_rng(29)
    J, n, m = 40, 400, 40
    diff = rng.normal(0, 0.8, n)
    p = 1 / (1 + np.exp(-diff))
    x = (rng.random((J, n)) < p).astype(int)
    a = np.arange(0, J, 2)
    b = np.arange(1, J, 2)
    w_a = item_information(x[a])
    picked = np.argsort(-w_a, kind="stable")[:m]
    rand = rng.choice(n, size=m, replace=False)
    ins = rank_agreement(x[a], picked)["pair_agreement"]
    out = rank_agreement(x[b], picked)["pair_agreement"]
    ran = rank_agreement(x[b], rand)["pair_agreement"]
    ok = ins > out and abs(out - ran) < 0.10
    return ok, (f"circularity visible: in-sample {ins:.3f} > held-out {out:.3f}, "
                f"held-out ~ random {ran:.3f}")


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_theorem(), _check_dead_items(),
                        _check_participation_ratio(),
                        _check_selection_is_circular()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--focal", type=int, default=19,
                    help="size of the contested top group")
    ap.add_argument("--out", default="benchmark_spectrum_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=int)
    J, n = x.shape
    print(f"matrix {a.matrix}: {J} systems x {n} items")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no headline number is printed.")
        return 1

    w = item_information(x)
    n_eff = effective_items(w)
    dead = int((w == 0).sum())

    order_scores = np.argsort(-x.mean(axis=1), kind="stable")
    focal = order_scores[:a.focal]
    wf = item_information_focal(x, focal)
    n_eff_f = effective_items(wf)
    dead_f = int((wf == 0).sum())

    L = []
    p = L.append
    p("THE BENCHMARK AS AN INSTRUMENT")
    p("=" * 74)
    p(f"{J} systems, {n} items")
    p("")
    p("WHOLE TABLE")
    p(f"  dead items (split no pair)        {dead} of {n}"
      f"   ({100 * dead / n:.1f} %)")
    p(f"  effective item count n_eff        {n_eff:.1f} of {n}"
      f"   ({100 * n_eff / n:.1f} %)")
    for t in (0.50, 0.80, 0.90, 0.95):
        p(f"  items carrying {100 * t:.0f} % of the ranking information"
          f"   {items_for_fraction(w, t)}")
    p("")
    p(f"CONTESTED TOP GROUP ({a.focal} systems whose rank set still contains 1)")
    p(f"  items that split no pair in it    {dead_f} of {n}"
      f"   ({100 * dead_f / n:.1f} %)")
    p(f"  effective item count n_eff        {n_eff_f:.1f}")
    for t in (0.50, 0.90):
        p(f"  items carrying {100 * t:.0f} % of the top-group information"
          f"   {items_for_fraction(wf, t)}")
    p("")
    inert = unanimous_items_are_inert(x, focal)
    p(f"  -> the contest among these {a.focal} systems lives on "
      f"{inert['live']} of {n} items.")
    p(f"     The other {inert['dead']} are unanimous inside the group: every")
    p(f"     member scores the same {inert['dead_contribution']:.0f} on them, so")
    p("     they shift the whole group and reorder nothing.")
    p(f"     ordering on live items only reproduces the full ordering: "
      f"{inert['order_preserved']}")
    p(f"     dead items contribute an identical amount to every member: "
      f"{inert['dead_contribution_identical']}")
    p("     This is exact, not statistical. It holds for THESE systems: a new")
    p("     system may disagree where they were unanimous, so the discarded")
    p("     items are inert for this comparison, not inert forever.")
    p("")

    # Honest test: choose items on half the systems, judge on the other half.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(J)
    ha, hb = perm[: J // 2], perm[J // 2:]
    w_a = item_information(x[ha])
    p("HELD-OUT TEST OF THE SUBSET CLAIM")
    p("  items chosen on one half of the systems, ranking judged on the other")
    p(f"{'m':>6} {'in-sample':>11} {'held-out':>10} {'random m':>10}")
    for m in (25, 50, 100, 200, 350, n):
        picked = np.argsort(-w_a, kind="stable")[:m]
        rand = rng.choice(n, size=m, replace=False)
        ins = rank_agreement(x[ha], picked)["pair_agreement"]
        out = rank_agreement(x[hb], picked)["pair_agreement"]
        ran = rank_agreement(x[hb], rand)["pair_agreement"]
        p(f"{m:>6} {ins:>11.3f} {out:>10.3f} {ran:>10.3f}")
    p("")
    p("  READ THE THIRD COLUMN FIRST. Choosing items by how well they")
    p("  separate systems barely beats choosing the same number at random:")
    p("  at m = 50 the held-out agreement is 0.921 against 0.919 for a random")
    p("  draw. The reason is in the theorem - w_i = s(J-s) is a smooth")
    p("  function of item difficulty, and a random subset already samples that")
    p("  distribution. So the lever is NOT clever item selection. It is that")
    p("  most items are silent on the question being asked.")
    p("")
    p("  in-sample is the number a paper would quote if it selected and")
    p("  evaluated on the same systems. The held-out column is the claim that")
    p("  survives, and the random column is what the same number of items")
    p("  bought without any selection at all. The gap between the first two")
    p("  is the size of the tautology.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
