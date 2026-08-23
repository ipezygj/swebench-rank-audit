"""Who could be first if the benchmark had been composed differently? All boards.

reweighting_polytope.py answers this exactly for SWE-bench: the score is a
linear function of the group weights, so "is there a weighting that puts
system t on top" is a linear-programming feasibility question. The weights
of every benchmark here are accidents of collection - 46 % of SWE-bench
Verified is django, MTEB has more classification tasks than anything else -
and nobody has had to defend them.

This runs the same LP on every board with a natural item grouping, using
the groupings already fixed in cluster_bootstrap.py.

The champion set is a different kind of statement from the rank set. The
rank set says which systems the NOISE cannot rule out. The champion set
says which systems the COMPOSITION cannot rule out - it is exact, with no
sampling in it at all.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the champion set is larger than 1 on every board except CASP14;
  * it is SMALLER than tie@1 on >= 3 of 4 boards - reweighting is a
    narrower freedom than sampling noise, because a champion has to lead
    under some fixed weighting rather than merely be unseparated;
  * on SWE-bench the champion set includes at least one system outside the
    printed top ten.

SELF-CHECKS
  * a system that leads every group must be the only champion;
  * a system dominated by another in every group can never be a champion.

    python composition_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import rank_sets as rs
from cluster_bootstrap import clusters_swebench, clusters_matharena, clusters_casp, clusters_mteb

SEED = 20260823
BOARDS = {
    "SWE-bench Verified": ("swebench_verified_matrix.csv", clusters_swebench),
    "MathArena 2025": ("matharena/matrix.csv", clusters_matharena),
    "CASP14": ("casp/matrix.csv", clusters_casp),
    "MTEB English v2": ("mteb_dated_matrix.csv", clusters_mteb),
}


def group_matrix(df, labels):
    """J x G matrix of group means, and the group sizes."""
    lab = np.array(labels)
    groups = sorted(set(lab))
    p = np.column_stack([df.to_numpy(dtype=float)[:, lab == g].mean(axis=1) for g in groups])
    sizes = np.array([int((lab == g).sum()) for g in groups], dtype=float)
    return p, sizes / sizes.sum(), groups


def can_lead(p, t, w0=None, factor=None):
    """Is there a weight vector putting system t first?

    With `factor` given, the weights are confined to within that factor of
    the board's actual composition w0 (each group between w0/factor and
    w0*factor, renormalised by the equality constraint). Without it, any
    point of the simplex is allowed, INCLUDING its corners - which means a
    system that leads on one single group is a champion, and the answer is
    close to vacuous. The unconstrained version was run first and is kept
    for comparison; the constrained one is the question a benchmark owner
    can actually be asked.
    """
    J, G = p.shape
    A = p - p[t][None, :]
    A = np.delete(A, t, axis=0)
    if factor is None or w0 is None:
        bounds = [(0, None)] * G
    else:
        bounds = [(float(w / factor), float(min(w * factor, 1.0))) for w in w0]
    res = linprog(c=np.zeros(G), A_ub=A, b_ub=np.zeros(A.shape[0]),
                  A_eq=np.ones((1, G)), b_eq=[1.0], bounds=bounds, method="highs")
    return bool(res.status == 0)


def _check_dominant():
    p = np.array([[0.9, 0.8, 0.7], [0.5, 0.4, 0.3], [0.2, 0.3, 0.1]])
    champs = [t for t in range(3) if can_lead(p, t)]
    return champs == [0], f"a system leading every group is the only champion: {champs}"


def _check_dominated():
    p = np.array([[0.9, 0.8, 0.7], [0.5, 0.4, 0.3], [0.4, 0.3, 0.2]])
    return not can_lead(p, 2), "a dominated system is never a champion"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_dominant(), _check_dominated()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p_ = L.append
    p_("WHO COULD BE FIRST UNDER A DIFFERENT COMPOSITION?")
    p_("=" * 86)
    p_(f"  {'board':<22} {'groups':>7} {'champions':>10} {'within 2x':>10} {'tie@1':>6} "
      f"{'lowest champion':>17} {'lowest within 2x':>18}")
    bigger, smaller, outside = 0, 0, 0
    for name, (path, fn) in BOARDS.items():
        df = pd.read_csv(path, index_col=0).dropna(axis=0)
        labels = fn(list(df.columns))
        p, w0, groups = group_matrix(df, labels)
        J = p.shape[0]
        champs = [t for t in range(J) if can_lead(p, t)]
        champs2 = [t for t in range(J) if can_lead(p, t, w0, factor=2.0)]
        r = rs.rank_sets(df.to_numpy(dtype=float), draws=800)
        tie1 = int((r["best"] == 1).sum())
        order = list(np.argsort(-df.to_numpy(dtype=float).mean(axis=1)))
        ranks = {int(j): i + 1 for i, j in enumerate(order)}
        worst_champ = max(ranks[c] for c in champs) if champs else 0
        bigger += len(champs) > 1
        smaller += len(champs) < tie1
        if name.startswith("SWE"):
            outside = worst_champ > 10
        worst2 = max(ranks[c] for c in champs2) if champs2 else 0
        p_(f"  {name:<22} {len(groups):>7} {len(champs):>10} {len(champs2):>10} {tie1:>6} "
           f"{f'rank {worst_champ}/{J}':>17} {f'rank {worst2}/{J}':>18}")
    p_("")
    p_(f"  champion set larger than one: {bigger}/4 (pre-registered: all but CASP14)")
    p_(f"  champion set smaller than tie@1: {smaller}/4 (pre-registered >= 3)")
    p_(f"  SWE-bench has a champion outside the printed top ten: {'yes' if outside else 'NO'}")
    p_("")
    p_("  A champion is a system for which SOME non-negative weighting of the item")
    p_("  groups puts it first. The weights a benchmark actually uses are its group")
    p_("  sizes, which are an artefact of collection. This is exact linear")
    p_("  programming - no sampling, no confidence level - and it answers a")
    p_("  different question from the rank sets: not what the noise allows, but")
    p_("  what the composition allows. The unconstrained column allows a corner of")
    p_("  the simplex - all weight on one repository - which is why it is nearly")
    p_("  vacuous. The 'within 2x' column keeps every group between half and twice")
    p_("  its actual share, which is the range a benchmark owner could defend.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("composition_all_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote composition_all_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
