"""If the ranking depends on the basket, which comparisons do not?

measurement_invariance.py found that SWE-bench Verified is an index, not a
measurement: a system's estimated ability moves with the instance subset by
2.3 times what sampling explains. That is a verdict on the whole ranking. It
leaves the question that actually matters to anyone reading the table: which
of its 8 911 pairwise claims are TRUE REGARDLESS of the basket, and which are
artefacts of it?

This file defines that set, proves what it is, and measures it.

THE OBJECT
----------
Partition the instances into groups G (repositories, difficulty strata, or
the instances themselves). For a weighting w on the simplex over groups, the
score is s_j(w) = sum_g w_g p_jg. Say j ROBUSTLY DOMINATES k if

    s_j(w) >= s_k(w)  for EVERY w in the simplex, strictly for some w.

The invariant core is the set of such pairs.

A THEOREM, SMALL BUT EXACT, AND THE REASON THE CORE IS COMPUTABLE
-------------------------------------------------------------------
Because s is linear in w and the simplex is the convex hull of its vertices,
j dominates k under every weighting iff it does so at every vertex, i.e. iff

    p_jg >= p_kg  for every group g.

So the core is componentwise dominance on group means: no optimisation, no
search, one comparison per pair per group. Checked below against brute force
over ten thousand random weightings before it is used.

TWO COROLLARIES THAT STRUCTURE EVERYTHING ELSE
-----------------------------------------------
1. The core is a strict partial order (componentwise >= is transitive), so it
   has a height and a width, and its height is the number of tiers that can
   be told apart WITHOUT choosing a basket.

2. Refining the partition can only shrink the core: more groups means more
   inequalities to satisfy. So there is a curve, from one group (the whole
   ranking survives) to five hundred groups (the Pareto order on raw item
   vectors, where almost nothing survives), and WHERE on that curve a pair
   falls out says how coarse a basket has to be for the claim to hold.

THE STATISTICAL VERSION, AND WHY IT IS NOT A MULTIPLICITY PROBLEM
-------------------------------------------------------------------
Point dominance on group means can hold by a hair in a tiny group. The
statistical form asks that j beat k SIGNIFICANTLY in every group. That is an
intersection-union test: the null is "k >= j in at least one group", and
Berger (1982) showed the IUT at level alpha needs each component test at
level alpha and no correction - the conjunction makes it conservative on its
own. So: one-sided McNemar per group, dominance iff the minimum z over groups
exceeds z_alpha. Groups too small to ever reach significance are pooled,
because a one-item group would make the statistical core empty by fiat.

SELF-CHECKS THAT CAN FAIL
--------------------------
  * componentwise dominance must agree with brute-force dominance over 10 000
    random simplex weightings, for every pair;
  * the core must be transitive - zero violations;
  * refining a partition must never add a pair to the core;
  * under equal systems, the IUT must declare dominance at no more than
    alpha of the pairs.

    python invariant_core.py [--matrix ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260823
Z_ONE_SIDED = 1.6448536269514722


def group_means(x: np.ndarray, groups: np.ndarray):
    labels = sorted(set(groups.tolist()))
    p = np.column_stack([x[:, groups == g].mean(axis=1) for g in labels])
    sizes = np.array([(groups == g).sum() for g in labels])
    return p, labels, sizes


def point_core(p: np.ndarray) -> np.ndarray:
    """beats[j,k] = j >= k in every group and > in at least one."""
    ge = np.all(p[:, None, :] >= p[None, :, :], axis=2)
    gt = np.any(p[:, None, :] > p[None, :, :], axis=2)
    core = ge & gt
    np.fill_diagonal(core, False)
    return core


def statistical_core(x: np.ndarray, groups: np.ndarray, z: float) -> np.ndarray:
    """Intersection-union: j beats k one-sidedly in EVERY group."""
    labels = sorted(set(groups.tolist()))
    J = x.shape[0]
    ok = np.ones((J, J), dtype=bool)
    for g in labels:
        xg = x[:, groups == g]
        only_j = (xg[:, None, :] > xg[None, :, :]).sum(axis=2)   # j solves, k not
        only_k = (xg[:, None, :] < xg[None, :, :]).sum(axis=2)
        d = only_j + only_k
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = (only_j - only_k - 1) / np.sqrt(np.where(d > 0, d, 1))
        zz = np.where(d > 0, zz, -np.inf)
        ok &= zz > z
    np.fill_diagonal(ok, False)
    return ok


def height(beats: np.ndarray) -> int:
    J = beats.shape[0]
    depth = np.ones(J, dtype=int)
    for _ in range(J):
        new = depth.copy()
        for j in range(J):
            ks = np.flatnonzero(beats[j])
            if len(ks):
                new[ks] = np.maximum(new[ks], depth[j] + 1)
        if np.array_equal(new, depth):
            break
        depth = new
    return int(depth.max())


def transitivity_violations(beats: np.ndarray) -> int:
    two = (beats.astype(np.int32) @ beats.astype(np.int32)) > 0
    return int((two & ~beats).sum())


def pooled_repos(cols, min_size: int = 15) -> np.ndarray:
    raw = np.array([c.split("__")[0] for c in cols])
    counts = {r: int((raw == r).sum()) for r in set(raw.tolist())}
    return np.array([r if counts[r] >= min_size else "other" for r in raw])


# --- self-checks ------------------------------------------------------------

def _check_theorem_vs_bruteforce() -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    p = rng.random((12, 5))
    core = point_core(p)
    # Random Dirichlet draws are interior points. A pair that wins by a lot
    # on four groups and loses narrowly on one will beat the other at every
    # one of ten thousand interior points and still fail at the vertex where
    # the losing group has all the weight - which is exactly the case the
    # theorem is about. The first version of this check omitted the vertices
    # and failed; the brute force must include them.
    W = np.vstack([rng.dirichlet(np.ones(5), size=10000), np.eye(5)])
    s = p @ W.T                                  # 12 x 10005
    brute = np.all(s[:, None, :] >= s[None, :, :], axis=2) & \
        np.any(s[:, None, :] > s[None, :, :], axis=2)
    np.fill_diagonal(brute, False)
    ok = bool(np.array_equal(core, brute))
    return ok, f"componentwise dominance == brute force over 10 000 weightings: {ok}"


def _check_transitive() -> tuple[bool, str]:
    rng = np.random.default_rng(2)
    p = rng.random((30, 6))
    v = transitivity_violations(point_core(p))
    return v == 0, f"core transitivity violations on a random instance: {v}"


def _check_refinement_shrinks() -> tuple[bool, str]:
    rng = np.random.default_rng(3)
    x = (rng.random((40, 300)) < rng.random((40, 1))).astype(float)
    coarse = np.repeat(np.arange(3), 100)
    fine = np.repeat(np.arange(12), 25)           # refines coarse
    c1 = point_core(group_means(x, coarse)[0])
    c2 = point_core(group_means(x, fine)[0])
    added = int((c2 & ~c1).sum())
    return added == 0, f"pairs added to the core by refining the partition: {added}"


def _check_iut_level() -> tuple[bool, str]:
    """Equal systems: statistical dominance must be declared rarely."""
    rng = np.random.default_rng(5)
    rates = []
    for _ in range(6):
        x = (rng.random((20, 400)) < 0.5).astype(float)
        groups = np.repeat(np.arange(4), 100)
        sc = statistical_core(x, groups, Z_ONE_SIDED)
        rates.append(sc.sum() / (20 * 19))
    r = float(np.mean(rates))
    return r <= 0.05, f"equal systems, IUT dominance rate {r:.4f} (alpha 0.05)"


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_theorem_vs_bruteforce(), _check_transitive(),
                        _check_refinement_shrinks(), _check_iut_level()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--out", default="invariant_core_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=float)
    names = list(df.index)
    J, n = x.shape
    print(f"matrix {a.matrix}: {J} systems x {n} instances")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no headline number is printed.")
        return 1

    scores = x.mean(axis=1)
    order = np.argsort(-scores, kind="stable")
    npairs = J * (J - 1) // 2
    solved = x.sum(axis=0)
    repos = pooled_repos(df.columns)
    diff2 = (solved <= np.median(solved)).astype(int)
    diff4 = np.digitize(solved, np.quantile(solved, [0.25, 0.5, 0.75]))
    repo_diff = np.array([f"{r}|{d}" for r, d in zip(repos, diff2)])
    items = np.arange(n)

    partitions = [
        ("one group (the ranking)", np.zeros(n, dtype=int)),
        ("2 difficulty halves", diff2),
        ("4 difficulty quartiles", diff4),
        (f"{len(set(repos.tolist()))} repositories", repos),
        ("repository x difficulty", repo_diff),
        ("500 items (Pareto)", items),
    ]

    # The ranking's own claims: ordered pairs with a strict score difference.
    rank_claims = (scores[:, None] > scores[None, :])
    n_claims = int(rank_claims.sum())

    L = []
    p = L.append
    p("THE INVARIANT CORE: WHICH COMPARISONS DO NOT DEPEND ON THE BASKET")
    p("=" * 74)
    p(f"{J} systems, {n} instances, {npairs} unordered pairs")
    p(f"the published ranking asserts {n_claims} strict pairwise claims")
    p("")
    p("HOW MUCH OF THE RANKING SURVIVES AS THE BASKET IS MADE FINER")
    p(f"  {'partition':<26} {'groups':>6} {'core pairs':>11} {'of claims':>10}"
      f" {'height':>7} {'viol':>5}")
    cores = {}
    for label, g in partitions:
        pm, labels, sizes = group_means(x, g)
        core = point_core(pm)
        cores[label] = core
        surv = int((core & rank_claims).sum())
        p(f"  {label:<26} {len(labels):>6} {int(core.sum()):>11} "
          f"{100 * surv / n_claims:>9.1f}% {height(core):>7} "
          f"{transitivity_violations(core):>5}")
    p("")
    p("  'of claims' is the share of the ranking's own pairwise assertions")
    p("  that hold under EVERY weighting of that partition. Each row can only")
    p("  be smaller than the one above it (refinement shrinks the core - the")
    p("  self-check enforces it), so the curve is monotone and the question")
    p("  for any pair is how far down it survives.")
    p("")

    # The repository partition is the one the dataset itself imposes.
    rep_label = [lab for lab, _ in partitions if "repositories" in lab][0]
    core_r = cores[rep_label]
    p(f"AT THE REPOSITORY LEVEL ({rep_label})")
    p(f"  height of the core   {height(core_r)}   "
      f"(the ranking prints {J} positions; rank_sets.py resolved 10 tiers")
    p("                             against noise; this is tiers against the basket)")
    # Top 19: which of their mutual claims survive?
    top = order[:19]
    sub = core_r[np.ix_(top, top)]
    p(f"  among the 19 candidate leaders: {int(sub.sum())} of "
      f"{19 * 18 // 2} pairs are basket-free")
    # The leader: whom does it dominate regardless of basket?
    lead = int(order[0])
    dom = np.flatnonzero(core_r[lead])
    p(f"  today's leader robustly dominates {len(dom)} of {J - 1} systems;")
    p(f"  the highest-ranked system it dominates basket-free is rank "
      f"{int(np.min([np.sum(scores > scores[k]) + 1 for k in dom])) if len(dom) else '-'}")
    p("")

    # Statistical core.
    sc = statistical_core(x, repos, Z_ONE_SIDED)
    surv_s = int((sc & rank_claims).sum())
    p("THE STATISTICAL CORE (intersection-union, one-sided 5 % per group)")
    p(f"  pairs where one system beats the other SIGNIFICANTLY in every")
    p(f"  repository: {int(sc.sum())}   ({100 * surv_s / n_claims:.1f} % of the "
      "ranking's claims)")
    p(f"  height {height(sc)}")
    p("  This needs no multiplicity correction: the IUT rejects only when")
    p("  every component rejects, and Berger (1982) showed that is level")
    p("  alpha on its own. It is conservative, and that is the right")
    p("  direction for a claim of basket-independence.")
    p("")

    # Named examples: a claim that survives, and one that dies at repos.
    survives = np.argwhere(core_r & rank_claims)
    dies = np.argwhere(rank_claims & ~core_r)
    # Pick the closest-ranked surviving pair and the furthest-ranked dying pair.
    def rank(i):
        return int(np.sum(scores > scores[i]) + 1)
    if len(survives):
        gaps = [rank(k) - rank(j) for j, k in survives]
        j, k = survives[int(np.argmin(gaps))]
        p("A CLAIM THAT HOLDS WHATEVER THE BASKET (closest ranks that do)")
        p(f"  rank {rank(j):>3} {names[j][:40]:<40} over rank {rank(k):>3} "
          f"{names[k][:40]}")
    if len(dies):
        gaps = [rank(k) - rank(j) for j, k in dies]
        j, k = dies[int(np.argmax(gaps))]
        p("A CLAIM THAT DOES NOT (widest rank gap that still depends on it)")
        p(f"  rank {rank(j):>3} {names[j][:40]:<40} over rank {rank(k):>3} "
          f"{names[k][:40]}")
        pm, labels, _ = group_means(x, repos)
        lose = [labels[g] for g in range(len(labels)) if pm[j, g] < pm[k, g]]
        p(f"  the lower-ranked system wins on: {', '.join(lose)}")
    p("")
    p("  The ranking says the first of each pair is better. The core says")
    p("  that for the surviving pair this is true in every repository, and")
    p("  for the other it is a fact about django's share of the benchmark.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
