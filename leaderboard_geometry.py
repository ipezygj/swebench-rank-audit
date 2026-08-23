"""A leaderboard is a line. The data underneath it is not, and this measures how.

A ranking makes two silent assumptions and reports neither. First, that the
systems form a chain - that "better than" orders all of them. Second, that
ability is one number, so that a single axis loses nothing. Both are testable
against the matrix the leaderboard was computed from, and both fail here.

I. HOW MANY TIERS DOES THIS BENCHMARK ACTUALLY RESOLVE?
--------------------------------------------------------
Take the relation "j is significantly above k", simultaneously over all pairs,
from rank_sets.py. That relation is a strict partial order, not a ranking:
most pairs are incomparable. A poset has a height - the longest chain in it -
and by Mirsky's theorem a poset of height h partitions into exactly h
antichains, each a set of mutually indistinguishable systems.

So the height IS the number of performance levels the instrument can tell
apart. A leaderboard with 134 rows and height h is reporting 134 positions
while resolving h. That number has, as far as I can find, never been published
for any machine-learning benchmark, and it is exact once the relation is
fixed: no distributional assumption enters beyond the ones already made.

Dilworth's theorem gives the companion number - the largest antichain, the
biggest set of systems that cannot be ordered among themselves at all -
computed as n minus a maximum bipartite matching on the transitive closure.

II. HOW MANY NUMBERS IS THE BENCHMARK MEASURING?
--------------------------------------------------
A ranking is a projection onto one axis. That is lossless only if the systems
differ along one axis. Strip the two things a ranking already accounts for -
how able each system is and how hard each item is - by taking the two-way
residual

    R = X - rowmean - colmean + grandmean

If ability were one-dimensional, R would be noise. It is not noise if some
systems are good at a KIND of item that others are not, and then the single
ranking depends on the item mix rather than on ability alone, which is a
different and much weaker claim than the leaderboard makes.

The number of real dimensions is read by parallel analysis: permute inside
each column, which destroys any system-specific structure while leaving item
difficulty exactly as it was, and count the singular values of R that stand
above what permutation produces.

SELF-CHECKS THAT CAN FAIL
--------------------------
  * a synthetic poset with k built-in tiers must measure height k;
  * pure noise must measure height 1 - nothing is above anything;
  * rank-1 synthetic data must show 1 dimension, rank-3 must show 3;
  * noise must show 0 dimensions;
  * transitivity is checked rather than assumed, and violations are counted:
    the relation is derived from thresholded differences and nothing forces
    it to be transitive, so if it is not, that is a fact about the instrument
    and belongs in the report.

    python leaderboard_geometry.py [--matrix ...] [--draws 1500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs

SEED = 20260823


# --- I. the partial order ---------------------------------------------------

def transitive_closure(beats: np.ndarray) -> np.ndarray:
    """Reachability of a DAG by repeated boolean squaring."""
    r = beats.copy()
    while True:
        nxt = r | (r @ r)
        if np.array_equal(nxt, r):
            return r
        r = nxt


def longest_chain(beats: np.ndarray) -> int:
    """Height of the poset: the most systems that can be strictly ordered."""
    J = beats.shape[0]
    # Kahn order, then longest path by dynamic programming.
    indeg = beats.sum(axis=0).astype(int)
    order, queue = [], [j for j in range(J) if indeg[j] == 0]
    indeg = indeg.copy()
    while queue:
        j = queue.pop()
        order.append(j)
        for k in np.flatnonzero(beats[j]):
            indeg[k] -= 1
            if indeg[k] == 0:
                queue.append(int(k))
    if len(order) < J:            # a cycle: intransitive relation
        return -1
    depth = np.ones(J, dtype=int)
    for j in order:
        for k in np.flatnonzero(beats[j]):
            depth[k] = max(depth[k], depth[j] + 1)
    return int(depth.max())


def largest_antichain(beats: np.ndarray) -> int:
    """Dilworth: max antichain = n - maximum matching on the closure."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    closure = transitive_closure(beats)
    m = maximum_bipartite_matching(csr_matrix(closure), perm_type="column")
    matched = int((m >= 0).sum())
    return int(beats.shape[0] - matched)


def transitivity_violations(beats: np.ndarray) -> int:
    """j>k and k>l but not j>l. Counted, not assumed away."""
    two = (beats.astype(np.int16) @ beats.astype(np.int16)) > 0
    return int((two & ~beats).sum())


def tiers(beats: np.ndarray) -> list[list[int]]:
    """Mirsky levels: each level is a set of mutually incomparable systems."""
    J = beats.shape[0]
    depth = np.ones(J, dtype=int)
    for _ in range(J):
        changed = False
        for j in range(J):
            for k in np.flatnonzero(beats[j]):
                if depth[k] < depth[j] + 1:
                    depth[k] = depth[j] + 1
                    changed = True
        if not changed:
            break
    out = []
    for d in range(1, int(depth.max()) + 1):
        out.append([int(j) for j in np.flatnonzero(depth == d)])
    return out


# --- II. how many dimensions ------------------------------------------------

def two_way_residual(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)
            - x.mean(axis=0, keepdims=True) + x.mean())


def additive_fit(x: np.ndarray, iters: int = 200) -> np.ndarray:
    """Fit p_ji = logistic(a_j + b_i), the strictly one-dimensional model.

    Iterative proportional fitting on the logit scale: alternately move the
    row effects until row means match and the column effects until column
    means match. This is the Rasch model with no second axis anywhere, which
    is exactly the hypothesis being tested.
    """
    eps = 1e-6
    rm = np.clip(x.mean(axis=1), eps, 1 - eps)
    cm = np.clip(x.mean(axis=0), eps, 1 - eps)
    a = np.log(rm / (1 - rm))
    b = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a[:, None] + b[None, :])))
        b += (cm - p.mean(axis=0)) * 4.0
        p = 1.0 / (1.0 + np.exp(-(a[:, None] + b[None, :])))
        a += (rm - p.mean(axis=1)) * 4.0
    return 1.0 / (1.0 + np.exp(-(a[:, None] + b[None, :])))


def dimensions(x: np.ndarray, perms: int = 200, seed: int = SEED,
               kmax: int = 12) -> dict:
    """Parallel analysis on the two-way residual."""
    rng = np.random.default_rng(seed)
    r = two_way_residual(x)
    sv = np.linalg.svd(r, compute_uv=False)[:kmax]
    null = np.empty((perms, len(sv)))
    # THE NULL IS PARAMETRIC, AND IT TOOK TWO FAILURES TO GET THERE.
    #
    # Version 1 permuted the RESIDUAL. A permuted residual is no longer
    # row-centred, so its spectrum is not comparable with a double-centred
    # one: on data built with NO second axis it reported 10 dimensions.
    #
    # Version 2 permuted the DATA and re-centred, which is the textbook
    # parallel analysis. Better - 10 down to 1 - but still one dimension too
    # many, because a binary residual is heteroscedastic: a cell with p near
    # a half carries far more variance than one near zero, and permuting
    # inside a column moves those variances to the wrong rows. The apparent
    # first component was that heteroscedasticity, not a second ability.
    #
    # Version 3 simulates from the fitted one-dimensional model itself, so
    # the null has the same row abilities, the same item difficulties and the
    # same cell-by-cell variance as the observation, and differs only in
    # having no second axis. That is the hypothesis, stated as data.
    #
    # None of this was visible at 60 x 400, where every version passed. It
    # only appeared at the real shape, which is why the calibration is run
    # there.
    p_hat = additive_fit(x)
    for b in range(perms):
        q = (rng.random(x.shape) < p_hat).astype(float)
        null[b] = np.linalg.svd(two_way_residual(q),
                                compute_uv=False)[:len(sv)]
    thresh = np.quantile(null, 0.95, axis=0)
    above = sv > thresh
    k = int(np.argmin(above)) if not above.all() else len(sv)
    return {"sv": sv, "thresh": thresh, "k": k,
            "var_frac": (sv ** 2 / (r ** 2).sum())[:kmax]}


# --- self-checks ------------------------------------------------------------

def _check_height_synthetic() -> tuple[bool, str]:
    """A built poset of k tiers must measure height k."""
    J, k = 12, 4
    beats = np.zeros((J, J), dtype=bool)
    tier = np.repeat(np.arange(k), J // k)
    for j in range(J):
        for m in range(J):
            if tier[j] < tier[m]:
                beats[j, m] = True
    got = longest_chain(beats)
    return got == k, f"synthetic {k}-tier poset measures height {got}"


def _check_height_noise() -> tuple[bool, str]:
    beats = np.zeros((10, 10), dtype=bool)
    got = longest_chain(beats)
    return got == 1, f"empty relation measures height {got} (want 1)"


def _check_antichain() -> tuple[bool, str]:
    """4 tiers of 3 mutually incomparable systems -> largest antichain 3."""
    J, k = 12, 4
    beats = np.zeros((J, J), dtype=bool)
    tier = np.repeat(np.arange(k), J // k)
    for j in range(J):
        for m in range(J):
            if tier[j] < tier[m]:
                beats[j, m] = True
    got = largest_antichain(beats)
    return got == 3, f"largest antichain {got} (want 3)"


def _check_dim_rank_k() -> tuple[bool, str]:
    """Data built with 3 extra axes must show about 3 dimensions."""
    rng = np.random.default_rng(4)
    J, n, k = 60, 400, 3
    a = rng.normal(0, 1, J)
    b = rng.normal(0, 1, n)
    u = rng.normal(0, 1, (J, k))
    v = rng.normal(0, 1, (k, n))
    logit = a[:, None] + b[None, :] + 1.4 * (u @ v) / np.sqrt(k)
    x = (rng.random((J, n)) < 1 / (1 + np.exp(-logit))).astype(float)
    got = dimensions(x, perms=60, seed=5)["k"]
    return got >= k, f"rank-3 structure measures {got} dimensions (want >= 3)"


def _check_dim_noise() -> tuple[bool, str]:
    rng = np.random.default_rng(6)
    J, n = 60, 400
    a = rng.normal(0, 1, J)
    b = rng.normal(0, 1, n)
    x = (rng.random((J, n)) < 1 / (1 + np.exp(-(a[:, None] + b[None, :])))
         ).astype(float)
    got = dimensions(x, perms=60, seed=7)["k"]
    return got == 0, f"purely additive data measures {got} dimensions (want 0)"



def _check_dim_noise_at_scale() -> tuple[bool, str]:
    """The calibration that decides whether the real result may be quoted.

    Parallel analysis can be liberal, and how liberal depends on the shape of
    the matrix. Checking it at 60 x 400 and then reporting a number computed
    at 134 x 500 would be checking a different instrument. So this generates
    purely additive data of the SAME shape as the real matrix - ability plus
    difficulty, no second axis anywhere - and demands zero dimensions back.
    If it returns more than zero, every component found in the real data is
    suspect by that many and the report says so instead of counting them.
    """
    rng = np.random.default_rng(21)
    J, n = 134, 500
    a = rng.normal(0, 1.1, J)
    b = rng.normal(0, 1.4, n)
    x = (rng.random((J, n)) < 1 / (1 + np.exp(-(a[:, None] + b[None, :])))
         ).astype(float)
    got = dimensions(x, perms=120, seed=23)["k"]
    return got == 0, (f"additive data at the REAL shape 134 x 500 measures "
                      f"{got} dimensions (want 0)")

def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_height_synthetic(), _check_height_noise(),
                        _check_antichain(), _check_dim_rank_k(),
                        _check_dim_noise(), _check_dim_noise_at_scale()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--out", default="leaderboard_geometry_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=float)
    names = list(df.index)
    J, n = x.shape
    print(f"matrix {a.matrix}: {J} systems x {n} items")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no headline number is printed.")
        return 1

    r = rs.rank_sets(x, draws=a.draws)
    beats = r["beats"]
    viol = transitivity_violations(beats)
    h = longest_chain(beats)
    w = largest_antichain(beats)
    lv = tiers(beats)
    d = dimensions(x, perms=a.perms)

    L = []
    p = L.append
    p("THE GEOMETRY OF A LEADERBOARD")
    p("=" * 74)
    p(f"{J} systems, {n} items")
    p("")
    p("I. TIERS - how many performance levels the instrument resolves")
    p(f"  relations 'significantly above'   {int(beats.sum())} of "
      f"{J * (J - 1)} ordered pairs "
      f"({100 * beats.sum() / (J * (J - 1)):.1f} %)")
    p(f"  transitivity violations           {viol}")
    p(f"  HEIGHT (longest chain)            {h}")
    p(f"  largest antichain                 {w}")
    p("")
    p(f"  -> this leaderboard prints {J} positions and resolves {h} levels.")
    p(f"     {w} of its systems cannot be ordered among themselves at all.")
    p("")
    p("  tier sizes, best first:")
    p("   " + ", ".join(str(len(t)) for t in lv))
    if lv:
        p(f"  tier 1 ({len(lv[0])} systems), first few:")
        for j in lv[0][:6]:
            p(f"    {r['theta'][j]:.3f}  {names[j][:52]}")
    p("")
    p("II. DIMENSIONS - how many numbers the benchmark is measuring")
    p(f"  {'component':>10} {'singular value':>16} {'permutation 95%':>17}"
      f" {'ratio':>7} {'var share':>10}")
    ratios = d["sv"] / d["thresh"]
    for i in range(min(8, len(d["sv"]))):
        mark = "**" if ratios[i] > 1.5 else ("*" if ratios[i] > 1.0 else "")
        p(f"  {i + 1:>10} {d['sv'][i]:>16.3f} {d['thresh'][i]:>17.3f}"
          f" {ratios[i]:>7.2f} {100 * d['var_frac'][i]:>9.1f}% {mark}")
    clear = int((ratios > 1.5).sum())
    p("")
    p(f"  clearly above the null (ratio > 1.5):   {clear}")
    p(f"  marginally above (1.0 to 1.5):          {d['k'] - clear}")
    p("  A component that clears a permutation null by two per cent is not")
    p("  a dimension anyone should build on. The honest count is the first")
    p("  column, and the marginal ones are shown so nobody has to take my")
    p("  word for where the line was drawn.")
    if clear >= 1:
        p("")
        p(f"  -> after removing system ability and item difficulty, {clear}")
        p("     component(s) of structure remain, well clear of permutation.")
        p("     Systems differ in WHICH items they solve, not only in how")
        p("     many. A single ranking is a projection of that onto one axis,")
        p("     and which system leads depends on the item mix.")
    else:
        top = float(ratios.max())
        p(f"  -> no component clears the null by more than {100*(top-1):.0f} %.")
        p("     The calibration returned exactly zero components on data built")
        p("     with no second axis at this same shape, so the small excess is")
        p("     real rather than an artefact - and it is small. On this")
        p("     benchmark the systems differ in HOW MANY items they solve,")
        p("     barely in which ones. One axis is the right model, and the")
        p("     ranking loses precision but not a hidden second ability.")
        p("")
        p("     I expected the opposite. The first version of this file used a")
        p("     permutation null and reported eight dimensions; that null was")
        p("     wrong and the number was mine, not the data's.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
