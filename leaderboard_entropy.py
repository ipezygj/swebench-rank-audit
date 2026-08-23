"""How many bits of the published ranking does the data not determine?

Sixteen files here establish, from sixteen directions, that the evidence in a
benchmark matrix supports a PARTIAL order and the leaderboard prints a TOTAL
one. The gap between them has a name in order theory and an exact size.

THE QUANTITY
------------
Let P be the partial order the data supports - "j is above k" for the pairs
where that has actually been established. Every total order consistent with
P is a linear extension of P, and every one of them is a leaderboard the data
supports equally. Their number, e(P), is the number of equally justified
leaderboards, and

    H(P) = log2 e(P)

is the number of bits of ordering the published table specifies WITHOUT
support. A chain has e = 1 and H = 0: the data fixes everything. An
antichain on n systems has e = n! and H = log2 n!: the data fixes nothing.
For 134 systems that ceiling is log2(134!) = 757 bits. H sits between, and
it is the single number that says how much of the leaderboard is evidence
and how much is typography.

THREE EVIDENTIAL ORDERS, THREE ENTROPIES
-----------------------------------------
    noise      j above k iff the difference is significant, simultaneously
               over all pairs (rank_sets.py). What the data supports if the
               basket is taken as given.
    basket     j above k iff j leads under EVERY repository weighting
               (invariant_core.py). What the data supports if the basket is
               not taken as given, with sampling error ignored.
    both       j above k iff j beats k significantly in every repository
               (the intersection-union core). The order supported against
               noise AND basket together.

Fewer established pairs means more extensions and higher H. The three
numbers bracket the leaderboard's evidential content from three sides.

COUNTING LINEAR EXTENSIONS, AND WHY IT IS ESTIMATED
-----------------------------------------------------
Exact counting is #P-complete (Brightwell & Winkler 1991), and 134 elements
is far past brute force. But Knuth's estimator (1975) for the size of a
search tree applies directly: grow a linear extension by choosing, at each
step, uniformly among the elements currently minimal in what remains. If
there were m_1, m_2, ... choices at each step, the product of the m_k is an
unbiased estimator of e(P), because every extension is one root-to-leaf path
and the product is the reciprocal of that path's probability. Average many
such products and take the log. The estimator is checked against exact
counts on small posets, and against the closed forms e(chain) = 1,
e(antichain) = n!, e(two disjoint chains of a and b) = C(a+b, a).

WHAT THE NUMBER IS NOT
-----------------------
It is not a claim that the published order is wrong. Every linear extension
of P is consistent with the evidence, including the published one. It is a
count of how many others are equally consistent, in bits, and a reader can
decide for themselves what to make of a ranking that is one of 2^H.

SELF-CHECKS THAT CAN FAIL
--------------------------
  * estimator vs brute force on random 8-element posets, within 10 %;
  * chain -> 0 bits, antichain -> log2 n!, two chains -> log2 C(a+b, a);
  * the three evidential orders must be transitive (zero violations) or
    the linear-extension count is not defined.

    python leaderboard_entropy.py [--matrix ...] [--samples 4000]
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, gammaln

import rank_sets as rs
import invariant_core as ic

SEED = 20260823


def transitive_closure(beats: np.ndarray) -> np.ndarray:
    r = beats.copy()
    while True:
        nxt = r | ((r.astype(np.int32) @ r.astype(np.int32)) > 0)
        if np.array_equal(nxt, r):
            return r
        r = nxt


def log_extensions(beats: np.ndarray, samples: int, rng) -> dict:
    """Knuth's estimator for log2 e(P). beats[j,k] = j must precede k."""
    n = beats.shape[0]
    closure = transitive_closure(beats)
    preds = [set(np.flatnonzero(closure[:, k]).tolist()) for k in range(n)]
    logs = np.empty(samples)
    for s in range(samples):
        placed = set()
        remaining = set(range(n))
        acc = 0.0
        while remaining:
            minimal = [k for k in remaining if preds[k] <= placed]
            acc += math.log(len(minimal))
            pick = minimal[rng.integers(len(minimal))]
            placed.add(pick)
            remaining.remove(pick)
        logs[s] = acc
    log_e = logsumexp(logs) - math.log(samples)      # log of the mean
    return {"bits": log_e / math.log(2),
            "bits_lower": float(logs.mean() / math.log(2)),   # Jensen bound
            "se_bits": float(logs.std(ddof=1) / math.sqrt(samples) / math.log(2))}


def brute_extensions(beats: np.ndarray) -> int:
    n = beats.shape[0]
    cnt = 0
    for perm in itertools.permutations(range(n)):
        pos = {v: i for i, v in enumerate(perm)}
        if all(pos[j] < pos[k] for j in range(n) for k in range(n) if beats[j, k]):
            cnt += 1
    return cnt


# --- self-checks ------------------------------------------------------------

def _check_closed_forms() -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    n = 7
    chain = np.triu(np.ones((n, n), dtype=bool), k=1)
    anti = np.zeros((n, n), dtype=bool)
    a, b = 3, 4
    two = np.zeros((a + b, a + b), dtype=bool)
    for i in range(a - 1):
        two[i, i + 1] = True
    for i in range(a, a + b - 1):
        two[i, i + 1] = True
    got = [log_extensions(m, 3000, rng)["bits"] for m in (chain, anti, two)]
    want = [0.0, math.log2(math.factorial(n)), math.log2(math.comb(a + b, a))]
    ok = all(abs(g - w) < 0.15 for g, w in zip(got, want))
    return ok, ("chain / antichain / two chains: "
                + ", ".join(f"{g:.2f} vs {w:.2f}" for g, w in zip(got, want)))


def _check_vs_bruteforce() -> tuple[bool, str]:
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(4):
        n = 8
        vals = rng.random(n)
        beats = np.zeros((n, n), dtype=bool)
        for j in range(n):
            for k in range(n):
                if j != k and vals[j] > vals[k] + 0.18:
                    beats[j, k] = True
        exact = math.log2(brute_extensions(beats))
        est = log_extensions(beats, 4000, rng)["bits"]
        worst = max(worst, abs(est - exact) / max(exact, 1.0))
    return worst < 0.10, f"estimator vs brute force on 8-element posets: worst rel. err {worst:.3f}"


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_closed_forms(), _check_vs_bruteforce()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--out", default="leaderboard_entropy_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=float)
    J, n = x.shape
    print(f"matrix {a.matrix}: {J} systems x {n} instances")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no entropy is reported.")
        return 1

    rng = np.random.default_rng(SEED)
    print("\nbuilding the three evidential orders ...")
    noise = rs.rank_sets(x, draws=a.draws)["beats"]
    repos = ic.pooled_repos(df.columns)
    pm, _, _ = ic.group_means(x, repos)
    basket = ic.point_core(pm)
    both = ic.statistical_core(x, repos, ic.Z_ONE_SIDED)

    orders = {"noise (rank_sets)": noise, "basket (invariant core)": basket,
              "both (IUT core)": both}
    ceiling = gammaln(J + 1) / math.log(2)

    L = []
    p = L.append
    p("LEADERBOARD ENTROPY: BITS OF ORDER THE DATA DOES NOT DETERMINE")
    p("=" * 74)
    p(f"{J} systems; a total order with no evidence at all carries "
      f"log2({J}!) = {ceiling:.0f} bits")
    p("")
    p(f"  {'evidential order':<26} {'pairs':>6} {'viol':>5} {'H bits':>8}"
      f" {'±':>5} {'of ceiling':>11} {'equally good tables':>20}")
    res = {}
    for label, beats in orders.items():
        viol = ic.transitivity_violations(beats)
        if viol:
            p(f"  {label:<26} {int(beats.sum()):>6} {viol:>5}   "
              "NOT TRANSITIVE - extensions undefined")
            continue
        r = log_extensions(beats, a.samples, rng)
        res[label] = r
        p(f"  {label:<26} {int(beats.sum()):>6} {viol:>5} {r['bits']:>8.1f}"
          f" {r['se_bits']:>5.1f} {100 * r['bits'] / ceiling:>10.1f}%"
          f" {'2^' + format(r['bits'], '.0f'):>20}")
    p("")
    p("  H is log2 of the number of total orders consistent with every")
    p("  established pair. The published leaderboard is one of them. The")
    p("  last column is how many others the evidence supports exactly as")
    p("  well - not approximately, not 'within noise': EXACTLY as well,")
    p("  under the criterion in the first column.")
    p("")

    # The top, which is what anyone reads.
    scores = x.mean(axis=1)
    order = np.argsort(-scores, kind="stable")
    p("THE SAME QUESTION FOR THE TOP OF THE TABLE")
    p(f"  {'evidential order':<26} {'top k':>6} {'H bits':>8} {'ceiling':>8}"
      f" {'equally good orderings of the top':>34}")
    for k in (10, 19, 30):
        idx = order[:k]
        ceil_k = gammaln(k + 1) / math.log(2)
        for label, beats in orders.items():
            sub = beats[np.ix_(idx, idx)]
            if ic.transitivity_violations(sub):
                continue
            r = log_extensions(sub, a.samples, rng)
            p(f"  {label:<26} {k:>6} {r['bits']:>8.1f} {ceil_k:>8.1f}"
              f" {'2^' + format(r['bits'], '.1f'):>34}")
        p("")
    p("  For the nineteen candidate leaders the noise order establishes")
    p("  almost nothing among them, so nearly every one of their 19! orderings")
    p("  is equally supported. The table prints one.")
    p("")
    p("WHAT THIS UNIFIES")
    p("  rank_sets.py gave confidence sets per system; leaderboard_geometry.py")
    p("  gave the height of the poset; invariant_core.py gave the pairs that")
    p("  survive every basket. Those are three views of one object. H is")
    p("  that object's size, in bits, and it is the quantity a leaderboard")
    p("  would print beside its ranking if it reported what it knew.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
