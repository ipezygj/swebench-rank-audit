"""What KIND of partial order is a leaderboard?

Every rank set in this repository is read off a relation: j beats k when their
difference clears a multiplicity-corrected threshold. That relation is
transitive on all nine boards - measured, closure adds zero edges - so it is a
strict partial order. But "partial order" is the widest possible description
and says almost nothing.

A threshold on a single latent score produces something far narrower. Give each
system an interval and declare j > k when j's interval lies entirely above k's:
that is an INTERVAL ORDER, characterised by containing no induced 2+2 (two
disjoint two-chains with no relation between them). Make every interval the
same width - one constant threshold for the whole board - and it is a SEMIORDER,
which additionally contains no induced 3+1 (a three-chain plus an element
incomparable to all of it).

So the shape of the relation is a measurement of the noise that produced it. A
board whose beats relation is a semiorder was resolved as if by one threshold. A
board that is an interval order but not a semiorder needed system-specific
widths. A board that is neither has pairwise noise that no interval
representation can reproduce, and its rank sets are not "system i sits in a
band" in any geometric sense - which is exactly how every report card in this
repository invites the reader to picture them.

Rank sets are built from per-PAIR standard errors, which vary across a board by
more than an order of magnitude, so there is no reason a priori for any of this
to hold. That is what makes it worth measuring.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  no board's beats relation is a semiorder. Per-pair sigma varies too much
      for one constant width to reproduce the relation.
  P2  at least 1 board IS an interval order. Interval orders permit
      system-specific widths, which is closer to what per-pair sigma gives.
  P3  the violation rate - the share of 4-subsets that are induced 2+2 -
      correlates with how much per-pair sigma varies across a board:
      Spearman over boards >= +0.5 between the 2+2 rate and the
      interquartile spread of sigma divided by its median.
  P4  HELM classic is the worst offender on P3's spread measure. It has n = 10
      items, so its pairwise standard errors are the least stable of any board.

  What a miss on P1 would mean: the whole relation collapses to one number, a
  single threshold on the observed scores, and everything built on per-pair
  standard errors is more machinery than the data needs.

SELF-CHECKS (no table if any fails)
  * the fast test must agree with direct enumeration of induced 2+2 and 3+1
    patterns on 300 random posets of 7 and 8 elements. The fast test uses a
    set-inclusion characterisation and the slow one looks at every 4-subset;
    they are different code answering the same question;
  * a poset built from a CONSTANT threshold on random scores must come back a
    semiorder every time - that is what a semiorder is, and a test that cannot
    confirm it is not testing;
  * a poset built from wildly varying per-pair thresholds must come back NOT an
    interval order at least sometimes, or the test cannot fire;
  * at least 8 boards must be measured.

    python order_shape.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import rank_sets as rs
from entropy_law_test import MATRICES

SEED = 20260825


# --- the fast test ----------------------------------------------------------

def _chain_by_inclusion(sets: list[set]) -> bool:
    """Are these sets linearly ordered by inclusion?

    Sorting by size makes it O(J log J) comparisons instead of O(J^2): if the
    family is a chain, each set must contain the one below it in size order.
    """
    order = sorted(range(len(sets)), key=lambda i: len(sets[i]))
    for a, b in zip(order, order[1:]):
        if not sets[a] <= sets[b]:
            return False
    return True


def shape(beats: np.ndarray) -> tuple[bool, bool]:
    """(is interval order, is semiorder), by set-inclusion characterisation.

    An interval order is exactly a poset whose strict down-sets form a chain
    under inclusion; a semiorder is one whose down-sets AND up-sets both do.
    Both are validated against direct 2+2 / 3+1 enumeration in the self-checks
    rather than taken on trust.
    """
    J = beats.shape[0]
    down = [set(np.flatnonzero(beats[:, k]).tolist()) for k in range(J)]
    up = [set(np.flatnonzero(beats[k, :]).tolist()) for k in range(J)]
    iv = _chain_by_inclusion(down)
    return iv, iv and _chain_by_inclusion(up)


# --- the slow test, for validating the fast one -----------------------------

def enumerate_patterns(beats: np.ndarray) -> tuple[int, int]:
    """Counts of induced 2+2 and 3+1 among all 4-subsets. O(J^4)."""
    J = beats.shape[0]
    b = beats
    n22 = n31 = 0
    for q in itertools.combinations(range(J), 4):
        rel = {(i, j) for i in q for j in q if b[i, j]}
        if not rel:
            continue
        # 2+2: two disjoint covering pairs, nothing else related
        if len(rel) == 2:
            (a, c), (e, g) = tuple(rel)
            if len({a, c, e, g}) == 4:
                n22 += 1
        # 3+1: a three-chain (3 relations by transitivity) and an isolated point
        if len(rel) == 3:
            pts = {i for pair in rel for i in pair}
            if len(pts) == 3 and len(set(q) - pts) == 1:
                n31 += 1
    return n22, n31


def _check_fast_against_slow(rng) -> tuple[bool, str]:
    bad_iv = bad_se = 0
    for _ in range(300):
        J = int(rng.integers(7, 9))
        theta = rng.normal(size=J)
        if rng.random() < 0.5:
            thr = float(rng.uniform(0.3, 1.2))
            b = (theta[:, None] - theta[None, :]) > thr
        else:
            t = rng.uniform(0.1, 2.0, (J, J))
            t = np.triu(t) + np.triu(t, 1).T
            b = (theta[:, None] - theta[None, :]) > t
        b &= ~np.eye(J, dtype=bool)
        iv, se = shape(b)
        n22, n31 = enumerate_patterns(b)
        bad_iv += iv != (n22 == 0)
        bad_se += se != (n22 == 0 and n31 == 0)
    return bad_iv == 0 and bad_se == 0, (
        f"300 random posets of 7-8 elements: interval-order disagreements "
        f"{bad_iv}, semiorder disagreements {bad_se}")


def _check_constant_threshold(rng) -> tuple[bool, str]:
    bad = 0
    for _ in range(200):
        J = int(rng.integers(10, 25))
        theta = rng.normal(size=J)
        b = (theta[:, None] - theta[None, :]) > 0.6
        if not shape(b)[1]:
            bad += 1
    return bad == 0, f"200 constant-threshold posets, not recognised as semiorders: {bad}"


def _check_can_fire(rng) -> tuple[bool, str]:
    fired = 0
    for _ in range(200):
        J = int(rng.integers(10, 25))
        theta = rng.normal(size=J)
        t = rng.uniform(0.05, 3.0, (J, J))
        t = np.triu(t) + np.triu(t, 1).T
        b = (theta[:, None] - theta[None, :]) > t
        b &= ~np.eye(J, dtype=bool)
        if not shape(b)[0]:
            fired += 1
    return fired > 0, f"200 varying-threshold posets, {fired} came back not an interval order"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print("self-checks ...")
    ok1, m1 = _check_fast_against_slow(np.random.default_rng(SEED + 1))
    print(f"  [{'ok  ' if ok1 else 'FAIL'}] {m1}")
    ok2, m2 = _check_constant_threshold(np.random.default_rng(SEED + 2))
    print(f"  [{'ok  ' if ok2 else 'FAIL'}] {m2}")
    ok3, m3 = _check_can_fire(np.random.default_rng(SEED + 3))
    print(f"  [{'ok  ' if ok3 else 'FAIL'}] {m3}")

    rows = []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        r = rs.rank_sets(x)
        b = r["beats"]
        J, n = x.shape
        iv, se = shape(b)
        iu = np.triu_indices(J, k=1)
        sig = r["sigma"][iu]
        sig = sig[np.isfinite(sig) & (sig > 0)]
        spread = float((np.percentile(sig, 75) - np.percentile(sig, 25))
                       / np.median(sig))
        rows.append({"name": name, "J": J, "n": n, "iv": iv, "se": se,
                     "spread": spread, "edges": int(b.sum())})
        print(f"  {name:<22} interval order {str(iv):<5} semiorder {se}")

    ok4 = len(rows) >= 8
    print(f"  [{'ok  ' if ok4 else 'FAIL'}] {len(rows)} boards measured (need >= 8)")

    if not (ok1 and ok2 and ok3 and ok4):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    # the 2+2 rate, exactly where the board is small enough and sampled otherwise
    rng = np.random.default_rng(SEED + 9)
    for r0 in rows:
        name = r0["name"]
        path = MATRICES[name]
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        b = rs.rank_sets(x)["beats"]
        J = b.shape[0]
        hits = trials = 0
        for _ in range(200000):
            q = rng.choice(J, 4, replace=False)
            sub = b[np.ix_(q, q)]
            rel = int(sub.sum())
            trials += 1
            if rel == 2:
                ii, jj = np.nonzero(sub)
                if len(set(ii.tolist() + jj.tolist())) == 4:
                    hits += 1
        r0["rate22"] = hits / trials

    L = []
    p = L.append
    p("WHAT KIND OF PARTIAL ORDER IS A LEADERBOARD?")
    p("=" * 92)
    p("  A threshold on one latent score gives an interval order; ONE constant")
    p("  threshold gives a semiorder. Rank sets are built from per-pair standard")
    p("  errors, so neither is guaranteed.")
    p("")
    p(f"  {'board':<22}{'J':>5}{'n':>6}{'interval order':>16}{'semiorder':>11}"
      f"{'2+2 rate':>11}{'sigma IQR/med':>15}")
    for r0 in rows:
        p(f"  {r0['name']:<22}{r0['J']:>5}{r0['n']:>6}"
          f"{('yes' if r0['iv'] else 'NO'):>16}{('yes' if r0['se'] else 'NO'):>11}"
          f"{r0['rate22']:>11.5f}{r0['spread']:>15.3f}")
    p("")
    n_iv = sum(1 for r0 in rows if r0["iv"])
    n_se = sum(1 for r0 in rows if r0["se"])
    rho, pv = spearmanr([r0["rate22"] for r0 in rows], [r0["spread"] for r0 in rows])
    worst = max(rows, key=lambda r0: r0["spread"])
    p(f"  P1  boards that are semiorders: {n_se} of {len(rows)}          "
      f"pre-registered = 0:    {'HIT' if n_se == 0 else 'MISS'}")
    p(f"  P2  boards that are interval orders: {n_iv} of {len(rows)}     "
      f"pre-registered >= 1:   {'HIT' if n_iv >= 1 else 'MISS'}")
    p(f"  P3  Spearman(2+2 rate, sigma spread) = {rho:+.3f} (p {pv:.3f})   "
      f"pre-registered >= +0.5: {'HIT' if rho >= 0.5 else 'MISS'}")
    p(f"  P4  widest sigma spread: {worst['name']} ({worst['spread']:.3f})   "
      f"pre-registered HELM classic: "
      f"{'HIT' if worst['name'] == 'HELM classic' else 'MISS'}")
    p("")
    p("  The 2+2 rate is estimated from 200 000 random 4-subsets per board, not")
    p("  enumerated: C(181,4) is 43 million and the estimate is what is needed.")
    p("")
    p("  Why this matters for how a report card reads. Every rank set in this")
    p("  repository is presented as a band - system i sits somewhere between")
    p("  rank a and rank b. A band is an interval, and a relation that is not an")
    p("  interval order cannot be produced by any assignment of intervals to")
    p("  systems. Where the answer below is NO, the picture the reader is being")
    p("  handed is not available, however sound the arithmetic underneath it.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("order_shape_results.txt").write_text(text + chr(10), encoding="utf-8",
                                               newline=chr(10))
    print(chr(10) + "wrote order_shape_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
