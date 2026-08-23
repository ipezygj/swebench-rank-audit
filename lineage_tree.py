"""The chase model with a family tree, which is what sibling_chase was missing.

sibling_chase.py made each chaser inherit the CURRENT RECORD-HOLDER's item
behaviour. It installed the right kappa on four boards of five but still
under-predicted P, and the diagnosis was in the results: on MTEB, where the
record changes hands often, the chain dilutes - each new chaser inherits
from a different predecessor, so nobody ends up sharing much with anybody.

Real entrants do not inherit from whoever holds the record. They inherit
from their own lineage: a lab's next model, a scaffold rebuilt on a newer
LLM, a method scaled up. This model gives the field a tree.

    * every entrant picks a PARENT from the systems already present,
      chosen among the top `k` by score (a new system is built on
      something that is already good, not on the median of the field);
    * its ability is the parent's plus Exp(lambda);
    * its residual is rho * parent residual + sqrt(1 - rho^2) * fresh,
      so it is sharp with its parent and, through the parent, with its
      siblings - which is what a family is.

rho comes from the board's measured frontier kappa (kappa^2 = 1 - rho),
lambda is fitted to total climb, k is fixed at 5 for every board. A and P
remain predictions.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * P within +-15 points on >= 3 of 5 boards (iid chase 2, bootstrap 1,
    sibling chase 3 - this must at least match the sibling chase);
  * the simulated frontier kappa is within 0.10 of the real one on >= 4 of 5,
    and specifically MTEB's simulated kappa falls below 0.70 (the sibling
    chain gave 0.88 against a real 0.53 - the dilution this model exists to
    fix);
  * A within +-30 % on >= 3 of 5.

SELF-CHECKS
  * with rho = 0 the tree reproduces the plain chase model's P within 6
    points (the tree changes abilities, not noise, when rho is zero);
  * the generated field's lineage structure is detectable: kappa within a
    simulated subtree is below kappa across subtrees.

    python lineage_tree.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from evidence_trajectory import load
from sota_audit import advances
from sota_twin import audit, fit_drift, sigma_p_of, synth_dates, SEED
from step_sizes import steps_u
from chase_model import BOARDS, chase_field, board_stats
from pair_sharpness import kappa_matrix
from sibling_chase import frontier_kappa

REPS = 8
TOPK = 5


def tree_field(J, n, dates, lam, sigma_item, rho, start, rng, k=TOPK):
    order = np.argsort(dates, kind="stable")
    ability = np.empty(J)
    resid = np.empty((J, n))
    placed = []
    for pos, idx in enumerate(order):
        fresh = rng.normal(0, sigma_item, n)
        if not placed:
            ability[idx] = start
            resid[idx] = fresh
        else:
            pool = sorted(placed, key=lambda j: -ability[j])[:k]
            par = int(rng.choice(pool))
            ability[idx] = ability[par] + rng.exponential(lam)
            resid[idx] = rho * resid[par] + math.sqrt(max(1 - rho ** 2, 0.0)) * fresh
        placed.append(int(idx))
    return ability[:, None] + resid


def climb_of(x, dates):
    sc = x.mean(axis=1)
    o = np.argsort(dates, kind="stable")
    return float(sc.max() - sc[o[0]])


def fit_lambda(J, n, dates, sigma_item, rho, start, target, rng, iters=16):
    lo, hi = 1e-6, max(target, 1e-3) * 2
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        cs = [climb_of(tree_field(J, n, dates, mid, sigma_item, rho, start,
                                  np.random.default_rng(int(rng.integers(1 << 31)))), dates) for _ in range(3)]
        if np.mean(cs) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def _check_rho_zero():
    rng = np.random.default_rng(SEED)
    J, n = 70, 200
    dates = synth_dates("2023-01-01", np.sort(rng.integers(0, 700, J)))
    pool = np.array([0.4])
    Pa = np.nanmean([audit(tree_field(J, n, dates, 0.01, 0.45, 0.0, 0.4, np.random.default_rng(10 + s)), dates, 20 + s)["P"]
                     for s in range(10)])
    Pb = np.nanmean([audit(chase_field(J, n, dates, 1.0, 0.01, 0.45, pool, np.random.default_rng(30 + s)), dates, 40 + s)["P"]
                     for s in range(10)])
    return abs(Pa - Pb) < 0.06, f"rho = 0: tree P {100 * Pa:.0f} % vs chase P {100 * Pb:.0f} %"


def _check_subtree():
    rng = np.random.default_rng(SEED + 2)
    J, n = 60, 300
    dates = synth_dates("2023-01-01", np.arange(J) * 5)
    x = tree_field(J, n, dates, 0.01, 0.45, 0.8, 0.4, rng)
    K = kappa_matrix(x)
    iu = np.triu_indices(J, k=1)
    # neighbours in arrival order are likelier to be relatives in this model
    near = np.array([K[i, i + 1] for i in range(J - 1)])
    far = np.array([K[i, j] for i, j in zip(*iu) if abs(i - j) > J // 3])
    return np.nanmedian(near) < np.nanmedian(far), \
        f"tree structure detectable: kappa near {np.nanmedian(near):.2f} vs far {np.nanmedian(far):.2f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_rho_zero(), _check_subtree()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("THE CHASE MODEL WITH A FAMILY TREE")
    p("=" * 88)
    p(f"  {'board':<20} {'kap real':>8} {'rho':>5} {'lambda':>8} | {'A real':>6} {'A sim':>6} | "
      f"{'P real':>6} {'P tree':>7} {'P sib':>6} | {'kap sim':>7}")
    okP, okK, okA = 0, 0, 0
    mteb_kap = None
    SIB = {"SWE-bench Verified": 0.16, "SWE-bench Lite": 0.22, "MTEB English v2": 0.17,
           "LiveBench": 0.20, "ProteinGym DMS": 0.34}          # sibling_chase_results.txt
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        J, n = x.shape
        sp = sigma_p_of(x)
        _, _, _, si = fit_drift(x, dates, sp)
        sc = x.mean(axis=1)
        o = np.argsort(dates, kind="stable")
        start = float(sc[o[0]])
        climb = climb_of(x, dates)
        real = board_stats(x, dates, SEED)
        kap_real = frontier_kappa(x, dates)
        rho = float(min(max(1 - kap_real ** 2, 0.0), 0.98))
        lam = fit_lambda(J, n, dates, si, rho, start, climb, np.random.default_rng(SEED + 5))
        sims = [tree_field(J, n, dates, lam, si, rho, start, np.random.default_rng(SEED + 1500 + s)) for s in range(REPS)]
        st = [board_stats(y, dates, SEED + 80 * s) for s, y in enumerate(sims)]
        A = float(np.mean([s["A"] for s in st])); P = float(np.nanmean([s["P"] for s in st]))
        kap_sim = float(np.nanmedian([frontier_kappa(y, dates) for y in sims]))
        if name.startswith("MTEB"):
            mteb_kap = kap_sim
        okP += abs(P - real["P"]) <= 0.15
        okK += abs(kap_sim - kap_real) <= 0.10
        okA += abs(A / real["A"] - 1) <= 0.30
        p(f"  {name:<20} {kap_real:>8.2f} {rho:>5.2f} {lam:>8.4f} | {real['A']:>6d} {A:>6.1f} | "
          f"{100 * real['P']:>5.0f}% {100 * P:>6.0f}% {100 * SIB.get(name, float('nan')):>5.0f}% | {kap_sim:>7.2f}")
    N = len(BOARDS)
    p("")
    p(f"  P within 15 points: {okP}/{N} (pre-registered >= 3; sibling chase 3)")
    p(f"  simulated frontier kappa within 0.10: {okK}/{N} (pre-registered >= 4)")
    p(f"  MTEB simulated kappa below 0.70: {'yes' if mteb_kap is not None and mteb_kap < 0.70 else 'NO'}"
      f" ({mteb_kap:.2f}; sibling chain gave 0.88 against a real 0.53)")
    p(f"  A within 30 %: {okA}/{N} (pre-registered >= 3)")
    p("")
    p("  Every entrant descends from one of the top five systems present when it")
    p("  arrives, inheriting its item behaviour to degree rho and its ability plus")
    p("  an exponential step. rho is set from the board's own frontier kappa;")
    p("  lambda is fitted to total climb; the parent pool is 5 on every board.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("lineage_tree_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote lineage_tree_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
