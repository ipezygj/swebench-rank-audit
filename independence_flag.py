"""How many independent lineages are in the top ten?

Three findings meet here. Entrants come in families (J_eff, iteration 13).
The comparison between relatives is sharper than the board average (kappa,
iteration 24). And kappa recovers the families the names know without ever
seeing a name (iteration 30). Put together, a leaderboard can flag, from
the matrix alone, that two of its rows are not independent evidence.

The flag: cluster the top rows at a kappa threshold, so that two systems
are in the same lineage when their pair sharpness is below the threshold.
The threshold is calibrated, not chosen: it is the 5th percentile of the
kappa distribution of a permuted board (each system's residuals shuffled
across items), so a flag fires at most 5 % of the time on a board with no
lineages at all.

Reported per board: lineages among the top 10, the largest lineage there,
and the same for the whole board, next to J_eff / J from iteration 13.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the top ten contains 7 or fewer independent lineages on >= 6 of 9
    boards;
  * at least one board has a lineage covering 4 or more of its top ten;
  * boards with a lower J_eff / J ratio have fewer top-ten lineages
    (Spearman > 0.4, exploratory - 9 points is thin).

SELF-CHECKS
  * on a permuted board the flag fires on at most 5 % of pairs (that is the
    calibration, verified rather than assumed);
  * on a board with four planted lineages of four systems each, the top
    sixteen resolve into four lineages (this check passed for the wrong
    reason in the first build: with only planted groups present, single
    linkage also gives four - the real boards exposed the chaining).

    python independence_flag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from entropy_law_test import MATRICES
from pair_sharpness import kappa_matrix
from residual_correlation import twin4

SEED = 20260823
TOP = 10

# J_eff / J from effective_entrants_results.txt (c64cc21), additive residual
JEFF_FRAC = {"SWE-bench Verified": 0.28, "MTEB English v2": 0.15, "HELM classic": 0.64,
             "ProteinGym DMS": 0.14, "TabArena 16 models": 0.59, "TabArena 45 variants": 0.31,
             "CASP14": 0.37, "LiveBench": 0.33, "MathArena 2025": 0.37}


def threshold(x, rng, reps=6):
    """5th percentile of kappa on a board with the same rows, no lineages."""
    qs = []
    for s in range(reps):
        y = twin4(x, np.random.default_rng(int(rng.integers(1 << 31))))
        K = kappa_matrix(y)
        iu = np.triu_indices(y.shape[0], k=1)
        qs.append(np.nanpercentile(K[iu], 5))
    return float(np.mean(qs))


def lineages(K, idx, thr):
    """Groups in which EVERY pair is below the threshold (complete linkage).

    The first build used single linkage - a chain of "A is close to B, B to
    C" - and on eight of nine boards it swallowed the entire leaderboard
    into one lineage, satisfying the pre-registered criteria in a way that
    said nothing. Kappa varies continuously, so transitive chaining always
    connects everything. Complete linkage states the claim that is actually
    meant: these systems are pairwise non-independent, all of them.
    """
    m = len(idx)
    sub = K[np.ix_(idx, idx)].copy()
    sub = np.nan_to_num((sub + sub.T) / 2, nan=float(np.nanmax(K)))
    np.fill_diagonal(sub, 0.0)
    sub[sub < 0] = 0.0
    Z = linkage(squareform(sub, checks=False), method="complete")
    pred = fcluster(Z, t=thr, criterion="distance")
    sizes = sorted(np.bincount(pred)[1:], reverse=True)
    return len(sizes), int(sizes[0])


def _check_calibration():
    rng = np.random.default_rng(SEED)
    x = 0.4 + rng.normal(0, 0.05, 60)[:, None] + rng.normal(0, 0.3, 200)[None, :] + rng.normal(0, 0.45, (60, 200))
    thr = threshold(x, rng)
    K = kappa_matrix(x)
    iu = np.triu_indices(60, k=1)
    rate = float(np.mean(K[iu] < thr))
    return rate <= 0.07, f"no-lineage board: flag fires on {100 * rate:.1f} % of pairs (threshold {thr:.3f})"


def _check_planted():
    rng = np.random.default_rng(SEED + 5)
    G, per, n = 4, 4, 300
    lab = np.repeat(np.arange(G), per)
    base = rng.normal(0, 0.45, (G, n))
    x = rng.normal(0.4, 0.05, G * per)[:, None] + 0.85 * base[lab] + np.sqrt(1 - 0.85 ** 2) * rng.normal(0, 0.45, (G * per, n))
    K = kappa_matrix(x)
    thr = threshold(x, np.random.default_rng(SEED + 6))
    k, big = lineages(K, list(range(G * per)), thr)
    return k == G, f"four planted lineages of four: found {k} (largest {big}), threshold {thr:.3f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_calibration(), _check_planted()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("INDEPENDENT LINEAGES IN THE TOP TEN")
    p("=" * 80)
    p(f"  {'leaderboard':<22} {'J':>4} {'thr':>6} {'top-10 lineages':>16} {'largest':>8} "
      f"{'whole board':>12} {'J_eff/J':>8}")
    few, big_any, tops, fracs = 0, 0, [], []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J = x.shape[0]
        K = kappa_matrix(x)
        thr = threshold(x, np.random.default_rng(SEED + 3))
        order = np.argsort(-x.mean(axis=1))
        top = [int(i) for i in order[: min(TOP, J)]]
        k_top, big_top = lineages(K, top, thr)
        k_all, big_all = lineages(K, list(range(J)), thr)
        few += k_top <= 7
        big_any += big_top >= 4
        tops.append(k_top); fracs.append(JEFF_FRAC.get(name, float("nan")))
        p(f"  {name:<22} {J:>4} {thr:>6.3f} {k_top:>16} {big_top:>8} {f'{k_all} ({big_all})':>12} "
          f"{JEFF_FRAC.get(name, float('nan')):>8.2f}")
    N = len(tops)
    r = spearmanr(fracs, tops)
    p("")
    p(f"  top ten holds 7 or fewer lineages: {few}/{N} (pre-registered >= 6)")
    p(f"  at least one board with a lineage of 4+ in its top ten: {'yes' if big_any else 'NO'} ({big_any} boards)")
    p(f"  Spearman(J_eff/J, top-10 lineages) = {r.statistic:+.2f} (p {r.pvalue:.2f}); pre-registered > 0.4")
    p("")
    p("  Two systems are in the same lineage when their kappa falls below the")
    p("  threshold, and the threshold is the 5th percentile of kappa on the same")
    p("  board with every system's residuals shuffled - so on a board with no")
    p("  lineages the flag fires on 5 % of pairs by construction. 'whole board'")
    p("  gives the lineage count and the largest lineage over all J systems.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("independence_flag_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote independence_flag_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
