"""Are the items independent? A cluster bootstrap says how much that assumption buys.

Every interval in the standard comes from resampling ITEMS as if they were
independent draws. They are not. SWE-bench instances come from twelve
repositories; MathArena problems come from six competitions; CASP14 targets
come in domains of the same protein; MTEB tasks come in kinds (classification,
retrieval, STS). If items within a cluster share difficulty, resampling them
one at a time understates the uncertainty - the effective item count is
closer to the number of clusters than to the number of items.

Cluster labels, read off the item names and fixed before running:
  SWE-bench Verified  the repository prefix of the instance id
  MathArena 2025      the competition prefix (aime_2025, hmmt_feb_2025, ...)
  CASP14              the target id before the domain suffix (T1030-D1 and
                      T1030-D2 are two domains of one target)
  MTEB English v2     the task-family suffix/prefix in the task name
                      (Classification, Clustering, Retrieval, STS, Reranking,
                      PairClassification, Summarization); anything else is
                      its own cluster

The comparison: the standard's rank sets against the same computation with
the bootstrap resampling CLUSTERS instead of items.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * clustered rank sets are wider on all four boards, and by at least 15 %
    (median width) on >= 3 of 4;
  * tie@1 rises on >= 3 of 4;
  * the reading does not change: no board's #1 vs #2 becomes separable, and
    CASP14's stays separable.

SELF-CHECKS
  * with every item in its own cluster, the clustered bootstrap must
    reproduce the standard's widths within 10 %;
  * on a matrix with a strong planted cluster effect, the clustered version
    must be clearly wider.

    python cluster_bootstrap.py
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs

SEED = 20260823
DRAWS = 1500
ALPHA = 0.05

MTEB_KINDS = ["Classification", "Clustering", "Retrieval", "STS", "Reranking",
              "PairClassification", "Summarization", "BitextMining"]


def clusters_swebench(cols):
    return [c.split("__")[0] for c in cols]


def clusters_matharena(cols):
    return [c.split("__")[0] for c in cols]


def clusters_casp(cols):
    return [c.split("-")[0] for c in cols]


def clusters_mteb(cols):
    out = []
    for c in cols:
        kind = next((k for k in MTEB_KINDS if k.lower() in c.lower()), None)
        out.append(kind or c)
    return out


BOARDS = {
    "SWE-bench Verified": ("swebench_verified_matrix.csv", clusters_swebench),
    "MathArena 2025": ("matharena/matrix.csv", clusters_matharena),
    "CASP14": ("casp/matrix.csv", clusters_casp),
    "MTEB English v2": ("mteb_dated_matrix.csv", clusters_mteb),
}


def cluster_rank_sets(x, labels, alpha=ALPHA, draws=DRAWS, seed=SEED):
    """Simultaneous rank sets with the bootstrap resampling clusters."""
    J, n = x.shape
    uniq = sorted(set(labels))
    idx_of = {u: np.flatnonzero(np.array(labels) == u) for u in uniq}
    G = len(uniq)
    theta = x.mean(axis=1)
    iu = np.triu_indices(J, k=1)
    diff = x[iu[0]] - x[iu[1]]
    d = theta[iu[0]] - theta[iu[1]]
    rng = np.random.default_rng(seed)
    boots = np.empty((draws, len(d)))
    for b in range(draws):
        pick = rng.integers(0, G, G)
        cols = np.concatenate([idx_of[uniq[k]] for k in pick])
        boots[b] = diff[:, cols].mean(axis=1)
    se = boots.std(axis=0, ddof=1)
    se = np.where(se > 0, se, np.inf)
    crit = float(np.quantile(np.max(np.abs(boots - d) / se, axis=1), 1 - alpha))
    beats = np.zeros((J, J), dtype=bool)
    sig = np.abs(d) / se > crit
    for k, (i, j) in enumerate(zip(*iu)):
        if sig[k]:
            if d[k] > 0:
                beats[i, j] = True
            else:
                beats[j, i] = True
    best = 1 + beats.sum(axis=0)
    worst = J - beats.sum(axis=1)
    return {"beats": beats, "best": best, "worst": worst, "G": G, "crit": crit}


def _check_singleton():
    rng = np.random.default_rng(11)
    x = 0.5 + rng.normal(0, 0.07, 25)[:, None] + rng.normal(0, 0.4, (25, 150))
    r = rs.rank_sets(x, draws=600)
    c = cluster_rank_sets(x, [str(i) for i in range(150)], draws=600)
    w1 = np.median(r["worst"] - r["best"] + 1)
    w2 = np.median(c["worst"] - c["best"] + 1)
    return abs(w2 / w1 - 1) <= 0.10, f"every item its own cluster: width {w1:.0f} vs {w2:.0f}"


def _check_planted():
    """Plant a cluster effect that survives differencing.

    The first version gave every item in a cluster the same difficulty shift.
    A shift common to all systems CANCELS in every pairwise difference, so it
    is invisible to a method built on differences - the clustered widths came
    out narrower (10 vs 6) and the check failed for the right reason. What
    has to be planted is a system x cluster interaction: some systems are
    better on some clusters. Then the difference between two systems really
    does vary by cluster, and resampling clusters is the honest interval.
    """
    rng = np.random.default_rng(13)
    J, nk, k = 25, 30, 5
    lab, cols = [], []
    ability = rng.normal(0, 0.06, J)
    for g in range(k):
        # each system has its own strength on this cluster, shared by all its items
        per_system = rng.normal(0, 0.25, J)
        block = (ability + per_system)[:, None] + rng.normal(0, 0.1, (J, nk))
        cols.append(block)
        lab += [f"g{g}"] * nk
    x = np.hstack(cols)
    r = rs.rank_sets(x, draws=600)
    c = cluster_rank_sets(x, lab, draws=600)
    w1 = np.median(r["worst"] - r["best"] + 1)
    w2 = np.median(c["worst"] - c["best"] + 1)
    return w2 > w1, f"planted system x cluster interaction: width {w1:.0f} -> {w2:.0f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_singleton(), _check_planted()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("ARE THE ITEMS INDEPENDENT? ITEM BOOTSTRAP VS CLUSTER BOOTSTRAP")
    p("=" * 92)
    p(f"  {'board':<22} {'n':>5} {'clusters':>9} {'width item':>11} {'cluster':>8} {'ratio':>6} "
      f"{'tie@1 item':>11} {'cluster':>8} {'#1v#2':>18}")
    wider, rose, reading = 0, 0, 0
    for name, (path, fn) in BOARDS.items():
        df = pd.read_csv(path, index_col=0).dropna(axis=0)
        x = df.to_numpy(dtype=float)
        labels = fn(list(df.columns))
        r = rs.rank_sets(x, draws=DRAWS)
        c = cluster_rank_sets(x, labels)
        w1 = float(np.median(r["worst"] - r["best"] + 1))
        w2 = float(np.median(c["worst"] - c["best"] + 1))
        t1 = int((r["best"] == 1).sum())
        t2 = int((c["best"] == 1).sum())
        order = np.argsort(-x.mean(axis=1))
        i1, i2 = int(order[0]), int(order[1])
        s1, s2 = bool(r["beats"][i1, i2]), bool(c["beats"][i1, i2])
        wider += (w2 / w1 - 1) >= 0.15
        rose += t2 > t1
        reading += s1 == s2
        p(f"  {name:<22} {x.shape[1]:>5} {c['G']:>9} {w1:>11.0f} {w2:>8.0f} {w2 / w1:>6.2f} "
          f"{t1:>11} {t2:>8} {('yes/yes' if s1 and s2 else 'no/no' if not s1 and not s2 else 'CHANGED'):>18}")
    p("")
    p(f"  clustered width at least 15 % wider: {wider}/4 (pre-registered >= 3)")
    p(f"  tie@1 rises: {rose}/4 (pre-registered >= 3)")
    p(f"  the #1 vs #2 reading is unchanged: {reading}/4 (pre-registered: all)")
    p("")
    p("  A cluster bootstrap resamples whole repositories, competitions, targets or")
    p("  task kinds. It is the honest interval when items inside a cluster share")
    p("  difficulty, and the gap between the two columns is what the independence")
    p("  assumption was buying.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("cluster_bootstrap_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote cluster_bootstrap_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
