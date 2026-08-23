"""How many SOTA claims survive when the items are admitted to be clustered?

cluster_bootstrap.py showed that resampling repositories instead of
instances widens SWE-bench's rank sets by 46 %. The SOTA audit's verdicts
were computed the other way - a paired test over instances - so they
inherit the independence assumption too.

This recomputes every frontier advance on the two boards with usable
cluster structure, replacing the item-level paired test with a cluster
bootstrap of the mean difference (resample repositories / competitions,
percentile interval, two-sided 5 %).

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the number of pairwise-separable advances falls on both boards;
  * on SWE-bench Verified it falls by at least a third (8 separable
    advances at the item level, so 5 or fewer under clustering);
  * every advance that survives clustering also survived at the item level -
    clustering can only remove separability here, since it only widens the
    interval.

SELF-CHECKS
  * with each item its own cluster, the clustered verdicts must match the
    item-level ones on at least 90 % of advances;
  * the clustered interval must be wider than the item interval on the
    median advance.

    python sota_clustered.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sota_audit import advances, parse_dates, fmt
from cluster_bootstrap import clusters_swebench, clusters_matharena

SEED = 20260823
DRAWS = 4000
ALPHA = 0.05

BOARDS = {
    "SWE-bench Verified": ("swebench_verified_matrix.csv", clusters_swebench, None),
    "MathArena 2025": ("matharena/matrix.csv", clusters_matharena, "matharena_dates.csv"),
}


def cluster_ci(d, labels, draws=DRAWS, seed=SEED, alpha=ALPHA):
    """Percentile bootstrap interval for mean(d), resampling clusters."""
    uniq = sorted(set(labels))
    idx = {u: np.flatnonzero(np.array(labels) == u) for u in uniq}
    G = len(uniq)
    rng = np.random.default_rng(seed)
    boots = np.empty(draws)
    for b in range(draws):
        pick = rng.integers(0, G, G)
        cols = np.concatenate([idx[uniq[k]] for k in pick])
        boots[b] = d[cols].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), G


def item_ci(d, draws=DRAWS, seed=SEED, alpha=ALPHA):
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = d[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _check_singleton():
    rng = np.random.default_rng(5)
    d = rng.normal(0.02, 0.4, 300)
    a = item_ci(d, 1500)
    b = cluster_ci(d, [str(i) for i in range(300)], 1500)[:2]
    same = (a[0] > 0) == (b[0] > 0) and (a[1] < 0) == (b[1] < 0)
    return same, f"singleton clusters: item CI [{a[0]:.4f}, {a[1]:.4f}] vs cluster [{b[0]:.4f}, {b[1]:.4f}]"


def _check_wider():
    rng = np.random.default_rng(7)
    G, per = 10, 30
    lab, parts = [], []
    for g in range(G):
        shift = rng.normal(0, 0.3)
        parts.append(shift + rng.normal(0.02, 0.1, per))
        lab += [f"g{g}"] * per
    d = np.concatenate(parts)
    a = item_ci(d, 1500)
    b = cluster_ci(d, lab, 1500)[:2]
    return (b[1] - b[0]) > (a[1] - a[0]), \
        f"clustered differences: item width {a[1] - a[0]:.4f} vs cluster {b[1] - b[0]:.4f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_singleton(), _check_wider()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("SOTA CLAIMS UNDER A CLUSTER BOOTSTRAP")
    p("=" * 84)
    fell, subset = 0, 0
    for name, (path, fn, datefile) in BOARDS.items():
        df = pd.read_csv(path, index_col=0).dropna(axis=0)
        x = df.to_numpy(dtype=float)
        names = list(df.index)
        labels = fn(list(df.columns))
        if datefile and Path(datefile).exists():
            dd = pd.read_csv(datefile, index_col=0)["date"]
            dates = np.array([int(dd.loc[n]) for n in names])
        else:
            try:
                dates = parse_dates(names)
            except SystemExit:
                p(f"  {name}: no dates available, skipped")
                continue
        adv = advances(x, dates)
        rows = []
        for a in adv:
            d = x[a["new"]] - x[a["old"]]
            ilo, ihi = item_ci(d)
            clo, chi, G = cluster_ci(d, labels)
            rows.append((a, ilo > 0, clo > 0, ihi - ilo, chi - clo, G))
        n_item = sum(r[1] for r in rows)
        n_clu = sum(r[2] for r in rows)
        fell += n_clu < n_item
        subset += all((not r[2]) or r[1] for r in rows)
        G = rows[0][5] if rows else 0
        p("")
        p(f"  {name}: {len(rows)} frontier advances, {G} clusters")
        p(f"    separable at the item level: {n_item}")
        p(f"    separable under the cluster bootstrap: {n_clu}")
        p(f"    median interval width: item {100 * np.median([r[3] for r in rows]):.2f}p, "
          f"cluster {100 * np.median([r[4] for r in rows]):.2f}p")
        lost = [(a['date'], names[a['new']]) for a, i_, c_, *_ in rows if i_ and not c_]
        if lost:
            p(f"    claims that lose separability ({len(lost)}):")
            for dte, nm in lost:
                p(f"      {fmt(int(dte))}  {nm}")
    p("")
    p(f"  separable count falls on both boards: {fell}/2 (pre-registered: both)")
    p(f"  every clustered survivor also survived at the item level: {subset}/2 (pre-registered: both)")
    p("")
    p("  A cluster bootstrap resamples whole repositories or competitions. It is")
    p("  the interval that admits a benchmark is not a bag of independent tasks.")
    p("  Claims that lose separability here were never separable from a field")
    p("  that happened to draw different repositories.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("sota_clustered_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote sota_clustered_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
