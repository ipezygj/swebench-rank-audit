"""What did dropping the incomplete rows do?

Every tool here starts with `.dropna(axis=0)`: a system that did not run on
every item is removed. That is the only way to keep the comparison paired,
and it is also a selection - the dropped systems are not a random subset,
they are the ones whose owners ran a cheaper evaluation, and on MTEB alone
it removes 150 of 331 rows.

Two questions, neither asked all evening:
  1. how many systems does each board lose, and are the survivors better
     than the dropped ones on the items they DID run?
  2. does the reading change if the dropped systems are kept, using only
     the items they all share (a smaller item set, more systems)?

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * on MTEB the dropped systems are WORSE on their shared items than the
    kept ones (dropping is not neutral, and it removes the weak tail);
  * keeping them on the shared item set leaves the top unchanged: the
    same system is first, and tie@1 does not fall;
  * on at least one board the complete-case matrix uses fewer than 60 % of
    the rows the board publishes.

SELF-CHECKS
  * a matrix with no missing values must give identical numbers both ways;
  * with missingness planted at random (not related to score), the two
    routes must agree on the leader.

    python missing_entries.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs

SEED = 20260823
DRAWS = 800
SOURCES = {
    "MTEB English v2": "mteb_eng_v2_wide.csv",
    "ProteinGym DMS": "proteingym/matrix.csv",
    "CASP14": "casp/matrix.csv",
    "TabArena 45 variants": "tabarena/matrix_all45.csv",
    "LMArena categories": "lmarena_matrix.csv",
}
MIN_ITEMS = 8


def routes(df):
    """Complete-case (drop rows) and shared-item (drop columns) views.

    The shared-item view keeps every system and uses only items that every
    system ran. If NO item is complete, that view does not exist - the first
    version silently fell back to the complete-case matrix and then reported
    that the two routes agreed, which was a comparison of a matrix with
    itself. It now returns None and the table says 'not available'.
    """
    cc = df.dropna(axis=0)
    keep_cols = [c for c in df.columns if df[c].notna().all()]
    if not keep_cols or len(keep_cols) < MIN_ITEMS:
        return cc, None
    return cc, df[keep_cols]


def _check_complete():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 0.4, (20, 60)))
    cc, si = routes(df)
    if si is None:
        return False, "no complete items in a matrix that has no missing values at all"
    return cc.shape == si.shape == df.shape, f"no missing values: {cc.shape} == {si.shape} == {df.shape}"


def _check_random_missing():
    rng = np.random.default_rng(5)
    x = 0.5 + rng.normal(0, 0.1, 30)[:, None] + rng.normal(0, 0.3, (30, 80))
    df = pd.DataFrame(x)
    mask = rng.random((30, 80)) < 0.05
    df = df.mask(mask)
    cc, si = routes(df)
    if si is None:
        return True, "planted missingness left no complete item; check skipped"
    if cc.shape[0] < 3 or si.shape[0] < 3 or cc.shape[1] < MIN_ITEMS or si.shape[1] < MIN_ITEMS:
        return True, "random missingness left too little to compare; check skipped"
    l1 = cc.mean(axis=1).idxmax()
    l2 = si.mean(axis=1).idxmax()
    return l1 == l2, f"missing at random: leader {l1} either way ({l1} vs {l2})"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_complete(), _check_random_missing()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHAT DROPPING THE INCOMPLETE ROWS DID")
    p("=" * 92)
    p(f"  {'board':<22} {'published':>10} {'complete':>9} {'kept':>6} | {'kept mean':>10} {'dropped':>9} "
      f"| {'shared-item J x n':>18} {'same leader':>12} {'tie@1 cc':>9} {'shared':>7}")
    worse, unchanged, thin = 0, 0, 0
    for name, path in SOURCES.items():
        if not Path(path).exists():
            continue
        df = pd.read_csv(path, index_col=0)
        J0 = df.shape[0]
        cc, si = routes(df)
        share = cc.shape[0] / J0
        thin += share < 0.60
        dropped = df.loc[[i for i in df.index if i not in set(cc.index)]]
        if len(dropped):
            # compare on the items the dropped rows did run, restricted to
            # the columns each dropped row has
            dm = float(np.nanmean(dropped.to_numpy(dtype=float)))
            km = float(np.nanmean(cc.to_numpy(dtype=float)))
        else:
            dm = km = float("nan")
        worse += (not np.isnan(dm)) and dm < km
        same = "-"
        t_cc = t_si = -1
        if si is None:
            xcc = cc.to_numpy(dtype=float)
            t_cc = int((rs.rank_sets(xcc, draws=DRAWS)["best"] == 1).sum())
            p(f"  {name:<22} {J0:>10} {cc.shape[0]:>9} {100 * share:>5.0f}% | {km:>10.3f} {dm:>9.3f} "
              f"| {'no complete item':>18} {'n/a':>12} {t_cc:>9} {'-':>7}")
            continue
        if si.shape[0] >= 3 and si.shape[1] >= MIN_ITEMS:
            xcc, xsi = cc.to_numpy(dtype=float), si.to_numpy(dtype=float)
            rcc = rs.rank_sets(xcc, draws=DRAWS)
            rsi = rs.rank_sets(xsi, draws=DRAWS)
            t_cc = int((rcc["best"] == 1).sum())
            t_si = int((rsi["best"] == 1).sum())
            lead_cc = cc.index[int(np.argmax(xcc.mean(axis=1)))]
            lead_si = si.index[int(np.argmax(xsi.mean(axis=1)))]
            same = "yes" if lead_cc == lead_si else "NO"
            unchanged += (same == "yes") and t_si >= t_cc
        p(f"  {name:<22} {J0:>10} {cc.shape[0]:>9} {100 * share:>5.0f}% | {km:>10.3f} {dm:>9.3f} "
          f"| {f'{si.shape[0]} x {si.shape[1]}':>18} {same:>12} {t_cc if t_cc >= 0 else '-':>9} "
          f"{t_si if t_si >= 0 else '-':>7}")
    p("")
    p(f"  dropped systems score lower than kept ones: {worse} of the boards with any dropped rows")
    p(f"  shared-item route keeps the same leader and no smaller tie@1: {unchanged}")
    p(f"  boards using under 60 % of their published rows: {thin} (pre-registered >= 1)")
    p("")
    p("  'complete' is the matrix every tool here uses: rows with a score on every")
    p("  item. 'shared-item' is the other way round - keep every system, use only")
    p("  the items all of them ran. Neither is neutral; the point is whether the")
    p("  reading depends on which one you take.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("missing_entries_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote missing_entries_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
