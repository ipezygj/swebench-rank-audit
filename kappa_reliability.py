"""Is pair sharpness a property of the pair, or of the items it was measured on?

The evening's main finding is that kappa - the ratio of a pair's difference
SD to what independence gives - is 0.44-0.94 for the pairs a leaderboard
argues about while the board average is 1.00. Before that goes into a
standard it has to survive the obvious objection: kappa is estimated from
the same items it is used on, and with J(J-1)/2 pairs some will look sharp
by chance.

Split-half reliability: draw the items at random into halves A and B,
compute kappa on each, and correlate. A property of the pair replicates; an
artefact of estimation does not. Reported with the Spearman-Brown
correction to the full item set, and repeated over 20 splits.

The same for the finding that matters: is the frontier's kappa deficit
present in BOTH halves, or only in the half that produced it?

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * split-half correlation of kappa above 0.5 on >= 7 of 9 boards;
  * the frontier kappa deficit (all-pair kappa minus frontier kappa) is
    positive in both halves on >= 4 of the 5 dated boards;
  * boards with few items (HELM n = 10, MTEB n = 41, CASP n = 42) have the
    lowest reliability - the estimate is noisier there.

SELF-CHECKS
  * on an iid field, kappa's split-half correlation is near zero (there is
    no pair property to recover) - this is the null the real boards must
    beat;
  * on a field with planted lineages, it is above 0.8.

    python kappa_reliability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from entropy_law_test import MATRICES
from evidence_trajectory import load
from sota_audit import advances
from pair_sharpness import kappa_matrix
from chase_model import BOARDS as DATED

SEED = 20260823
SPLITS = 20


def split_half_r(x, rng, splits=SPLITS):
    J, n = x.shape
    iu = np.triu_indices(J, k=1)
    rs = []
    for _ in range(splits):
        perm = rng.permutation(n)
        A, B = perm[: n // 2], perm[n // 2:]
        ka = kappa_matrix(x[:, A])[iu]
        kb = kappa_matrix(x[:, B])[iu]
        m = np.isfinite(ka) & np.isfinite(kb)
        if m.sum() > 10:
            rs.append(spearmanr(ka[m], kb[m]).statistic)
    r = float(np.mean(rs))
    sb = 2 * r / (1 + r) if r > -1 else float("nan")     # Spearman-Brown
    return r, sb


def frontier_deficit(x, dates, items):
    K = kappa_matrix(x[:, items])
    J = x.shape[0]
    iu = np.triu_indices(J, k=1)
    allk = float(np.nanmedian(K[iu]))
    fr = [K[a["new"], a["old"]] for a in advances(x, dates)]
    return allk - float(np.nanmedian(fr))


def _check_iid():
    rng = np.random.default_rng(SEED)
    x = 0.4 + rng.normal(0, 0.05, 50)[:, None] + rng.normal(0, 0.3, 400)[None, :] + rng.normal(0, 0.45, (50, 400))
    r, _ = split_half_r(x, rng, splits=8)
    return abs(r) < 0.15, f"iid field: split-half r {r:+.3f} (must be near zero)"


def _check_planted():
    rng = np.random.default_rng(SEED + 2)
    G, per, n = 5, 10, 400
    lab = np.repeat(np.arange(G), per)
    base = rng.normal(0, 0.45, (G, n))
    x = rng.normal(0.4, 0.05, G * per)[:, None] + 0.8 * base[lab] + np.sqrt(1 - 0.8 ** 2) * rng.normal(0, 0.45, (G * per, n))
    r, _ = split_half_r(x, rng, splits=8)
    return r > 0.8, f"planted lineages: split-half r {r:+.3f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_iid(), _check_planted()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("IS PAIR SHARPNESS RELIABLE? SPLIT-HALF OVER ITEMS")
    p("=" * 72)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>4} {'split-half r':>13} {'Spearman-Brown':>15}")
    good, rows = 0, []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        r, sb = split_half_r(x, np.random.default_rng(SEED + 1))
        good += r > 0.5
        rows.append((name, x.shape[1], r))
        p(f"  {name:<22} {x.shape[0]:>4} {x.shape[1]:>4} {r:>13.2f} {sb:>15.2f}")
    N = len(rows)
    p("")
    p(f"  split-half r above 0.5: {good}/{N} (pre-registered >= 7)")
    small = sorted(rows, key=lambda t: t[1])[:3]
    p("  fewest items: " + ", ".join(f"{n_} (n {ni}, r {r_:.2f})" for n_, ni, r_ in small))
    p("")
    p(f"  {'dated board':<22} {'deficit half A':>15} {'deficit half B':>15} {'both positive':>14}")
    both = 0
    for name, (path, dc) in DATED.items():
        x, dates = load(path, dc)
        rng = np.random.default_rng(SEED + 7)
        perm = rng.permutation(x.shape[1])
        A, B = perm[: x.shape[1] // 2], perm[x.shape[1] // 2:]
        da, db = frontier_deficit(x, dates, A), frontier_deficit(x, dates, B)
        both += da > 0 and db > 0
        p(f"  {name:<22} {da:>15.3f} {db:>15.3f} {'yes' if da > 0 and db > 0 else 'NO':>14}")
    p("")
    p(f"  frontier deficit positive in both halves: {both}/{len(DATED)} (pre-registered >= 4)")
    p("")
    p("  Kappa is estimated from the items, so a pair can look sharp by chance.")
    p("  A property of the pair shows up in an independent half of the items; an")
    p("  estimation artefact does not. Spearman-Brown projects the half-length")
    p("  correlation back to the full item set.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("kappa_reliability_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote kappa_reliability_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
