"""A tenth board, untouched all evening: LMArena categories x models.

Everything tonight was measured on nine matrices that have been in this
repo since before the loop started. A held-out board tests the whole
toolkit at once - the two laws, kappa, the pairing dividend - on data none
of the thresholds were chosen against.

LMArena: 124 models x 28 category win rates, extracted for a different
project (~/bio-eval/invE_matrices) and never opened by any tool here. It is
also a different kind of measurement: human preference votes aggregated per
category, not items a system passes or fails.

PRE-REGISTERED EXPECTATION (2026-08-23, before loading the matrix)
  1 kappa over all pairs is 1.00 +- 0.05 (independence holds on average);
  2 kappa of the #1 vs #2 pair is below 0.90;
  3 at least 5 models have a rank set containing rank 1;
  4 law 1: the observed established share is within 5 points of
    Phibar(c sigma_p / (sqrt(2n) tau));
  5 law 2: H / ceiling is within 5 points of the Gaussian twin's;
  6 the pairing dividend (rank-set width without co-movement, over width
    with it) exceeds 1.3;
  7 split-half reliability of kappa is above 0.5 - with 28 categories this
    is the one I expect to fail, since 41 tasks gave 0.70 and 10 gave 0.15.

Six of the seven are the evening's findings applied blind. The seventh is
a stated expectation of failure, so that the run cannot be read as a clean
sweep either way.

    python tenth_board.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm

import rank_sets as rs
import leaderboard_entropy as le
from entropy_law_test import gaussian_twin, stats_of
from pair_sharpness import kappa_matrix
from pairing_dividend import widths
from kappa_reliability import split_half_r

SRC = Path.home() / "bio-eval" / "invE_matrices"
SEED = 20260823


def build():
    a = np.load(SRC / "arena_categories_by_models.npy", allow_pickle=True)   # categories x models
    meta = json.loads((SRC / "arena_meta.json").read_text(encoding="utf-8"))
    models = meta["models"]
    cats = meta["categories"]
    x = np.asarray(a, dtype=float).T                                          # models x categories
    assert x.shape == (len(models), len(cats)), (x.shape, len(models), len(cats))
    df = pd.DataFrame(x, index=models, columns=cats).dropna(axis=0)
    return df


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    df = build()
    x = df.to_numpy(dtype=float)
    J, n = x.shape
    out = Path("lmarena_matrix.csv")
    df.to_csv(out)
    L = []
    p = L.append
    p("TENTH BOARD, HELD OUT ALL EVENING: LMARENA CATEGORIES")
    p("=" * 78)
    p(f"  {J} models x {n} category win rates, from {SRC.name}; written to {out.name}")
    p(f"  range {x.min():.3f} .. {x.max():.3f}")

    rng = np.random.default_rng(SEED)
    r = rs.rank_sets(x, draws=1500)
    beats = r["beats"]
    iu = np.triu_indices(J, k=1)
    sc = x.mean(axis=1)
    tau = float(sc.std(ddof=1))
    sig = r["sigma"][iu]
    sigma_p = float(np.median(sig))
    crit = float(r["crit"])
    estab = float(beats.sum() / (J * (J - 1)))
    pred = float(norm.sf(crit * sigma_p / (math.sqrt(2 * n) * tau)))
    H = le.log_extensions(beats, 2000, rng)["bits"] / (gammaln(J + 1) / math.log(2))
    tw = [stats_of(gaussian_twin(J, n, tau, sigma_p, np.random.default_rng(SEED + 10 * s + 1)), 800, 1000, rng)["H_frac"]
          for s in range(2)]
    Ht = float(np.mean(tw))
    K = kappa_matrix(x)
    order = np.argsort(-sc)
    k_all = float(np.nanmedian(K[iu]))
    k_top = float(K[int(order[0]), int(order[1])])
    wa, ra = widths(x, True, 1200)
    wb, rb = widths(x, False, 1200)
    ratio = float(np.median(wb) / np.median(wa))
    rel, sb, rho = split_half_r(x, np.random.default_rng(SEED + 1), splits=12)
    tie1 = int((r["best"] == 1).sum())

    checks = [
        ("1 kappa over all pairs 1.00 +-0.05", abs(k_all - 1) <= 0.05, f"{k_all:.3f}"),
        ("2 kappa(#1,#2) below 0.90", k_top < 0.90, f"{k_top:.3f}"),
        ("3 at least 5 rank sets contain 1", tie1 >= 5, f"{tie1}"),
        ("4 law 1 within 5 points", abs(pred - estab) <= 0.05, f"observed {100 * estab:.1f} %, law {100 * pred:.1f} %"),
        ("5 law 2 within 5 points", abs(H - Ht) <= 0.05, f"real {100 * H:.1f} %, twin {100 * Ht:.1f} %"),
        ("6 pairing dividend above 1.3", ratio > 1.3, f"{ratio:.2f} (paired {np.median(wa):.0f}, independent {np.median(wb):.0f})"),
        ("7 kappa split-half above 0.5 (expected to FAIL at n=28)", rel > 0.5, f"r {rel:.2f}"),
    ]
    p("")
    for label, okk, detail in checks:
        p(f"  [{'yes' if okk else 'NO ':>3}] {label:<52} {detail}")
    passed = sum(1 for _, o, _ in checks[:6] if o)
    p("")
    p(f"  first six (blind application of the evening's findings): {passed}/6")
    p(f"  seventh, predicted to fail: {'failed as predicted' if not checks[6][1] else 'PASSED against the prediction'}")
    p("")
    p(f"  tie@1 {tie1} of {J} · median rank-set width {int(np.median(r['worst'] - r['best']))} · "
      f"entropy {100 * H:.1f} % of ceiling · tau {tau:.4f} · sigma_p {sigma_p:.4f}")
    p(f"  leader as printed: {df.index[int(order[0])]}  ·  runner-up: {df.index[int(order[1])]}")
    text = chr(10).join(L)
    print(text)
    Path("tenth_board_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote tenth_board_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
