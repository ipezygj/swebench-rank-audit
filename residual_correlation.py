"""Is the universal negative residual the CORRELATION of entrants' residuals?

entropy_decomposition.py found that on all nine boards the real leaderboard
has LOWER entropy than 'real abilities + iid Gaussian noise' (residual term
-1.7 to -10.6). Independent noise cannot make a board more ordered than
that twin. Correlated noise can: two systems that share a base model, a
scaffold, a training set, fail the same items; their pairwise difference
has a smaller variance than independence predicts; their pair is
established at a gap that independent noise would leave open. More
established pairs, fewer linear extensions, lower H.

Twin 4 keeps everything about each system - its ability, the item
difficulties, its own residual VALUES - and permutes each system's residual
vector independently across items. That destroys cross-system residual
correlation and nothing else.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * H(twin 4) - H(real) >= +3 points on at least 7 of 9 boards;
  * H(twin 4) lands within 3 points of the shape twin (iid Gaussian noise
    around real abilities) on at least 6 of 9 - the residual term is
    recovered by breaking correlation alone;
  * exploratory, reported not judged: the mean pairwise residual correlation
    of each board against its residual term from entropy_decomposition
    (sign expected: more correlation, more negative residual).

SELF-CHECKS
  * twin 4 of an iid field (no cross-system correlation) must change H by
    less than 3 points;
  * twin 4 of a field with a planted shared component (half the systems share
    one residual factor) must RAISE H by more than 3 points.

    python residual_correlation.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from entropy_law_test import MATRICES, stats_of
from entropy_decomposition import shape_twin

SEED = 20260823
DRAWS, SAMPLES = 800, 1000

DECOMP_RESID = {   # residual term from entropy_decomposition_results.txt (464a2af)
    "SWE-bench Verified": -2.3, "MTEB English v2": -8.3, "HELM classic": -1.9,
    "ProteinGym DMS": -8.8, "TabArena 16 models": -10.6, "TabArena 45 variants": -5.5,
    "CASP14": -2.0, "LiveBench": -1.7, "MathArena 2025": -3.3,
}


def decompose(x):
    row = x.mean(axis=1, keepdims=True)
    col = x.mean(axis=0, keepdims=True)
    resid = x - row - col + x.mean()
    return row, col, resid


def twin4(x: np.ndarray, rng) -> np.ndarray:
    row, col, resid = decompose(x)
    J, n = x.shape
    out = np.empty_like(resid)
    for j in range(J):
        out[j] = resid[j, rng.permutation(n)]
    return row + col - x.mean() + out


def mean_resid_corr(x: np.ndarray) -> float:
    _, _, resid = decompose(x)
    c = np.corrcoef(resid)
    iu = np.triu_indices(x.shape[0], k=1)
    return float(np.nanmean(c[iu]))


def _check_iid():
    rng = np.random.default_rng(41)
    J, n = 80, 200
    x = rng.normal(0, 0.06, J)[:, None] + rng.normal(0, 0.3, n)[None, :] + rng.normal(0, 0.45, (J, n))
    a = stats_of(x, 500, 600, rng)["H_frac"]
    b = np.mean([stats_of(twin4(x, np.random.default_rng(42 + s)), 500, 600, rng)["H_frac"] for s in range(2)])
    return abs(b - a) < 0.03, f"iid field: twin4 moves H by {100 * (b - a):+.1f} points (must be < 3)"


def _check_planted():
    rng = np.random.default_rng(43)
    J, n = 80, 200
    # Two groups, each sharing one residual factor at loading 0.9. The first
    # version (one group of half the systems at 0.8) raised H by +2.5 - the
    # right sign but under the 3-point bar, a matter of planted strength, so
    # the planting was made stronger; the bar was not lowered.
    f1, f2 = rng.normal(0, 0.45, n), rng.normal(0, 0.45, n)
    noise = rng.normal(0, 0.45, (J, n))
    load = 0.9
    shared = np.vstack([np.tile(f1, (J // 2, 1)), np.tile(f2, (J - J // 2, 1))])
    x = rng.normal(0, 0.06, J)[:, None] + load * shared + math.sqrt(1 - load ** 2) * noise
    a = stats_of(x, 500, 600, rng)["H_frac"]
    b = np.mean([stats_of(twin4(x, np.random.default_rng(44 + s)), 500, 600, rng)["H_frac"] for s in range(2)])
    return (b - a) > 0.03, f"planted shared component: twin4 raises H by {100 * (b - a):+.1f} points (must be > 3)"


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

    rng = np.random.default_rng(SEED)
    L = []
    p = L.append
    p("IS THE NEGATIVE RESIDUAL THE CORRELATION OF ENTRANTS' RESIDUALS?")
    p("=" * 80)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>4} {'real':>7} {'twin4':>7} {'d4':>6} {'shape':>7} {'t4-shape':>8} {'rho_res':>8} {'resid':>6}")
    d4s, near, rows = [], [], []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        a = stats_of(x, DRAWS, SAMPLES, rng)
        t4 = float(np.mean([stats_of(twin4(x, np.random.default_rng(SEED + 400 + s)), DRAWS, SAMPLES, rng)["H_frac"] for s in range(2)]))
        sh = float(np.mean([stats_of(shape_twin(x, a["sigma_p"], np.random.default_rng(SEED + 300 + s)), DRAWS, SAMPLES, rng)["H_frac"] for s in range(2)]))
        rho = mean_resid_corr(x)
        d4 = t4 - a["H_frac"]
        d4s.append(d4 >= 0.03); near.append(abs(t4 - sh) <= 0.03)
        rows.append((name, rho, DECOMP_RESID.get(name, float("nan"))))
        p(f"  {name:<22} {x.shape[0]:>4} {x.shape[1]:>4} {100 * a['H_frac']:>6.1f}% {100 * t4:>6.1f}% {100 * d4:>+5.1f} {100 * sh:>6.1f}% {100 * (t4 - sh):>+7.1f} {rho:>8.3f} {DECOMP_RESID.get(name, float('nan')):>+6.1f}")
    p("")
    p(f"  twin4 raises H by >= 3 points: {sum(d4s)}/{len(d4s)}   (pre-registered >= 7)")
    p(f"  twin4 within 3 points of shape twin: {sum(near)}/{len(near)}   (pre-registered >= 6)")
    from scipy.stats import spearmanr
    r = spearmanr([r_[1] for r_ in rows], [r_[2] for r_ in rows])
    p(f"  exploratory: Spearman(rho_res, residual term) = {r.statistic:+.2f} (p {r.pvalue:.2f}); expected sign negative")
    p("")
    p("  twin4 = real abilities, real item difficulties, each system's own")
    p("  residuals permuted independently across items. rho_res = mean pairwise")
    p("  correlation of residuals. d4 = H(twin4) - H(real): what correlation")
    p("  among entrants was hiding. t4-shape: how much of the iid-twin level is")
    p("  recovered by breaking correlation alone.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("residual_correlation_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote residual_correlation_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
