"""The fifth number: can one shape statistic close law 1's residual?

The paper states the gap in its own limitations: "the Gaussian assumption fails
on skewed fields, and we have no theory for that case. Substituting an
IQR-based tau repairs the numbers but is a patch, not an account: the honest
fix is a Phibar that carries the field's skew, and we do not have one." This
tests whether one measurable shape statistic supplies it.

FIRST, A CORRECTION TO HOW THE PAPER DESCRIBES ITS OWN FAILURE

The paper calls this skew. It cannot be. Law 1 is about the distribution of
s_j - s_k, and the difference of two INDEPENDENT DRAWS FROM THE SAME
DISTRIBUTION is symmetric whatever that distribution is - skew cancels exactly.
What survives is the SHAPE: a field with a long tail has a tau inflated by the
tail while its bulk sits closer together, so a threshold in units of tau
separates fewer pairs than a Gaussian of the same tau would. That is a kurtosis
statement, not a skew one, and the paper's wording will be corrected either way
this run comes out.

THE CANDIDATE

    r = IQR(scores) / (1.349 * SD(scores))

which is 1.0 for a Gaussian field, below 1 for one with a heavy tail, above 1
for one with a light tail or a gap. It costs nothing: both quantities are
already computed. The question is whether the law-1 residual is a function of
it, and whether a one-parameter correction fitted on eight boards helps on the
ninth.

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  Spearman(residual under the Gaussian tau, r) across the nine boards has
      |rho| >= 0.6. The residual should be large exactly where r is small.
  P2  the two TabArena boards have the two smallest r of the nine.
  P3  a one-parameter correction fitted by leave-one-out - each board predicted
      by a rule fitted on the other eight - beats the uncorrected law on the
      held-out board on at least 6 of 9.
  P4  the correction survives its own control: with r SHUFFLED across boards,
      the leave-one-out improvement must vanish, and specifically must beat
      the uncorrected law on at most 4 of 9 in the median over 199 shuffles.

  P4 is the one that matters. Nine points and one fitted parameter is a regime
  where something will always look like it helps, and a permutation control is
  the only thing standing between a fifth number and a fitted curve.

  Not predicted: whether the correction beats the IQR-robust version, which
  already uses part of the same information.

SELF-CHECKS (no table if any fails)
  * r must be 1.00 within 0.03 on Gaussian samples of the same sizes as the
    real fields. The raw ratio is not: it reads 0.953 at J = 16, which is the
    size of the board with the most extreme value, so it is divided by its own
    Gaussian expectation at each J before anything is read from it;
  * the leave-one-out harness must be able to fail: fed a residual that is pure
    noise, it must not beat the uncorrected law on more than 6 of 9;
  * every board's residual must reproduce the committed figure in
    resolution_law_test_results.txt exactly.

    python shape_correction.py
"""
from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import rank_sets as rs

SEED = 20260824
DRAWS = 800
PERMS = 199

MATRICES = {
    "SWE-bench Verified": "swebench_verified_matrix.csv",
    "MTEB English v2": "mteb_eng_v2_wide.csv",
    "HELM classic": "helm_winrate_matrix.csv",
    "ProteinGym DMS": "proteingym/matrix.csv",
    "TabArena 16 models": "tabarena/matrix_one_per_model.csv",
    "TabArena 45 variants": "tabarena/matrix_all45.csv",
    "CASP14": "casp/matrix.csv",
    "LiveBench": "livebench/matrix.csv",
    "MathArena 2025": "matharena/matrix.csv",
}


def load(path):
    return pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)


_GAUSS_R = {}


def _raw_ratio(sc: np.ndarray) -> float:
    sd = float(sc.std(ddof=1))
    if sd <= 0:
        return float("nan")
    q1, q3 = np.percentile(sc, [25, 75])
    return float((q3 - q1) / (1.349 * sd))


def gauss_expectation(J: int, reps: int = 4000) -> float:
    """What r averages on a Gaussian field of exactly this many systems.

    The ratio is biased low in small samples - 0.953 at J = 16 - and the boards
    here run from 16 systems to 181, so an uncorrected r would rank them partly
    by size. Cached because it depends only on J.
    """
    if J not in _GAUSS_R:
        rng = np.random.default_rng(1000 + J)
        _GAUSS_R[J] = float(np.mean([_raw_ratio(rng.normal(0, 1, J)) for _ in range(reps)]))
    return _GAUSS_R[J]


def shape_ratio(sc: np.ndarray) -> float:
    """IQR over 1.349 SD, divided by its Gaussian expectation at this J.

    1.0 for a Gaussian field of any size; below 1 when a tail inflates the SD
    while the bulk stays close; above 1 for a light-tailed or gapped field.
    """
    raw = _raw_ratio(sc)
    e = gauss_expectation(len(sc))
    return raw / e if e > 0 else float("nan")


def law1(tau, sigma_p, n, c):
    return float(norm.sf(c * sigma_p / (math.sqrt(2 * n) * tau)))


def committed_errors():
    """err (sd) per board from the committed results file."""
    out = {}
    txt = Path("resolution_law_test_results.txt").read_text(encoding="utf-8", errors="replace")
    for line in txt.splitlines():
        m = re.match(r"\s{2}(\S.*?)\s{2,}(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%\s+"
                     r"([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)", line)
        if m:
            out[m.group(1).strip()] = float(m.group(8))
    return out


def measure():
    rows = {}
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = load(path)
        J, n = x.shape
        r = rs.rank_sets(x, draws=DRAWS)
        sc = x.mean(axis=1)
        iu = np.triu_indices(J, k=1)
        tau = float(sc.std(ddof=1))
        sigma_p = float(np.median(r["sigma"][iu]))
        obs = float(r["beats"].sum() / (J * (J - 1)))
        pred = law1(tau, sigma_p, n, r["crit"])
        rows[name] = {"J": J, "n": n, "tau": tau, "sigma_p": sigma_p, "c": r["crit"],
                      "obs": obs, "pred": pred, "resid": 100 * (pred - obs),
                      "r": shape_ratio(sc)}
    return rows


def loo_scores(resid, shape):
    """Leave-one-out: fit resid ~ a + b*shape on the others, score the held-out.

    Returns how many held-out boards the correction beats the uncorrected law
    on, in absolute error.
    """
    resid = np.asarray(resid, dtype=float)
    shape = np.asarray(shape, dtype=float)
    wins = 0
    for i in range(len(resid)):
        m = np.ones(len(resid), dtype=bool)
        m[i] = False
        b, a = np.polyfit(shape[m], resid[m], 1)
        corrected = resid[i] - (a + b * shape[i])
        wins += abs(corrected) < abs(resid[i])
    return wins


def _check_shape_unbiased() -> tuple[bool, str]:
    rng = np.random.default_rng(3)
    worst, where = 0.0, ""
    for J in (16, 35, 45, 90, 134, 181):
        vals = [shape_ratio(rng.normal(0, 1, J)) for _ in range(400)]
        m = float(np.mean(vals))
        if abs(m - 1.0) > worst:
            worst, where = abs(m - 1.0), f"J={J} gives {m:.3f}"
    return worst <= 0.03, f"r is 1.00 on Gaussian fields, worst {where}"


def _check_loo_can_fail() -> tuple[bool, str]:
    rng = np.random.default_rng(5)
    wins = [loo_scores(rng.normal(0, 5, 9), rng.normal(0, 1, 9)) for _ in range(200)]
    med = float(np.median(wins))
    return med <= 6, f"on pure noise the leave-one-out wins a median of {med:.0f} of 9"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = measure()
    names = list(rows)

    print("self-checks ...")
    checks = [_check_shape_unbiased(), _check_loo_can_fail()]
    committed = committed_errors()
    bad = [(k, round(rows[k]["resid"], 1), committed[k]) for k in names
           if k in committed and abs(rows[k]["resid"] - committed[k]) > 0.06]
    checks.append((not bad, "residuals reproduce the committed file"
                   + ("" if not bad else "  off: " + "; ".join(f"{k} {a} vs {b}" for k, a, b in bad))))
    ok = True
    for passed, msg in checks:
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    resid = np.array([rows[k]["resid"] for k in names])
    shape = np.array([rows[k]["r"] for k in names])
    rho, pv = spearmanr(shape, resid)
    wins = loo_scores(resid, shape)

    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(PERMS):
        s = shape.copy()
        rng.shuffle(s)
        null.append(loo_scores(resid, s))
    null_med = float(np.median(null))
    null_p = float((np.array(null) >= wins).mean())

    order = np.argsort(shape)
    two_smallest = {names[i] for i in order[:2]}

    L = []
    p = L.append
    p("THE FIFTH NUMBER: IS LAW 1'S RESIDUAL A FUNCTION OF THE FIELD'S SHAPE?")
    p("=" * 92)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>5} {'r = IQR/1.349SD':>16} {'observed':>9} "
      f"{'law 1':>7} {'residual':>9}")
    for k in sorted(names, key=lambda z: rows[z]["r"]):
        v = rows[k]
        p(f"  {k:<22} {v['J']:>4} {v['n']:>5} {v['r']:>16.3f} {100 * v['obs']:>8.1f}% "
          f"{100 * v['pred']:>6.1f}% {v['resid']:>+9.1f}")
    p("")
    p(f"  P1  Spearman(r, residual) = {rho:+.2f} (p {pv:.3f})        "
      f"pre-registered |rho| >= 0.6:  {'HIT' if abs(rho) >= 0.6 else 'MISS'}")
    p(f"  P2  the two smallest r are {', '.join(sorted(two_smallest))}")
    p(f"      pre-registered: both TabArena boards:  "
      f"{'HIT' if two_smallest == {'TabArena 16 models', 'TabArena 45 variants'} else 'MISS'}")
    p(f"  P3  leave-one-out correction beats the plain law on {wins} of {len(names)}   "
      f"pre-registered >= 6:  {'HIT' if wins >= 6 else 'MISS'}")
    p(f"  P4  with r shuffled, the median is {null_med:.0f} of {len(names)} and "
      f"{100 * null_p:.0f} % of shuffles reach {wins}")
    p(f"      pre-registered median <= 4:  {'HIT' if null_med <= 4 else 'MISS'}")
    p("")
    p("  r is the interquartile range over 1.349 standard deviations: 1.0 for a")
    p("  Gaussian field, below 1 when a tail inflates the SD while the bulk stays")
    p("  close, above 1 for a light-tailed or gapped field.")
    p("")
    p("  The paper calls this failure skew. It cannot be: law 1 concerns the")
    p("  difference of two draws from the same field, and that difference is")
    p("  symmetric whatever the field looks like - skew cancels exactly. What")
    p("  survives is the shape of the bulk against the tail, which is what r")
    p("  measures and what the wording should say.")
    p("")
    p("  P4 is the load-bearing check. Nine boards and one fitted parameter is a")
    p("  regime where a correction will look like it helps whatever it is fitted")
    p("  to, so the same leave-one-out is run with the shape statistic shuffled")
    p("  between boards. If the shuffled version wins about as often, the fifth")
    p("  number is a fitted curve and not a fifth number.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("shape_correction_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                    newline=chr(10))
    print(chr(10) + "wrote shape_correction_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
