"""How much of the top score is just the fact that it is a maximum?

THE QUESTION
------------
A leaderboard's headline number is a MAXIMUM. Even if every system on it were
equally good, the best observed score would exceed the common truth, by an
amount that grows with how many systems were tried. Finance has priced this for
a decade: the Deflated Sharpe Ratio of Bailey and Lopez de Prado subtracts the
expected maximum of N trials before calling a strategy good. Nobody deflates a
benchmark score.

WHAT IS IDENTIFIED AND WHAT IS NOT - STATED FIRST
--------------------------------------------------
NOT identified: the number of unreported trials N. The publication-bias
literature identifies a selection function from variation in precision across
studies (Andrews & Kasy 2019). On a leaderboard every system is scored on the
SAME items, so the standard errors barely vary and that variation is not
available. Any point estimate of N here would be invented, and this file does
not produce one.

Identified, and computed here:
  (1) the inflation of the maximum GIVEN the J systems actually shown, from
      the observed correlation structure, with no distributional assumption;
  (2) a SENSITIVITY CURVE in N, and the breakdown point - the smallest number
      of trials at which the leader's advantage is fully explained by
      selection. That single number is the honest answer to "how fragile is
      this lead", in the spirit of a Rosenbaum sensitivity bound rather than a
      point estimate.

THE CORRECTION THAT DSR MISSES FOR LEADERBOARDS
------------------------------------------------
DSR assumes independent trials. Leaderboard systems are NOT independent: they
solve the same easy instances and fail the same hard ones, so their scores are
strongly positively correlated. Write the null scores as equicorrelated,

    theta_j = mu + sigma * ( sqrt(rho) Z0 + sqrt(1 - rho) Z_j )

with Z0 shared across systems and Z_j private. The maximum over N is

    max_j theta_j = mu + sigma sqrt(rho) Z0 + sigma sqrt(1 - rho) max_j Z_j

and the shared term has mean zero, so

    E[max] - mu = sigma * sqrt(1 - rho) * a_N,        a_N = E[max of N N(0,1)]

The winner's curse on a leaderboard is smaller than the independent-trials
formula says, by exactly sqrt(1 - rho). With rho measured at 0.6 the inflation
is 63 % of what DSR would charge; at rho = 0.9 it is 32 %. Applying the finance
formula unchanged to a benchmark OVER-deflates. rho is measurable, so this is
an arithmetic correction, not a judgement call.

THE NULL, AND WHY IT NEEDS NO DISTRIBUTIONAL ASSUMPTION
--------------------------------------------------------
For the top group we also run an exact permutation test: within each item,
permute the outcomes among the k systems being compared. Under the hypothesis
that those k systems are equally able, they are exchangeable, so this leaves
the distribution unchanged - while preserving every item's difficulty and the
whole dependence structure. The observed maximum is then compared with its own
permutation distribution.

    python deflated_benchmark.py [--matrix ...] [--top 19] [--perms 2000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260823


def expected_max_normal(N: int, draws: int = 200_000,
                        rng: np.random.Generator | None = None) -> float:
    """E[max of N standard normals], by simulation.

    The textbook sqrt(2 ln N) is an asymptotic approximation and is off by
    more than 10 % at the sizes a leaderboard actually has (N < 200), which is
    exactly the range that matters here.
    """
    if N <= 1:
        return 0.0
    rng = rng or np.random.default_rng(SEED)
    m = 0.0
    block = max(1, draws // 20)
    total = 0
    while total < draws:
        b = min(block, draws - total)
        m += rng.standard_normal((b, N)).max(axis=1).sum()
        total += b
    return float(m / draws)


def group_stats(x: np.ndarray) -> dict:
    """mu, sigma and the mean pairwise correlation of the group's scores."""
    k, n = x.shape
    theta = x.mean(axis=1)
    u = x - theta[:, None]
    cov_items = (u @ u.T) / n              # item-level covariance
    var_score = np.diag(cov_items) / n     # Var of each system's mean score
    sd = np.sqrt(np.maximum(var_score, 0.0))
    denom = np.sqrt(np.outer(np.diag(cov_items), np.diag(cov_items)))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov_items / denom
    iu = np.triu_indices(k, k=1)
    rho = float(np.nanmean(corr[iu])) if k > 1 else 0.0
    return {"theta": theta, "mu": float(theta.mean()),
            "sigma": float(np.nanmean(sd)), "rho": rho, "k": k, "n": n}


def permutation_null(x: np.ndarray, perms: int = 2000,
                     seed: int = SEED) -> np.ndarray:
    """Max score under within-item permutation among the rows of `x`.

    Exact under the hypothesis that these systems are exchangeable, i.e.
    equally able. Item difficulty and the dependence structure are untouched
    because only the labels move.
    """
    k, n = x.shape
    rng = np.random.default_rng(seed)
    out = np.empty(perms)
    for b in range(perms):
        # One independent permutation of the k labels per item.
        idx = np.argsort(rng.random((n, k)), axis=1)      # n x k
        permuted = np.take_along_axis(x.T, idx, axis=1).T  # k x n
        out[b] = permuted.mean(axis=1).max()
    return out


def sensitivity(mu: float, sigma: float, rho: float, observed_max: float,
                ns=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 20000),
                seed: int = SEED) -> tuple[list, int | None]:
    """Deflated score and lead for a range of assumed trial counts."""
    rng = np.random.default_rng(seed)
    shrink = float(np.sqrt(max(0.0, 1.0 - rho)))
    rows, breakdown = [], None
    for N in ns:
        a = expected_max_normal(N, draws=60_000, rng=rng)
        expected = mu + sigma * shrink * a
        naive = mu + sigma * a          # what DSR's independence would give
        rows.append({"N": N, "a_N": a, "expected_max": expected,
                     "naive_expected_max": naive,
                     "lead_left": observed_max - expected})
        if breakdown is None and expected >= observed_max:
            breakdown = N
    return rows, breakdown



def model_vs_permutation(g: dict, null: np.ndarray, seed: int = SEED) -> dict:
    """Does the equicorrelated normal model reproduce the exact null?

    The sensitivity curve extrapolates to trial counts nobody observed, and it
    does that through a parametric model: equicorrelated normals with mu,
    sigma and rho estimated from the data. The permutation null needs no such
    model and is exact at the one point where both can be evaluated, N = k.
    If the two disagree there, the extrapolation is not entitled to belief and
    the curve is withheld.

    Reported as a fraction of the quantity being explained, the observed
    lead over the group mean, because an absolute score difference of 0.001
    means something very different on a tight leaderboard than on a loose one.
    """
    a_k = expected_max_normal(g["k"], draws=200_000,
                              rng=np.random.default_rng(seed))
    shrink = float(np.sqrt(max(0.0, 1.0 - g["rho"])))
    model_mean = g["mu"] + g["sigma"] * shrink * a_k
    perm_mean = float(null.mean())
    lead = float(g["theta"].max() - g["mu"])
    gap = abs(model_mean - perm_mean)
    return {"model_mean": model_mean, "perm_mean": perm_mean, "gap": gap,
            "rel": gap / lead if lead > 0 else float("inf"),
            "a_k": a_k, "shrink": shrink}

# ---------------------------------------------------------------------------
# Self-checks: these decide whether any headline number is printed.
# ---------------------------------------------------------------------------

def _check_expected_max() -> tuple[bool, str]:
    """E[max of N] must match known values."""
    got2 = expected_max_normal(2, draws=200_000)
    want2 = 1.0 / np.sqrt(np.pi)                 # exact for N = 2
    ok = abs(got2 - want2) < 0.01
    return ok, f"E[max of 2 normals] {got2:.4f} vs exact {want2:.4f}"


def _check_rho_zero() -> tuple[bool, str]:
    """Independent systems: the shrink factor must be ~1, i.e. no discount."""
    rng = np.random.default_rng(5)
    x = (rng.random((12, 4000)) < 0.5).astype(float)
    g = group_stats(x)
    ok = abs(g["rho"]) < 0.05
    return ok, f"independent systems -> rho {g['rho']:+.3f} (want ~0)"


def _check_rho_one() -> tuple[bool, str]:
    """Identical systems: rho ~ 1, so the winner's curse must vanish."""
    rng = np.random.default_rng(7)
    row = (rng.random(2000) < 0.5).astype(float)
    x = np.tile(row, (10, 1))
    g = group_stats(x)
    shrink = np.sqrt(max(0.0, 1 - g["rho"]))
    ok = g["rho"] > 0.99 and shrink < 0.05
    return ok, f"identical systems -> rho {g['rho']:.3f}, shrink {shrink:.3f}"


def _check_permutation_null_calibrated() -> tuple[bool, str]:
    """With no real leader the observed max must sit inside its own null."""
    rng = np.random.default_rng(9)
    n, k = 500, 8
    diff = rng.normal(0, 0.6, n)
    p = 1 / (1 + np.exp(-diff))
    x = (rng.random((k, n)) < p).astype(float)
    null = permutation_null(x, perms=400, seed=13)
    obs = x.mean(axis=1).max()
    pval = float((null >= obs).mean())
    ok = pval > 0.05
    return ok, f"no real leader -> permutation p = {pval:.3f} (want > 0.05)"


def _check_permutation_detects() -> tuple[bool, str]:
    """A genuinely better system must be detected."""
    rng = np.random.default_rng(17)
    n, k = 500, 8
    diff = rng.normal(0, 0.6, n)
    p = 1 / (1 + np.exp(-diff))
    x = (rng.random((k, n)) < p).astype(float)
    strong = 1 / (1 + np.exp(-(diff + 1.5)))
    x[0] = (rng.random(n) < strong).astype(float)
    null = permutation_null(x, perms=400, seed=19)
    obs = x.mean(axis=1).max()
    pval = float((null >= obs).mean())
    ok = pval < 0.05
    return ok, f"injected real leader -> permutation p = {pval:.3f} (want < 0.05)"


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_expected_max(), _check_rho_zero(),
                        _check_rho_one(), _check_permutation_null_calibrated(),
                        _check_permutation_detects()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--top", type=int, default=19,
                    help="size of the candidate-best group (default: the 19 "
                         "systems whose rank set includes 1)")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--out", default="deflated_benchmark_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x_all = df.to_numpy(dtype=float)
    names = list(df.index)
    print(f"matrix {a.matrix}: {x_all.shape[0]} systems x {x_all.shape[1]} items")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no headline number is printed.")
        return 1

    order = np.argsort(-x_all.mean(axis=1), kind="stable")
    top = order[:a.top]
    x = x_all[top]
    g = group_stats(x)
    observed_max = float(g["theta"].max())
    shrink = float(np.sqrt(max(0.0, 1.0 - g["rho"])))

    null = permutation_null(x, perms=a.perms)
    pval = float((null >= observed_max).mean())

    cross = model_vs_permutation(g, null)
    curve_ok = cross["rel"] <= 0.25
    rows, breakdown = sensitivity(g["mu"], g["sigma"], g["rho"], observed_max)

    L = []
    p = L.append
    p("HOW MUCH OF THE TOP SCORE IS JUST THE MAXIMUM?")
    p("=" * 74)
    p(f"top group: {a.top} systems, {g['n']} items")
    p(f"best observed score          {observed_max:.4f}  ({names[top[0]][:44]})")
    p(f"group mean                   {g['mu']:.4f}")
    p(f"per-system standard error    {g['sigma']:.4f}")
    p(f"mean pairwise correlation    {g['rho']:.3f}")
    p(f"winner's-curse shrink factor {shrink:.3f}   "
      f"= sqrt(1 - rho)")
    p("")
    p("EXACT PERMUTATION TEST (are the top systems exchangeable?)")
    p(f"  observed maximum           {observed_max:.4f}")
    p(f"  null mean / 95th pct       {null.mean():.4f} / "
      f"{np.quantile(null, 0.95):.4f}")
    p(f"  p-value over {a.perms} permutations   {pval:.4f}")
    if pval > 0.05:
        p("  -> the top group is consistent with all of them being equally able;")
        p("     the leader's margin is what a maximum over this many correlated")
        p("     systems produces on its own.")
    else:
        p("  -> the leader is above what exchangeability alone would produce.")
    p("")
    p("MODEL CHECKED AGAINST THE EXACT NULL AT N = k")
    p(f"  permutation null mean      {cross['perm_mean']:.4f}")
    p(f"  equicorrelated model       {cross['model_mean']:.4f}")
    p(f"  gap                        {cross['gap']:.4f}  "
      f"= {100 * cross['rel']:.1f}% of the lead being explained")
    if not curve_ok:
        p("  -> the model does NOT reproduce the assumption-free null, so the")
        p("     extrapolation below is withheld. Nothing is reported that the")
        p("     data cannot carry.")
    p("")
    p("SENSITIVITY TO UNREPORTED TRIALS (N is not identified; this is a curve)")
    p(f"{'N':>7} {'E[max] here':>12} {'if independent':>15} {'lead left':>11}")
    if curve_ok:
        for r in rows:
            p(f"{r['N']:>7} {r['expected_max']:>12.4f} "
              f"{r['naive_expected_max']:>15.4f} {r['lead_left']:>11.4f}")
    else:
        p("  withheld - see the model check above")
    p("")
    if not curve_ok:
        p("BREAKDOWN POINT: withheld with the curve.")
    elif breakdown:
        p(f"BREAKDOWN POINT: N = {breakdown}. If at least this many comparable")
        p("attempts stood behind the leaderboard, the best score is entirely")
        p("explained by taking a maximum, with no ability difference at all.")
    else:
        p("BREAKDOWN POINT: not reached within the range tried - the leader's")
        p("score survives selection even under extreme assumed trial counts.")
    p("")
    p("The 'if independent' column is what the Deflated Sharpe Ratio formula")
    p(f"would charge. It over-deflates here by a factor of 1/{shrink:.3f} = "
      f"{1/shrink:.2f}x,")
    p("because leaderboard systems share items and are strongly correlated.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
