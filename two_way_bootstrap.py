"""Which bootstrap? The answer depends on what you claim to generalise to.

An evaluation matrix is crossed: systems x items, and BOTH could have come out
differently. Three resampling schemes are in use, they target three different
parameters, and picking the wrong one is not imprecision - it is generalising
to a population you did not mean.

    per-system items   each system gets its own independent item resample.
                       This is what you get by putting an error bar on each
                       system separately and comparing them. It DESTROYS the
                       pairing: the whole reason paired comparison is powerful
                       is that the same items are shared, and this throws that
                       away.

    shared items       one item resample used by every system, systems fixed.
                       This is the unit rank_sets.py uses. It answers: would
                       this ranking survive a fresh draw of items, for THESE
                       systems?

    two-way            items and systems both resampled (Owen's pigeonhole
                       bootstrap, 2007). It answers: would this hold for a
                       fresh draw of items AND a fresh set of systems?

NO EXACT BOOTSTRAP EXISTS HERE, AND THAT IS NOT A DETAIL
---------------------------------------------------------
McCullagh (2000) showed there is no exact bootstrap for crossed random
effects, and Owen notes the pigeonhole scheme can overstate variance because
resampling with replacement duplicates rows and columns, manufacturing perfect
correlations that were not in the data. So this file does not assert that the
two-way scheme is correct. It MEASURES the coverage of all three against a
simulated truth and reports which is right for which question, including when
the two-way one is conservative.

WHAT IS MEASURED
----------------
Two estimands with deliberately different sensitivity:

    grand mean          barely depends on which systems were drawn
    between-system sd   the leaderboard's own spread, which depends on the
                        system draw as much as on the item draw

and two populations:

    systems fixed       the same systems every replication, items redrawn
    systems random      systems drawn afresh from a super-population

The prediction before running: with systems fixed, shared-items is right and
two-way is conservative. With systems random, shared-items under-covers on the
spread and two-way is needed. If the numbers say otherwise, the numbers win
and the docstring is wrong.

    python two_way_bootstrap.py [--matrix ...] [--reps 300] [--boot 400]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260823
SCHEMES = ("per-system items", "shared items", "two-way")


# --- estimands --------------------------------------------------------------

def grand_mean(x: np.ndarray) -> float:
    return float(x.mean())


def between_sd(x: np.ndarray) -> float:
    """Spread of system scores: what a leaderboard's range actually is."""
    return float(x.mean(axis=1).std(ddof=1))


ESTIMANDS = {"grand mean": grand_mean, "between-system sd": between_sd}


# --- resampling -------------------------------------------------------------

def resample(x: np.ndarray, scheme: str, rng: np.random.Generator) -> np.ndarray:
    J, n = x.shape
    if scheme == "per-system items":
        out = np.empty_like(x)
        for j in range(J):
            out[j] = x[j, rng.integers(0, n, n)]
        return out
    if scheme == "shared items":
        return x[:, rng.integers(0, n, n)]
    if scheme == "two-way":
        return x[rng.integers(0, J, J)][:, rng.integers(0, n, n)]
    raise ValueError(scheme)


def boot_ci(x: np.ndarray, stat, scheme: str, boot: int, alpha: float,
            rng: np.random.Generator) -> tuple[float, float]:
    vals = np.array([stat(resample(x, scheme, rng)) for _ in range(boot)])
    return (float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


# --- simulated world --------------------------------------------------------

def draw_world(rng, J, n, sys_sd, item_sd, systems=None):
    """Crossed random effects: ability_j + difficulty_i, logit link."""
    a = rng.normal(0.0, sys_sd, J) if systems is None else systems
    b = rng.normal(0.0, item_sd, n)
    p = 1.0 / (1.0 + np.exp(-(a[:, None] + b[None, :])))
    return (rng.random((J, n)) < p).astype(float), a


def truth(stat, a, item_sd, rng, big_n=40_000):
    """The estimand on a very large item draw, for the given abilities."""
    b = rng.normal(0.0, item_sd, big_n)
    p = 1.0 / (1.0 + np.exp(-(a[:, None] + b[None, :])))
    return stat((rng.random((len(a), big_n)) < p).astype(float))


def coverage_study(reps, boot, J, n, sys_sd, item_sd, alpha, seed,
                   systems_random):
    """Coverage of each scheme, for each estimand."""
    rng = np.random.default_rng(seed)
    fixed = None if systems_random else rng.normal(0.0, sys_sd, J)
    hits = {(s, e): 0 for s in SCHEMES for e in ESTIMANDS}
    widths = {(s, e): 0.0 for s in SCHEMES for e in ESTIMANDS}
    # Totuus ei riipu toistosta, joten se lasketaan KERRAN ja tarkasti.
    # Kierroksittain laskettuna se olisi ollut sekä hidasta että kohinaista,
    # ja kohina totuudessa nakyy peitossa aivan kuin se olisi menetelman vika.
    truths = {}
    for ename, stat in ESTIMANDS.items():
        if systems_random:
            truths[ename] = float(np.mean([
                truth(stat, rng.normal(0.0, sys_sd, J), item_sd, rng,
                      big_n=6000) for _ in range(40)]))
        else:
            truths[ename] = float(np.mean([
                truth(stat, fixed, item_sd, rng, big_n=20000)
                for _ in range(3)]))
    for _ in range(reps):
        x, a = draw_world(rng, J, n, sys_sd, item_sd, systems=fixed)
        for ename, stat in ESTIMANDS.items():
            # With systems random the estimand is a super-population quantity,
            # so the truth must average over system draws too; with systems
            # fixed it is conditional on those abilities.
            t = truths[ename]
            for scheme in SCHEMES:
                lo, hi = boot_ci(x, stat, scheme, boot, alpha, rng)
                hits[(scheme, ename)] += int(lo <= t <= hi)
                widths[(scheme, ename)] += hi - lo
    return ({k: v / reps for k, v in hits.items()},
            {k: v / reps for k, v in widths.items()})


# --- self-checks ------------------------------------------------------------

def _check_resample_shapes() -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    x = (rng.random((7, 30)) < 0.5).astype(float)
    ok = all(resample(x, s, rng).shape == x.shape for s in SCHEMES)
    return ok, f"all three schemes return the original shape: {ok}"


def _check_pairing_destroyed() -> tuple[bool, str]:
    """per-system resampling must break the cross-system correlation."""
    rng = np.random.default_rng(3)
    n = 600
    b = rng.normal(0, 1.5, n)
    p = 1 / (1 + np.exp(-b))
    x = (rng.random((2, n)) < p).astype(float)

    def rho(m):
        u = m - m.mean(axis=1, keepdims=True)
        c = (u @ u.T) / m.shape[1]
        return c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])

    real = rho(x)
    shared = np.mean([rho(resample(x, "shared items", rng)) for _ in range(60)])
    per = np.mean([rho(resample(x, "per-system items", rng)) for _ in range(60)])
    ok = real > 0.2 and abs(shared - real) < 0.1 and abs(per) < 0.1
    return ok, (f"correlation kept by shared ({shared:+.3f} vs real {real:+.3f}) "
                f"and destroyed by per-system ({per:+.3f})")


def _check_two_way_moves_with_systems() -> tuple[bool, str]:
    """Two-way must show more spread uncertainty than item-only."""
    rng = np.random.default_rng(5)
    x, _ = draw_world(rng, 24, 400, 1.0, 1.0)
    lo1, hi1 = boot_ci(x, between_sd, "shared items", 300, 0.05, rng)
    lo2, hi2 = boot_ci(x, between_sd, "two-way", 300, 0.05, rng)
    ok = (hi2 - lo2) > (hi1 - lo1)
    return ok, (f"between-system sd interval: shared {hi1-lo1:.4f}, "
                f"two-way {hi2-lo2:.4f}")


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_resample_shapes(), _check_pairing_destroyed(),
                        _check_two_way_moves_with_systems()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="two_way_bootstrap_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    print("self-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no headline number is printed.")
        return 1

    L = []
    p = L.append
    p("WHICH BOOTSTRAP, AND WHAT IT COSTS TO PICK THE WRONG ONE")
    p("=" * 74)
    p(f"simulated coverage, nominal {1 - a.alpha:.2f}, "
      f"{a.reps} replications x {a.boot} bootstrap draws")
    for systems_random in (False, True):
        label = ("SYSTEMS RANDOM (a fresh set of systems could have been built)"
                 if systems_random else
                 "SYSTEMS FIXED (these systems, a fresh draw of items)")
        cov, wid = coverage_study(a.reps, a.boot, 16, 300, 1.0, 1.2,
                                  a.alpha, SEED + int(systems_random),
                                  systems_random)
        p("")
        p(label)
        p(f"  {'scheme':<20} {'estimand':<20} {'coverage':>9} {'width':>9}")
        for ename in ESTIMANDS:
            for scheme in SCHEMES:
                p(f"  {scheme:<20} {ename:<20} "
                  f"{cov[(scheme, ename)]:>9.3f} {wid[(scheme, ename)]:>9.4f}")

    # The real matrix: what the choice costs on the actual data.
    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    p("")
    p(f"ON THE REAL MATRIX ({x.shape[0]} systems x {x.shape[1]} items)")
    p(f"  {'scheme':<20} {'between-system sd 95% interval':<34} {'width':>8}")
    for scheme in SCHEMES:
        lo, hi = boot_ci(x, between_sd, scheme, a.boot, a.alpha, rng)
        p(f"  {scheme:<20} [{lo:.4f}, {hi:.4f}]"
          f"{'':<14} {hi - lo:>8.4f}")
    p(f"  point estimate {between_sd(x):.4f}")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
