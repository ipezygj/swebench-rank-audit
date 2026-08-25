"""One place to draw sub-poset samples, so nobody reports a single one again.

Every exact result in this repository is computed on an induced sub-poset,
because both counting problems are dynamic programs over 2^J subsets. Until
2026-08-25 each file drew ONE subset per board with

    np.random.default_rng(SEED + J).choice(J, SUB, replace=False)

and reported what it found there as the board's value. slack_draws.py measured
what that costs: a standard deviation of 0.53 to 1.00 bits across draws, the
seeded draw sitting outside the central half on 5 of 8 boards, SWE-bench
Verified's published 2.721 above all 25 fresh draws, CASP14's published 0.000
the minimum of its own distribution, and eight board pairs whose ordering
reverses. The single draw was not an estimate of anything.

This module is the repair. A result computed on sub-posets reports a median and
an interquartile range over R draws, not a point, and the draws come from here
so the seed discipline is in one file instead of six.

    for q in subsets(J, 18, R=25, seed=SEED + J):
        ...
"""
from __future__ import annotations

import numpy as np

R_DEFAULT = 25


def subsets(J: int, sub: int, R: int = R_DEFAULT, seed: int = 0):
    """R independent sorted index arrays of `sub` systems out of J.

    One generator per (board, call site) so the draws are reproducible, and R
    of them so what comes out has a spread rather than a value.
    """
    rng = np.random.default_rng(seed)
    for _ in range(R):
        yield np.sort(rng.choice(J, sub, replace=False))


def summarise(values) -> dict:
    """median, IQR and spread of a per-draw quantity."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    if not len(v):
        return {"n": 0, "median": float("nan"), "q1": float("nan"),
                "q3": float("nan"), "mean": float("nan"), "sd": float("nan"),
                "min": float("nan"), "max": float("nan")}
    q1, q3 = np.percentile(v, [25, 75])
    return {"n": int(len(v)), "median": float(np.median(v)), "q1": float(q1),
            "q3": float(q3), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def fmt(s: dict, dp: int = 2) -> str:
    """median [q1, q3] in a fixed width, for tables."""
    if not s["n"]:
        return f"{'-':>18}"
    return f"{s['median']:.{dp}f} [{s['q1']:.{dp}f}, {s['q3']:.{dp}f}]"
