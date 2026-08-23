"""Entropy law, second attempt: a twin that keeps the NOISE PROFILE.

entropy_law_test.py matched a leaderboard's H/ceiling with a Gaussian twin on
6 of 9 boards, and evidence_trajectory.py showed the MTEB miss grows with J
and goes the wrong way (real BELOW twin). An exploratory reading after that
run (recorded in commit 7db67bf, not pre-registered) found that the SIZE of
the miss tracks how unequal the per-pair noise is: the two most homogeneous
boards (sigma_p CV 0.11, 0.13) fit best, the two least (0.47, 0.53) worst.
The first twin gives every system the same item noise. Real systems do not
have the same item noise.

Twin 2 keeps J, n, tau AND each system's own item-noise level s_j, and
nothing else. Two variants, to separate two hypotheses:

  2a  profile SHUFFLED  - the s_j are reassigned to random abilities. Tests
      whether unequal noise alone explains the miss.
  2b  profile ALIGNED   - s_j stays with the ability of the same RANK as the
      real system it came from. Tests whether it is the LINK between noise
      and ability (e.g. weak systems being noisier) that matters.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * twin 2a brings MTEB within 5 points (was -3.4 cross-section, -7.5 in the
    replay) and at least halves |dev| on TabArena 16 and 45;
  * twin 2a changes SWE-bench, LiveBench and HELM by no more than 2 points
    (they were already homogeneous);
  * count within 5 points: >= 8 of 9 for twin 2a (was 6 of 9);
  * if 2a fails where 2b succeeds, the noise-ability link is a fifth input
    the law needs, and the law is weaker than hoped: it would then require a
    shape of the field, not four numbers.

SELF-CHECKS
  * twin 2a of a heteroscedastic Gaussian field reproduces its H/ceiling
    within 3 points and its median sigma_p within 10 %;
  * twin 1 (homogeneous) on the same field must miss by MORE than twin 2a
    does, or the test has no power to distinguish them.

    python entropy_law_twin2.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import rank_sets as rs
import leaderboard_entropy as le
from entropy_law_test import MATRICES, gaussian_twin, stats_of

SEED = 20260823
DRAWS, SAMPLES = 800, 1000


def twin2(x: np.ndarray, rng, aligned: bool) -> np.ndarray:
    """Same J, n, tau and per-system item noise as x; Gaussian otherwise."""
    J, n = x.shape
    sc = x.mean(axis=1)
    tau = float(sc.std(ddof=1))
    s = x.std(axis=1, ddof=1)                       # per-system item SD
    latent = max(tau ** 2 - float(np.mean(s ** 2)) / n, 0.0) ** 0.5
    ability = rng.normal(0.0, latent, J)
    if aligned:
        # noise level travels with rank: the noisiest real system's s goes to
        # the twin ability of the same rank
        order_real = np.argsort(np.argsort(sc))     # rank of each real system
        order_twin = np.argsort(np.argsort(ability))
        s_use = np.empty(J)
        s_by_rank = s[np.argsort(sc)]               # s sorted by real score
        s_use = s_by_rank[order_twin]
    else:
        s_use = rng.permutation(s)
    return ability[:, None] + rng.normal(0.0, 1.0, (J, n)) * s_use[:, None]


def hfrac(x, rng):
    return stats_of(x, DRAWS, SAMPLES, rng)["H_frac"]


def _check_hetero_field():
    rng = np.random.default_rng(21)
    J, n = 80, 200
    ability = rng.normal(0, 0.06, J)
    s = rng.choice([0.15, 0.6], size=J)             # strongly unequal noise
    x = ability[:, None] + rng.normal(0, 1, (J, n)) * s[:, None]
    a = stats_of(x, 500, 600, rng)
    y = twin2(x, np.random.default_rng(22), aligned=False)
    b = stats_of(y, 500, 600, rng)
    z = gaussian_twin(a["J"], a["n"], a["tau"], a["sigma_p"], np.random.default_rng(23))
    c = stats_of(z, 500, 600, rng)
    g2, g1 = abs(a["H_frac"] - b["H_frac"]), abs(a["H_frac"] - c["H_frac"])
    ok = g2 < 0.03 and abs(b["sigma_p"] / a["sigma_p"] - 1) < 0.1 and g1 > g2
    return ok, (f"hetero field H/ceil {100 * a['H_frac']:.1f}: twin2a {100 * b['H_frac']:.1f} "
                f"(gap {100 * g2:.1f}), twin1 {100 * c['H_frac']:.1f} (gap {100 * g1:.1f}); "
                f"sigma_p ratio {b['sigma_p'] / a['sigma_p']:.2f}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok, msg = _check_hetero_field()
    print(f"  [{'ok  ' if ok else 'FAIL'}] {msg}")
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rng = np.random.default_rng(SEED)
    L = []
    p = L.append
    p("ENTROPY LAW, TWIN 2: SAME J, n, tau AND PER-SYSTEM NOISE PROFILE")
    p("=" * 86)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>4} {'sig CV':>7} {'real':>7} {'twin1':>7} {'dev1':>6}"
      f" {'twin2a':>7} {'dev2a':>6} {'twin2b':>7} {'dev2b':>6}")
    rows = {}
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        a = stats_of(x, DRAWS, SAMPLES, rng)
        J, n = x.shape
        r = rs.rank_sets(x, draws=300)
        iu = np.triu_indices(J, k=1)
        cv = float(r["sigma"][iu].std() / r["sigma"][iu].mean())
        t1 = np.mean([hfrac(gaussian_twin(J, n, a["tau"], a["sigma_p"], np.random.default_rng(SEED + 10 * s + 1)), rng) for s in range(2)])
        t2a = np.mean([hfrac(twin2(x, np.random.default_rng(SEED + 100 + s), False), rng) for s in range(2)])
        t2b = np.mean([hfrac(twin2(x, np.random.default_rng(SEED + 200 + s), True), rng) for s in range(2)])
        rows[name] = (a["H_frac"] - t1, a["H_frac"] - t2a, a["H_frac"] - t2b, cv)
        p(f"  {name:<22} {J:>4} {n:>4} {cv:>7.2f} {100 * a['H_frac']:>6.1f}% {100 * t1:>6.1f}% {100 * (a['H_frac'] - t1):>+5.1f}"
          f" {100 * t2a:>6.1f}% {100 * (a['H_frac'] - t2a):>+5.1f} {100 * t2b:>6.1f}% {100 * (a['H_frac'] - t2b):>+5.1f}")
    p("")
    w1 = sum(abs(v[0]) <= 0.05 for v in rows.values())
    w2a = sum(abs(v[1]) <= 0.05 for v in rows.values())
    w2b = sum(abs(v[2]) <= 0.05 for v in rows.values())
    p(f"  within 5 points: twin1 {w1}/{len(rows)}   twin2a {w2a}/{len(rows)}   twin2b {w2b}/{len(rows)}")
    p("  pre-registered for twin 2a: >= 8 of 9; MTEB within 5; TabArena |dev| at least halved;")
    p("  SWE-bench, LiveBench, HELM moved by <= 2 points.")
    checks = []
    m = rows.get("MTEB English v2")
    if m:
        checks.append(f"MTEB within 5: {'yes' if abs(m[1]) <= 0.05 else 'NO'} ({100 * m[1]:+.1f})")
    for k in ("TabArena 16 models", "TabArena 45 variants"):
        if k in rows:
            checks.append(f"{k} halved: {'yes' if abs(rows[k][1]) <= 0.5 * abs(rows[k][0]) else 'NO'} ({100 * rows[k][0]:+.1f} -> {100 * rows[k][1]:+.1f})")
    for k in ("SWE-bench Verified", "LiveBench", "HELM classic"):
        if k in rows:
            checks.append(f"{k.split()[0]} moved <= 2: {'yes' if abs(rows[k][1] - rows[k][0]) <= 0.02 else 'NO'} ({100 * (rows[k][1] - rows[k][0]):+.1f})")
    for c in checks:
        p("    " + c)
    p("")
    p("  twin1 = same J, n, tau, one noise level for all. twin2a = same J, n,")
    p("  tau and each system's own item-noise level, assigned at random. twin2b")
    p("  = the same levels kept with the ability of the same rank. dev = real -")
    p("  twin. If 2a closes the gap the law needs the noise PROFILE (a vector,")
    p("  still measurable before ranking anyone). If only 2b closes it, the law")
    p("  needs the noise-ability LINK, which is a property of the field.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("entropy_law_twin2_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote entropy_law_twin2_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
