"""Recompute every board under a construction that holds its coverage.

tie_coverage_boards.py measured the multiplier bootstrap undercovering on eight
of twelve boards under exact ties - HELM classic at 0.013, MTEB English v2 at
0.540 against a nominal 0.95 - while Holm on directional t-tests holds coverage
on every shape. The bias narrows rank sets, so every count of possible first
places on those boards is too low and every established share too high.

This recomputes the three quantities the repo's claims rest on, under all three
constructions, on all twelve boards:

  bootstrap   what the repo has been using
  holm        directional paired t-tests with Holm's FWER control
  union       per system, the wider of the two sets. Its simultaneous coverage
              is at least the larger of the two by construction: if either
              method contains every true rank, so does the union.

and re-tests law 1 against the corrected observed shares, because the law is
the piece PRIOR_ART.md left standing and it is fitted to nothing - its
prediction Phibar(1/SNR) moves only through c, the simultaneous critical value
the construction actually used.

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  tie@1 RISES on at least 6 of the 8 boards flagged as undercovering.
      Narrow sets hide ties; correcting them should surface ties.
  P2  tie@1 does not FALL on any of those 8.
  P3  the established share falls on at least 6 of the 8.
  P4  on the four boards where the bootstrap holds coverage - SWE-bench
      Verified, Lite, Test and MathArena - tie@1 moves by at most 2 systems
      under Holm. Where the machinery is valid the two constructions should
      agree.
  P5  law 1's mean absolute error across the nine boards does not worsen by
      more than 5 points under the corrected observed shares. If it worsens by
      more, the law was partly an artefact of the biased construction and
      PRIOR_ART.md's one surviving contribution has to be withdrawn.

  Not predicted: anything about entropy, which is reported for completeness
  but whose law was validated with a different tool and needs its own rerun.

SELF-CHECKS (no table if any fails)
  * the union must be no narrower than either input, everywhere;
  * Holm's beats matrix must be antisymmetric and consistent with the score
    order - a rejected pair must be recorded in the direction of the higher
    score, on every board;
  * the bootstrap column must reproduce the committed benchmark_health tie@1
    figures exactly on the boards that file covers, so any difference below is
    the construction and not a changed input.

    python holm_recompute.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import gammaln

import rank_sets as rs
import leaderboard_entropy as le
from tie_coverage import holm_rank_sets

SEED = 20260824
DRAWS = 800
SAMPLES = 800

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
    "SWE-bench Lite": "swebench_lite_matrix.csv",
    "SWE-bench Test": "swebench_test_matrix.csv",
    "LMArena categories": "lmarena_matrix.csv",
}

# From tie_coverage_boards_results.txt
UNDERCOVERS = {"HELM classic", "MTEB English v2", "CASP14", "LMArena categories",
               "TabArena 45 variants", "LiveBench", "ProteinGym DMS",
               "TabArena 16 models"}
SOUND = {"SWE-bench Verified", "SWE-bench Lite", "SWE-bench Test", "MathArena 2025"}

COMMITTED_TIE1 = {
    "SWE-bench Verified": 19, "MTEB English v2": 16, "HELM classic": 15,
    "ProteinGym DMS": 3, "TabArena 16 models": 8, "TabArena 45 variants": 12,
    "CASP14": 1, "LiveBench": 8, "MathArena 2025": 11,
}


def load(path):
    return pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)


def holm_full(x, alpha=0.05):
    """Holm rank sets from the single implementation in rank_sets.py.

    An earlier version of this file carried its own copy, and the two
    disagreed: on HELM classic, 10 items and 4 005 pairs, the local copy used
    the normal reference and reported 21 possible first places where the shared
    one, using t with n-1 degrees of freedom, reports 50. At a Bonferroni-level
    threshold the t with 9 degrees of freedom needs |t| about 8.5, which ten
    items essentially cannot reach - so the normal version was claiming a
    precision the sample size does not support. Two implementations of the same
    procedure that disagree by 29 systems is a defect in itself, so there is now
    one.
    """
    r = rs.rank_sets(x, method="holm", alpha=alpha)
    return {"beats": r["beats"], "best": r["best"], "worst": r["worst"],
            "crit": r["crit"], "theta": r["theta"]}


def union_of(a, b):
    return {"best": np.minimum(a["best"], b["best"]),
            "worst": np.maximum(a["worst"], b["worst"]),
            "beats": a["beats"] & b["beats"], "crit": max(a["crit"], b["crit"])}


def summarise(res, x, rng):
    J = x.shape[0]
    beats = res["beats"]
    H = le.log_extensions(beats, SAMPLES, rng)
    return {"tie1": int((res["best"] == 1).sum()),
            "estab": float(beats.sum() / (J * (J - 1))),
            "Hfrac": H["bits"] / (gammaln(J + 1) / math.log(2)),
            "crit": res["crit"]}


def law1(tau, sigma_p, n, c):
    return float(norm.sf(c * sigma_p / (math.sqrt(2 * n) * tau)))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    rows, bad_union, bad_holm = {}, [], []

    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        print(f"  {name} ...")
        x = load(path)
        J, n = x.shape
        rb = rs.rank_sets(x, draws=DRAWS)
        rh = holm_full(x)
        ru = union_of(rb, rh)
        if np.any(ru["best"] > rb["best"]) or np.any(ru["worst"] < rb["worst"]) \
           or np.any(ru["best"] > rh["best"]) or np.any(ru["worst"] < rh["worst"]):
            bad_union.append(name)
        if np.any(rh["beats"] & rh["beats"].T):
            bad_holm.append(name)
        else:
            a, b = np.nonzero(rh["beats"])
            if len(a) and np.any(rh["theta"][a] < rh["theta"][b]):
                bad_holm.append(name)
        sc = x.mean(axis=1)
        iu = np.triu_indices(J, k=1)
        rows[name] = {
            "J": J, "n": n, "tau": float(sc.std(ddof=1)),
            "sigma_p": float(np.median(rb["sigma"][iu])),
            "boot": summarise(rb, x, rng),
            "holm": summarise(rh, x, rng),
            "union": summarise(ru, x, rng)}

    print("self-checks ...")
    ok_u = not bad_union
    print(f"  [{'ok  ' if ok_u else 'FAIL'}] the union is never narrower than its inputs"
          + ("" if ok_u else "  off: " + ", ".join(bad_union)))
    ok_h = not bad_holm
    print(f"  [{'ok  ' if ok_h else 'FAIL'}] Holm's beats matrix is antisymmetric and "
          f"score-ordered" + ("" if ok_h else "  off: " + ", ".join(bad_holm)))
    par = [(k, rows[k]["boot"]["tie1"], COMMITTED_TIE1[k])
           for k in rows if k in COMMITTED_TIE1]
    badp = [t for t in par if t[1] != t[2]]
    ok_p = not badp
    print(f"  [{'ok  ' if ok_p else 'FAIL'}] bootstrap column reproduces benchmark_health: "
          f"{len(par) - len(badp)} of {len(par)}"
          + ("" if ok_p else "  " + "; ".join(f"{k} {a} vs {b}" for k, a, b in badp)))
    if not (ok_u and ok_h and ok_p):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("EVERY BOARD RECOMPUTED UNDER A CONSTRUCTION THAT HOLDS ITS COVERAGE")
    p("=" * 108)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>5} {'cover':>6} | "
      f"{'tie@1 boot':>10} {'holm':>5} {'union':>6} | "
      f"{'estab boot':>10} {'holm':>7} | {'H boot':>7} {'holm':>7}")
    for name, v in rows.items():
        flag = "BAD" if name in UNDERCOVERS else "ok"
        p(f"  {name:<22} {v['J']:>4} {v['n']:>5} {flag:>6} | "
          f"{v['boot']['tie1']:>10} {v['holm']['tie1']:>5} {v['union']['tie1']:>6} | "
          f"{100 * v['boot']['estab']:>9.1f}% {100 * v['holm']['estab']:>6.1f}% | "
          f"{100 * v['boot']['Hfrac']:>6.1f}% {100 * v['holm']['Hfrac']:>6.1f}%")
    p("")
    aff = [k for k in rows if k in UNDERCOVERS]
    rose = sum(1 for k in aff if rows[k]["holm"]["tie1"] > rows[k]["boot"]["tie1"])
    fell = sum(1 for k in aff if rows[k]["holm"]["tie1"] < rows[k]["boot"]["tie1"])
    estab_fell = sum(1 for k in aff if rows[k]["holm"]["estab"] < rows[k]["boot"]["estab"])
    sound = [k for k in rows if k in SOUND]
    agree = sum(1 for k in sound if abs(rows[k]["holm"]["tie1"] - rows[k]["boot"]["tie1"]) <= 2)
    p(f"  P1  tie@1 rises on {rose} of the {len(aff)} undercovering boards      "
      f"pre-registered >= 6:  {'HIT' if rose >= 6 else 'MISS'}")
    p(f"  P2  tie@1 falls on {fell} of them                            "
      f"pre-registered = 0:   {'HIT' if fell == 0 else 'MISS'}")
    p(f"  P3  established share falls on {estab_fell} of {len(aff)}                "
      f"pre-registered >= 6:  {'HIT' if estab_fell >= 6 else 'MISS'}")
    p(f"  P4  the four sound boards agree within 2 on {agree} of {len(sound)}       "
      f"pre-registered = {len(sound)}:   {'HIT' if agree == len(sound) else 'MISS'}")
    p("")
    p("  LAW 1 AGAINST THE CORRECTED SHARES")
    p(f"  {'leaderboard':<22} {'c boot':>7} {'c holm':>7} | {'obs boot':>9} {'pred':>7} "
      f"{'err':>6} | {'obs holm':>9} {'pred':>7} {'err':>6}")
    eb, eh = [], []
    for name, v in rows.items():
        if name not in COMMITTED_TIE1:
            continue
        pb = law1(v["tau"], v["sigma_p"], v["n"], v["boot"]["crit"])
        ph = law1(v["tau"], v["sigma_p"], v["n"], v["holm"]["crit"])
        db = 100 * (v["boot"]["estab"] - pb)
        dh = 100 * (v["holm"]["estab"] - ph)
        eb.append(abs(db))
        eh.append(abs(dh))
        p(f"  {name:<22} {v['boot']['crit']:>7.2f} {v['holm']['crit']:>7.2f} | "
          f"{100 * v['boot']['estab']:>8.1f}% {100 * pb:>6.1f}% {db:>+6.1f} | "
          f"{100 * v['holm']['estab']:>8.1f}% {100 * ph:>6.1f}% {dh:>+6.1f}")
    mb, mh = float(np.mean(eb)), float(np.mean(eh))
    p("")
    p(f"  mean |error|: bootstrap {mb:.1f} points, Holm {mh:.1f} points")
    p(f"  P5  law 1 does not worsen by more than 5 points: {mh - mb:+.1f}   "
      f"{'HIT' if mh - mb <= 5 else 'MISS'}")
    p("")
    p("  cover marks the eight boards tie_coverage_boards.py found undercovering.")
    p("  The union column is the wider of the two sets per system; its coverage")
    p("  is at least the better of the two by construction, and it is the")
    p("  conservative choice where neither is proven at that shape.")
    p("")
    p("  c is the simultaneous critical value each construction actually used -")
    p("  the bootstrap's multiplier quantile, and for Holm the z of the largest")
    p("  p-value it rejected. Law 1's prediction moves only through c, so the")
    p("  two prediction columns are the same formula asked of two instruments.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("holm_recompute_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                  newline=chr(10))
    print(chr(10) + "wrote holm_recompute_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
