"""Why the laws reproduce a board and still miss its top by a factor of six.

Both laws work on the aggregate. A Gaussian field with SWE-bench Verified's
(J, n, tau, sigma_p) reproduces its established share to 0.4 points and its
entropy to 1.5, and then says three systems could be first where nineteen can
(`target_board.py`). LAWS.md explains that away in one sentence - "every real
board has a dense cluster at the top that a Gaussian field does not" - which
is an assertion, not a measurement. This measures it, and splits the miss.

TWO CANDIDATE CAUSES, AND THE TWIN THAT ISOLATES EACH

  shape        the real field's top is compressed: the gaps between the best
               systems are smaller than a Gaussian field of the same spread
               puts there. Isolated by keeping the real score SHAPE (rescaled
               to the same latent spread) and giving it synthetic independent
               item noise.
  correlation  the real per-item outcomes are correlated across systems and
               unequally noisy, which changes how far apart two systems must
               be to separate. Isolated by keeping the real RESIDUAL matrix
               and hanging it on a Gaussian ability vector.

Both twins have the same J, n and latent spread as the plain Gaussian twin, so
whichever moves tie@1 towards the real number is carrying the information the
four numbers do not.

WHY THE GAP STATISTIC IS A PERCENTILE AND NOT A RATIO
------------------------------------------------------
The first version of this file compared the real gap between ranks k and k+1
against the median gap of a Gaussian twin, as a ratio. Its own Gaussian-null
self-check failed at 1.93, and the reason is not the field but the estimator:
in a Gaussian sample of 100 the top gap has sd/mean 0.91 and p90/p50 2.93 -
it is very nearly an exponential draw - while the median gap in the middle
third has sd/mean 0.23. A ratio of one draw to a median of that distribution
is noise with a skew, and a threshold on it tests almost nothing. The gap is
therefore reported as the real gap's PERCENTILE within the twin's own
distribution of the same gap, over 999 twins. That statistic is uniform under
the null by construction, which is what makes the null check able to fail.

The measurement is worth keeping for its own sake: "we beat second place by X
points" is one draw from a distribution whose standard deviation equals its
mean, before any question of item noise is asked.

PRE-REGISTERED (2026-08-24)
  Written and committed before the run. P1-P3 are unchanged from the first
  commit of this file; P4-P5 replace two predictions about the ratio
  estimator that its own self-check retired before any board was read.

  P1  T_shape recovers at least 60 % of the tie@1 miss, median over the boards
      where the plain twin misses by at least 2 systems.
  P2  T_corr recovers at most 40 %, same boards, same median.
  P3  T_shape lands closer to the real tie@1 than T_corr on at least 6 of
      those boards.
  P4  the real top gap sits below the twin median - percentile < 0.50 - on at
      least 7 of the 9 boards. (Under the null each board is uniform, so 7 of
      9 one-sided has p = 0.09; 8 or 9 would be p = 0.02 or 0.004.)
  P5  the top gap's percentile is at least 0.25 below the bulk gap's on at
      least 6 of the 9 boards - the compression is specific to the top and not
      a board-wide scale error.

  Not predicted: the direction on CASP14, whose top t is 9.89 and whose single
  possible first place leaves nothing to compress; and nothing about LMArena,
  which the nine-board tools hold out.

  Note against my own interest: T_shape is built from OBSERVED scores, whose
  spread already contains measurement noise. Rescaling to the latent spread
  removes the excess spread but not the noise in the ordering, and a noisy
  ordering has LARGER top gaps than the truth. P1 is therefore tested against
  a twin biased towards separating, not towards tying.

SELF-CHECKS (the table is not printed if any fails)
  * real tie@1 must reproduce the committed benchmark_health figure on all
    nine boards, exactly;
  * calibration: on 20 synthetic Gaussian boards the top-gap percentile must
    be below 0.50 on 5 to 15 of them and average between 0.30 and 0.70 - if
    the statistic is not uniform under the null, every board's number is a
    reading of the estimator;
  * on a synthetic Gaussian board the shape and Gaussian twins must agree on
    tie@1 within 3 systems;
  * T_shape's realised score spread must be within 10 % of T_gauss's on every
    board, or the comparison is a spread comparison wearing a shape label.

    python top_compression.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs

SEED = 20260824
DRAWS = 800          # the setting the standard's report cards use
REPS_TIE = 25        # twin replicates for tie@1 (each needs a bootstrap)
REPS_GAP = 999       # twin replicates for the gap null (means only, cheap)
KTOP = 5

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

# From benchmark_health_results.txt, committed 2026-08-23. Parity, not input.
COMMITTED_TIE1 = {
    "SWE-bench Verified": 19, "MTEB English v2": 16, "HELM classic": 15,
    "ProteinGym DMS": 3, "TabArena 16 models": 8, "TabArena 45 variants": 12,
    "CASP14": 1, "LiveBench": 8, "MathArena 2025": 11,
}


def load(path: str) -> np.ndarray:
    return pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)


def field(x: np.ndarray) -> dict:
    """The four numbers the twins are allowed to know, plus the residuals."""
    J, n = x.shape
    r = rs.rank_sets(x, draws=DRAWS)
    sc = x.mean(axis=1)
    iu = np.triu_indices(J, k=1)
    sigma_p = float(np.median(r["sigma"][iu]))
    sigma_item = sigma_p / math.sqrt(2.0)
    tau = float(sc.std(ddof=1))
    latent = max(tau ** 2 - sigma_item ** 2 / n, 0.0) ** 0.5
    return {"J": J, "n": n, "sc": sc, "tau": tau, "sigma_item": sigma_item,
            "latent": latent, "resid": x - sc[:, None],
            "tie1": int((r["best"] == 1).sum())}


def twin_gauss(f: dict, rng) -> np.ndarray:
    a = rng.normal(0.0, f["latent"], f["J"])
    return a[:, None] + rng.normal(0.0, f["sigma_item"], (f["J"], f["n"]))


def twin_shape(f: dict, rng) -> np.ndarray:
    """Real ordering and real relative gaps, rescaled to the latent spread."""
    dev = f["sc"] - f["sc"].mean()
    a = dev * (f["latent"] / f["tau"]) if f["tau"] > 0 else dev
    return a[:, None] + rng.normal(0.0, f["sigma_item"], (f["J"], f["n"]))


def twin_corr(f: dict, rng) -> np.ndarray:
    """Gaussian ordering carrying the real item-level residuals."""
    a = rng.normal(0.0, f["latent"], f["J"])
    return a[:, None] + f["resid"]


def gap_stats(sc: np.ndarray, ktop: int = KTOP) -> tuple[np.ndarray, float]:
    """Top gaps 1..ktop, and the median gap in the middle third."""
    s = np.sort(sc)[::-1]
    top = s[:ktop] - s[1:ktop + 1]
    lo, hi = len(s) // 3, (2 * len(s)) // 3
    bulk = s[lo:hi] - s[lo + 1:hi + 1]
    return top, (float(np.median(bulk)) if len(bulk) else float("nan"))


def gap_percentiles(f: dict, seed: int, reps: int = REPS_GAP) -> dict:
    """Where the real gaps fall inside the Gaussian twin's own distribution."""
    real_top, real_bulk = gap_stats(f["sc"])
    T = np.empty((reps, KTOP))
    B = np.empty(reps)
    rng = np.random.default_rng(seed)
    sigma_mean = f["sigma_item"] / math.sqrt(f["n"])
    for s in range(reps):
        # only the score means matter here, so draw them directly
        sc = rng.normal(0.0, f["latent"], f["J"]) + rng.normal(0.0, sigma_mean, f["J"])
        T[s], B[s] = gap_stats(sc)
    q_top = (T < real_top[None, :]).mean(axis=0)
    q_bulk = float((B < real_bulk).mean())
    return {"q1": float(q_top[0]), "q_top": float(np.mean(q_top)), "q_bulk": q_bulk,
            "real_top1": float(real_top[0]), "twin_top1": float(np.median(T[:, 0]))}


def tie_twins(f: dict, seed: int) -> dict:
    offset = {"gauss": 1, "shape": 2, "corr": 3}
    out, taus = {}, {}
    for kind, maker in (("gauss", twin_gauss), ("shape", twin_shape), ("corr", twin_corr)):
        ties, tt = [], []
        for s in range(REPS_TIE):
            y = maker(f, np.random.default_rng(seed + 100 * s + offset[kind]))
            ties.append(int((rs.rank_sets(y, draws=DRAWS)["best"] == 1).sum()))
            tt.append(float(y.mean(axis=1).std(ddof=1)))
        out[kind] = float(np.median(ties))
        taus[kind] = float(np.median(tt))
    out["tau_gauss"], out["tau_shape"] = taus["gauss"], taus["shape"]
    return out


def _check_calibration() -> tuple[bool, str]:
    """The gap percentile must be uniform on fields that really are Gaussian."""
    qs = []
    for b in range(20):
        rng = np.random.default_rng(500 + b)
        x = rng.normal(0.0, 0.05, 100)[:, None] + rng.normal(0.0, 0.4, (100, 300))
        f = field(x)
        qs.append(gap_percentiles(f, seed=900 + b, reps=299)["q1"])
    below = sum(q < 0.5 for q in qs)
    mean = float(np.mean(qs))
    ok = 5 <= below <= 15 and 0.30 <= mean <= 0.70
    return ok, f"null calibration: {below} of 20 below 0.50, mean percentile {mean:.2f}"


def _check_gaussian_twins() -> tuple[bool, str]:
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 0.05, 100)[:, None] + rng.normal(0.0, 0.4, (100, 300))
    t = tie_twins(field(x), seed=7)
    ok = abs(t["shape"] - t["gauss"]) <= 3
    return ok, f"Gaussian board: tie@1 gauss {t['gauss']:.0f}, shape {t['shape']:.0f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks 1 and 2 (null calibration, Gaussian twins) ...")
    ok_cal, msg_cal = _check_calibration()
    print(f"  [{'ok  ' if ok_cal else 'FAIL'}] {msg_cal}")
    ok_gt, msg_gt = _check_gaussian_twins()
    print(f"  [{'ok  ' if ok_gt else 'FAIL'}] {msg_gt}")

    rows = {}
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        print(f"  measuring {name} ...")
        f = field(load(path))
        v = {"J": f["J"], "n": f["n"], "real": f["tie1"]}
        v.update(gap_percentiles(f, seed=SEED))
        v.update(tie_twins(f, seed=SEED))
        rows[name] = v

    parity = [(k, v["real"], COMMITTED_TIE1[k]) for k, v in rows.items() if k in COMMITTED_TIE1]
    bad_parity = [t for t in parity if t[1] != t[2]]
    ok_parity = not bad_parity
    print(f"  [{'ok  ' if ok_parity else 'FAIL'}] parity with benchmark_health: "
          f"{len(parity) - len(bad_parity)} of {len(parity)} exact"
          + ("" if ok_parity else "  " + "; ".join(f"{k} {a} vs {b}" for k, a, b in bad_parity)))

    bad_tau = [k for k, v in rows.items()
               if v["tau_gauss"] > 0 and abs(v["tau_shape"] / v["tau_gauss"] - 1) > 0.10]
    ok_tau = not bad_tau
    print(f"  [{'ok  ' if ok_tau else 'FAIL'}] shape twin's spread within 10 % of the "
          f"Gaussian twin's on {len(rows) - len(bad_tau)} of {len(rows)}"
          + ("" if ok_tau else "  off: " + ", ".join(bad_tau)))

    if not (ok_cal and ok_gt and ok_parity and ok_tau):
        print("\nA CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHY THE LAWS MISS THE TOP: SHAPE OR CORRELATION")
    p("=" * 104)
    p(f"  {'leaderboard':<22} {'J':>4} {'real':>5} {'gauss':>6} {'shape':>6} {'corr':>6} "
      f"{'rec.shape':>10} {'rec.corr':>9} {'q top':>7} {'q 1-5':>7} {'q bulk':>7}")
    rec_s, rec_c, closer, qual = [], [], 0, []
    q1_below, top_below_bulk = 0, 0
    for name, v in rows.items():
        miss = v["real"] - v["gauss"]
        if abs(miss) >= 2:
            qual.append(name)
            a = (v["shape"] - v["gauss"]) / miss
            b = (v["corr"] - v["gauss"]) / miss
            rec_s.append(a)
            rec_c.append(b)
            if abs(v["shape"] - v["real"]) < abs(v["corr"] - v["real"]):
                closer += 1
            rec_txt = f"{100 * a:>9.0f}% {100 * b:>8.0f}%"
        else:
            rec_txt = f"{'-':>10} {'-':>9}"
        if v["q1"] < 0.50:
            q1_below += 1
        if v["q1"] <= v["q_bulk"] - 0.25:
            top_below_bulk += 1
        p(f"  {name:<22} {v['J']:>4} {v['real']:>5.0f} {v['gauss']:>6.0f} {v['shape']:>6.0f} "
          f"{v['corr']:>6.0f} {rec_txt} {v['q1']:>7.3f} {v['q_top']:>7.3f} {v['q_bulk']:>7.3f}")
    p("")
    p(f"  boards where the plain twin misses by >= 2 systems: {len(qual)}")
    med_s = float(np.median(rec_s)) if rec_s else float("nan")
    med_c = float(np.median(rec_c)) if rec_c else float("nan")
    p(f"  P1  shape twin recovers {100 * med_s:.0f} % of the miss (median)    "
      f"pre-registered >= 60 %:  {'HIT' if med_s >= 0.60 else 'MISS'}")
    p(f"  P2  corr twin recovers {100 * med_c:.0f} % of the miss (median)     "
      f"pre-registered <= 40 %:  {'HIT' if med_c <= 0.40 else 'MISS'}")
    p(f"  P3  shape closer than corr on {closer} of {len(qual)}              "
      f"pre-registered >= 6:     {'HIT' if closer >= 6 else 'MISS'}")
    p(f"  P4  top gap below the twin median on {q1_below} of {len(rows)}       "
      f"pre-registered >= 7:     {'HIT' if q1_below >= 7 else 'MISS'}")
    p(f"  P5  top percentile >= 0.25 below bulk on {top_below_bulk} of {len(rows)}   "
      f"pre-registered >= 6:     {'HIT' if top_below_bulk >= 6 else 'MISS'}")
    p("")
    p("  q top is where the real gap between the printed first and second falls")
    p("  inside the distribution of that gap in 999 Gaussian fields of the same")
    p("  spread: 0.02 means only 2 % of Gaussian fields put their top two that")
    p("  close together. q 1-5 averages the first five gaps, q bulk is the same")
    p("  reading for the median gap in the middle third of the board.")
    p("")
    p("  rec.shape and rec.corr are the share of the twin's tie@1 miss closed by")
    p("  keeping the real score shape, and by keeping the real residual matrix,")
    p("  each with everything else Gaussian and the same latent spread.")
    text = "\n".join(L)
    print("\n" + text)
    Path("top_compression_results.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    print("\nwrote top_compression_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
