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

PRE-REGISTERED (2026-08-24, third and final registration)

  Written and committed before any board was read with these statistics. Two
  earlier registrations in this file were retired by its own self-checks,
  both for the same defect and both before any board was read:

    first   the gap statistic was one observation over a twin MEDIAN. In a
            Gaussian field of 100 the top gap has sd/mean 0.91 and p90/p50
            2.93 - very nearly exponential - so that ratio is noise with a
            skew. Replaced by the gap's percentile inside the twin's own
            distribution, which is uniform under the null and therefore has
            a null check that can fail. It now passes at 9 of 20 below 0.50.

    second  tie@1 had the same defect one level up. The shape twin holds ONE
            ability configuration and redraws only noise; the Gaussian twin
            redrew ability too, so their medians were not comparable. And
            tie@1 turns out to be wildly configuration-dependent: on one
            field spec, 60 Gaussian ability draws gave tie@1 sd 16.4 and a
            5-95 range of 26 to 79. Every twin quantity is therefore reported
            as an INTERVAL here, and the question becomes containment rather
            than distance.

  That second failure has a consequence beyond this file. LAWS.md and
  target_board.py state the top-of-board miss as "3 against 19" - two point
  values. If 19 falls inside the twin's own 5-95 interval, that sentence is
  not supported by its own measurement and has to be corrected. P1 tests
  exactly that, and I have not looked.

  P1  the real tie@1 lies OUTSIDE the Gaussian twin's central 90 % interval
      on at least 6 of the 9 boards. A miss here refutes the claim that the
      laws miss the top, and LAWS.md gets rewritten rather than defended.
  P2  the real tie@1 lies INSIDE the shape twin's central 90 % interval on at
      least 6 of the 9 boards.
  P3  the shape twin's median is closer to the real tie@1 than the
      correlation twin's, on at least 6 of the boards where the Gaussian
      twin's median is at least 2 systems away.
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
  ordering has LARGER top gaps than the truth. P2 is therefore tested against
  a twin biased towards separating, not towards tying.

SELF-CHECKS (the table is not printed if any fails)
  * real tie@1 must reproduce the committed benchmark_health figure on all
    nine boards, exactly;
  * gap calibration: on 20 synthetic Gaussian boards the top-gap percentile
    must be below 0.50 on 5 to 15 of them and average between 0.30 and 0.70;
  * containment calibration: on 5 synthetic Gaussian boards, the board's own
    tie@1 must fall inside its Gaussian twin's central 90 % interval on at
    least 4 - if a Gaussian field falls outside its own twin, P1 is measuring
    the machinery and not the field;
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
REPS_TIE = 99        # twin replicates for tie@1 (each needs a bootstrap)
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
    """Full tie@1 distribution for each twin, not a point.

    gauss redraws ability AND noise, so its spread is the spread of fields of
    this spec. shape holds the real ability shape and redraws only noise, so
    its spread is the spread of one field re-measured. corr redraws ability
    and keeps the real residual matrix.
    """
    offset = {"gauss": 1, "shape": 2, "corr": 3}
    out = {}
    for kind, maker in (("gauss", twin_gauss), ("shape", twin_shape), ("corr", twin_corr)):
        ties, tt = [], []
        for s in range(REPS_TIE):
            y = maker(f, np.random.default_rng(seed + 1000 * s + offset[kind]))
            ties.append(int((rs.rank_sets(y, draws=DRAWS)["best"] == 1).sum()))
            tt.append(float(y.mean(axis=1).std(ddof=1)))
        v = np.array(ties, dtype=float)
        out[kind] = float(np.median(v))
        out[kind + "_lo"] = float(np.percentile(v, 5))
        out[kind + "_hi"] = float(np.percentile(v, 95))
        out["tau_" + kind] = float(np.median(tt))
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


def _check_containment() -> tuple[bool, str]:
    """A Gaussian field must fall inside its own Gaussian twin's interval."""
    inside = 0
    for b in range(5):
        rng = np.random.default_rng(700 + b)
        x = rng.normal(0.0, 0.05, 100)[:, None] + rng.normal(0.0, 0.4, (100, 300))
        f = field(x)
        t = tie_twins(f, seed=1300 + b)
        if t["gauss_lo"] <= f["tie1"] <= t["gauss_hi"]:
            inside += 1
    return inside >= 4, f"containment calibration: {inside} of 5 Gaussian boards inside their own twin"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks on synthetic fields ...")
    ok_cal, msg_cal = _check_calibration()
    print(f"  [{'ok  ' if ok_cal else 'FAIL'}] {msg_cal}")
    ok_ct, msg_ct = _check_containment()
    print(f"  [{'ok  ' if ok_ct else 'FAIL'}] {msg_ct}")

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

    if not (ok_cal and ok_ct and ok_parity and ok_tau):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHY THE LAWS MISS THE TOP - AND WHETHER THEY MISS IT AT ALL")
    p("=" * 108)
    p(f"  {'leaderboard':<22} {'J':>4} {'real':>5} {'gaussian twin':>17} {'shape twin':>17} "
      f"{'corr twin':>16} {'q top':>7} {'q bulk':>7}")
    outside_g, inside_s, closer, qual = 0, 0, 0, []
    q1_below, top_below_bulk = 0, 0
    for name, v in rows.items():
        og = not (v["gauss_lo"] <= v["real"] <= v["gauss_hi"])
        isx = v["shape_lo"] <= v["real"] <= v["shape_hi"]
        outside_g += og
        inside_s += isx
        if abs(v["real"] - v["gauss"]) >= 2:
            qual.append(name)
            if abs(v["shape"] - v["real"]) < abs(v["corr"] - v["real"]):
                closer += 1
        if v["q1"] < 0.50:
            q1_below += 1
        if v["q1"] <= v["q_bulk"] - 0.25:
            top_below_bulk += 1
        g = f"{v['gauss']:.0f} [{v['gauss_lo']:.0f}-{v['gauss_hi']:.0f}]" + ("*" if og else "")
        sh = f"{v['shape']:.0f} [{v['shape_lo']:.0f}-{v['shape_hi']:.0f}]" + ("" if isx else "*")
        c = f"{v['corr']:.0f} [{v['corr_lo']:.0f}-{v['corr_hi']:.0f}]"
        p(f"  {name:<22} {v['J']:>4} {v['real']:>5.0f} {g:>17} {sh:>17} {c:>16} "
          f"{v['q1']:>7.3f} {v['q_bulk']:>7.3f}")
    p("")
    p("  Intervals are the twin's central 90 % over 99 replicates. A star on the")
    p("  Gaussian column means the real board falls outside it; a star on the")
    p("  shape column means it falls outside that one too.")
    p("")
    p(f"  P1  real outside the Gaussian twin's interval on {outside_g} of {len(rows)}      "
      f"pre-registered >= 6:  {'HIT' if outside_g >= 6 else 'MISS'}")
    p(f"  P2  real inside the shape twin's interval on {inside_s} of {len(rows)}          "
      f"pre-registered >= 6:  {'HIT' if inside_s >= 6 else 'MISS'}")
    p(f"  P3  shape median closer than corr on {closer} of {len(qual)}                 "
      f"pre-registered >= 6:  {'HIT' if closer >= 6 else 'MISS'}")
    p(f"  P4  top gap below the twin median on {q1_below} of {len(rows)}               "
      f"pre-registered >= 7:  {'HIT' if q1_below >= 7 else 'MISS'}")
    p(f"  P5  top percentile >= 0.25 below bulk on {top_below_bulk} of {len(rows)}         "
      f"pre-registered >= 6:  {'HIT' if top_below_bulk >= 6 else 'MISS'}")
    p("")
    p("  q top is where the real gap between the printed first and second falls")
    p("  inside the distribution of that gap in 999 Gaussian fields of the same")
    p("  spread: 0.02 means only 2 % of Gaussian fields put their top two that")
    p("  close together. q bulk is the same reading for the median gap in the")
    p("  middle third of the board.")
    p("")
    p("  The Gaussian interval is the correction this file owes its own repo:")
    p("  target_board.py prints the twin's tie@1 as a single number and LAWS.md")
    p("  quotes it as one. Whether a board's top departs from a Gaussian field")
    p("  is a containment question, and only the interval can answer it.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("top_compression_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote top_compression_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
