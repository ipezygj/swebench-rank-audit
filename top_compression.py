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

The compression is also measured directly, without any twin arithmetic: the
gap between the k-th and (k+1)-th ranked systems, real over Gaussian, for k in
1..5 and for the middle third of the board. A field that is merely scaled
wrongly gives the same ratio everywhere; a field with a crowded top gives a
ratio below one at the top and near one in the bulk.

PRE-REGISTERED (2026-08-24, written and committed before the run)
  P1  T_shape recovers at least 60 % of the tie@1 miss, median over the boards
      where the plain twin misses by at least 2 systems.
  P2  T_corr recovers at most 40 %, same boards, same median.
  P3  T_shape lands closer to the real tie@1 than T_corr on at least 6 of
      those boards.
  P4  the top gap ratio C1 is below 1 on at least 7 of the 9 boards.
  P5  the median ratio over k=1..5 is at least 0.15 below the bulk ratio on at
      least 6 of the 9 boards - i.e. the compression is specific to the top
      and not a global scale error.

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
  * on a synthetic Gaussian board the two twins must agree within 3 systems
    and both gap ratios must sit in [0.6, 1.6] - if a Gaussian field's own
    "shape" already moves tie@1, the decomposition is measuring noise;
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
REPS = 9             # twin replicates per variant; odd, so the median is one of them
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


def tie1(x: np.ndarray) -> int:
    return int((rs.rank_sets(x, draws=DRAWS)["best"] == 1).sum())


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


def gap_profile(sc: np.ndarray, ktop: int = KTOP) -> tuple[np.ndarray, float]:
    """Top gaps 1..ktop, and the median gap in the middle third."""
    s = np.sort(sc)[::-1]
    top = s[:ktop] - s[1:ktop + 1]
    lo, hi = len(s) // 3, (2 * len(s)) // 3
    bulk = s[lo:hi] - s[lo + 1:hi + 1]
    return top, float(np.median(bulk)) if len(bulk) else float("nan")


def run_board(x: np.ndarray, seed: int) -> dict:
    f = field(x)
    out = {"J": f["J"], "n": f["n"], "real": f["tie1"]}
    real_top, real_bulk = gap_profile(f["sc"])
    reps = {"gauss": [], "shape": [], "corr": []}
    taus = {"gauss": [], "shape": []}
    gtop, gbulk = [], []
    # fixed per-kind offsets: Python's string hash is randomised per process
    offset = {"gauss": 1, "shape": 2, "corr": 3}
    for s in range(REPS):
        for kind, maker in (("gauss", twin_gauss), ("shape", twin_shape), ("corr", twin_corr)):
            y = maker(f, np.random.default_rng(seed + 100 * s + offset[kind]))
            reps[kind].append(tie1(y))
            sc = y.mean(axis=1)
            if kind in taus:
                taus[kind].append(float(sc.std(ddof=1)))
            if kind == "gauss":
                t, b = gap_profile(sc)
                gtop.append(t)
                gbulk.append(b)
    for kind in reps:
        out[kind] = float(np.median(reps[kind]))
    out["tau_gauss"] = float(np.median(taus["gauss"]))
    out["tau_shape"] = float(np.median(taus["shape"]))
    gtop = np.median(np.array(gtop), axis=0)
    gbulk = float(np.median(gbulk))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["C"] = np.where(gtop > 0, real_top / gtop, np.nan)
        out["C_bulk"] = real_bulk / gbulk if gbulk > 0 else float("nan")
    out["C1"] = float(out["C"][0])
    out["C_top"] = float(np.nanmedian(out["C"]))
    return out


def _check_gaussian_null() -> tuple[bool, str]:
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 0.05, 100)[:, None] + rng.normal(0.0, 0.4, (100, 300))
    r = run_board(x, seed=7)
    ok = (abs(r["shape"] - r["gauss"]) <= 3
          and 0.6 <= r["C1"] <= 1.6 and 0.6 <= r["C_bulk"] <= 1.6)
    return ok, (f"Gaussian null: tie@1 gauss {r['gauss']:.0f} shape {r['shape']:.0f}, "
                f"C1 {r['C1']:.2f}, bulk {r['C_bulk']:.2f}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-check 1 of 3 (Gaussian null) ...")
    ok_null, msg_null = _check_gaussian_null()
    print(f"  [{'ok  ' if ok_null else 'FAIL'}] {msg_null}")

    rows = {}
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        print(f"  measuring {name} ...")
        rows[name] = run_board(load(path), seed=SEED)

    parity = [(k, v["real"], COMMITTED_TIE1[k]) for k, v in rows.items()
              if COMMITTED_TIE1.get(k) is not None]
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

    if not (ok_null and ok_parity and ok_tau):
        print("\nA CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHY THE LAWS MISS THE TOP: SHAPE OR CORRELATION")
    p("=" * 100)
    p(f"  {'leaderboard':<22} {'J':>4} {'real':>5} {'gauss':>6} {'shape':>6} {'corr':>6} "
      f"{'rec.shape':>10} {'rec.corr':>9} {'C1':>6} {'C1-5':>6} {'bulk':>6}")
    rec_s, rec_c, closer, qual = [], [], 0, []
    c1_below, top_below_bulk = 0, 0
    for name, v in rows.items():
        miss = v["real"] - v["gauss"]
        if abs(miss) >= 2:
            qual.append(name)
            rs_ = (v["shape"] - v["gauss"]) / miss
            rc_ = (v["corr"] - v["gauss"]) / miss
            rec_s.append(rs_)
            rec_c.append(rc_)
            if abs(v["shape"] - v["real"]) < abs(v["corr"] - v["real"]):
                closer += 1
            rec_txt = (f"{100 * rs_:>9.0f}% {100 * rc_:>8.0f}%")
        else:
            rec_txt = f"{'-':>10} {'-':>9}"
        if v["C1"] < 1:
            c1_below += 1
        if not math.isnan(v["C_bulk"]) and v["C_top"] <= v["C_bulk"] - 0.15:
            top_below_bulk += 1
        p(f"  {name:<22} {v['J']:>4} {v['real']:>5.0f} {v['gauss']:>6.0f} {v['shape']:>6.0f} "
          f"{v['corr']:>6.0f} {rec_txt} {v['C1']:>6.2f} {v['C_top']:>6.2f} {v['C_bulk']:>6.2f}")
    p("")
    p(f"  boards where the plain twin misses by >= 2 systems: {len(qual)}")
    med_s = float(np.median(rec_s)) if rec_s else float("nan")
    med_c = float(np.median(rec_c)) if rec_c else float("nan")
    p(f"  P1  shape twin recovers {100 * med_s:.0f} % of the miss (median)   "
      f"pre-registered >= 60 %:  {'HIT' if med_s >= 0.60 else 'MISS'}")
    p(f"  P2  corr twin recovers {100 * med_c:.0f} % of the miss (median)    "
      f"pre-registered <= 40 %:  {'HIT' if med_c <= 0.40 else 'MISS'}")
    p(f"  P3  shape closer than corr on {closer} of {len(qual)}             "
      f"pre-registered >= 6:     {'HIT' if closer >= 6 else 'MISS'}")
    p(f"  P4  top gap ratio below 1 on {c1_below} of {len(rows)}              "
      f"pre-registered >= 7:     {'HIT' if c1_below >= 7 else 'MISS'}")
    p(f"  P5  top ratio at least 0.15 below bulk on {top_below_bulk} of {len(rows)} "
      f"pre-registered >= 6:     {'HIT' if top_below_bulk >= 6 else 'MISS'}")
    p("")
    p("  C1 is the gap between the printed first and second, divided by the gap a")
    p("  Gaussian field of the same spread puts there. C1-5 is the median of the")
    p("  first five such ratios, bulk the same ratio in the middle third of the")
    p("  board. A field that is simply scaled wrongly moves all three together.")
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
