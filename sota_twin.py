"""Is the share of 'real' SOTA advances predictable from the field's drift?

sota_audit.py found that of 60 frontier advances on three dated boards, 18
were pairwise-separable from the previous leader and 7 simultaneously. Those
are audit numbers. This asks whether they are also LAW numbers: is the share
determined by a few measurable quantities of the field, with nothing about
which systems arrived when?

The dated twin: keep J, n, the ENTRY DATES, the per-pair noise sigma_p, and
a linear drift fitted to the real scores over time,

    score_j = a + beta * t_j + N(0, tau_res^2),    t in years

and nothing else. Generate the field, replay it with the identical audit
(running max, sign-flip pairwise test, simultaneous beats among the systems
present at that date), and compare three numbers with the real board:

    A  number of frontier advances
    P  share of advances pairwise-separable (p < 0.05)
    S  share of advances simultaneously separable

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * A within +-30 % of the real count on each of the three boards;
  * P and S each within +-10 points of the real share on each board
    (binomial noise alone at n=12-32 advances is 5-9 points);
  * the twin will OVER-predict A on SWE-bench: its real history is slow
    then fast (agent scaffolds arrived in 2024), a linear drift spreads the
    climb evenly and produces more records early;
  * no prediction for the last-unambiguous-leader date.

SYMMETRY NOTE  The real board is re-audited here with the same sign-flip
test used on the twin (binary or not), so real and twin are judged by one
instrument. The real SWE-bench pairwise count may therefore differ from
sota_audit_results.txt, which used McNemar exact for binary data.

SELF-CHECKS
  * with beta = 0 the twin's advance count must be within 25 % of the
    harmonic number H_J (records of an exchangeable sequence);
  * a twin of a twin must reproduce A within 30 % and P, S within 10 points.

    python sota_twin.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from sota_audit import advances, paired_test, fmt
from evidence_trajectory import load, DATED

SEED = 20260823
DRAWS = 300
TWINS = 10


def years(dates: np.ndarray) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates.astype(int).astype(str)), format="%Y%m%d")
    return ((d - d.min()).dt.days / 365.25).to_numpy()


def audit(x: np.ndarray, dates: np.ndarray, seed: int) -> dict:
    """A, P, S for one matrix with the sign-flip pairwise test throughout."""
    adv = advances(x, dates)
    pair = sim = 0
    for k, ad in enumerate(adv):
        new, old, d = ad["new"], ad["old"], ad["date"]
        # force the continuous test for both binary and continuous data
        dd = x[new] - x[old]
        dd = dd[dd != 0]
        if len(dd):
            rng = np.random.default_rng(seed + k)
            flips = rng.choice([-1.0, 1.0], size=(4000, len(dd)))
            null = np.abs((flips * dd[None, :]).mean(axis=1))
            pv = float((np.sum(null >= abs(dd.mean()) - 1e-15) + 1) / (len(null) + 1))
        else:
            pv = 1.0
        pair += pv < 0.05
        present = np.flatnonzero(dates <= d)
        r = rs.rank_sets(x[present], draws=DRAWS, seed=seed + 1000 + k)
        pi = {int(s): i for i, s in enumerate(present)}
        sim += bool(r["beats"][pi[new], pi[old]])
    A = len(adv)
    return {"A": A, "P": pair / A if A else float("nan"), "S": sim / A if A else float("nan")}


def fit_drift(x: np.ndarray, dates: np.ndarray, sigma_p: float):
    sc = x.mean(axis=1)
    t = years(dates)
    X = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(X, sc, rcond=None)
    resid = sc - X @ coef
    n = x.shape[1]
    sigma_item = sigma_p / math.sqrt(2)
    tau_res = max(float(resid.var(ddof=2)) - sigma_item ** 2 / n, 0.0) ** 0.5
    return float(coef[0]), float(coef[1]), tau_res, sigma_item


def dated_twin(J, n, dates, a, beta, tau_res, sigma_item, rng):
    t = years(dates)
    ability = a + beta * t + rng.normal(0, tau_res, J)
    return ability[:, None] + rng.normal(0, sigma_item, (J, n))


def sigma_p_of(x):
    r = rs.rank_sets(x, draws=DRAWS, seed=SEED)
    J = x.shape[0]
    iu = np.triu_indices(J, k=1)
    return float(np.median(r["sigma"][iu]))


def _check_no_drift():
    rng = np.random.default_rng(7)
    J, n = 80, 150
    dates = 20230101 + np.arange(J)
    x = dated_twin(J, n, dates, 0.5, 0.0, 0.05, 0.4, rng)
    A = len(advances(x, dates))
    HJ = sum(1 / k for k in range(1, J + 1))
    # average over a few fields - a single one is too noisy
    As = [len(advances(dated_twin(J, n, dates, 0.5, 0.0, 0.05, 0.4, np.random.default_rng(70 + s)), dates)) for s in range(40)]
    ok = abs(np.mean(As) / HJ - 1) < 0.25
    return ok, f"no-drift records: mean A {np.mean(As):.2f} vs H_J {HJ:.2f}"


def _check_twin_of_twin():
    rng = np.random.default_rng(9)
    J, n = 60, 150
    dates = 20230101 + np.sort(rng.integers(0, 700, J))
    x = dated_twin(J, n, dates, 0.3, 0.15, 0.04, 0.4, rng)
    real = audit(x, dates, 100)
    sp = sigma_p_of(x)
    a, b, tr, si = fit_drift(x, dates, sp)
    res = [audit(dated_twin(J, n, dates, a, b, tr, si, np.random.default_rng(200 + s)), dates, 300 + s) for s in range(6)]
    A = np.mean([r["A"] for r in res]); P = np.nanmean([r["P"] for r in res]); S = np.nanmean([r["S"] for r in res])
    ok = abs(A / real["A"] - 1) < 0.3 and abs(P - real["P"]) < 0.10 and abs(S - real["S"]) < 0.10
    return ok, (f"twin of twin: A {real['A']} vs {A:.1f}; P {100 * real['P']:.0f} vs {100 * P:.0f}; "
                f"S {100 * real['S']:.0f} vs {100 * S:.0f}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_no_drift(), _check_twin_of_twin()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("IS THE SHARE OF SEPARABLE SOTA ADVANCES A FUNCTION OF THE FIELD'S DRIFT?")
    p("=" * 86)
    p(f"  {'leaderboard':<20} {'J':>4} {'beta/yr':>8} {'tau_res':>8} {'sigma_p':>8} "
      f"{'A real':>7} {'A twin':>8} {'P real':>7} {'P twin':>7} {'S real':>7} {'S twin':>7}")
    verdicts = []
    for name, (path, dc) in DATED.items():
        x, dates = load(path, dc)
        J, n = x.shape
        sp = sigma_p_of(x)
        a, beta, tau_res, si = fit_drift(x, dates, sp)
        real = audit(x, dates, SEED)
        tw = [audit(dated_twin(J, n, dates, a, beta, tau_res, si, np.random.default_rng(SEED + 50 + s)), dates, SEED + 500 + 100 * s)
              for s in range(TWINS)]
        A = float(np.mean([t["A"] for t in tw])); Alo, Ahi = np.min([t["A"] for t in tw]), np.max([t["A"] for t in tw])
        P = float(np.nanmean([t["P"] for t in tw])); S = float(np.nanmean([t["S"] for t in tw]))
        okA = abs(A / real["A"] - 1) <= 0.30
        okP = abs(P - real["P"]) <= 0.10
        okS = abs(S - real["S"]) <= 0.10
        verdicts.append((name, okA, okP, okS, A / real["A"] - 1))
        p(f"  {name:<20} {J:>4} {beta:>+8.3f} {tau_res:>8.4f} {sp:>8.3f} "
          f"{real['A']:>7d} {A:>5.1f}[{Alo}-{Ahi}] {100 * real['P']:>6.0f}% {100 * P:>6.0f}% {100 * real['S']:>6.0f}% {100 * S:>6.0f}%")
    p("")
    p("  pre-registered: A within +-30 %, P and S within +-10 points, on each board;")
    p("  twin OVER-predicts A on SWE-bench (linear drift vs slow-then-fast history).")
    for name, okA, okP, okS, dA in verdicts:
        p(f"    {name:<20} A {'ok' if okA else 'MISS'} ({100 * dA:+.0f} %)   P {'ok' if okP else 'MISS'}   S {'ok' if okS else 'MISS'}")
    sw = next(v for v in verdicts if v[0].startswith("SWE"))
    p(f"    SWE-bench over-prediction of A: {'yes' if sw[4] > 0 else 'NO'}")
    p("")
    p("  The twin keeps J, n, the entry dates, sigma_p and a LINEAR drift with")
    p("  its residual spread - nothing about which system arrived when. Where")
    p("  P and S match, the share of a board's SOTA claims that survive the")
    p("  standard was set by how fast the field rose relative to how finely")
    p("  the benchmark resolves - before anyone claimed anything.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("sota_twin_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote sota_twin_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
