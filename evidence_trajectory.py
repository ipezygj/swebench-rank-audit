"""Do the two laws hold THROUGH TIME inside one leaderboard?

resolution_law_test.py and entropy_law_test.py tested the laws across nine
leaderboards at one moment each. A leaderboard with dated entries gives a
second, sharper test: replay it. At each checkpoint date take the systems
that existed, run the identical machinery, and compare the observed
established share with Phibar(1/SNR(t)) and the observed H/ceiling with the
Gaussian twin's. Inside one leaderboard n and sigma_p are fixed; only J(t)
and tau(t) move. If the laws hold at every checkpoint, the whole history of
the leaderboard's resolving power is a function of how many systems there
were and how spread out they were - nothing about WHICH systems.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * law 1 (established share): within 5 points at >= 80 % of checkpoints
    on each of SWE-bench, MTEB, LiveBench - all three held cross-sectionally;
  * early checkpoints (J < 15) will be the worst: tau from few systems is a
    poor estimate of the field's spread, and c(J) is small;
  * law 2 (entropy vs twin): within 5 points at >= 70 % of checkpoints;
    misses, if any, in the direction real > twin (clustered fields);
  * no directional prediction for how established share moves over time -
    reported descriptively, as a reading, not a result.

SELF-CHECKS
  * the final checkpoint (all systems) must reproduce the full-matrix
    established share of resolution_law_test within 1.5 points (different
    draws, same data);
  * a Gaussian field replayed with growing J must satisfy law 1 within 3
    points at every checkpoint with J >= 15.

    python evidence_trajectory.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm

import rank_sets as rs
import leaderboard_entropy as le
from sota_audit import parse_dates, fmt
from entropy_law_test import gaussian_twin

SEED = 20260823
DRAWS, SAMPLES = 600, 600
DATED = {
    "SWE-bench Verified": ("swebench_verified_matrix.csv", None),
    "MTEB English v2": ("mteb_dated_matrix.csv", "mteb_dates.csv"),
    "LiveBench": ("livebench/matrix.csv", "livebench/dates.csv"),
}


def load(path, dates_csv):
    df = pd.read_csv(path, index_col=0).dropna(axis=0)
    names = list(df.index)
    if dates_csv:
        dd = pd.read_csv(dates_csv, index_col=0)["date"]
        dates = np.array([int(dd.loc[n]) for n in names])
    else:
        dates = parse_dates(names)
    return df.to_numpy(dtype=float), dates


def checkpoints(dates, k=8, jmin=10):
    J = len(dates)
    ds = np.sort(dates)
    out = []
    for target in np.linspace(jmin, J, k):
        t = int(round(target))
        d = int(ds[t - 1])
        if not out or d != out[-1]:
            out.append(d)
    return out


def measure(x, rng, entropy=True):
    J, n = x.shape
    r = rs.rank_sets(x, draws=DRAWS)
    sc = x.mean(axis=1)
    tau = float(sc.std(ddof=1))
    iu = np.triu_indices(J, k=1)
    sigma_p = float(np.median(r["sigma"][iu]))
    c = float(r["crit"])
    obs = float(r["beats"].sum() / (J * (J - 1)))
    snr = tau * math.sqrt(2 * n) / (c * sigma_p)
    pred = float(norm.sf(1 / snr)) if snr > 0 else 0.0
    d = {"J": J, "n": n, "tau": tau, "sigma_p": sigma_p, "crit": c,
         "obs": obs, "snr": snr, "pred": pred}
    if entropy:
        H = le.log_extensions(r["beats"], SAMPLES, rng)["bits"]
        d["H"] = H / (gammaln(J + 1) / math.log(2))
        tw = []
        for s in range(2):
            y = gaussian_twin(J, n, tau, sigma_p, np.random.default_rng(SEED + 7 * s + 3))
            rt = rs.rank_sets(y, draws=DRAWS)
            tw.append(le.log_extensions(rt["beats"], SAMPLES, rng)["bits"] / (gammaln(J + 1) / math.log(2)))
        d["H_twin"] = float(np.mean(tw))
    return d


def _check_gaussian_replay():
    rng = np.random.default_rng(11)
    J, n = 100, 300
    x = 0.5 + rng.normal(0, 0.08, J)[:, None] + rng.normal(0, 0.45, (J, n))
    x = np.clip(x, -1, 1)
    dates = 20230101 + np.arange(J)          # one entrant per "day", any order
    worst = 0.0
    for d in checkpoints(dates, k=6, jmin=15):
        m = measure(x[dates <= d], rng, entropy=False)
        worst = max(worst, abs(m["pred"] - m["obs"]))
    return worst < 0.03, f"Gaussian replay, worst |pred-obs| over checkpoints with J>=15: {100 * worst:.1f} points"


def _check_final_matches_full():
    ref = Path("resolution_law_test_results.txt")
    if not ref.exists():
        return True, "resolution_law_test_results.txt absent; check skipped"
    want = {}
    for line in ref.read_text(encoding="utf-8").splitlines():
        for k in DATED:
            if line.strip().startswith(k):
                toks = line.split()
                # observed% is the first token ending with '%'
                want[k] = float(next(t for t in toks if t.endswith("%")).rstrip("%")) / 100
    rng = np.random.default_rng(1)
    worst, msgs = 0.0, []
    for k, (p, dc) in DATED.items():
        if k not in want:
            continue
        x, _ = load(p, dc)
        m = measure(x, rng, entropy=False)
        gap = abs(m["obs"] - want[k])
        worst = max(worst, gap)
        msgs.append(f"{k.split()[0]} {100 * m['obs']:.1f} vs {100 * want[k]:.1f}")
    return worst < 0.015, "final checkpoint vs full-matrix table: " + "; ".join(msgs)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_gaussian_replay(), _check_final_matches_full()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rng = np.random.default_rng(SEED)
    L = []
    p = L.append
    p("DO THE LAWS HOLD THROUGH TIME? REPLAYED LEADERBOARDS")
    p("=" * 84)
    summary = {}
    for name, (path, dc) in DATED.items():
        x, dates = load(path, dc)
        p("")
        p(f"  {name}   ({x.shape[0]} systems, {x.shape[1]} items, {fmt(int(dates.min()))} to {fmt(int(dates.max()))})")
        p(f"  {'date':>10} {'J':>4} {'tau':>7} {'SNR':>6} {'estab':>7} {'law':>7} {'err':>6}"
          f" {'H/ceil':>7} {'twin':>7} {'err':>6}")
        e1, e2, e1_big = [], [], []
        for d in checkpoints(dates):
            m = measure(x[dates <= d], rng)
            a, b = m["pred"] - m["obs"], m["H"] - m["H_twin"]
            e1.append(abs(a)); e2.append(abs(b))
            if m["J"] >= 15:
                e1_big.append(abs(a))
            p(f"  {fmt(d):>10} {m['J']:>4} {m['tau']:>7.4f} {m['snr']:>6.2f} {100 * m['obs']:>6.1f}%"
              f" {100 * m['pred']:>6.1f}% {100 * a:>+5.1f} {100 * m['H']:>6.1f}% {100 * m['H_twin']:>6.1f}% {100 * b:>+5.1f}")
        summary[name] = (np.mean(np.array(e1) <= 0.05), np.mean(np.array(e2) <= 0.05),
                         np.mean(np.array(e1_big) <= 0.05) if e1_big else float("nan"))
    p("")
    p("  pre-registered: law 1 within 5 points at >= 80 % of checkpoints; law 2 at >= 70 %")
    for k, (s1, s2, s1b) in summary.items():
        v1 = "HELD" if s1 >= 0.8 else "MISSED"
        v2 = "HELD" if s2 >= 0.7 else "MISSED"
        p(f"  {k:<22} law 1 {100 * s1:>3.0f} % ({v1}; J>=15 only: {100 * s1b:.0f} %)   law 2 {100 * s2:>3.0f} % ({v2})")
    p("")
    p("  'estab' and 'law' are ORDERED-pair shares (ceiling 50 %). Inside one")
    p("  leaderboard n and sigma_p are fixed, so the law's only moving inputs")
    p("  are J(t) through c(J) and tau(t). Where the error column stays small the")
    p("  replayed history of the board's resolving power was determined by how")
    p("  many systems there were and how spread they were, not by which.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("evidence_trajectory_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote evidence_trajectory_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
