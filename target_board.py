"""What would a benchmark that can name a winner look like?

Every measurement tonight says what existing boards cannot do. A benchmark
owner needs the other half: the numbers a board WOULD have if it were built
to resolve its top. Law 1 gives the design equation - the established share
is Phibar(1/SNR) with SNR = tau sqrt(2n) / (c sigma_p) - so the target can
be constructed rather than wished for.

Simulated boards of the same size as SWE-bench Verified (134 systems, 500
items), varying only the field's spread relative to the item noise, and run
through the same report card. The output is a lookup table: what SNR buys
what.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * at SNR 6 the board has tie@1 = 1 and entropy below 20 % of ceiling;
  * SWE-bench Verified's own SNR (3.1) reproduces roughly its own numbers
    (tie@1 near 19, entropy near 54 %) - the simulation is calibrated
    against the real board it is modelled on;
  * the tie@1 curve is steep between SNR 3 and 6 and flat after.

SELF-CHECKS
  * the simulated established share matches law 1's prediction within 3
    points at every SNR;
  * two seeds at the same SNR give tie@1 within 3 of each other.

    python target_board.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm

import rank_sets as rs
import leaderboard_entropy as le
import leaderboard_geometry as lg

SEED = 20260823
J, N = 134, 500
SIGMA_ITEM = 0.34            # SWE-bench Verified's own per-item SD, roughly
SNRS = (1.5, 2.0, 3.1, 4.0, 6.0, 9.0)
DRAWS, SAMPLES = 1000, 1500


def build(snr, seed):
    """A Gaussian board whose SNR is `snr` at this J, n and item noise."""
    rng = np.random.default_rng(seed)
    # SNR = tau sqrt(2n) / (c sigma_p); sigma_p = sqrt(2) sigma_item, and c
    # is not known before the run, so it is taken from a pilot at tau = 0.05
    pilot = 0.5 + rng.normal(0, 0.05, J)[:, None] + rng.normal(0, SIGMA_ITEM, (J, N))
    c = float(rs.rank_sets(pilot, draws=400)["crit"])
    sigma_p = math.sqrt(2) * SIGMA_ITEM
    tau = snr * c * sigma_p / math.sqrt(2 * N)
    x = 0.5 + rng.normal(0, tau, J)[:, None] + rng.normal(0, SIGMA_ITEM, (J, N))
    return x, c, tau


def card(x, rng):
    r = rs.rank_sets(x, draws=DRAWS)
    H = le.log_extensions(r["beats"], SAMPLES, rng)["bits"]
    tiers = len(lg.tiers(r["beats"]))
    order = np.argsort(-x.mean(axis=1))
    i1, i2 = int(order[0]), int(order[1])
    d = x[i1] - x[i2]
    t = float(d.mean() / (d.std(ddof=1) / math.sqrt(x.shape[1])))
    return {"tie1": int((r["best"] == 1).sum()),
            "estab": float(r["beats"].sum() / (J * (J - 1))),
            "H": H / (gammaln(J + 1) / math.log(2)),
            "tiers": tiers, "t": t,
            "width": float(np.median(r["worst"] - r["best"] + 1))}


def _check_law(rng):
    worst = 0.0
    for snr in (2.0, 4.0):
        x, c, tau = build(snr, SEED + int(snr * 10))
        r = rs.rank_sets(x, draws=600)
        obs = float(r["beats"].sum() / (J * (J - 1)))
        pred = float(norm.sf(1 / snr))
        worst = max(worst, abs(obs - pred))
    return worst < 0.03, f"law 1 on the simulated boards: worst error {100 * worst:.1f} points"


def _check_seeds(rng):
    a = card(build(4.0, 11)[0], rng)["tie1"]
    b = card(build(4.0, 12)[0], rng)["tie1"]
    return abs(a - b) <= 3, f"two seeds at SNR 4: tie@1 {a} and {b}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    print("self-checks")
    ok = True
    for passed, msg in (_check_law(rng), _check_seeds(rng)):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHAT A BOARD THAT CAN NAME A WINNER LOOKS LIKE")
    p("=" * 84)
    p(f"  {J} systems, {N} items, per-item SD {SIGMA_ITEM} - the shape of SWE-bench Verified.")
    p(f"  Only the spread of the field changes.")
    p("")
    p(f"  {'SNR':>5} {'tau':>8} {'estab':>8} {'tie@1':>7} {'entropy':>9} {'tiers':>7} {'median width':>13} {'top t':>7}")
    rows = []
    for snr in SNRS:
        x, c, tau = build(snr, SEED + int(snr * 100))
        m = card(x, rng)
        rows.append((snr, m))
        p(f"  {snr:>5.1f} {tau:>8.4f} {100 * m['estab']:>7.1f}% {m['tie1']:>7} {100 * m['H']:>8.1f}% "
          f"{m['tiers']:>7} {m['width']:>13.0f} {m['t']:>7.2f}")
    p("")
    six = next(m for s, m in rows if s == 6.0)
    three = next(m for s, m in rows if abs(s - 3.1) < 0.01)
    p(f"  at SNR 6: tie@1 {six['tie1']} and entropy {100 * six['H']:.1f} % "
      f"(pre-registered tie@1 = 1 and entropy < 20 %)")
    p(f"  at SWE-bench's own SNR 3.1: tie@1 {three['tie1']}, entropy {100 * three['H']:.1f} % "
      f"(the real board: 19 and 54.2 %)")
    p("")
    p("  Read as a design table. A board with SWE-bench's size and item noise")
    p("  needs its systems spread over roughly twice the range they currently")
    p("  occupy before its top becomes a single name - or, equivalently, four")
    p("  times the items at the same spread, since SNR grows as sqrt(n).")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("target_board_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote target_board_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
