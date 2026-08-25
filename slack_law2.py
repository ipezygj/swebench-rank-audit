"""slack_law.py fitted a curve in density alone. That curve cannot be right.

Its four predictions all hit and it predicted eight real boards from simulation
with an error of 0.46 bits, so it is not wrong about what it measured. It is
wrong about what it is. Every board in it has 18 systems, and slack was written
as a function of edge density with no J in it at all.

slack_scaling.py had already made that untenable and neither file noticed.
Growing a board's sub-poset from 8 systems to 24 leaves its density almost
where it started - SWE-bench Verified 0.821 to 0.804, HELM classic 0.107 to
0.098, LiveBench 0.714 to 0.609 - while the slack goes from 0.00 to 2.18, from
0.72 to 6.50, from 0.00 to 3.62. Density held, slack multiplied. A function of
density alone cannot do that.

Recorded before writing this file, and stated as measured rather than
predicted: across the eight boards, the density at 24 systems differs from the
density at 8 by at most 0.36 and by 0.09 in the median, while the slack rises on
every one of them.

So the curve needs a second variable, and the natural candidate is J itself.
This sweeps simulated boards across BOTH axes - 7 sizes from 10 to 22, spreads
chosen to spread the density - fits both forms, and compares them on boards
never used in the fit, simulated and real.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  at matched density, slack rises with J: comparing the smallest and
      largest sizes present, the larger has more slack in at least 3 of the 4
      density bins.
  P2  the dependence is close to proportional: slack divided by J varies less
      across sizes, within a density bin, than slack does, in at least 3 of 4
      bins.
  P3  on HELD-OUT simulated boards, the two-variable form J * g(density) has a
      lower mean absolute error than the one-variable form f(density). The
      split is by simulation cell, not by row.
  P4  and on the eight REAL 22-system sub-posets, fitted on simulation only,
      the two-variable form also wins. This is the one that matters: a form
      that only wins on its own simulations has not earned anything.

  What a miss on P4 would mean: the extra variable is fitting the simulator
  rather than the phenomenon, and slack_law.py's single-size curve should be
  left standing as the honest limit of what is known.

SELF-CHECKS (no table if any fails)
  * each density bin must contain boards from at least 5 distinct sizes, or
    "slack rises with J at matched density" is not a measurement;
  * at the degenerate ends slack must still be exactly zero at every size;
  * the held-out split must remove whole simulation cells, verified by
    asserting no held-out cell appears in the training set;
  * at least 200 simulated boards.

    python slack_law2.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from band_slack import band_matrix, bands_of, permanent01
from entropy_law_test import MATRICES
from exact_extensions import exact_log2

SEED = 20260825
SIZES = (10, 12, 14, 16, 18, 20, 22)
NS = (20, 40, 80, 160, 320, 640)
SPREADS = (0.006, 0.012, 0.022, 0.04, 0.07, 0.12, 0.2, 0.32)
REPS = 2
REAL_J = 22
BINS = (0.0, 0.25, 0.45, 0.65, 1.01)


def slack_of(beats: np.ndarray) -> float:
    best, worst = bands_of(beats)
    return (math.log2(permanent01(band_matrix(best, worst)))
            - math.log2(exact_log2(beats)[0]))


def board(J: int, n: int, tau: float, rng) -> np.ndarray:
    p = np.clip(0.5 + rng.normal(0, tau, J), 0.02, 0.98)
    return (rng.random((J, n)) < p[:, None]).astype(float)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)

    print("self-checks ...")
    deg = []
    for J in SIZES:
        theta = np.linspace(0, 1, J)
        deg.append(slack_of((theta[:, None] - theta[None, :]) > 0))
        deg.append(slack_of(np.zeros((J, J), dtype=bool)))
    ok_deg = all(v == 0.0 for v in deg)
    print(f"  [{'ok  ' if ok_deg else 'FAIL'}] slack exactly zero at both degenerate "
          f"ends for all {len(SIZES)} sizes")

    sim = []          # (J, density, slack, cell id)
    for J in SIZES:
        for n in NS:
            for tau in SPREADS:
                cell = (J, n, tau)
                for _ in range(REPS):
                    b = rs.rank_sets(board(J, n, tau, rng))["beats"]
                    d = b.sum() / (J * (J - 1) / 2)
                    sim.append((J, d, slack_of(b), cell))
        print(f"  simulated J={J}")
    Js = np.array([r[0] for r in sim], float)
    D = np.array([r[1] for r in sim], float)
    S = np.array([r[2] for r in sim], float)
    cells = [r[3] for r in sim]

    binid = np.digitize(D, BINS) - 1
    sizes_per_bin = [len(set(Js[binid == k].tolist())) for k in range(len(BINS) - 1)]
    ok_bins = all(v >= 5 for v in sizes_per_bin)
    print(f"  [{'ok  ' if ok_bins else 'FAIL'}] distinct sizes per density bin: "
          f"{sizes_per_bin} (need >= 5 each)")
    ok_n = len(sim) >= 200
    print(f"  [{'ok  ' if ok_n else 'FAIL'}] {len(sim)} simulated boards (need >= 200)")

    uniq = sorted(set(cells))
    hold = set(uniq[::4])
    tr = np.array([c not in hold for c in cells])
    ok_split = not (set(np.array(cells, dtype=object)[~tr].tolist())
                    & set(np.array(cells, dtype=object)[tr].tolist()))
    print(f"  [{'ok  ' if ok_split else 'FAIL'}] the held-out split removes whole cells: "
          f"{int(tr.sum())} train, {int((~tr).sum())} held out, no cell in both")

    if not (ok_deg and ok_bins and ok_n and ok_split):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    # the two forms, fitted on the training rows only
    c1 = np.polyfit(D[tr], S[tr], 4)                      # f(density)
    c2 = np.polyfit(D[tr], S[tr] / Js[tr], 4)             # J * g(density)

    def p1(d, J):
        return np.polyval(c1, d)

    def p2(d, J):
        return J * np.polyval(c2, d)

    mae1 = float(np.mean(np.abs(p1(D[~tr], Js[~tr]) - S[~tr])))
    mae2 = float(np.mean(np.abs(p2(D[~tr], Js[~tr]) - S[~tr])))

    real = []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        Jb = x.shape[0]
        if Jb < REAL_J:
            continue
        b = rs.rank_sets(x)["beats"]
        pick = np.sort(np.random.default_rng(SEED + Jb).permutation(Jb)[:REAL_J])
        sub = b[np.ix_(pick, pick)]
        real.append((name, sub.sum() / (REAL_J * (REAL_J - 1) / 2), slack_of(sub)))
    rd = np.array([r[1] for r in real])
    rs_ = np.array([r[2] for r in real])
    rmae1 = float(np.mean(np.abs(p1(rd, REAL_J) - rs_)))
    rmae2 = float(np.mean(np.abs(p2(rd, REAL_J) - rs_)))

    L = []
    p = L.append
    p("SLACK IS NOT A FUNCTION OF DENSITY ALONE")
    p("=" * 96)
    p(f"  {len(sim)} simulated boards across {len(SIZES)} sizes ({min(SIZES)} to "
      f"{max(SIZES)} systems) and {len(NS)} item counts.")
    p("")
    p(f"  mean slack by size and density bin")
    p(f"  {'J':>4}" + "".join(f"{f'{BINS[k]:.2f}-{BINS[k+1]:.2f}':>14}"
                              for k in range(len(BINS) - 1)))
    for J in SIZES:
        cellsrow = []
        for k in range(len(BINS) - 1):
            m = (Js == J) & (binid == k)
            cellsrow.append(f"{S[m].mean():>14.2f}" if m.sum() else f"{'-':>14}")
        p(f"  {J:>4}" + "".join(cellsrow))
    p("")
    p(f"  the same divided by J")
    p(f"  {'J':>4}" + "".join(f"{f'{BINS[k]:.2f}-{BINS[k+1]:.2f}':>14}"
                              for k in range(len(BINS) - 1)))
    for J in SIZES:
        cellsrow = []
        for k in range(len(BINS) - 1):
            m = (Js == J) & (binid == k)
            cellsrow.append(f"{(S[m] / J).mean():>14.3f}" if m.sum() else f"{'-':>14}")
        p(f"  {J:>4}" + "".join(cellsrow))
    p("")
    rises = 0
    tighter = 0
    for k in range(len(BINS) - 1):
        m = binid == k
        if not m.sum():
            continue
        lo, hi = Js[m].min(), Js[m].max()
        a = S[m & (Js == lo)].mean()
        b_ = S[m & (Js == hi)].mean()
        rises += b_ > a
        v1 = np.std([S[m & (Js == J)].mean() for J in SIZES if (m & (Js == J)).sum()])
        v2 = np.std([(S[m & (Js == J)] / J).mean() for J in SIZES
                     if (m & (Js == J)).sum()])
        tighter += (v2 / max(np.mean([(S[m & (Js == J)] / J).mean() for J in SIZES
                                      if (m & (Js == J)).sum()]), 1e-9)
                    < v1 / max(np.mean([S[m & (Js == J)].mean() for J in SIZES
                                        if (m & (Js == J)).sum()]), 1e-9))
    nb = len(BINS) - 1
    p(f"  P1  slack rises with J in {rises} of {nb} density bins        "
      f"pre-registered >= 3:  {'HIT' if rises >= 3 else 'MISS'}")
    p(f"  P2  slack/J varies relatively less in {tighter} of {nb} bins   "
      f"pre-registered >= 3:  {'HIT' if tighter >= 3 else 'MISS'}")
    p(f"  P3  held-out simulated MAE: f(d) {mae1:.3f}, J*g(d) {mae2:.3f}   "
      f"pre-registered J*g wins:  {'HIT' if mae2 < mae1 else 'MISS'}")
    p(f"  P4  real boards at J={REAL_J}: f(d) {rmae1:.3f}, J*g(d) {rmae2:.3f}    "
      f"pre-registered J*g wins:  {'HIT' if rmae2 < rmae1 else 'MISS'}")
    p("")
    p(f"  {'board':<22}{'density':>10}{'slack':>9}{'f(d)':>9}{'J*g(d)':>10}")
    for (name, d, s), q1, q2 in zip(real, p1(rd, REAL_J), p2(rd, REAL_J)):
        p(f"  {name:<22}{d:>10.3f}{s:>9.3f}{q1:>9.3f}{q2:>10.3f}")
    p("")
    p("  RECORDED, NOT PREDICTED, measured before this file was written. Growing")
    p("  a real board's sub-poset from 8 systems to 24 barely moves its density")
    p("  - SWE-bench Verified 0.821 to 0.804, HELM classic 0.107 to 0.098,")
    p("  LiveBench 0.714 to 0.609 - while the slack rises on every board, from")
    p("  0.00 to 2.18, from 0.72 to 6.50, from 0.00 to 3.62. That alone rules")
    p("  out any function of density alone, and it was sitting in")
    p("  slack_scaling_results.txt and slack_law_results.txt at the same time,")
    p("  in two files that never had their numbers put side by side.")
    p("")
    p("  slack_law.py is not retracted. Its four predictions hold and its curve")
    p("  predicts real 18-system sub-posets to 0.46 bits. What it is, is a")
    p("  section through a surface at J = 18, and it was written as though the")
    p("  surface had only one axis.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("slack_law2_results.txt").write_text(text + chr(10), encoding="utf-8",
                                              newline=chr(10))
    print(chr(10) + "wrote slack_law2_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
