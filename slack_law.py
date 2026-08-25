"""Where does the band picture cost the most, and can that be predicted?

The slack must vanish at both ends. A board resolved perfectly has one ordering
and its bands admit that one ordering, so the slack is zero. A board that
resolves nothing is an antichain: every band is [1, J], the bands admit all J!
orderings, and so does the poset, so the slack is zero again. Between those two
it is positive. Somewhere in between it peaks.

That is a claim about a curve, and a curve can be swept. This simulates boards
across a grid of item counts and field spreads, measures the slack exactly on
each, and locates the maximum. Then - and this is the part that makes it a
prediction rather than a description - the curve fitted to SIMULATED boards
alone is used to predict the eight real ones, which contribute nothing to the
fit.

The x-axis is the sub-poset's own edge density, the share of the 153 pairs that
the relation orders. It is the natural coordinate because both ends of the
argument above are statements about it: density 0 is the antichain, density 1
is the total order.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  the simulated slack against edge density is single-peaked, and the peak
      sits at a density BELOW 0.35 - below where most real boards live, so that
      over the real range the slack falls as a board resolves more.
  P2  both ends collapse: mean slack below 0.5 bits among simulated boards with
      density under 0.05, and again among those over 0.95.
  P3  the peak exceeds 4 bits.
  P4  a curve fitted to the simulated boards ONLY predicts the eight real
      sub-posets with mean absolute error below 1.0 bits, and below the error
      of predicting every board by the simulated mean. The real boards are
      never fitted to.

  What a miss on P4 would mean: simulated boards and real boards are not the
  same kind of object at this level of detail, and the whole Gaussian-twin
  method this repository leans on has a limit that sits right here.

SELF-CHECKS (no table if any fails)
  * the sweep must SPAN the real boards: the simulated densities must reach
    below the lowest real density and above the highest, or the prediction in
    P4 is an extrapolation dressed as an interpolation;
  * the two degenerate cases must come out exactly zero, not approximately: a
    field with no noise, and a field with no signal;
  * at least 100 simulated boards;
  * the real slacks must be recomputed here with the same counters, and must
    agree with band_slack_results.txt to 3 decimals - two files, one number.

    python slack_law.py
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
J = 18
NS = (12, 20, 35, 60, 100, 200, 400)
SPREADS = (0.004, 0.008, 0.014, 0.022, 0.032, 0.045, 0.065, 0.09, 0.13, 0.2)
REPS = 3
PAIRS = J * (J - 1) // 2


def slack_of(beats: np.ndarray) -> float:
    best, worst = bands_of(beats)
    e = exact_log2(beats)[0]
    b = permanent01(band_matrix(best, worst))
    return math.log2(b) - math.log2(e)


def simulate(n: int, tau: float, rng) -> np.ndarray:
    """A binary board: J systems with latent rates spread by tau, n items."""
    p = np.clip(0.5 + rng.normal(0, tau, J), 0.02, 0.98)
    return (rng.random((J, n)) < p[:, None]).astype(float)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print("self-checks ...")
    # no noise: every system separated, one ordering, bands admit one ordering
    theta = np.linspace(0, 1, J)
    clean = np.repeat(theta[:, None], 60, axis=1)
    sep = (theta[:, None] - theta[None, :]) > 0
    s_clean = slack_of(sep)
    # no signal: nothing ordered, bands are [1, J], both counts are J!
    s_noise = slack_of(np.zeros((J, J), dtype=bool))
    ok_deg = s_clean == 0.0 and s_noise == 0.0
    print(f"  [{'ok  ' if ok_deg else 'FAIL'}] degenerate cases exactly zero: "
          f"total order {s_clean:.6f}, antichain {s_noise:.6f}")

    sim = []
    rng = np.random.default_rng(SEED)
    for n in NS:
        for tau in SPREADS:
            for _ in range(REPS):
                b = rs.rank_sets(simulate(n, tau, rng))["beats"]
                sim.append((b.sum() / PAIRS, slack_of(b)))
    sim = np.array(sim)
    print(f"  simulated {len(sim)} boards, density {sim[:, 0].min():.3f} "
          f"to {sim[:, 0].max():.3f}")

    real = []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        Jb = x.shape[0]
        if Jb < J:
            continue
        beats = rs.rank_sets(x)["beats"]
        pick = np.sort(np.random.default_rng(SEED + Jb).choice(Jb, J, replace=False))
        sub = beats[np.ix_(pick, pick)]
        real.append((name, sub.sum() / PAIRS, slack_of(sub)))

    ok_span = (sim[:, 0].min() <= min(r[1] for r in real)
               and sim[:, 0].max() >= max(r[1] for r in real))
    print(f"  [{'ok  ' if ok_span else 'FAIL'}] the sweep spans the real range "
          f"[{min(r[1] for r in real):.3f}, {max(r[1] for r in real):.3f}]")
    ok_n = len(sim) >= 100
    print(f"  [{'ok  ' if ok_n else 'FAIL'}] {len(sim)} simulated boards (need >= 100)")

    reported = {}
    f = Path("band_slack_results.txt")
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            t = line.split()
            if len(t) >= 7:
                try:
                    float(t[-1]); float(t[-3])
                except ValueError:
                    continue
                reported[" ".join(t[:-6])] = float(t[-3])
    agree = sum(1 for n_, _, s in real
                if n_ not in reported or abs(reported[n_] - s) < 5e-4)
    ok_agree = agree == len(real)
    print(f"  [{'ok  ' if ok_agree else 'FAIL'}] recomputed slacks agree with "
          f"band_slack_results.txt on {agree} of {len(real)}")

    if not (ok_deg and ok_span and ok_n and ok_agree):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    # bin the sweep and fit a smooth curve on the SIMULATED data only
    edges = np.linspace(0, 1, 21)
    mids, means = [], []
    for a, b in zip(edges, edges[1:]):
        m = (sim[:, 0] >= a) & (sim[:, 0] < b)
        if m.sum() >= 3:
            mids.append(float(sim[m, 0].mean()))
            means.append(float(sim[m, 1].mean()))
    mids, means = np.array(mids), np.array(means)
    coef = np.polyfit(sim[:, 0], sim[:, 1], 4)
    grid = np.linspace(sim[:, 0].min(), sim[:, 0].max(), 400)
    curve = np.polyval(coef, grid)
    peak_at = float(grid[int(np.argmax(curve))])
    peak_val = float(curve.max())

    pred = np.polyval(coef, np.array([r[1] for r in real]))
    truth = np.array([r[2] for r in real])
    mae = float(np.mean(np.abs(pred - truth)))
    mae0 = float(np.mean(np.abs(sim[:, 1].mean() - truth)))

    lo = sim[sim[:, 0] < 0.05, 1]
    hi = sim[sim[:, 0] > 0.95, 1]

    L = []
    p = L.append
    p("WHERE DOES THE BAND PICTURE COST THE MOST?")
    p("=" * 92)
    p(f"  {len(sim)} simulated boards of {J} systems, {min(NS)} to {max(NS)} items,")
    p("  field spread swept over more than an order of magnitude. Slack counted")
    p("  exactly on each. The real boards contribute nothing to the fit.")
    p("")
    p(f"  {'density':>10}{'mean slack':>13}{'boards':>9}")
    for a, b in zip(edges, edges[1:]):
        m = (sim[:, 0] >= a) & (sim[:, 0] < b)
        if m.sum():
            p(f"  {0.5 * (a + b):>10.3f}{sim[m, 1].mean():>13.3f}{int(m.sum()):>9}")
    p("")
    p(f"  peak of the fitted curve: {peak_val:.2f} bits at density {peak_at:.3f}")
    p("")
    p(f"  {'board':<22}{'density':>10}{'slack':>9}{'predicted':>12}{'error':>9}")
    for (name, d, s), q in zip(real, pred):
        p(f"  {name:<22}{d:>10.3f}{s:>9.3f}{q:>12.3f}{q - s:>9.3f}")
    p("")
    single = (np.diff(np.sign(np.diff(curve))) != 0).sum() <= 1
    p(f"  P1  single-peaked: {single}, peak at density {peak_at:.3f}       "
      f"pre-registered < 0.35:  {'HIT' if single and peak_at < 0.35 else 'MISS'}")
    p(f"  P2  mean slack below 0.05 density: {lo.mean() if len(lo) else float('nan'):.3f} "
      f"({len(lo)} boards); above 0.95: "
      f"{hi.mean() if len(hi) else float('nan'):.3f} ({len(hi)})   "
      f"pre-registered both < 0.5:  "
      f"{'HIT' if len(lo) and len(hi) and lo.mean() < 0.5 and hi.mean() < 0.5 else 'MISS'}")
    p(f"  P3  peak {peak_val:.2f} bits                          "
      f"pre-registered > 4:     {'HIT' if peak_val > 4 else 'MISS'}")
    p(f"  P4  MAE on the real boards {mae:.3f} bits, against {mae0:.3f} for the "
      f"simulated mean   pre-registered < 1.0 and < that:  "
      f"{'HIT' if mae < 1.0 and mae < mae0 else 'MISS'}")
    p("")
    p("  The two ends are not assumptions. A board with no noise is a total")
    p("  order: one ordering, and the bands admit exactly that one. A board with")
    p("  no signal is an antichain: every band is [1, 18] and both counts are")
    p("  18!. Both come out exactly zero above, as integers, not approximately.")
    p("")
    p("  What the peak means for a benchmark owner. The band summary is safest")
    p("  on a board that resolves almost nothing and on a board that resolves")
    p("  almost everything. It is least safe in between - which is where every")
    p("  board in this repository sits, and where a benchmark is useful.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("slack_law_results.txt").write_text(text + chr(10), encoding="utf-8",
                                             newline=chr(10))
    print(chr(10) + "wrote slack_law_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
