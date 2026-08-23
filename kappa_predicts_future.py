"""Does pair sharpness predict the FUTURE, or only describe the past?

Everything measured tonight is retrospective: kappa computed on the same
matrix whose claims it judges. The sharpest test of a measurement is
whether it says something about data it has not seen.

Here is one it can be asked. Take the systems present up to a cut-off date.
Compute, for each, its mean kappa with the systems already on the board -
how much it behaves like the field it entered. Then look at what happened
AFTER the cut-off: did the systems that behaved most like the incumbents go
on to be passed sooner? A model that is a variation on the field should be
overtaken faster than one that is doing something different, because the
field's next step reproduces it easily.

Design, fixed before running:
  * cut-off = the date at which two thirds of the systems have arrived;
  * for each system present at the cut-off and not the leader, x = its mean
    kappa with the systems that arrived BEFORE it (its incumbents), and
    y = the number of later systems that end up above it;
  * Spearman(x, y), one value per board; a NEGATIVE correlation means low
    kappa (behaves unlike the field) is followed by MORE systems passing it,
    a POSITIVE one means the derivative systems get passed faster.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the correlation is positive on >= 3 of the 5 dated boards: systems that
    resemble the field they entered are overtaken by more later entrants;
  * the effect survives conditioning on score - within the top half of the
    board at the cut-off, the sign is the same on >= 3 of 5;
  * if the sign is negative instead, the reading reverses and is reported
    as such: distinctive systems are the ones that get buried.

SELF-CHECKS
  * on a simulated board where kappa is unrelated to arrival order, the
    correlation must be within +-0.2 of zero;
  * the "later systems above it" count must equal a direct recomputation
    from scores and dates.

    python kappa_predicts_future.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from evidence_trajectory import load
from pair_sharpness import kappa_matrix
from chase_model import BOARDS
from sota_twin import synth_dates

SEED = 20260823


def board_stats(x, dates):
    J = x.shape[0]
    K = kappa_matrix(x)
    sc = x.mean(axis=1)
    cut = int(np.quantile(dates, 2 / 3))
    present = np.flatnonzero(dates <= cut)
    later = np.flatnonzero(dates > cut)
    if len(present) < 8 or len(later) < 5:
        return None
    xs, ys, ss = [], [], []
    for j in present:
        inc = [k for k in present if dates[k] < dates[j] and k != j]
        if len(inc) < 3:
            continue
        xs.append(float(np.nanmean([K[j, k] for k in inc])))
        ys.append(int(np.sum(sc[later] > sc[j])))
        ss.append(float(sc[j]))
    return np.array(xs), np.array(ys), np.array(ss), len(present), len(later)


def _check_unrelated():
    rng = np.random.default_rng(SEED)
    J, n = 90, 300
    dates = synth_dates("2023-01-01", np.sort(rng.integers(0, 900, J)))
    x = 0.4 + rng.normal(0, 0.06, J)[:, None] + rng.normal(0, 0.45, (J, n))
    out = board_stats(x, dates)
    r = spearmanr(out[0], out[1]).statistic
    return abs(r) < 0.2, f"kappa unrelated to arrival: Spearman {r:+.2f}"


def _check_count():
    rng = np.random.default_rng(SEED + 1)
    J, n = 40, 200
    dates = synth_dates("2023-01-01", np.sort(rng.integers(0, 400, J)))
    x = 0.4 + rng.normal(0, 0.06, J)[:, None] + rng.normal(0, 0.45, (J, n))
    xs, ys, ss, npres, nlate = board_stats(x, dates)
    sc = x.mean(axis=1)
    cut = int(np.quantile(dates, 2 / 3))
    later = np.flatnonzero(dates > cut)
    direct = []
    for j in np.flatnonzero(dates <= cut):
        inc = [k for k in np.flatnonzero(dates <= cut) if dates[k] < dates[j] and k != j]
        if len(inc) >= 3:
            direct.append(int(np.sum(sc[later] > sc[j])))
    return np.array_equal(ys, np.array(direct)), f"overtake counts match a direct recomputation ({len(ys)} systems)"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_unrelated(), _check_count()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("DOES KAPPA AT ENTRY PREDICT BEING OVERTAKEN LATER?")
    p("=" * 80)
    p(f"  {'board':<20} {'at cutoff':>10} {'later':>6} {'n used':>7} {'Spearman':>9} {'p':>7} "
      f"{'top half':>9} {'p':>7}")
    pos, pos_top = 0, 0
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        out = board_stats(x, dates)
        if out is None:
            p(f"  {name:<20} too few systems on one side of the cut-off")
            continue
        xs, ys, ss, npres, nlate = out
        r = spearmanr(xs, ys)
        half = ss >= np.median(ss)
        rt = spearmanr(xs[half], ys[half]) if half.sum() > 5 else None
        pos += r.statistic > 0
        pos_top += bool(rt and rt.statistic > 0)
        p(f"  {name:<20} {npres:>10} {nlate:>6} {len(xs):>7} {r.statistic:>+9.2f} {r.pvalue:>7.3f} "
          f"{(f'{rt.statistic:+.2f}' if rt else 'n/a'):>9} {(f'{rt.pvalue:.3f}' if rt else ''):>7}")
    p("")
    p(f"  positive on the whole field: {pos}/5 (pre-registered >= 3)")
    p(f"  positive within the top half: {pos_top}/5 (pre-registered >= 3)")
    p("")
    p("  x = a system's mean kappa with the systems that were already on the")
    p("  board when it arrived; y = how many later arrivals ended up above it.")
    p("  Low kappa means it behaves unlike the incumbents. A positive")
    p("  correlation says derivative systems get passed by more newcomers.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("kappa_predicts_future_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote kappa_predicts_future_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
