"""Fifth dated board, from a DIFFERENT FIELD: ProteinGym DMS (96 x 217).

The four dated boards so far are three LLM boards and a second SWE-bench
split. The open question after commit dd6415d was whether the step-size
reading survives on a dated board from another field. ProteinGym's 96
variant-effect predictors can be dated from their reference URLs
(proteingym/dates.py, 96/96: 49 preprint dates, 47 publication dates).

The pre-registered checks are the ones written in c9ff2df before any fourth
or fifth board was sought, plus the flat/drift decomposition of db6292a:

    median u in [0.20, 0.60]        share u < 1 >= 60 %
    A_real > A_twin                 P_real < P_twin
    real median u > flat-twin median u
    progress fraction pf in [0, 1]

ADDITIONAL PRE-REGISTRATION FOR THIS BOARD (2026-08-23, before running)
  * ProteinGym is NOT a race: entrants are published methods, not
    submissions chasing a leaderboard, and 47 of them are dated by
    publication rather than preprint. If "the frontier is chased" is what
    drives A_real > A_twin, this board is where it should be WEAKEST:
    predicted A_real/A_twin below the 1.4-2.4 of the LLM boards.
  * ROBUSTNESS: rerun with the 49 preprint dates shifted +12 months, so all
    dates are publication-like. Every verdict above must survive; if a
    verdict flips, it is reported as date-sensitive and not counted.
  * ties: 15 models share 2021-07-01 etc. (year-only sources). Ties are
    broken by the file order, which is score order - the WORST case for
    A_real, since it lets a tied group produce several records. Reported.

    python fifth_board.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evidence_trajectory import load
from sota_twin import audit, fit_drift, dated_twin, sigma_p_of, SEED, TWINS
from step_sizes import steps_u
from sota_luck_law import twin_us, _flat

MATRIX, DATES = "proteingym/matrix.csv", "proteingym/dates.csv"


def shift_preprints(dates_csv, months=12):
    d = pd.read_csv(dates_csv, index_col=0)
    out = d.copy()
    pre = d["source"].isin(["biorxiv", "arxiv"])
    ts = pd.to_datetime(d.loc[pre, "date"].astype(str), format="%Y%m%d") + pd.DateOffset(months=months)
    out.loc[pre, "date"] = ts.dt.strftime("%Y%m%d").astype(int)
    return out


def run(x, dates, label, L):
    J, n = x.shape
    sp = sigma_p_of(x)
    u = steps_u(x, dates, sp)
    real = audit(x, dates, SEED)
    a, beta, tau_res, si = fit_drift(x, dates, sp)
    tw = [audit(dated_twin(J, n, dates, a, beta, tau_res, si, np.random.default_rng(SEED + 50 + s)), dates, SEED + 500 + 100 * s)
          for s in range(TWINS)]
    A_t = float(np.mean([t["A"] for t in tw]))
    P_t = float(np.nanmean([t["P"] for t in tw]))
    uf = twin_us(_flat, x, dates)
    ud = twin_us(dated_twin, x, dates)
    mr, mf, md = float(np.median(u)), float(np.median(uf)), float(np.median(ud))
    pf = (mr - mf) / (md - mf) if md != mf else float("nan")
    v = {
        "median u in [0.20, 0.60]": (0.20 <= mr <= 0.60, f"{mr:.2f}"),
        "share u < 1 >= 60 %": (float(np.mean(u < 1)) >= 0.6, f"{100 * np.mean(u < 1):.0f} %"),
        "A_real > A_twin": (real["A"] > A_t, f"{real['A']} vs {A_t:.1f}  (ratio {real['A'] / A_t:.2f})"),
        "P_real < P_twin": (real["P"] < P_t, f"{100 * real['P']:.0f} % vs {100 * P_t:.0f} %"),
        "real median > flat median": (mr > mf, f"{mr:.2f} vs {mf:.2f}"),
        "pf in [0, 1]": (0 <= pf <= 1, f"{pf:.2f}"),
    }
    L.append("")
    L.append(f"  {label}: {J} systems x {n} items, {len(u)} frontier advances, beta {beta:+.3f}/yr")
    for k, (okk, txt) in v.items():
        L.append(f"    {k:<28} {'yes' if okk else 'NO':>3}   {txt}")
    L.append(f"    (S real {100 * real['S']:.0f} %; drift median {md:.2f}; q25 {np.percentile(u, 25):.2f} q75 {np.percentile(u, 75):.2f})")
    return {k: b for k, (b, _) in v.items()}, real["A"] / A_t


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    L = []
    p = L.append
    p("FIFTH DATED BOARD, DIFFERENT FIELD: PROTEINGYM DMS")
    p("=" * 80)
    x, dates = load(MATRIX, DATES)
    base, ratio = run(x, dates, "ProteinGym, dates as published", L)

    shifted = shift_preprints(DATES)
    tmp = Path("proteingym/dates_shifted.csv")
    shifted.to_csv(tmp)
    x2, dates2 = load(MATRIX, str(tmp))
    rob, ratio2 = run(x2, dates2, "ROBUSTNESS: preprint dates +12 months", L)

    p("")
    flips = [k for k in base if base[k] != rob[k]]
    p(f"  verdicts surviving the date shift: {sum(base[k] == rob[k] for k in base)}/{len(base)}"
      + (f"   date-sensitive: {', '.join(flips)}" if flips else ""))
    p(f"  A_real/A_twin: {ratio:.2f} (shifted {ratio2:.2f}); LLM boards were 1.4-2.4")
    p(f"  pre-registered: this ratio should be LOWER here (not a race) -> {'yes' if max(ratio, ratio2) < 1.4 else 'NO'}")
    p("")
    p("  Ties: year-only date sources put up to 15 models on one day; ties resolve")
    p("  in file order, which is score order - the case most generous to A_real.")
    text = chr(10).join(L)
    print(text)
    Path("fifth_board_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote fifth_board_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
