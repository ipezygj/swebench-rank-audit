"""Does the frontier move within families?

effective_entrants.py: a board of J entrants carries the evidence of J_eff
independent ones, because entrants come in families (shared base models,
scaffolds, methods). sota_audit.py: most frontier advances are too small to
separate. Put together: is a new leader typically a SIBLING of the old one
- the same family, iterated - or an outsider?

Measure: for each frontier advance, the residual correlation between the new
and the old leader, placed as a percentile among ALL pairwise residual
correlations of the board. A sibling sits high in that distribution; an
outsider near the middle.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the median percentile of frontier pairs is above 50 on at least 3 of the
    4 dated boards (SWE-bench Verified, SWE-bench Lite, MTEB, LiveBench);
  * on the SWE-bench boards it is above 70: advances there are overwhelmingly
    scaffold iterations on the same few LLMs;
  * exploratory, reported not judged: whether sibling advances (percentile
    > 75) are SMALLER in u than outsider advances (percentile < 50).

SELF-CHECKS
  * on an iid field with random dates the median percentile of frontier
    pairs must lie within 50 +- 15 over 30 fields;
  * the percentile is invariant to affine rescaling of the matrix.

    python sota_families.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from sota_audit import advances
from residual_correlation import decompose
from evidence_trajectory import load
from sota_twin import sigma_p_of, synth_dates
from step_sizes import steps_u

BOARDS = {
    "SWE-bench Verified": ("swebench_verified_matrix.csv", None),
    "SWE-bench Lite": ("swebench_lite_matrix.csv", None),
    "MTEB English v2": ("mteb_dated_matrix.csv", "mteb_dates.csv"),
    "LiveBench": ("livebench/matrix.csv", "livebench/dates.csv"),
}


def frontier_percentiles(x, dates, control=False):
    """Percentile of corr(new leader, old leader) among all pairs.

    control=True (added after the first run, NOT pre-registered, to meet a
    confound): two strong systems share positive residuals on the hard items
    whatever their lineage, so the frontier pair could score high on ability
    alone. The control pair is (new leader, runner-up to the old leader at
    that date) - nearly the same ability gap, no frontier relation. If the
    control percentiles are as high as the frontier ones, the reading is
    ability, not family.
    """
    _, _, resid = decompose(x)
    c = np.nan_to_num(np.corrcoef(resid))
    J = x.shape[0]
    iu = np.triu_indices(J, k=1)
    allc = np.sort(c[iu])
    sc = x.mean(axis=1)
    out = []
    for a in advances(x, dates):
        other = a["old"]
        if control:
            present = np.flatnonzero(dates <= a["date"])
            present = present[(present != a["new"]) & (present != a["old"])]
            if len(present) == 0:
                continue
            other = int(present[np.argmax(sc[present])])
        v = c[a["new"], other]
        out.append(100.0 * np.searchsorted(allc, v) / len(allc))
    return np.array(out)


def _check_iid():
    meds = []
    for s in range(30):
        rng = np.random.default_rng(100 + s)
        J, n = 60, 150
        x = 0.5 + rng.normal(0, 0.08, J)[:, None] + rng.normal(0, 0.45, (J, n))
        dates = synth_dates("2023-01-01", rng.permutation(J) * 7)
        pc = frontier_percentiles(x, dates)
        if len(pc):
            meds.append(np.median(pc))
    m = float(np.mean(meds))
    return abs(m - 50) < 15, f"iid fields: mean median percentile of frontier pairs {m:.1f} (must be 50 +- 15)"


def _check_affine():
    rng = np.random.default_rng(3)
    x = rng.random((40, 100))
    dates = synth_dates("2023-01-01", np.arange(40) * 3)
    a = frontier_percentiles(x, dates)
    b = frontier_percentiles(3.0 * x + 7.0, dates)
    return np.allclose(a, b), "affine invariance of the percentile"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_iid(), _check_affine()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("DOES THE FRONTIER MOVE WITHIN FAMILIES?")
    p("=" * 78)
    p(f"  {'leaderboard':<20} {'advances':>8} {'median pct':>10} {'>75':>6} {'<50':>6} {'u sib':>7} {'u out':>7} {'CONTROL':>8}")
    meds = {}
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        pc = frontier_percentiles(x, dates)
        ctrl = frontier_percentiles(x, dates, control=True)
        u = steps_u(x, dates, sigma_p_of(x))
        sib, out = u[pc > 75], u[pc < 50]
        meds[name] = float(np.median(pc))
        p(f"  {name:<20} {len(pc):>8} {np.median(pc):>10.0f} {100 * np.mean(pc > 75):>5.0f}% {100 * np.mean(pc < 50):>5.0f}% "
          f"{(np.median(sib) if len(sib) else float('nan')):>7.2f} {(np.median(out) if len(out) else float('nan')):>7.2f} {np.median(ctrl):>8.0f}")
    p("")
    above = sum(v > 50 for v in meds.values())
    p(f"  median percentile > 50: {above}/4 boards (pre-registered >= 3)")
    swe = [v for k, v in meds.items() if k.startswith("SWE")]
    p(f"  SWE-bench boards > 70: {'yes' if all(v > 70 for v in swe) else 'NO'} ({', '.join(f'{v:.0f}' for v in swe)})")
    p("")
    p("  CONTROL = same percentile for (new leader, runner-up to the old leader at")
    p("  that date): similar ability gap, no frontier relation. Added after the")
    p("  first run to meet the ability confound; if CONTROL is as high as the")
    p("  frontier median, the family reading does not stand.")
    p("  percentile = where corr(new leader, old leader) sits among all pairwise")
    p("  residual correlations of the board. u sib / u out = median step size in")
    p("  resolution units for sibling (>75) and outsider (<50) advances.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("sota_families_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote sota_families_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
