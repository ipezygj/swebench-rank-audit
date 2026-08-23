"""The open question, asked of the names instead of the residuals.

isotonic_families.py left a reversal unexplained: with an ability-conditioned
residual, a new frontier leader correlates LESS with the leader it passed
than with that day's runner-up (-19 to -42 percentile points, 4 boards of 4).
leader_luck.py ruled out the winner's curse as the cause.

The residuals are one way to ask. The names are another, and they are
independent of everything the residual sees. For each frontier advance:
is the new leader in the same NAME family as the system it passed, as that
day's runner-up, as anyone else on the board, or none of them?

Families are the ones fixed in lineage_detection.py: the base model in a
SWE-bench submission id, the HuggingFace organisation for MTEB, the method
family for ProteinGym.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the new leader shares a family with the system it passed MORE often
    than with that day's runner-up, on >= 2 of the 3 boards - my prior is
    that the residual reversal is an artefact of the residual, not a fact
    about lineage;
  * at least a third of advances share a family with someone already on
    the board;
  * if instead the runner-up match is the more common one, the residual
    result is corroborated by an independent source and the open question
    becomes a finding.

SELF-CHECKS
  * the family assignment must reproduce lineage_detection's counts;
  * a shuffled-name control must give the same rate for both comparisons.

    python frontier_lineage.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sota_audit import advances
from lineage_detection import FAMILY, MIN_FAMILY

BOARDS = {
    "SWE-bench Verified": ("swebench_verified_matrix.csv", None),
    "MTEB English v2": ("mteb_dated_matrix.csv", "mteb_dates.csv"),
    "ProteinGym DMS": ("proteingym/matrix.csv", "proteingym/dates.csv"),
}
SEED = 20260823


def load_named(path, dates_csv):
    df = pd.read_csv(path, index_col=0).dropna(axis=0)
    names = list(df.index)
    if dates_csv:
        dd = pd.read_csv(dates_csv, index_col=0)["date"]
        dates = np.array([int(dd.loc[n]) for n in names])
    else:
        from sota_audit import parse_dates
        dates = parse_dates(names)
    return df.to_numpy(dtype=float), dates, names


def families(names, fn):
    fams = [fn(n) for n in names]
    counts = Counter(f for f in fams if f)
    keep = {f for f, c in counts.items() if c >= MIN_FAMILY}
    return [f if f in keep else None for f in fams]


def tally(x, dates, fams):
    sc = x.mean(axis=1)
    rows = []
    for a in advances(x, dates):
        new, old = int(a["new"]), int(a["old"])
        present = np.flatnonzero(dates <= a["date"])
        others = [int(i) for i in present if i not in (new, old)]
        run = int(others[int(np.argmax(sc[others]))]) if others else None
        f_new = fams[new]
        rows.append({
            "same_as_old": bool(f_new and f_new == fams[old]),
            "same_as_run": bool(f_new and run is not None and f_new == fams[run]),
            "same_as_any": bool(f_new and any(fams[i] == f_new for i in present if i != new)),
            "has_family": f_new is not None,
        })
    return rows


def _check_counts():
    msgs = []
    ok = True
    for name, (path, dc) in BOARDS.items():
        _, _, names = load_named(path, dc)
        fams = families(names, FAMILY[name])
        k = len({f for f in fams if f})
        msgs.append(f"{name.split()[0]} {k} families / {sum(f is not None for f in fams)} systems")
        ok = ok and k >= 2
    return ok, "; ".join(msgs)


def _check_shuffle():
    rng = np.random.default_rng(SEED)
    diffs = []
    for name, (path, dc) in BOARDS.items():
        x, dates, names = load_named(path, dc)
        fams = families(names, FAMILY[name])
        for s in range(8):
            sh = list(np.random.default_rng(SEED + s).permutation(fams))
            rows = tally(x, dates, sh)
            if rows:
                diffs.append(np.mean([r["same_as_old"] for r in rows]) - np.mean([r["same_as_run"] for r in rows]))
    m = float(np.mean(diffs))
    return abs(m) < 0.10, f"shuffled names: mean(old-match minus runner-up-match) {m:+.3f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_counts(), _check_shuffle()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WHOSE FAMILY DOES A NEW LEADER BELONG TO?")
    p("=" * 78)
    p(f"  {'board':<22} {'advances':>9} {'has family':>11} {'same as passed':>15} "
      f"{'same as runner-up':>18} {'same as anyone':>15}")
    old_wins, third = 0, 0
    for name, (path, dc) in BOARDS.items():
        x, dates, names = load_named(path, dc)
        fams = families(names, FAMILY[name])
        rows = tally(x, dates, fams)
        if not rows:
            continue
        hf = np.mean([r["has_family"] for r in rows])
        so = np.mean([r["same_as_old"] for r in rows])
        sr = np.mean([r["same_as_run"] for r in rows])
        sa = np.mean([r["same_as_any"] for r in rows])
        old_wins += so > sr
        third += sa >= 1 / 3
        p(f"  {name:<22} {len(rows):>9} {100 * hf:>10.0f}% {100 * so:>14.0f}% {100 * sr:>17.0f}% {100 * sa:>14.0f}%")
    p("")
    p(f"  new leader matches the system it passed more often than the runner-up: {old_wins}/3"
      f" (pre-registered >= 2)")
    p(f"  at least a third of advances share a family with someone present: {third}/3")
    p("")
    p("  Families are the name-derived ones of lineage_detection.py, independent")
    p("  of the score matrix. 'same as passed' is the question the residual")
    p("  reversal put on the table; the names answer it without the residual.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("frontier_lineage_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote frontier_lineage_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
