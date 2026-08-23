"""Was it progress, or just more attempts? A leaderboard is a time series.

Every SWE-bench submission carries its date in its name, so the leaderboard is
a 26-month record, and the picture everyone shows is the running maximum: the
frontier climbing from 0.4 % to 79 %. That curve rises for two reasons that
nobody separates. Systems got better, and more systems were tried. A running
maximum climbs on its own even when nothing improves, because the maximum of
many draws exceeds the maximum of few.

THE NULL, AND THE THING THAT MAKES IT EXACT
--------------------------------------------
Permute the dates. Under the hypothesis that when a system was submitted says
nothing about how good it is, the assignment of scores to dates is
exchangeable, so shuffling the date labels leaves the distribution alone. No
model, no distributional assumption, and the SAME 134 systems in every draw.

THE CONSEQUENCE THAT INVERTS THE TEST
--------------------------------------
Permutation fixes the final maximum exactly: after all 134 systems, the
running maximum is the best score whatever order they came in. So the test has
no power at the endpoint and all of it in the interior, and it is not asking
"did the field improve" but "did the good systems arrive in date order".

That inverts the direction, which is worth stating plainly because it is easy
to get backwards. If progress is real, the best systems are LATE, so the
running maximum stays low for longer and the observed curve sits BELOW the
permutation band. A curve ABOVE the band would mean the good ones came early -
the field peaking and then filling in behind itself.

WHAT IS REPORTED
----------------
A counterfactual with dates in it, which is the form the question is actually
asked in: the field passed 70 % on such a date; had the same 134 systems
arrived in random order it would have passed 70 % on this other date. The gap
is the part of the frontier's climb that is chronology rather than
accumulation.

SELF-CHECKS THAT CAN FAIL
--------------------------
  * scores unrelated to date must land inside the band at about the nominal
    rate;
  * scores rising with date must land BELOW the band, which is also the
    check that the inverted direction above is right and not a rationalisation;
  * the running maximum must be monotone;
  * the permutation must preserve the multiset of scores exactly, so that the
    endpoint is identical in every draw.

    python progress_or_selection.py [--matrix ...] [--perms 4000]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260823
DATE = re.compile(r"^(20\d{2})(\d{2})(\d{2})_")


def parse_dates(names) -> np.ndarray:
    out = []
    for nm in names:
        m = DATE.match(nm)
        if not m:
            raise SystemExit(f"no date in {nm!r} - refusing to guess one")
        out.append(int(m.group(1)) * 10000 + int(m.group(2)) * 100
                   + int(m.group(3)))
    return np.array(out)


def fmt(d: int) -> str:
    return f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}"


def running_max(scores: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(scores)


def crossing_index(rm: np.ndarray, level: float) -> int:
    """First position at which the running maximum reaches `level`."""
    idx = np.flatnonzero(rm >= level)
    return int(idx[0]) if len(idx) else -1


def null_curves(scores: np.ndarray, perms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((perms, len(scores)))
    for b in range(perms):
        out[b] = running_max(rng.permutation(scores))
    return out


# --- self-checks ------------------------------------------------------------

def _check_monotone() -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    s = rng.random(50)
    rm = running_max(s)
    ok = bool(np.all(np.diff(rm) >= 0))
    return ok, f"running maximum is monotone: {ok}"


def _check_endpoint_fixed() -> tuple[bool, str]:
    rng = np.random.default_rng(2)
    s = rng.random(40)
    nc = null_curves(s, 50, 3)
    ok = bool(np.allclose(nc[:, -1], s.max()))
    return ok, f"every permutation ends at the same maximum: {ok}"


def _check_no_trend_inside() -> tuple[bool, str]:
    """Scores unrelated to date must sit inside the band most of the time."""
    rng = np.random.default_rng(5)
    hits, reps = 0, 200
    for _ in range(reps):
        s = rng.random(60)
        rm = running_max(s)
        nc = null_curves(s, 300, int(rng.integers(1 << 30)))
        lo = np.quantile(nc, 0.025, axis=0)
        hi = np.quantile(nc, 0.975, axis=0)
        mid = slice(5, 55)
        hits += int(np.all((rm[mid] >= lo[mid]) & (rm[mid] <= hi[mid])))
    frac = hits / reps
    return frac >= 0.75, (f"no-trend curves stay inside the band in "
                          f"{frac:.2f} of runs")


def _check_trend_below() -> tuple[bool, str]:
    """Rising ability must put the observed curve BELOW the band.

    This is the check that the inverted direction is real. If it fails, the
    reasoning in the docstring is wrong and the report must not be printed.
    """
    rng = np.random.default_rng(7)
    n = 60
    s = np.sort(rng.random(n)) + rng.normal(0, 0.02, n)
    rm = running_max(s)
    nc = null_curves(s, 800, 11)
    lo = np.quantile(nc, 0.025, axis=0)
    mid = slice(5, 45)
    below = float(np.mean(rm[mid] < lo[mid]))
    return below > 0.5, (f"rising ability puts the curve below the band in "
                         f"{below:.2f} of the interior")


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_monotone(), _check_endpoint_fixed(),
                        _check_no_trend_inside(), _check_trend_below()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--perms", type=int, default=4000)
    ap.add_argument("--out", default="progress_or_selection_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=float)
    dates = parse_dates(df.index)
    order = np.argsort(dates, kind="stable")
    dates, names = dates[order], np.array(df.index)[order]
    scores = x[order].mean(axis=1)
    J = len(scores)
    print(f"matrix {a.matrix}: {J} systems, "
          f"{fmt(dates[0])} to {fmt(dates[-1])}")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no headline number is printed.")
        return 1

    rm = running_max(scores)
    nc = null_curves(scores, a.perms, SEED)
    lo = np.quantile(nc, 0.025, axis=0)
    hi = np.quantile(nc, 0.975, axis=0)
    mid = np.mean(nc, axis=0)

    # Spearman between date rank and score, with its own permutation p.
    rank_date = np.arange(J)
    rank_score = np.argsort(np.argsort(scores))
    rho = float(np.corrcoef(rank_date, rank_score)[0, 1])
    rng = np.random.default_rng(SEED + 1)
    nullrho = np.array([float(np.corrcoef(rank_date,
                                          rng.permutation(rank_score))[0, 1])
                        for _ in range(a.perms)])
    p_rho = float((nullrho >= rho).mean())

    L = []
    p = L.append
    p("PROGRESS, OR JUST MORE ATTEMPTS?")
    p("=" * 74)
    p(f"{J} systems, {fmt(dates[0])} to {fmt(dates[-1])}")
    p(f"first score {scores[0]:.3f}, best {scores.max():.3f}")
    p("")
    p(f"date-vs-score rank correlation   {rho:+.3f}   "
      f"permutation p = {p_rho:.4f}")
    p("")
    p("WHEN THE FRONTIER PASSED EACH LEVEL, AND WHEN IT WOULD HAVE IN")
    p("RANDOM ORDER (same 134 systems, dates shuffled)")
    p(f"  {'level':>6} {'actual':>12} {'random order, median':>22} {'gap':>10}")
    for lvl in (0.20, 0.40, 0.50, 0.60, 0.70, 0.75):
        i = crossing_index(rm, lvl)
        if i < 0:
            continue
        nulls = np.array([crossing_index(nc[b], lvl) for b in range(a.perms)])
        nulls = nulls[nulls >= 0]
        med = int(np.median(nulls))
        p(f"  {lvl:>6.2f} {fmt(dates[i]):>12} {fmt(dates[med]):>22}"
          f" {(i - med):>+7} systems")
    p("")
    p("  The gap is in submissions, not days, and it is the number to read:")
    p("  several systems share a date (four RAG baselines are all 2023-10-10),")
    p("  so a median crossing at submission 0, 1 or 2 prints the same date.")
    p("")
    p("  A positive gap means the level was reached LATER than a random")
    p("  ordering of the same systems would have reached it. That is what")
    p("  progress looks like here: the good systems arrived last, so the")
    p("  frontier had to wait for them instead of stumbling on them early.")
    p("")
    inside = float(np.mean((rm >= lo) & (rm <= hi)))
    below = float(np.mean(rm < lo))
    p(f"  observed frontier inside the 95 % band   {100 * inside:.1f} % of the record")
    p(f"  observed frontier BELOW the band         {100 * below:.1f} %")
    p("")
    p("THE FRONTIER AGAINST ITS OWN SHUFFLED SELF")
    p(f"  {'after':>6} {'date':>12} {'actual':>8} {'random median':>14}"
      f" {'2.5%':>7} {'97.5%':>7}")
    for i in (9, 19, 39, 59, 79, 99, 119, J - 1):
        p(f"  {i + 1:>6} {fmt(dates[i]):>12} {rm[i]:>8.3f} {mid[i]:>14.3f}"
          f" {lo[i]:>7.3f} {hi[i]:>7.3f}")
    p("")
    p("  The permutation fixes the endpoint: after all 134 systems every")
    p("  ordering has the same maximum, so the last row carries no test and")
    p("  the evidence is entirely in the interior. This measures whether the")
    p("  good systems came in date order, not whether the field improved -")
    p("  those are different questions and only the first one is decidable")
    p("  from a leaderboard that reports no confidence in its own ordering.")
    p("")
    p("  And it cannot say WHY the good ones came late. A stronger base model,")
    p("  a better scaffold and a benchmark leaking into training data all")
    p("  produce the same curve. This separates chronology from accumulation,")
    p("  which is one question, and leaves the causal one open.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
