"""Was the top of the board ever decisive, and when did it stop being so?

R10 gives one number per board: the t of the #1 vs #2 comparison today.
Across ten boards it runs from 9.89 (CASP14, AlphaFold2) to 0.00
(SWE-bench Verified). The dated boards let the same number be computed at
every point in their history, which turns a snapshot into a question with a
date: how long has the top of this leaderboard been undecidable?

At each checkpoint, among the systems present: the paired t of the two best,
the gap in points, and how many systems are within one SE of the leader.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * the top-pair t declines over time on >= 3 of the 5 dated boards
    (Spearman with date negative) - as a field crowds, the two best get
    closer relative to the noise;
  * on >= 3 boards there exists an early checkpoint with t > 2 (the top was
    once decidable) followed by a final t < 1.5;
  * the number of systems within one SE of the leader grows on >= 4 of 5.

SELF-CHECKS
  * on a simulated board with a fixed clear leader and growing field, t
    must NOT decline;
  * on a simulated board where new entrants crowd the leader, it must.

    python decisiveness_history.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from evidence_trajectory import load, checkpoints
from chase_model import BOARDS
from sota_twin import synth_dates
from sota_audit import fmt

SEED = 20260823


def top_pair(x):
    sc = x.mean(axis=1)
    o = np.argsort(-sc)
    i1, i2 = int(o[0]), int(o[1])
    d = x[i1] - x[i2]
    n = x.shape[1]
    se = float(d.std(ddof=1) / math.sqrt(n))
    t = float(d.mean() / se) if se > 0 else float("inf")
    # systems within one SE of the leader, each with its own SE against it
    near = 0
    for j in range(x.shape[0]):
        if j == i1:
            continue
        dj = x[i1] - x[j]
        sej = float(dj.std(ddof=1) / math.sqrt(n))
        if sej > 0 and dj.mean() / sej < 1.0:
            near += 1
    return t, float(sc[i1] - sc[i2]), near


def series(x, dates, k=8):
    out = []
    for d in checkpoints(dates, k=k, jmin=8):
        xs = x[dates <= d]
        if xs.shape[0] < 3:
            continue
        t, gap, near = top_pair(xs)
        out.append((d, xs.shape[0], t, gap, near))
    return out


def _check_clear_leader():
    rng = np.random.default_rng(SEED)
    J, n = 60, 300
    dates = synth_dates("2023-01-01", np.arange(J) * 5)
    x = 0.4 + rng.normal(0, 0.05, J)[:, None] + rng.normal(0, 0.4, (J, n))
    x[0] += 0.35                                    # a permanent, clear leader
    s = series(x, dates)
    ts = [a[2] for a in s]
    r = spearmanr([a[0] for a in s], ts).statistic
    # With a permanent leader and a permanent runner-up, t is literally
    # constant across checkpoints and Spearman is undefined - which is the
    # strongest possible form of "does not decline", not a failure. The first
    # run returned nan here and the check rejected it.
    flat = float(np.std(ts)) < 1e-9
    ok = flat or r > -0.5
    return ok, (f"fixed clear leader: t {'constant at ' + format(ts[0], '.2f') if flat else 'Spearman ' + format(r, '+.2f')}"
                f" (must not fall sharply)")


def _check_crowding():
    rng = np.random.default_rng(SEED + 1)
    J, n = 60, 300
    dates = synth_dates("2023-01-01", np.arange(J) * 5)
    ability = np.concatenate([[0.75], 0.4 + rng.normal(0, 0.05, 19), np.linspace(0.5, 0.749, J - 20)])
    x = ability[:, None] + rng.normal(0, 0.4, (J, n))
    s = series(x, dates)
    r = spearmanr([a[0] for a in s], [a[2] for a in s]).statistic
    return r < -0.3, f"crowding field: Spearman(date, t) {r:+.2f} (must fall)"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_clear_leader(), _check_crowding()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("HOW LONG HAS THE TOP BEEN UNDECIDABLE?")
    p("=" * 78)
    decl, once, grew = 0, 0, 0
    for name, (path, dc) in BOARDS.items():
        x, dates = load(path, dc)
        s = series(x, dates)
        if len(s) < 3:
            p(f"  {name}: too few checkpoints")
            continue
        r = spearmanr([a[0] for a in s], [a[2] for a in s])
        rn = spearmanr([a[0] for a in s], [a[4] for a in s])
        decl += r.statistic < 0
        once += any(a[2] > 2 for a in s[:-1]) and s[-1][2] < 1.5
        grew += rn.statistic > 0
        p("")
        p(f"  {name}   Spearman(date, t) = {r.statistic:+.2f} (p {r.pvalue:.2f}); "
          f"within-1-SE count trend {rn.statistic:+.2f}")
        p(f"    {'date':>12} {'J':>4} {'top t':>7} {'gap':>8} {'within 1 SE':>12}")
        for d, J, t, gap, near in s:
            p(f"    {fmt(d):>12} {J:>4} {t:>7.2f} {100 * gap:>7.2f}p {near:>12}")
    p("")
    p(f"  top-pair t declines over time: {decl}/5 (pre-registered >= 3)")
    p(f"  was once above 2 and is now below 1.5: {once}/5 (pre-registered >= 3)")
    p(f"  number of systems within one SE of the leader grows: {grew}/5 (pre-registered >= 4)")
    p("")
    p("  t is the paired statistic of the two best systems present at that date,")
    p("  computed on that board's own items. 'within 1 SE' counts systems whose")
    p("  own paired comparison with the leader is under one standard error - the")
    p("  crowd a claim has to be separated from, not just the runner-up.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("decisiveness_history_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote decisiveness_history_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
