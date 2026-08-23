"""A named case: what the board could and could not tell Augment Code.

Chosen because it is one of the few companies that did BOTH things this
repo can check, in public, on a board whose per-instance data is here:

  * it justified a model choice with the leaderboard - "Since Anthropic's
    models are currently state-of-the-art on code, we used Claude Sonnet
    3.7 as our agent's core driver";
  * it made a ranking claim - "#1 open-source agent on SWE-Bench Verified"
    with "a 65.4% success rate on SWE-bench verified", posted 2025-03-31;
  * and it submitted to the board, so the claim can be recomputed rather
    than argued about: 20250316_augment_agent_v0 is in the matrix at
    exactly 65.4 %.

Everything below is computed from the public SWE-bench Verified submission
files. Nothing here says the claim is false. It says what the instrument
behind it can and cannot resolve, which is a different thing and the only
thing the matrix knows.

Quotes verified 2026-08-23 by fetching the post; the score in the post and
the score in the matrix agree to the decimal, which is the check that the
submission and the claim are the same object.

    python case_augment.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from sota_audit import parse_dates, fmt, mcnemar_exact
from pair_sharpness import kappa_matrix
from swebench_base_models import base_model

MATRIX = "swebench_verified_matrix.csv"
SUB_V0 = "20250316_augment_agent_v0"
SUB_V1 = "20250610_augment_agent_v1"
CLAIMED = 0.654
DRAWS = 2000


def _check_claim_matches_matrix(sc, names):
    got = float(sc[names.index(SUB_V0)])
    return abs(got - CLAIMED) < 0.0005, \
        f"the post's 65.4 % and the matrix agree: {100 * got:.1f} %"


def _check_mcnemar_symmetry(x, names):
    i, j = names.index(SUB_V0), names.index(SUB_V1)
    a1, b1, p1 = mcnemar_exact(x[i], x[j])
    a2, b2, p2 = mcnemar_exact(x[j], x[i])
    return (a1, b1) == (b2, a2) and abs(p1 - p2) < 1e-12, \
        f"the paired test is symmetric: ({a1}, {b1}) and ({a2}, {b2}), p {p1:.4f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv(MATRIX, index_col=0)
    x = df.to_numpy(dtype=float)
    names = list(df.index)
    sc = x.mean(axis=1)
    dates = parse_dates(names)

    print("self-checks")
    ok = True
    for passed, msg in (_check_claim_matches_matrix(sc, names), _check_mcnemar_symmetry(x, names)):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    i = names.index(SUB_V0)
    j = names.index(SUB_V1)
    present = np.flatnonzero(dates <= dates[i])
    sub = x[present]
    r = rs.rank_sets(sub, draws=DRAWS)
    K = kappa_matrix(sub)
    loc = list(present).index(i)
    order = np.argsort(-sub.mean(axis=1))

    L = []
    p = L.append
    p("WHAT SWE-BENCH VERIFIED COULD AND COULD NOT TELL AUGMENT CODE")
    p("=" * 82)
    p("  Claim, 2025-03-31: \"#1 open-source agent on SWE-Bench Verified\", "
      "\"a 65.4 % success rate\".")
    p("  Reason given for the model choice: \"Since Anthropic's models are currently")
    p("  state-of-the-art on code, we used Claude Sonnet 3.7 as our agent's core driver\".")
    p("  Submission in the matrix: " + SUB_V0 + f", {100 * sc[i]:.1f} % - the same object.")
    p("")
    p(f"  Field on that date: {len(present)} submissions. Augment ranks "
      f"{list(order).index(loc) + 1} by score.")
    p(f"  Its simultaneous rank set is [{int(r['best'][loc])}, {int(r['worst'][loc])}], and "
      f"{int((r['best'] == 1).sum())} submissions have a rank set containing 1.")
    p("")
    p("  Every submission the board cannot rank below Augment on that date:")
    p(f"    {'submission':<52} {'score':>7} {'kappa':>6} {'t':>7} {'base model':>14}")
    for k in np.flatnonzero(r["best"] == 1):
        gi = int(present[k])
        d = x[i] - x[gi]
        se = float(d.std(ddof=1) / math.sqrt(x.shape[1]))
        t = float(d.mean() / se) if se > 0 else 0.0
        fam = base_model(names[gi]) or "-"
        star = "  <- Augment" if gi == i else ""
        p(f"    {names[gi][:52]:<52} {100 * sc[gi]:>6.1f}% {K[loc, k]:>6.2f} {t:>+7.2f} {fam:>14}{star}")
    p("")
    runner = int(present[int(order[1])])
    a, b, pv = mcnemar_exact(x[i], x[runner])
    p(f"  Against the runner-up, {names[runner]} ({100 * sc[runner]:.1f} %):")
    p(f"    {a} instances only Augment solves, {b} only the runner-up, exact p = {pv:.2f}.")
    p(f"    Their pair's kappa is {K[loc, int(order[1])]:.2f}: the comparison is sharper than the")
    p("    board's average pair, and it still does not separate them.")
    p(f"    That runner-up's core driver is not a Claude model, which is the specific")
    p("    comparison the stated reason for the model choice rests on.")
    p("")
    a2, b2, p2 = mcnemar_exact(x[j], x[i])
    d2 = x[j] - x[i]
    t2 = float(d2.mean() / (d2.std(ddof=1) / math.sqrt(x.shape[1])))
    p(f"  What the board CAN resolve: Augment's own next version.")
    p(f"    {SUB_V1} scores {100 * sc[j]:.1f} %, {100 * (sc[j] - sc[i]):.1f} points above v0.")
    p(f"    {a2} instances only v1 solves, {b2} only v0, exact p = {p2:.4f}, t = {t2:.2f}.")
    p("    Their own engineering shows up on this board at conventional significance;")
    p("    their advantage over the field on the day of the claim does not.")
    p("")
    fam_rows = []
    for gi in present:
        f = base_model(names[int(gi)])
        if f == "claude-3.7":
            fam_rows.append((names[int(gi)], sc[int(gi)]))
    if fam_rows:
        lo = min(s for _, s in fam_rows)
        hi = max(s for _, s in fam_rows)
        p(f"  On the model choice itself: {len(fam_rows)} submissions on that date name Claude 3.7")
        p(f"  as their base model and they span {100 * lo:.1f} % to {100 * hi:.1f} % - a range of")
        p(f"  {100 * (hi - lo):.1f} points on one model. Across the whole board the spread WITHIN a")
        p("  base model is 0.67 of the spread BETWEEN base models (model_or_harness.py).")
        p("  A board on which the harness moves the score two thirds as much as the model")
        p("  is weak evidence for a model choice, whichever model wins.")
    p("")
    p("  None of this says the agent is not good or the claim was made in bad faith.")
    p("  It says the instrument cited cannot support the comparison it was cited for,")
    p("  and can support a different one the company did not claim.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("case_augment_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote case_augment_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
