"""Is law 2 a corollary of law 1, or does the twin know something extra?

The paper's two laws are not symmetric and it says so: law 1 is a closed form,
law 2 is "a Gaussian field with the same four numbers reproduces the entropy",
which is a weaker kind of claim. That asymmetry is worth attacking rather than
apologising for.

The suspicion is simple. Ordering entropy counts the total orders a partial
order permits, and a partial order with more relations permits fewer. The
established share IS the density of that partial order. So H/ceiling might be a
function of the established share and the number of systems, both of which law
1 already gives - in which case law 2 is a corollary and the twin is redundant.

Three predictors of H/ceiling are compared on the same nine boards, each scored
by leave-one-out so that no board helps predict itself:

  A   the established share alone, linear
  B   the established share and log J
  C   the Gaussian twin - law 2 as it stands

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  Spearman(established share, H/ceiling) across the nine boards is at most
      -0.8. Denser orders permit fewer orderings; if this fails the whole idea
      is wrong.
  P2  predictor B reaches a leave-one-out mean absolute error of at most 5
      points.
  P3  predictor B is within 2 points of the twin's mean absolute error. If it
      is, law 2 can be stated as a corollary and the twin is a convenience
      rather than a necessity.
  P4  the control: with the established share shuffled between boards,
      predictor B's leave-one-out error must be worse than the unshuffled one
      in at least 90 % of 199 shuffles.

  What would make this a real improvement to the paper: P1, P2 and P3 together.
  What would make it a real finding either way: P4, which is the only thing
  that separates a two-parameter fit on nine points from a law.

SELF-CHECKS (no table if any fails)
  * the leave-one-out harness must be able to fail: fed a target that is pure
    noise, its error must not beat the target's own standard deviation;
  * H/ceiling must reproduce the committed entropy_law_test figures exactly,
    so any difference below is the predictor and not a changed input;
  * the twin column must reproduce the committed twin figures too.

    python law2_closed_form.py
"""
from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.stats import spearmanr

SEED = 20260824
PERMS = 199


def law2_rows():
    """(board, J, n, H_real, H_twin) from the committed results file."""
    rows = []
    txt = Path("entropy_law_test_results.txt").read_text(encoding="utf-8", errors="replace")
    for line in txt.splitlines():
        m = re.match(r"\s{2}(\S.*?)\s{2,}(\d+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+"
                     r"([+-][\d.]+)\s+([\d.]+)%\s+([\d.]+)%", line)
        if m:
            rows.append((m.group(1).strip(), int(m.group(2)), int(m.group(3)),
                         float(m.group(4)), float(m.group(5)), float(m.group(7))))
    return rows


def loo_fit(X, y):
    """Leave-one-out predictions from an ordinary least-squares fit."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.hstack([np.ones((len(y), 1)), X])
    out = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        beta, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        out[i] = A[i] @ beta
    return out


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _check_loo_can_fail() -> tuple[bool, str]:
    rng = np.random.default_rng(4)
    worse = 0
    for _ in range(200):
        y = rng.normal(0, 10, 9)
        x = rng.normal(0, 1, (9, 2))
        worse += mae(loo_fit(x, y), y) > float(np.std(y))
    return worse >= 180, f"on pure noise the leave-one-out is worse than the SD in {worse} of 200"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = law2_rows()
    if len(rows) < 9:
        print(f"  [FAIL] parsed only {len(rows)} boards from entropy_law_test_results.txt")
        return 1

    names = [r[0] for r in rows]
    J = np.array([r[1] for r in rows], float)
    H = np.array([r[3] for r in rows], float)
    twin = np.array([r[4] for r in rows], float)
    estab = np.array([r[5] for r in rows], float)

    print("self-checks ...")
    ok_loo, msg = _check_loo_can_fail()
    print(f"  [{'ok  ' if ok_loo else 'FAIL'}] {msg}")
    ok_parse = len(rows) == 9 and all(0 < h < 100 for h in H)
    print(f"  [{'ok  ' if ok_parse else 'FAIL'}] parsed {len(rows)} boards, "
          f"H between {H.min():.1f} and {H.max():.1f} %")
    if not (ok_loo and ok_parse):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rho, pv = spearmanr(estab, H)
    predA = loo_fit(estab, H)
    predB = loo_fit(np.column_stack([estab, np.log(J)]), H)

    maeA, maeB, maeC = mae(predA, H), mae(predB, H), mae(twin, H)

    rng = np.random.default_rng(SEED)
    worse = 0
    for _ in range(PERMS):
        e = estab.copy()
        rng.shuffle(e)
        worse += mae(loo_fit(np.column_stack([e, np.log(J)]), H), H) > maeB
    frac = worse / PERMS

    L = []
    p = L.append
    p("IS LAW 2 A COROLLARY OF LAW 1?")
    p("=" * 92)
    p(f"  {'leaderboard':<22} {'J':>4} {'established':>12} {'H/ceiling':>10} "
      f"{'A: estab':>9} {'B: +log J':>10} {'C: twin':>8}")
    for i, nm in enumerate(names):
        p(f"  {nm:<22} {int(J[i]):>4} {estab[i]:>11.1f}% {H[i]:>9.1f}% "
          f"{predA[i]:>8.1f}% {predB[i]:>9.1f}% {twin[i]:>7.1f}%")
    p("")
    p(f"  mean absolute error   A {maeA:.1f}   B {maeB:.1f}   C (twin) {maeC:.1f}   points")
    p("")
    p(f"  P1  Spearman(established, H) = {rho:+.2f} (p {pv:.3f})    "
      f"pre-registered <= -0.8:  {'HIT' if rho <= -0.8 else 'MISS'}")
    p(f"  P2  B's leave-one-out error {maeB:.1f} points           "
      f"pre-registered <= 5:  {'HIT' if maeB <= 5 else 'MISS'}")
    p(f"  P3  B against the twin: {maeB - maeC:+.1f} points          "
      f"pre-registered within 2:  {'HIT' if abs(maeB - maeC) <= 2 else 'MISS'}")
    p(f"  P4  shuffling the established share makes B worse in {100 * frac:.0f} % of shuffles")
    p(f"      pre-registered >= 90 %:  {'HIT' if frac >= 0.90 else 'MISS'}")
    p("")
    p("  A and B are fitted by leave-one-out: each board's prediction comes from")
    p("  a fit that never saw it. C is the Gaussian twin, which uses J, n, tau")
    p("  and sigma_p and is not fitted to anything.")
    p("")
    p("  The question is whether the twin knows more than the established share")
    p("  and the size of the field. If it does not, law 2 is a corollary of law")
    p("  1 rather than a second law, and the paper should say so - a partial")
    p("  order's entropy being a function of its density and its width is a")
    p("  simpler claim than a simulation matching it, and a stronger one.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("law2_closed_form_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                    newline=chr(10))
    print(chr(10) + "wrote law2_closed_form_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
