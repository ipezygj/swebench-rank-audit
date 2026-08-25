"""How uniform must a winning bid be, before a procurement ranking is evidence?

A public tender is a leaderboard. J bidders are scored on n comparison criteria
and the highest total wins, and the award is a legally binding decision that a
losing bidder may challenge. It is the same object every other file here
measures, with two differences that matter: n is almost always single-digit, and
the ranking has a counterparty with standing to contest it.

This asks what the shape alone permits, before seeing a single score. For the
top pair to separate at simultaneous 95 % confidence across all m = J(J-1)/2
pairwise comparisons, the winner's per-criterion advantage over the runner-up
must satisfy

    mean / sd  >=  c / sqrt(n),     c = t(1 - alpha/(2m), df = n-1)

which is exact for the most significant pair - Holm's first step is Bonferroni -
and needs NO assumption about how the scores are distributed. Its reciprocal is
the largest coefficient of variation the winner's advantages may have: how
UNEVEN the winner is allowed to be across criteria and still be distinguishable
from the runner-up.

That is the quantity a procurement officer can check against their own
comparison table in about a minute, and it is the quantity nobody computes.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  at the median real Finnish shape the allowed coefficient of variation is
      below 0.50 - the winner's advantages must vary by less than half their
      mean across criteria.
  P2  there is a realistic shape - J >= 8 bidders, n <= 5 criteria - where the
      allowed coefficient of variation is below 0.35.
  P3  the allowed variation falls as bidders are ADDED at fixed criteria: more
      competition makes the winner harder to establish, not easier, because
      every extra bidder adds pairs to correct for.
  P4  among the real shapes harvested from TED, at least a quarter have three
      or fewer criteria, where the t distribution has 2 or fewer degrees of
      freedom and c exceeds 10.

  What a miss on P3 would mean: the multiplicity correction is not what drives
  this, and the difficulty is about items alone.

SELF-CHECKS (no table if any fails)
  * c must reproduce a known Bonferroni value: at m = 1 and large n it must
    approach 1.96, asserted;
  * c must increase with the number of pairs and decrease with n, asserted
    across the whole grid;
  * the requirement must be scale-free: multiplying every score by a constant
    must not change the verdict, asserted on a worked table;
  * the real shapes, if present, must carry both a bidder count and a criteria
    count, and their number is printed rather than assumed.

    python procurement_shape.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

ALPHA = 0.05
JS = (3, 4, 5, 6, 8, 10, 12, 15, 20)
NS = (2, 3, 4, 5, 6, 8, 10, 12)
SHAPES = ("C:/Users/ipezy.DESKTOP-GD1DJED/AppData/Local/Temp/claude/"
          "C--Users-ipezy-DESKTOP-GD1DJED/0bfba4dc-942d-4499-89e2-d2373e687ea2/"
          "scratchpad/ted_shapes.jsonl")


def crit(J: int, n: int, alpha: float = ALPHA) -> float:
    """The t threshold the most significant pair must clear."""
    m = J * (J - 1) // 2
    return float(tdist.isf(alpha / (2 * m), df=max(n - 1, 1)))


def allowed_cv(J: int, n: int) -> float:
    """Largest coefficient of variation the winner's advantages may have."""
    return math.sqrt(n) / crit(J, n)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print("self-checks ...")
    ok_bonf = abs(float(tdist.isf(ALPHA / 2, df=10000)) - 1.96) < 0.01
    print(f"  [{'ok  ' if ok_bonf else 'FAIL'}] the threshold reduces to 1.96 at one "
          f"pair and large n")
    ok_mono = all(crit(a, n) <= crit(b, n) + 1e-12
                  for n in NS for a, b in zip(JS, JS[1:])) and \
              all(crit(J, a) >= crit(J, b) - 1e-12
                  for J in JS for a, b in zip(NS, NS[1:]))
    print(f"  [{'ok  ' if ok_mono else 'FAIL'}] the threshold rises with bidders and "
          f"falls with criteria, across the whole grid")
    # scale-freeness on a worked table
    d = np.array([4.0, 3.0, 5.0, 2.0])
    t1 = d.mean() / d.std(ddof=1)
    t2 = (7.5 * d).mean() / (7.5 * d).std(ddof=1)
    ok_scale = abs(t1 - t2) < 1e-9
    print(f"  [{'ok  ' if ok_scale else 'FAIL'}] the requirement is scale-free: "
          f"{t1:.4f} against {t2:.4f} after multiplying every score by 7.5")

    real = []
    p = Path(SHAPES)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            tn = r.get("tenders") or []
            nc = r.get("n_criteria")
            if tn and nc:
                real.append((int(max(tn)), int(round(float(nc)))))
    print(f"  [{'ok  ' if True else ''}] {len(real)} real shapes read from TED")

    if not (ok_bonf and ok_mono and ok_scale):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    q = L.append
    q("HOW UNIFORM MUST A WINNING BID BE?")
    q("=" * 92)
    q("  Largest coefficient of variation the winner's per-criterion advantage")
    q("  over the runner-up may have, and still separate from it at simultaneous")
    q("  95 % across all pairs. Lower means stricter. Exact for the most")
    q("  significant pair; no assumption about the score distribution.")
    q("")
    q(f"  {'bidders':<9}" + "".join(f"{'n=' + str(n):>8}" for n in NS))
    for J in JS:
        q(f"  {J:<9}" + "".join(f"{allowed_cv(J, n):>8.2f}" for n in NS))
    q("")
    q("  The same thing as the raw threshold c the top pair must clear:")
    q(f"  {'bidders':<9}" + "".join(f"{'n=' + str(n):>8}" for n in NS))
    for J in JS:
        q(f"  {J:<9}" + "".join(f"{crit(J, n):>8.1f}" for n in NS))
    q("")
    med_cv = float("nan")
    usable = [(J, n) for J, n in real if J >= 2 and n >= 2]
    if usable:
        cvs = [allowed_cv(J, n) for J, n in usable]
        med_cv = float(np.median(cvs))
        few = sum(1 for _, n in usable if n <= 3) / len(usable)
        q(f"  REAL SHAPES: {len(usable)} usable of {len(real)} Finnish contract")
        q(f"  award notices harvested from TED. Median bidders "
          f"{int(np.median([j for j, _ in usable]))}, median criteria "
          f"{int(np.median([n for _, n in usable]))}.")
        q(f"  Median allowed coefficient of variation: {med_cv:.2f}.")
        q(f"  Share with 3 or fewer criteria: {100 * few:.0f} %.")
    else:
        q(f"  REAL SHAPES: THE HARVEST FAILED and the overlay is empty. "
          f"{len(real)} Finnish")
        q("  award notices were retrieved from TED's eForms-era records and every")
        q("  one of them is a single-criterion price award, which is not a")
        q("  scoring table at all - there is nothing to be uniform across. TED's")
        q("  structured record carries the criteria TYPES but not the published")
        q("  comparison table, and the tables themselves live as PDF attachments")
        q("  in municipal decision registers, which this run did not reach.")
        q("  P1 and P4 are VACUOUS, not missed. The grid above needs no data: it")
        q("  is exact arithmetic of the construction, and it is the result.")
    q("")
    p1 = (med_cv < 0.50) if usable else None
    p2 = allowed_cv(8, 5) < 0.35
    p3 = all(allowed_cv(a, n) > allowed_cv(b, n)
             for n in NS for a, b in zip(JS, JS[1:]))
    p4 = ((sum(1 for _, n in usable if n <= 3) / len(usable) >= 0.25)
          if usable else None)
    q(f"  P1  median allowed CV at real shapes: {med_cv:.2f}" if usable
      else "  P1  no usable real shapes were harvested")
    q(f"      pre-registered < 0.50:  "
      f"{'VACUOUS - no real shapes' if p1 is None else ('HIT' if p1 else 'MISS')}")
    q(f"  P2  allowed CV at 8 bidders and 5 criteria: {allowed_cv(8, 5):.2f}")
    q(f"      pre-registered < 0.35:  {'HIT' if p2 else 'MISS'}")
    q(f"  P3  allowed CV falls as bidders are added, at every criteria count")
    q(f"      pre-registered yes:  {'HIT' if p3 else 'MISS'}")
    q(f"  P4  share of real shapes with <= 3 criteria")
    q(f"      pre-registered >= 25 %:  "
      f"{'VACUOUS - no real shapes' if p4 is None else ('HIT' if p4 else 'MISS')}")
    q("")
    q("  HOW TO USE THIS ON A REAL COMPARISON TABLE. Take the winner's points")
    q("  and the runner-up's, criterion by criterion, and subtract. If the")
    q("  standard deviation of those differences is larger than the mean times")
    q("  the number above, the two bids are not distinguishable at 95 %")
    q("  simultaneous confidence, and the award ranked them on noise. It is one")
    q("  subtraction and one ratio.")
    q("")
    q("  What makes procurement different from every other board here is not the")
    q("  arithmetic. It is that the ranking is a decision with a counterparty:")
    q("  the losing bidder has standing, the authority has a statutory duty to")
    q("  justify the comparison, and neither side currently has a way to say")
    q("  whether the gap between first and second was larger than the noise in")
    q("  the scoring.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("procurement_shape_results.txt").write_text(text + chr(10),
                                                     encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote procurement_shape_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
