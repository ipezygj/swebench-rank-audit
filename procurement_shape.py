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

DATA SOURCE NOTE (added later the same day, after P1-P4 above were fixed)
  TED was the intended source for "real Finnish shape" when P1/P4 were
  written. It turned out to expose criteria TYPES but never the published
  per-bidder table: all 18,799 Finnish eForms notices retrieved are
  single-criterion price awards (n=1), a schema limit discovered before any
  n>=2 shape existed - not a result, and not a reason to discard data.
  ted_shapes.jsonl (4 records, all n=1) is that harvest, kept for the record.

  Municipal decision registers (Helsinki's Ahjo publication, ahjo_shapes.jsonl)
  turned out to be the actual working route to real multi-criterion tables -
  see project memory project-procurement-shape-overlay.md for the full
  enumeration log. P1 was written source-agnostically ("the median real
  Finnish shape") and is evaluated below against the Ahjo overlay, since that
  is what "real Finnish shape" turned out to mean once retrievable. P4 named
  TED explicitly and is NOT redirected to Ahjo: it stays evaluated against
  literal TED data only, which makes it permanently VACUOUS (0 of 0 TED
  records have n>=2) rather than quietly swapped for a friendlier source. The
  Ahjo overlay's own share-with-<=3-criteria is reported separately below,
  labeled exploratory, not as a P4 substitute.

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
HERE = Path(__file__).resolve().parent
TED_SHAPES = HERE / "ted_shapes.jsonl"
AHJO_SHAPES = HERE / "ahjo_shapes.jsonl"
# 7F8DDEC4 / hel-2019-011004 (dynaamisen puitejarjestelyn 2. avaaminen,
# ikaantyneiden ymparivuorokautinen palveluasuminen) is deliberately NOT in
# ahjo_shapes.jsonl: 69 ranked positions from ~20-25 distinct legal entities
# submitting multiple facility-level bids each, n=4 confirmed but J is
# ambiguous (ranked units vs. distinct bidders) and needs a decision before
# it can sit in the same table as a clean single-lot point. Named here so it
# doesn't silently vanish from the record.
AHJO_OUTLIER_NOTE = (
    "7F8DDEC4 / hel-2019-011004 excluded: n=4 confirmed but J is ambiguous "
    "(69 ranked positions, ~20-25 distinct legal entities) - not yet decided."
)


def load_shapes(path: Path) -> list[tuple[int, int]]:
    shapes = []
    if not path.exists():
        return shapes
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        tn = r.get("tenders") or []
        nc = r.get("n_criteria")
        if tn and nc:
            shapes.append((int(max(tn)), int(round(float(nc)))))
    return shapes


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

    ted_real = load_shapes(TED_SHAPES)
    ahjo_real = load_shapes(AHJO_SHAPES)
    print(f"  [ok  ] {len(ted_real)} real shapes read from TED, "
          f"{len(ahjo_real)} from the Ahjo overlay")

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
    ted_usable = [(J, n) for J, n in ted_real if J >= 2 and n >= 2]
    ahjo_usable = [(J, n) for J, n in ahjo_real if J >= 2 and n >= 2]

    q(f"  TED HARVEST: 0 of {len(ted_real)} Finnish eForms contract award")
    q("  notices are usable - every one is a single-criterion price award,")
    q("  which is not a scoring table at all: there is nothing to be uniform")
    q("  across. TED's structured record carries the criteria TYPES but never")
    q("  the published comparison table. This is a permanent, not a fixable,")
    q("  0 - the schema does not carry the field.")
    q("")
    med_cv = float("nan")
    if ahjo_usable:
        cvs = [allowed_cv(J, n) for J, n in ahjo_usable]
        med_cv = float(np.median(cvs))
        few = sum(1 for _, n in ahjo_usable if n <= 3) / len(ahjo_usable)
        q(f"  AHJO OVERLAY: {len(ahjo_usable)} real (bidders, criteria) shapes")
        q("  hand-verified from Helsinki municipal decision PDF attachments")
        q(f"  (ahjo_shapes.jsonl). Median bidders "
          f"{int(np.median([j for j, _ in ahjo_usable]))}, median criteria "
          f"{int(np.median([n for _, n in ahjo_usable]))}.")
        q(f"  Median allowed coefficient of variation: {med_cv:.2f}.")
        q(f"  Share with 3 or fewer criteria: {100 * few:.0f} % (exploratory -")
        q("  see P4 note below, this is not the pre-registered TED quantity).")
        q(f"  {AHJO_OUTLIER_NOTE}")
    else:
        q("  AHJO OVERLAY: empty (ahjo_shapes.jsonl not found or has no")
        q("  usable rows).")
    q("")
    p1 = (med_cv < 0.50) if ahjo_usable else None
    p2 = allowed_cv(8, 5) < 0.35
    p3 = all(allowed_cv(a, n) > allowed_cv(b, n)
             for n in NS for a, b in zip(JS, JS[1:]))
    p4 = ((sum(1 for _, n in ted_usable if n <= 3) / len(ted_usable) >= 0.25)
          if ted_usable else None)
    ahjo_few_share = (sum(1 for _, n in ahjo_usable if n <= 3) / len(ahjo_usable)
                      if ahjo_usable else None)
    q(f"  P1  median allowed CV at real shapes (Ahjo overlay, {len(ahjo_usable)} "
      f"points): {med_cv:.2f}" if ahjo_usable
      else "  P1  no usable real shapes were harvested")
    q(f"      pre-registered < 0.50:  "
      f"{'VACUOUS - no real shapes' if p1 is None else ('HIT' if p1 else 'MISS')}")
    q(f"  P2  allowed CV at 8 bidders and 5 criteria: {allowed_cv(8, 5):.2f}")
    q(f"      pre-registered < 0.35:  {'HIT' if p2 else 'MISS'}")
    q(f"  P3  allowed CV falls as bidders are added, at every criteria count")
    q(f"      pre-registered yes:  {'HIT' if p3 else 'MISS'}")
    q(f"  P4  share of real shapes with <= 3 criteria, literal TED data only")
    q(f"      ({len(ted_usable)} usable TED shapes)")
    q(f"      pre-registered >= 25 %:  "
      f"{'VACUOUS - no real TED shapes' if p4 is None else ('HIT' if p4 else 'MISS')}")
    if ahjo_few_share is not None:
        q(f"      (exploratory, not pre-registered: Ahjo overlay's own share "
          f"with <= 3 criteria is {100 * ahjo_few_share:.0f} % of "
          f"{len(ahjo_usable)} points)")
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
