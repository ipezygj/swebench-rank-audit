"""Every number in a generated document should exist in a results file.

LAWS.md is built by a script, but not all of it is derived. Its tables are read
out of results files; its prose is typed. Today the correction box at the top -
the part that tells a reader which numbers to distrust - was found to contain
six figures that contradicted the file it cites, and all six traced to the same
cause: they were copied from the Holm implementation with a normal reference,
which was discarded when the two implementations were reconciled onto t with
n-1 degrees of freedom. HELM's "21 possible first places" and its critical
value of "4.26" were that discarded run. The committed file said 50 and 8.45.

Nothing in the repository could have caught that. staleness_audit.py asks
whether a results file is older than its tool and whether a document cites a
stale file; it never asks whether the document's numbers are the file's
numbers.

This asks. It reads the generated documents, takes every numeric literal that
looks like a measurement, and searches the results files for it. A number that
appears in none of them was typed rather than derived, and typed numbers drift.

A flag is a question, not a verdict. Some prose numbers are legitimately not
measurements - a count of laws, a section number, a year - and the filters
below remove the obvious ones, but not all. What the output removes is the case
where nobody is looking.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  run against the PRE-FIX LAWS.md from git, the checker flags at least 4 of
      the 6 figures known to have been wrong: 4.6, +0.2, 21, 4.26, 18, 16.
      A checker that cannot find the case it was built for is worthless.
  P2  run against the current LAWS.md, those specific figures are gone.
  P3  at least 3 further unmatched numbers remain somewhere in the generated
      documents. This was a class, not one bad paragraph.
  P4  LAWS.md carries more unmatched numbers per hundred than the other
      generated documents, because it is the one with a hand-written
      correction box.

  What a miss on P3 would mean: the correction box was the only hand-typed
  passage in the repository's documents and the rest is derived, which would be
  the good outcome and should be said as plainly as the bad one.

SELF-CHECKS (no table if any fails)
  * a number planted in a copy of a document, chosen not to occur in any
    results file, must be flagged;
  * a number lifted verbatim OUT of a results file and inserted into that copy
    must NOT be flagged;
  * the results corpus must be non-empty and must be read: at least 90 files
    and at least 20 000 numeric tokens across them;
  * at least 3 documents parsed and at least 50 candidate numbers examined.

    python doc_numbers.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DOCS = ("LAWS.md", "LEADERBOARD_STANDARD.md", "README.md", "PRIOR_ART.md")

# A measurement looks like a decimal, a percentage, or a signed number. Bare
# small integers are usually counts in a sentence and are skipped; years and
# arXiv-style identifiers are skipped explicitly.
CAND = re.compile(r"(?<![\w.])([+-]?\d+\.\d+|[+-]?\d+\s?%|[+-]\d+)(?![\w])")
NUMTOK = re.compile(r"(?<![\w.])[+-]?\d+\.?\d*%?(?![\w])")
SKIP = re.compile(r"^(19|20)\d\d$|^\d{4}\.\d{5}$")


def corpus() -> set[str]:
    """Every numeric token appearing in any results file, as text."""
    out = set()
    for f in Path(".").glob("*_results.txt"):
        for m in NUMTOK.finditer(f.read_text(encoding="utf-8", errors="replace")):
            t = m.group().replace(" ", "")
            out.add(t)
            out.add(t.lstrip("+"))
            out.add(t.rstrip("%"))
            out.add(t.lstrip("+").rstrip("%"))
    return out


def unmatched(text: str, known: set[str]) -> list[str]:
    """Candidate measurements in prose that appear in no results file.

    Table rows are skipped: they are generated from the files by construction,
    and including them would bury the typed numbers under hundreds of derived
    ones.
    """
    bad = []
    for line in text.splitlines():
        st = line.strip()
        if st.startswith("|") or st.startswith("```") or st.startswith("    "):
            continue
        for m in CAND.finditer(line):
            t = m.group().replace(" ", "")
            if SKIP.match(t.lstrip("+-").rstrip("%")):
                continue
            if t in known or t.lstrip("+") in known or t.rstrip("%") in known \
                    or t.lstrip("+").rstrip("%") in known:
                continue
            bad.append(t)
    return bad


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print("self-checks ...")
    files = sorted(Path(".").glob("*_results.txt"))
    known = corpus()
    ok_corpus = len(files) >= 90 and len(known) >= 20000
    print(f"  [{'ok  ' if ok_corpus else 'FAIL'}] {len(files)} results files, "
          f"{len(known)} distinct numeric tokens (need >= 90 and >= 20000)")

    doc = Path("LAWS.md").read_text(encoding="utf-8", errors="replace")
    planted = doc + "\n\nThe measured value was 8675.309 points.\n"
    lifted = doc + "\n\nThe measured value was " + \
        sorted(t for t in known if "." in t and len(t) > 4)[len(known) // 2] + " points.\n"
    ok_plant = "8675.309" in unmatched(planted, known)
    ok_lift = len(unmatched(lifted, known)) == len(unmatched(doc, known))
    print(f"  [{'ok  ' if ok_plant else 'FAIL'}] a planted number absent from every "
          f"results file is flagged")
    print(f"  [{'ok  ' if ok_lift else 'FAIL'}] a number lifted verbatim out of a "
          f"results file is not flagged")

    present = [d for d in DOCS if Path(d).exists()]
    cands = sum(len(list(CAND.finditer(l))) for d in present
                for l in Path(d).read_text(encoding="utf-8", errors="replace").splitlines()
                if not l.strip().startswith("|"))
    ok_docs = len(present) >= 3 and cands >= 50
    print(f"  [{'ok  ' if ok_docs else 'FAIL'}] {len(present)} documents, "
          f"{cands} candidate numbers (need >= 3 and >= 50)")

    old = subprocess.run(["git", "show", "HEAD:LAWS.md"], capture_output=True,
                         text=True, encoding="utf-8", errors="replace").stdout
    ok_ctrl = bool(old)
    print(f"  [{'ok  ' if ok_ctrl else 'FAIL'}] the pre-fix control is readable from git")

    if not (ok_corpus and ok_plant and ok_lift and ok_docs and ok_ctrl):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    WRONG = ("4.6", "+0.2", "21", "4.26", "18", "16")
    pre = unmatched(old, known)
    post = unmatched(doc, known)
    caught = [w for w in WRONG if w in pre]
    still = [w for w in WRONG if w in post]

    per_doc = {d: unmatched(Path(d).read_text(encoding="utf-8", errors="replace"), known)
               for d in present}

    L = []
    p = L.append
    p("NUMBERS IN GENERATED DOCUMENTS THAT NO RESULTS FILE CONTAINS")
    p("=" * 92)
    p(f"  {len(files)} results files supply {len(known)} distinct numeric tokens.")
    p("  Table rows are skipped - they are derived by construction. What is")
    p("  checked is prose.")
    p("")
    p(f"  CONTROL: LAWS.md as committed, before today's fix")
    p(f"    of the 6 figures known to have been wrong, flagged: "
      f"{', '.join(caught) if caught else 'none'}")
    p(f"    still present after the fix: {', '.join(still) if still else 'none'}")
    p("")
    p(f"  {'document':<28}{'prose numbers':>15}{'unmatched':>12}{'rate':>9}")
    for d in present:
        tot = sum(1 for l in Path(d).read_text(encoding="utf-8",
                                               errors="replace").splitlines()
                  if not l.strip().startswith("|")
                  for _ in CAND.finditer(l))
        p(f"  {d:<28}{tot:>15}{len(per_doc[d]):>12}"
          f"{(100 * len(per_doc[d]) / tot if tot else 0):>8.1f}%")
    p("")
    for d in present:
        if per_doc[d]:
            p(f"  {d}: {', '.join(sorted(set(per_doc[d])))}")
    p("")
    other = sum(len(per_doc[d]) for d in present if d != "LAWS.md")
    p(f"  P1  pre-fix control flagged {len(caught)} of 6 known-wrong figures   "
      f"pre-registered >= 4:  {'HIT' if len(caught) >= 4 else 'MISS'}")
    p(f"  P2  those figures still unmatched after the fix: {len(still)}      "
      f"pre-registered = 0:   {'HIT' if not still else 'MISS'}")
    p(f"  P3  further unmatched numbers across the documents: "
      f"{sum(len(v) for v in per_doc.values())}   "
      f"pre-registered >= 3:  "
      f"{'HIT' if sum(len(v) for v in per_doc.values()) >= 3 else 'MISS'}")
    p(f"  P4  LAWS.md unmatched {len(per_doc.get('LAWS.md', []))}, "
      f"the others {other} between them")
    p("")
    p("  A flag is a question. A number in prose that appears in no results")
    p("  file was typed rather than derived, and a typed number cannot be")
    p("  regenerated - it stays as it was on the day it was typed while the")
    p("  measurement underneath it moves. Some of these are legitimate: counts")
    p("  in a sentence, a version, an identifier. The point is that somebody")
    p("  has to look, and until now nothing asked.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("doc_numbers_results.txt").write_text(text + chr(10), encoding="utf-8",
                                               newline=chr(10))
    print(chr(10) + "wrote doc_numbers_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
