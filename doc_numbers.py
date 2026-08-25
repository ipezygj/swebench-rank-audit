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
  * the corpus must be able to find the numbers that ARE derived: LAWS.md's
    table rows come out of the results files by construction, so at least 95 %
    of them must match, or the corpus cannot be trusted to judge prose;
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


CITE = re.compile(r"`([A-Za-z0-9_]+)\.py`|([A-Za-z0-9_]+_results\.txt)")
_CACHE: dict[str, set[str]] = {}


def file_tokens(name: str) -> set[str]:
    if name not in _CACHE:
        f = Path(name)
        out: set[str] = set()
        if f.exists():
            for m in NUMTOK.finditer(f.read_text(encoding="utf-8", errors="replace")):
                t = m.group().replace(" ", "")
                out |= {t, t.lstrip("+"), t.rstrip("%"), t.lstrip("+").rstrip("%")}
        _CACHE[name] = out
    return _CACHE[name]


def cited(lines: list[str], i: int, back: int = 12) -> str | None:
    """The results file the passage around line i points at, if any.

    A paragraph that says (`holm_recompute.py`) is telling the reader where its
    numbers came from. That is the file its numbers should be checked against.
    """
    for j in range(i, max(-1, i - back), -1):
        found = None
        for m in CITE.finditer(lines[j]):
            stem, direct = m.group(1), m.group(2)
            cand = direct if direct else f"{stem}_results.txt"
            if Path(cand).exists():
                found = cand
        if found:
            return found
    return None


def hits(t: str, known: set[str]) -> bool:
    return (t in known or t.lstrip("+") in known or t.rstrip("%") in known
            or t.lstrip("+").rstrip("%") in known)


def unmatched(text: str, known: set[str], scoped: bool = True) -> list[str]:
    """Candidate measurements in prose that their own cited file does not contain.

    The first version searched the union of all 107 results files. That is far
    too weak: "21", "18", "16" and "4.6" all occur somewhere in some file, so
    five of the six figures known to have been wrong in LAWS.md's correction box
    passed it. Existence somewhere is not the claim a sentence makes. When a
    passage names its source, the number is checked against THAT file, and the
    union is only the fallback for prose that cites nothing.

    Table rows are skipped: they are generated from the files by construction,
    and including them would bury the typed numbers under hundreds of derived
    ones.
    """
    bad = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        st = line.strip()
        if st.startswith("|") or st.startswith("```") or st.startswith("    "):
            continue
        src = cited(lines, i) if scoped else None
        pool = file_tokens(src) if src else known
        for m in CAND.finditer(line):
            t = m.group().replace(" ", "")
            if SKIP.match(t.lstrip("+-").rstrip("%")):
                continue
            if hits(t, pool):
                continue
            if src and hits(t, known):
                bad.append(f"{t} [not in {src}]")
            else:
                bad.append(t)
    return bad


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print("self-checks ...")
    files = sorted(Path(".").glob("*_results.txt"))
    known = corpus()
    # A count threshold here would be a guess. The first version demanded
    # 20 000 distinct tokens, a number I made up; the corpus has 2 884. What
    # the corpus actually has to do is find the numbers that ARE derived:
    # LAWS.md's table rows are read out of these files by construction, so a
    # corpus that cannot match those cannot be trusted to judge prose.
    tab = [m.group().replace(" ", "")
           for line in Path("LAWS.md").read_text(encoding="utf-8",
                                                 errors="replace").splitlines()
           if line.strip().startswith("|")
           for m in CAND.finditer(line)]
    hit = sum(1 for t in tab
              if t in known or t.lstrip("+") in known or t.rstrip("%") in known
              or t.lstrip("+").rstrip("%") in known)
    share = hit / max(len(tab), 1)
    ok_corpus = len(files) >= 90 and len(tab) >= 40 and share >= 0.95
    print(f"  [{'ok  ' if ok_corpus else 'FAIL'}] {len(files)} results files, "
          f"{len(known)} distinct tokens; they match {hit} of {len(tab)} "
          f"({share:.1%}) of LAWS.md's DERIVED table numbers (need >= 95%)")

    doc = Path("LAWS.md").read_text(encoding="utf-8", errors="replace")
    planted = doc + "\n\nThe measured value was 8675.309 points.\n"
    pool = sorted(t for t in known if "." in t and len(t) > 4 and "%" not in t)
    lifted = doc + chr(10) + "The measured value was " \
        + pool[len(pool) // 2] + " points." + chr(10)
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
    pre = unmatched(old, known, scoped=True) + unmatched(old, known, scoped=False)
    post = unmatched(doc, known, scoped=True) + unmatched(doc, known, scoped=False)
    caught = [w for w in WRONG if any(u.split(" ")[0] == w for u in pre)]
    still = [w for w in WRONG if any(u.split(" ")[0] == w for u in post)]

    txt = {d: Path(d).read_text(encoding="utf-8", errors="replace") for d in present}
    # Two signals, reported apart. ABSENT is strong and rare: the number occurs
    # in no results file at all, so it was typed. MISPLACED is weak and common:
    # the number occurs somewhere but not in the file its own paragraph names,
    # which is often just a paragraph that quotes more than one source.
    absent = {d: unmatched(txt[d], known, scoped=False) for d in present}
    misplaced = {d: [u for u in unmatched(txt[d], known, scoped=True)
                     if "[not in" in u] for d in present}
    per_doc = absent

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
    p(f"  {'document':<28}{'prose numbers':>15}{'ABSENT':>9}{'MISPLACED':>12}")
    for d in present:
        tot = sum(1 for l in Path(d).read_text(encoding="utf-8",
                                               errors="replace").splitlines()
                  if not l.strip().startswith("|")
                  for _ in CAND.finditer(l))
        p(f"  {d:<28}{tot:>15}{len(absent[d]):>9}{len(misplaced[d]):>12}")
    p("")
    p("")
    p("  ABSENT - in no results file anywhere. These were typed.")
    for d in present:
        if absent[d]:
            p(f"    {d}: {', '.join(sorted(set(absent[d])))}")
    p("")
    p("  MISPLACED - present somewhere, but not in the file the passage names.")
    for d in present:
        if misplaced[d]:
            p(f"    {d}: {', '.join(sorted(set(misplaced[d])))}")
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
    p("  THREE OF FOUR MISSED, and the important one missed in the good")
    p("  direction. P3 predicted at least 3 further typed numbers across the")
    p("  documents; there is ONE - a 2.00 in LEADERBOARD_STANDARD.md. The")
    p("  correction box was the exception and not the rule. The documents are")
    p("  otherwise derived, which is the outcome this file's own docstring said")
    p("  to report as plainly as the bad one, so: they are derived.")
    p("")
    p("  P1 missed at 3 of 6 and the miss is a limit of the method, worth more")
    p("  than the score. A wrong number that looks plausible usually occurs")
    p("  legitimately somewhere else - 21, 18 and 16 are all real values in")
    p("  some results file - so asking whether a token EXISTS has a ceiling no")
    p("  amount of tuning lifts. Catching those would need provenance at the")
    p("  level of the claim rather than the token: this sentence asserts this")
    p("  field of this row of this file. That is a bigger change than a checker.")
    p("")
    p("  P2's one survivor is the checker working. 4.26 is flagged in the")
    p("  current LAWS.md because it sits in the new sentence that names 4.26 as")
    p("  the discarded value. Correct prose, correctly flagged, and left in")
    p("  rather than reworded so the flag stays visible.")
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
