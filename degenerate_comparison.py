"""A comparison whose two arms never differ has not been run.

tie_coverage_results.txt was committed with twelve rows in which the bootstrap
column equalled the Holm column to three decimals, the widths were equal, and
the difference column was zero twelve times. It read as "the two constructions
agree". It was one construction printed twice: the results file was generated
before the line that named the second arm. The evidence was in the output and
nothing looked at it.

This looks at it, across every results file in the repository. It parses the
fixed-width numeric tables the tools print, and for each table reports every
pair of numeric columns that is identical in every row. A pair of columns that
never disagrees over many rows is either two names for one number - which is
worth knowing - or a comparison that was not run.

The check cannot tell those two apart and does not try. Some identical columns
are honest: a width that really is the same under both constructions, a count
that is structurally equal. What it does is put them in front of a reader, with
the header line, so the ones that are not honest can be seen. A flag is a
question, not a verdict.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  the check fires on the PRE-FIX tie_coverage_results.txt, fetched from git
      history. A check that cannot catch the case it was built for is worthless,
      and this is the only way to know it can.
  P2  the check does NOT fire on the fixed tie_coverage_results.txt. If it fires
      on both, it is flagging the shape of the table rather than the defect.
  P3  at least 1 other results file in the repository carries an identical
      column pair over 4 or more rows.
  P4  of everything flagged outside tie_coverage, most are honest - structurally
      equal columns rather than unrun comparisons. Predicted: at most half of
      the flagged files turn out to be real degenerate comparisons on reading.

  What a miss on P3 would mean: this was a single accident rather than a class,
  and the check earns its place only as a regression guard.

SELF-CHECKS (no table if any fails)
  * the parser must find numeric tables in at least 40 of the results files. A
    parser that reads nothing reports nothing and looks like a clean bill;
  * a synthetic table whose two columns differ in exactly one row of twenty must
    NOT be flagged - an off-by-one tolerance would flag everything;
  * a synthetic table whose two columns are identical MUST be flagged;
  * the pre-fix control must actually differ from the current file, or P1 and P2
    are being run on the same input.

    python degenerate_comparison.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MIN_ROWS = 4          # below this, identical columns are coincidence
MIN_FILES = 40        # the parser must see at least this many tables
CONTROL = "tie_coverage_results.txt"
CONTROL_REV = "f4d6fe2~1"   # the commit before the fix

NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)%?$")


def cells(line: str) -> list[str]:
    return line.split()


def is_num(tok: str) -> bool:
    return bool(NUM.match(tok))


def tables(text: str) -> list[tuple[str, list[list[str]]]]:
    """Split a results file into blocks of consecutive lines with equal token count.

    A tool's table is a run of lines that tokenise to the same width. The line
    immediately above the run is kept as its header, which is what makes a flag
    readable: the column names are the whole point.
    """
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        toks = cells(lines[i])
        if len(toks) < 3:
            i += 1
            continue
        w = len(toks)
        j = i
        block = []
        while j < len(lines) and len(cells(lines[j])) == w:
            block.append(cells(lines[j]))
            j += 1
        if len(block) >= MIN_ROWS:
            header = lines[i - 1] if i > 0 else ""
            out.append((header.strip(), block))
        i = j if j > i else i + 1
    return out


def identical_pairs(block: list[list[str]]) -> list[tuple[int, int, int]]:
    """Column indices that are numeric in every row and equal in every row."""
    w = len(block[0])
    numeric = [c for c in range(w) if all(is_num(r[c]) for r in block)]
    found = []
    for a_i, a in enumerate(numeric):
        for b in numeric[a_i + 1:]:
            if all(r[a] == r[b] for r in block):
                # a column that is constant down its own length is not a
                # comparison at all, it is a repeated label; skip those.
                if len({r[a] for r in block}) == 1:
                    continue
                found.append((a, b, len(block)))
    return found


def scan(text: str) -> list[tuple[str, int, int, int]]:
    hits = []
    for header, block in tables(text):
        for a, b, n in identical_pairs(block):
            hits.append((header, a, b, n))
    return hits


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    here = Path(".").resolve()

    print("self-checks ...")
    files = sorted(here.glob("*_results.txt"))
    parsed = sum(1 for f in files
                 if tables(f.read_text(encoding="utf-8", errors="replace")))
    ok_parse = parsed >= MIN_FILES
    print(f"  [{'ok  ' if ok_parse else 'FAIL'}] the parser found a table in "
          f"{parsed} of {len(files)} results files (need >= {MIN_FILES})")

    near = "\n".join(["  h a b"] + [f"  r{i} {i} {i}" for i in range(19)] + ["  r19 19 20"])
    same = "\n".join(["  h a b"] + [f"  r{i} {i} {i}" for i in range(20)])
    ok_near = not scan(near)
    ok_same = bool(scan(same))
    print(f"  [{'ok  ' if ok_near else 'FAIL'}] a table differing in 1 row of 20 is not flagged")
    print(f"  [{'ok  ' if ok_same else 'FAIL'}] a table identical in 20 of 20 is flagged")

    cur = (here / CONTROL).read_text(encoding="utf-8", errors="replace")
    old = subprocess.run(["git", "show", f"{CONTROL_REV}:{CONTROL}"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace").stdout
    ok_ctrl = bool(old) and old.replace("\r", "") != cur.replace("\r", "")
    print(f"  [{'ok  ' if ok_ctrl else 'FAIL'}] the pre-fix control differs from the current file")

    if not (ok_parse and ok_near and ok_same and ok_ctrl):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    pre = scan(old)
    post = scan(cur)

    flagged = {}
    for f in files:
        hits = scan(f.read_text(encoding="utf-8", errors="replace"))
        if hits:
            flagged[f.name] = hits

    L = []
    p = L.append
    p("A COMPARISON WHOSE ARMS NEVER DIFFER HAS NOT BEEN RUN")
    p("=" * 88)
    p(f"  {len(files)} results files, {parsed} of them carrying a numeric table of "
      f"{MIN_ROWS}+ rows.")
    p("")
    p("  CONTROL: tie_coverage_results.txt, the file this check was built from")
    p(f"    before the fix: {len(pre)} identical column pair(s)")
    for h, a, b, n in pre[:4]:
        p(f"      cols {a} and {b}, equal in all {n} rows   under: {h[:56]}")
    p(f"    after the fix:  {len(post)} identical column pair(s)")
    for h, a, b, n in post[:4]:
        p(f"      cols {a} and {b}, equal in all {n} rows   under: {h[:56]}")
    p("")
    p(f"  ELSEWHERE: {len([k for k in flagged if k != CONTROL])} other results "
      f"file(s) carry an identical numeric column pair over {MIN_ROWS}+ rows.")
    p("")
    for name in sorted(k for k in flagged if k != CONTROL):
        p(f"  {name}")
        for h, a, b, n in flagged[name][:3]:
            p(f"      cols {a} and {b}, equal in all {n} rows")
            p(f"      header: {h[:76]}")
    p("")
    p(f"  P1  fires on the pre-fix control: {len(pre)} pair(s)         "
      f"pre-registered > 0:   {'HIT' if pre else 'MISS'}")
    p(f"  P2  silent on the fixed control:  {len(post)} pair(s)         "
      f"pre-registered = 0:   {'HIT' if not post else 'MISS'}")
    n_other = len([k for k in flagged if k != CONTROL])
    p(f"  P3  other files flagged: {n_other}                       "
      f"pre-registered >= 1:  {'HIT' if n_other >= 1 else 'MISS'}")
    p("  P4  how many of those are real unrun comparisons rather than columns")
    p("      that are structurally equal is a reading, not a count, and it is")
    p("      written under the table rather than scored here.")
    p("")
    p("  A flag is a question. Two columns that never disagree are either two")
    p("  names for one number - honest, and worth saying out loud - or a")
    p("  comparison that was never run. The output cannot tell those apart, so")
    p("  it prints the header line and lets a reader decide. What it removes is")
    p("  the third case: nobody looking at all.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("degenerate_comparison_results.txt").write_text(text + chr(10),
                                                         encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote degenerate_comparison_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
