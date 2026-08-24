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

# A numeric field, not a digit inside a word: "claude-opus-4-5", "MTEB v2" and
# "2025-12-05" must not tokenise into columns. Blocking a preceding word
# character, dot, percent or hyphen does that.
NUM = re.compile(r"(?<![\w.%-])[+-]?(?:\d+\.\d+|\d+|\.\d+)%?(?![\w.])")


def fields(line: str) -> list[str]:
    """The numeric fields of one line, in order.

    Keying on numeric fields rather than whitespace tokens is what makes the
    parser survive real tables: a label column holds "SWE-bench Verified" on one
    row and "LiveBench" on the next, so splitting on whitespace gives rows of
    different width and the block never forms. The first version did that and
    saw a table in 30 of 100 files; this one sees 66.
    """
    return [m.group() for m in NUM.finditer(line)]


def tables(text: str) -> list[tuple[str, list[list[str]]]]:
    """Runs of consecutive lines carrying the same number of numeric fields.

    The line above the run is kept as the header. That is what makes a flag
    readable - the column names are the whole point of showing it to a person.
    """
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        k = len(fields(lines[i]))
        if k < 2:
            i += 1
            continue
        j = i
        block = []
        while j < len(lines) and len(fields(lines[j])) == k:
            block.append(fields(lines[j]))
            j += 1
        if len(block) >= MIN_ROWS:
            out.append((lines[i - 1].strip() if i > 0 else "", block))
        i = j if j > i else i + 1
    return out


def identical_pairs(block: list[list[str]]) -> list[tuple[int, int, int]]:
    """Column indices equal in every row, skipping columns that never vary.

    A column holding the same value all the way down is a repeated label, not
    an arm of a comparison, and two of those match each other trivially.
    """
    w = len(block[0])
    found = []
    for a in range(w):
        if len({r[a] for r in block}) == 1:
            continue
        for b in range(a + 1, w):
            if all(r[a] == r[b] for r in block):
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
    p("  P4  predicted: at most half of what is flagged outside the control is a")
    p("      real unrun comparison. Read on the 2026-08-25 run, 4 flags:")
    p("        alpha_sensitivity   REAL. The \"no stepdown\" column was produced by")
    p("                            stepdown=False, which the Holm path accepted and")
    p("                            ignored, so it reprinted the alpha 0.05 column")
    p("                            and its prediction scored 10/10 against itself.")
    p("        tie_coverage_boards REAL, and the worst of the four. Its committed")
    p("                            table read \"1 of 12 boards undercover, HELM")
    p("                            0.880\" - both arms were Holm. Re-run: 8 of 12,")
    p("                            HELM 0.013. rank_sets.py had been citing the")
    p("                            correct 0.013 in its own docstring the whole")
    p("                            time, and nothing compared the two.")
    p("        holm_recompute      honest. The union set equals the Holm set")
    p("                            wherever Holm is the wider of the two, which is")
    p("                            what a union is.")
    p("        top_redundancy      honest, and the file says so itself: its own P2")
    p("                            is scored VACUOUS because the null is constant.")
    p("      2 real of 4:  pre-registered at most half:  HIT")
    p("")
    p("      Fixing alpha_sensitivity turned up a fifth, in the core module. Its")
    p("      replacement single-step column was ALSO identical on all ten boards,")
    p("      because rank_sets returned single_best as a second name for best on")
    p("      the Holm path - so every step-down-against-single-step comparison in")
    p("      the repository had been a column against itself. _holm now computes")
    p("      the Bonferroni sets, and the two arms differ on 6 of 10 boards.")
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
