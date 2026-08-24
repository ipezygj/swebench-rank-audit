"""Which results files are older than the tool that writes them?

The compile guard added in the last iteration exists because a failed build
left the previous PDF in place and every check afterwards read it and reported
that all was well. That failure has a general form, and this repository is full
of the conditions for it: ninety-nine tools each writing a results file, three
generated documents parsing figures out of those files, and several tools
reading each other's.

A results file older than its own tool is a measurement from a version of the
code that no longer exists. Nothing warns about it, every document quoting it
looks fine, and the number can be wrong in either direction.

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  at least 3 results files are older than the tool that writes them.
  P2  at least one generated document - LAWS.md, README.md, the standard, the
      paper - quotes a figure from a stale file.
  P3  none of the results files the paper parses is stale. The pipeline was
      rerun end to end two iterations ago, so if any of those five is stale the
      rerun did not do what it claimed.
  P4  after re-running the stale tools, none remains stale.

SELF-CHECKS (no table if any fails)
  * the detector must find a planted staleness: touching a tool must make its
    pair report stale, and touching the results file must clear it;
  * the detector must not flag a pair it has just refreshed;
  * mtimes are local. Git does not preserve them, so on a fresh clone every
    file carries the checkout time and this audit says nothing. The report
    states that rather than implying otherwise.

    python staleness_audit.py [--fix]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

DOCS = ["LAWS.md", "README.md", "LEADERBOARD_STANDARD.md", "laws_paper/paper.tex"]
SKIP_RERUN = {"selection_sbi.py"}      # known to fail its own calibration check


def pairs():
    """(tool, results file) for every results file with a tool of that name."""
    out = []
    for r in sorted(Path(".").glob("*_results.txt")):
        tool = Path(r.name.replace("_results.txt", ".py"))
        if tool.exists():
            out.append((tool, r))
    return out


def stale(tool: Path, res: Path) -> bool:
    return tool.stat().st_mtime > res.stat().st_mtime


def _check_detector() -> tuple[bool, str]:
    """Plant a staleness and confirm it is seen, then clear it."""
    t = Path("_stale_probe.py")
    r = Path("_stale_probe_results.txt")
    try:
        r.write_text("probe\n", encoding="utf-8")
        time.sleep(0.05)
        t.write_text("# probe\n", encoding="utf-8")
        planted = stale(t, r)
        time.sleep(0.05)
        r.write_text("probe\n", encoding="utf-8")
        cleared = not stale(t, r)
        return planted and cleared, (f"planted staleness detected: {planted}; "
                                     f"cleared after rewriting: {cleared}")
    finally:
        for p in (t, r):
            if p.exists():
                p.unlink()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    fix = "--fix" in sys.argv

    print("self-checks ...")
    ok, msg = _check_detector()
    print(f"  [{'ok  ' if ok else 'FAIL'}] {msg}")
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    ps = pairs()
    bad = [(t, r) for t, r in ps if stale(t, r)]

    doc_hits = []
    for d in DOCS:
        p = Path(d)
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        for t, r in bad:
            stem = r.name.replace("_results.txt", "")
            if stem in txt:
                doc_hits.append((d, stem))

    repaired, changed = [], []
    if fix and bad:
        for t, r in bad:
            if t.name in SKIP_RERUN:
                continue
            before = r.read_text(encoding="utf-8", errors="replace")
            print(f"  re-running {t.name} ...")
            run = subprocess.run([sys.executable, t.name], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=1800)
            after = r.read_text(encoding="utf-8", errors="replace")
            repaired.append(t.name)
            if before.splitlines() != after.splitlines():
                changed.append(t.name)
        bad = [(t, r) for t, r in pairs() if stale(t, r)]

    L = []
    p = L.append
    p("RESULTS FILES OLDER THAN THE TOOL THAT WRITES THEM")
    p("=" * 88)
    p(f"  {len(ps)} tool/results pairs examined.")
    p("")
    if bad:
        p(f"  {len(bad)} stale:")
        for t, r in bad:
            age = (t.stat().st_mtime - r.stat().st_mtime) / 3600.0
            p(f"    {t.name:<34} tool is {age:>7.1f} h newer than its results")
    else:
        p("  none stale.")
    p("")
    if doc_hits:
        p("  generated documents quoting a stale file:")
        for d, stem in sorted(set(doc_hits)):
            p(f"    {d:<26} cites {stem}")
    else:
        p("  no generated document quotes a stale file.")
    if fix:
        p("")
        p(f"  re-ran {len(repaired)} tools; {len(changed)} produced different output"
          + (": " + ", ".join(changed) if changed else ""))
    p("")
    p("  A results file older than its tool is a measurement from a version of")
    p("  the code that no longer exists. Nothing warns about it: the file parses,")
    p("  the documents that quote it render, and the number can be wrong in")
    p("  either direction.")
    p("")
    p("  This audit reads modification times, which git does not preserve. On a")
    p("  fresh clone every file carries the checkout time and this table is")
    p("  empty for that reason rather than because the repository is clean. It")
    p("  is a check for a working tree, not for a release.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("staleness_audit_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                   newline=chr(10))
    print(chr(10) + "wrote staleness_audit_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
