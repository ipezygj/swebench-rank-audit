"""What changed across the whole repo when the construction was corrected.

run_all.py was run twice on identical inputs, once with the multiplier
bootstrap and once with RANK_SETS_METHOD=holm, and this compares the two sets
of results files. The point is not that numbers moved - they must - but WHICH
claims moved and by how much, so that a reader can tell a conclusion that
survives the correction from one that depended on it.

    python holm_pipeline_diff.py <bootstrap_dir> <holm_dir>
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

NUM = re.compile(r"-?\d+\.?\d*")


def numbers(line):
    return [float(v) for v in NUM.findall(line)]


def biggest_move(a_lines, b_lines):
    """Largest absolute change in any number on a line that differs."""
    worst, where = 0.0, ""
    sm = difflib.SequenceMatcher(None, a_lines, b_lines)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        for a, b in zip(a_lines[i1:i2], b_lines[j1:j2]):
            na, nb = numbers(a), numbers(b)
            if len(na) != len(nb):
                continue
            for x, y in zip(na, nb):
                if abs(y - x) > worst:
                    worst, where = abs(y - x), a.strip()[:52]
    return worst, where


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    old, new = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for f in sorted(old.glob("*_results.txt")):
        g = new / f.name
        if not g.exists():
            rows.append((f.name, "MISSING in holm run", 0.0, ""))
            continue
        a = f.read_text(encoding="utf-8", errors="replace").splitlines()
        b = g.read_text(encoding="utf-8", errors="replace").splitlines()
        if a == b:
            rows.append((f.name, "identical", 0.0, ""))
        else:
            changed = sum(1 for line in difflib.unified_diff(a, b, n=0)
                          if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
            worst, where = biggest_move(a, b)
            rows.append((f.name, f"{changed} lines", worst, where))

    same = [r for r in rows if r[1] == "identical"]
    diff = [r for r in rows if r[1] not in ("identical",) and "MISSING" not in r[1]]
    miss = [r for r in rows if "MISSING" in r[1]]
    diff.sort(key=lambda r: -r[2])

    L = []
    p = L.append
    p("THE WHOLE PIPELINE, BOOTSTRAP AGAINST HOLM")
    p("=" * 96)
    p(f"  {len(rows)} results files compared: {len(same)} identical, {len(diff)} changed, "
      f"{len(miss)} missing from the Holm run.")
    p("")
    p(f"  {'results file':<40} {'changed':>10} {'largest move':>13}  on")
    for name, status, worst, where in diff[:40]:
        p(f"  {name:<40} {status:>10} {worst:>13.3f}  {where}")
    if len(diff) > 40:
        p(f"  ... and {len(diff) - 40} more changed files")
    if miss:
        p("")
        p("  missing from the Holm run (the tool failed or was skipped):")
        for name, _, _, _ in miss:
            p(f"    {name}")
    p("")
    p("  identical files are the ones whose claims do not touch rank sets at")
    p("  all - item-side measurements, family clustering, the harness")
    p("  decomposition - and they are as much of the answer as the changed ones.")
    text = "\n".join(L)
    print(text)
    Path("holm_pipeline_diff_results.txt").write_text(text + "\n", encoding="utf-8",
                                                      newline="\n")
    print("\nwrote holm_pipeline_diff_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
