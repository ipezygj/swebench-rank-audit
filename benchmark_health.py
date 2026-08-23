"""One card per benchmark: what it can and cannot decide.

The evening produced a dozen measures of the same ten boards. This
assembles them into one table, with each column read from the results file
of the tool that computed it, so the card cannot disagree with the
measurement it summarises.

Columns, and what each one answers:
  tie@1          how many systems could be first (R2)
  top t          the paired statistic of #1 vs #2 (R10)
  crown          share of item bootstraps in which the printed leader stays
                 first
  flips          share of random item halves whose two halves name
                 different first places
  reversed       share of established pairs one half reverses in the other
  cluster        how much wider the rank sets get when items are resampled
                 in their natural groups
  items          share of items needed to recover 90 % of the established
                 pairs

No composite score is computed. A single number would hide exactly the
disagreements between these columns that make them worth reporting - a
board can be stable at the top and unable to order its middle, or the
reverse.

    python benchmark_health.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BOARDS = ["SWE-bench Verified", "MTEB English v2", "HELM classic", "ProteinGym DMS",
          "TabArena 16 models", "TabArena 45 variants", "CASP14", "LiveBench",
          "MathArena 2025", "LMArena categories"]


def read(name):
    p = Path(name)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def col_tie1_and_t():
    """From the standard's report cards."""
    txt = read("leaderboard_standard_results.txt")
    tie, t = {}, {}
    name = None
    for line in txt.splitlines():
        m = re.match(r"LEADERBOARD REPORT CARD - (.+)", line)
        if m:
            name = m.group(1).strip()
        m2 = re.search(r"(\d+) system\(s\) could be first", line)
        if m2 and name:
            tie[name] = int(m2.group(1))
        m3 = re.search(r"R10 pair resolution.*\(t ([\d.-]+)\)", line)
        if m3 and name:
            t[name] = float(m3.group(1))
    return tie, t


def col_crown():
    out = {}
    for line in read("crown_stability_results.txt").splitlines():
        m = re.match(r"\s{2}(\S.*?)\s{2,}\d+\s+\d+\s+([\d.]+)%", line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


def col_split():
    flips, rev = {}, {}
    for line in read("half_split_replication_results.txt").splitlines():
        m = re.match(r"\s{2}(\S.*?)\s{2,}\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)%\s+([\d.]+)\s+(\d+)%", line)
        if m:
            rev[m.group(1).strip()] = float(m.group(2))
            flips[m.group(1).strip()] = float(m.group(4))
    return flips, rev


def col_cluster():
    out = {}
    for line in read("cluster_bootstrap_results.txt").splitlines():
        m = re.match(r"\s{2}(\S.*?)\s{2,}\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s", line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


def col_items():
    out = {}
    for line in read("minimal_benchmark_results.txt").splitlines():
        m = re.match(r"\s{2}(\S.*?)\s{2,}\d+\s+\d+\s+\d+\s+(\d+)%", line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


def main() -> int:
    tie, t = col_tie1_and_t()
    crown = col_crown()
    flips, rev = col_split()
    clus = col_cluster()
    items = col_items()
    missing = []
    L = []
    p = L.append
    p("BENCHMARK HEALTH: TEN BOARDS, SEVEN QUESTIONS")
    p("=" * 96)
    p(f"  {'leaderboard':<22} {'tie@1':>6} {'top t':>7} {'crown':>7} {'top flips':>10} "
      f"{'reversed':>9} {'cluster x':>10} {'items for 90 %':>15}")
    for b in BOARDS:
        row = [tie.get(b), t.get(b), crown.get(b), flips.get(b), rev.get(b), clus.get(b), items.get(b)]
        if row[0] is None or row[1] is None:
            missing.append(b)
        def f(v, fmt, suffix=""):
            return "-" if v is None else format(v, fmt) + suffix
        p(f"  {b:<22} {f(row[0], 'd'):>6} {f(row[1], '.2f'):>7} {f(row[2], '.0f', ' %'):>7} "
          f"{f(row[3], '.0f', ' %'):>10} {f(row[4], '.2f', ' %'):>9} {f(row[5], '.2f'):>10} "
          f"{f(row[6], '.0f', ' %'):>15}")
    p("")
    p("  tie@1      systems whose simultaneous rank set contains rank 1")
    p("  top t      paired statistic of the printed #1 against the printed #2")
    p("  crown      item bootstraps in which the printed leader stays first")
    p("  top flips  random item halves whose halves name different leaders")
    p("  reversed   established pairs that one item half orders the other way")
    p("  cluster x  rank-set widening when items are resampled in their groups")
    p("  items      share of items that recovers 90 % of the established pairs")
    p("")
    p("  Read the columns against each other. CASP14 has one system that could be")
    p("  first, a top t of 9.89, a crown that never moves and halves that never")
    p("  disagree - a board whose headline is evidence. SWE-bench Verified has")
    p("  nineteen possible firsts, a top t of 0.00, a crown that survives a third")
    p("  of resamples and halves that disagree every time - a board whose headline")
    p("  is a coin toss and whose middle is nevertheless ordered reliably")
    p("  (0.00 % reversed).")
    p("")
    p("  LMArena has only the two columns the standard's own run produces: it was")
    p("  held out while the laws were fixed and was added to the reference")
    p("  implementation afterwards, so the nine-board tools have not been rerun")
    p("  with it. Adding it to them would change the pre-registered nine-board")
    p("  tables, which is not worth one extra row.")
    if missing:
        p("")
        p(f"  MISSING from the source files: {', '.join(missing)} - rerun the tool that owns the column")
    text = "\n".join(L)
    print(text)
    Path("benchmark_health_results.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    print("\nwrote benchmark_health_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
