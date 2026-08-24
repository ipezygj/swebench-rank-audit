"""Run every tool and record whether it still works.

Eighty-two iterations of editing in one evening, with several tools patched
after other tools were built on top of them. Before the repo is left for the
night, everything runs once more from a clean interpreter and the outcome is
recorded: exit code, wall time, and whether the self-checks passed.

A tool counts as PASS only if it exits 0 and its output does not contain the
string that every tool here prints when a check fails.

    python run_all.py [--quick]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

KNOWN = {
    "selection_sbi.py": "its own SBC check fails (KS p 0.0003) and it refuses to print - "
                        "a pre-existing calibration problem in the neural posterior, not a "
                        "regression from this evening; recorded rather than silenced",
}
SKIP = {
    "run_all.py",              # this file
    "build_readme.py",         # regenerates documents, run separately
    "build_laws_md.py",
    "mteb_dates.py",           # network
    "swebench_matrix.py",      # needs the raw submission tree
    "all_splits.py",
    "helm_matrix.py",
    "lmarena_resolution.py",
    "_aa_scrape.py", "_add_cites.py",
}
ARGS = {"leaderboard_standard.py": ["--all"]}   # tools that need an argument to do anything
SLOW = {"top_compression.py", "effective_items.py", "family_clustering.py", "family_generalises.py", "top_redundancy.py", "redundancy_power.py", "detectable_difference.py", "tie_coverage.py", "tie_coverage_boards.py", "holm_recompute.py", "leaderboard_standard.py", "chase_refit.py", "chase_correlated.py", "lineage_tree.py", "target_board.py",
        "method_independence.py", "sibling_chase.py", "chase_model.py", "cluster_bootstrap.py"}
FAIL_MARK = "A CHECK FAILED"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    quick = "--quick" in sys.argv
    tools = sorted(p.name for p in Path(".").glob("*.py")
                   if p.name not in SKIP and not p.name.startswith("_"))
    if quick:
        tools = [t for t in tools if t not in SLOW]
    rows, t0 = [], time.perf_counter()
    for t in tools:
        start = time.perf_counter()
        try:
            r = subprocess.run([sys.executable, t, *ARGS.get(t, [])], capture_output=True, text=True,
                               timeout=900, encoding="utf-8", errors="replace")
            out = (r.stdout or "") + (r.stderr or "")
            ok = r.returncode == 0 and FAIL_MARK not in out
            note = "" if ok else (out.strip().splitlines()[-1][:70] if out.strip() else f"exit {r.returncode}")
        except subprocess.TimeoutExpired:
            ok, note = False, "TIMEOUT after 900 s"
        el = time.perf_counter() - start
        if not ok and t in KNOWN:
            note = "KNOWN: " + KNOWN[t]
        rows.append((t, ok, el, note))
        print(f"  [{'ok  ' if ok else 'FAIL'}] {t:<32} {el:>7.1f}s  {note}")
    total = time.perf_counter() - t0
    bad = [r for r in rows if not r[1]]
    L = [f"FULL RUN {time.strftime('%Y-%m-%d %H:%M')}" + (" (quick)" if quick else ""),
         "=" * 72,
         f"  {len(rows)} tools run, {len(rows) - len(bad)} passed, {len(bad)} failed, "
         f"{total:.0f} s total"]
    for t, ok, el, note in rows:
        L.append(f"  {'ok  ' if ok else 'FAIL'}  {t:<34} {el:>7.1f}s  {note}")
    if bad:
        L.append("")
        L.append("  FAILURES:")
        for t, _, _, note in bad:
            L.append(f"    {t}: {note}")
    Path("run_all_results.txt").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"\n{len(rows) - len(bad)}/{len(rows)} passed in {total:.0f} s; wrote run_all_results.txt")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
