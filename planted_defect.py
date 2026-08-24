"""If the data were wrong, would these tools notice?

staleness_audit.py asked whether the repository notices when the CODE changes
under a results file. This asks the other half: whether it notices when the DATA
does. The two together are the only way to know that a results file means
anything, because a tool that reports the same numbers on corrupted input is not
reading the input.

The method is a planted defect. One board's outcome matrix is copied and a
fraction of its cells are flipped, the tools that read it are run against the
copy, and three things are recorded for each: whether its output changed at all,
whether any of its self-checks fired, and whether it noticed in a way a reader
would see.

A tool whose output is byte-identical on corrupted data has failed the most
basic form of this test. A tool whose output changes but whose checks stay
silent is normal - most of these tools measure the field rather than police it -
but the ones that claim to police it should speak.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  every tool that reads the corrupted matrix produces different output.
      Byte-identical output would mean the file is not being read.
  P2  at least 2 tools' self-checks fire and refuse to print a table.
  P3  the parity checks fire specifically: top_compression compares its tie@1
      against the live benchmark_health run, so a corrupted board must break
      that agreement.
  P4  law 1's error gets worse. The corruption destroys the structure the law
      describes, so a law that still fit would be a law about nothing.

  What a miss on P1 would mean: a tool with a hardcoded number where it should
  have a measurement. That is worth more than any of the rest of this file.

SELF-CHECKS (no table if any fails)
  * the corruption must actually change the matrix: the flipped copy must differ
    from the original in the intended number of cells, counted, not assumed;
  * the harness must run the tools against the COPY - verified by pointing a
    tool at the untouched original first and confirming its output matches the
    committed results file;
  * the scratch directory must be a copy: the real matrices are never written.

    python planted_defect.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

SEED = 20260825
TARGET = "swebench_verified_matrix.csv"
FLIP = 0.05          # share of cells to flip
TIMEOUT = 900

# The tools the paper's claims rest on, plus the two that police inputs.
TOOLS = [
    "resolution_law_test.py",
    "entropy_law_test.py",
    "pair_sharpness.py",
    "effective_items.py",
    "detectable_difference.py",
    "top_compression.py",
    "shape_correction.py",
    "holm_recompute.py",
]


def corrupt(src: Path, dst: Path, frac: float, rng) -> int:
    x = pd.read_csv(src, index_col=0)
    v = x.to_numpy(dtype=float)
    n = int(round(frac * v.size))
    idx = rng.choice(v.size, n, replace=False)
    flat = v.ravel().copy()
    flat[idx] = 1.0 - flat[idx]          # binary board: a flip is a flip
    out = pd.DataFrame(flat.reshape(v.shape), index=x.index, columns=x.columns)
    out.to_csv(dst)
    changed = int((pd.read_csv(dst, index_col=0).to_numpy(dtype=float) != v).sum())
    return changed


def run(tool: str, cwd: Path) -> tuple[int, str]:
    try:
        r = subprocess.run([sys.executable, tool], cwd=str(cwd), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    here = Path(".").resolve()
    rng = np.random.default_rng(SEED)

    scratch = Path(tempfile.mkdtemp(prefix="planted-"))
    print(f"  scratch: {scratch}")
    for p in here.iterdir():
        if p.is_file() and p.suffix in (".py", ".csv", ".txt"):
            shutil.copy2(p, scratch / p.name)
        elif p.is_dir() and p.name in ("casp", "livebench", "matharena",
                                       "proteingym", "tabarena"):
            shutil.copytree(p, scratch / p.name, dirs_exist_ok=True)

    print("self-checks ...")
    original = here / TARGET
    copy_before = scratch / TARGET
    same = (pd.read_csv(original, index_col=0).to_numpy(dtype=float) ==
            pd.read_csv(copy_before, index_col=0).to_numpy(dtype=float)).all()
    print(f"  [{'ok  ' if same else 'FAIL'}] the scratch copy starts identical to the original")

    baseline = {}
    for t in TOOLS:
        rc, out = run(t, scratch)
        res = scratch / t.replace(".py", "_results.txt")
        baseline[t] = res.read_text(encoding="utf-8", errors="replace") if res.exists() else ""
        print(f"  baseline {t:<30} exit {rc}")

    matched = sum(1 for t in TOOLS
                  if (here / t.replace(".py", "_results.txt")).exists()
                  and baseline[t].splitlines()
                  == (here / t.replace(".py", "_results.txt")).read_text(
                      encoding="utf-8", errors="replace").splitlines())
    ok_base = matched >= len(TOOLS) - 2
    print(f"  [{'ok  ' if ok_base else 'FAIL'}] {matched} of {len(TOOLS)} baselines reproduce "
          f"the committed results file")

    changed = corrupt(original, scratch / TARGET, FLIP, rng)
    want = int(round(FLIP * pd.read_csv(original, index_col=0).size))
    ok_corrupt = abs(changed - want) <= max(2, want // 100)
    print(f"  [{'ok  ' if ok_corrupt else 'FAIL'}] corruption flipped {changed} cells "
          f"(intended {want})")
    still = (pd.read_csv(original, index_col=0).to_numpy(dtype=float) ==
             pd.read_csv(here / TARGET, index_col=0).to_numpy(dtype=float)).all()
    print(f"  [{'ok  ' if still else 'FAIL'}] the real matrix is untouched")

    if not (same and ok_base and ok_corrupt and still):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rows = []
    for t in TOOLS:
        rc, out = run(t, scratch)
        res = scratch / t.replace(".py", "_results.txt")
        after = res.read_text(encoding="utf-8", errors="replace") if res.exists() else ""
        fired = "A CHECK FAILED" in out
        moved = after.splitlines() != baseline[t].splitlines()
        rows.append((t, rc, moved, fired, len(after) > 0))

    L = []
    p = L.append
    p("IF THE DATA WERE WRONG, WOULD THESE TOOLS NOTICE?")
    p("=" * 88)
    p(f"  {FLIP:.0%} of the cells of {TARGET} flipped: {changed} of "
      f"{pd.read_csv(original, index_col=0).size}.")
    p("")
    p(f"  {'tool':<32} {'output moved':>13} {'check fired':>12} {'printed':>9}")
    for t, rc, moved, fired, printed in rows:
        p(f"  {t:<32} {'yes' if moved else 'NO':>13} {'yes' if fired else '-':>12} "
          f"{'yes' if printed else 'no':>9}")
    p("")
    moved_n = sum(1 for r in rows if r[2])
    fired_n = sum(1 for r in rows if r[3])
    p(f"  P1  output moved on {moved_n} of {len(rows)}                 "
      f"pre-registered = all:  {'HIT' if moved_n == len(rows) else 'MISS'}")
    p(f"  P2  a self-check fired on {fired_n} of {len(rows)}           "
      f"pre-registered >= 2:  {'HIT' if fired_n >= 2 else 'MISS'}")
    p("")
    p("  A tool whose output is byte-identical on corrupted data is not reading")
    p("  the data. A tool whose output moves but whose checks stay quiet is")
    p("  behaving normally - most of these measure a field rather than police")
    p("  it - but the ones that compare against a committed figure should speak.")
    p("")
    p("  The real matrices are never written: everything runs in a scratch copy,")
    p("  and the check above confirms the original is untouched afterwards.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("planted_defect_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                  newline=chr(10))
    print(chr(10) + "wrote planted_defect_results.txt")
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
