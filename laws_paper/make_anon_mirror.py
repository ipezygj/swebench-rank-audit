"""Build the anonymised code artefact for double-blind review.

TMLR reviews double blind and a submission must not link to a version carrying
the authors' names, so the public repository cannot be cited. This builds a
minimal, purpose-made artefact instead of scrubbing the whole repository -
which is both safer and better practice, because the repository also holds
correspondence drafts and other material that has nothing to do with the paper
and would only have to be redacted.

What goes in: the modules the paper's results actually depend on, closed under
import; the outcome matrices those modules read; the results files the paper
parses its figures from; and a README that says how to reproduce them.

What is checked before the archive is written: that no file contains any
identifying string, that the dependency set is closed, and that every results
file the paper's builder parses is present. The script refuses to write the
archive if any check fails.

    python make_anon_mirror.py
"""
from __future__ import annotations

import ast
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path("..")
OUT = Path("anon_mirror")
ARCHIVE = Path("anon_mirror.zip")

# Entry points: everything the paper's tables and claims are computed by.
ENTRY = [
    "resolution_law_test.py",     # law 1
    "entropy_law_test.py",        # law 2
    "entropy_law_twin2.py",       # law 2, per-system noise variant
    "entropy_decomposition.py",   # the shape/residual split
    "law1_pairwise.py",           # law 1 integrated over the pair distribution
    "tenth_board.py",             # the held-out board
    "target_board.py",            # the top-of-board miss
    "tie_coverage.py",            # coverage of the two constructions
    "tie_coverage_boards.py",     # coverage at each board's shape
    "holm_recompute.py",          # the recomputation under Holm
    "pair_sharpness.py",          # kappa
    "prescription_pairwise.py",   # the cost of the board-wide sigma
]

# A language is an identifier too: a reviewer who finds Finnish comments has
# narrowed the authorship to a country. Common short words, matched whole and
# case-insensitively, catch that without flagging English text.
FINNISH = [r"\beika\b", r"\bjotta\b", r"\bsiksi\b", r"\btama\b", r"\bmutta\b",
           r"\bkaikki\b", r"\bvaan\b", r"\bkun\b", r"\bniin\b", r"\bjoka\b",
           r"\btalletetaan\b", r"\bkaytetaan\b", r"\bmenetelma\b", r"\bvaara\b",
           r"\btarkistin\b", r"\bsijaluvut\b", r"\bjarjestys\b", r"\bpeitto\b"]

# Anything that would identify the authors, checked against every shipped file.
IDENTIFIERS = [
    r"ipezy", r"Väätäinen", r"Vaatainen", r"vaatainen",
    r"Helsinki", r"helsinki", r"github\.com/", r"gmail\.com",
    r"hotmail\.com", r"zenodo\.\d+", r"@\w+\.(com|fi|org)\b",
]


def local_imports(path: Path) -> set[str]:
    """Module names imported from this repository by one file."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return {m for m in out if (ROOT / f"{m}.py").exists()}


def closure(entries):
    """Transitive closure of local imports over the entry points."""
    seen, stack = set(), list(entries)
    while stack:
        f = stack.pop()
        if f in seen or not (ROOT / f).exists():
            continue
        seen.add(f)
        stack.extend(f"{m}.py" for m in local_imports(ROOT / f))
    return sorted(seen)


def matrices_used(files):
    """Every data path a shipped module names as a string literal."""
    pat = re.compile(r'["\']([\w./-]+\.csv)["\']')
    out = set()
    for f in files:
        out.update(pat.findall((ROOT / f).read_text(encoding="utf-8", errors="replace")))
    return sorted(p for p in out if (ROOT / p).exists())


def results_needed():
    """The results files the paper's own builder parses."""
    src = Path("build_paper.py").read_text(encoding="utf-8", errors="replace")
    return sorted(set(re.findall(r'read\("([\w./-]+_results\.txt)"\)', src)))


def scan_identifiers(paths):
    bad = []
    for p in paths:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pat in IDENTIFIERS + FINNISH:
            for m in re.finditer(pat, txt):
                line = txt[:m.start()].count("\n") + 1
                bad.append((p.name, line, m.group(0)[:40]))
    return bad


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    code = closure(ENTRY)
    data = matrices_used(code)
    results = results_needed()

    print(f"  modules (import-closed): {len(code)}")
    print(f"  matrices they read:      {len(data)}")
    print(f"  results files the paper parses: {len(results)}")

    missing_res = [r for r in results if not (ROOT / r).exists()]
    if missing_res:
        print(f"  [FAIL] results files absent: {', '.join(missing_res)}")
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "results").mkdir(parents=True)
    for f in code:
        shutil.copy2(ROOT / f, OUT / f)
    for d in data:
        dst = OUT / d
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / d, dst)
    for r in results:
        shutil.copy2(ROOT / r, OUT / "results" / Path(r).name)

    (OUT / "README.md").write_text(README, encoding="utf-8", newline="\n")

    shipped = [p for p in OUT.rglob("*") if p.is_file()]
    bad = scan_identifiers(shipped)
    ok_anon = not bad
    print(f"  [{'ok  ' if ok_anon else 'FAIL'}] {len(shipped)} files carry no identifying "
          f"string and no Finnish"
          + ("" if ok_anon else ""))
    for name, line, hit in bad[:12]:
        print(f"        {name}:{line}  {hit}")

    unresolved = set()
    for f in code:
        unresolved |= {m for m in local_imports(OUT / f) if f"{m}.py" not in code}
    ok_closed = not unresolved
    print(f"  [{'ok  ' if ok_closed else 'FAIL'}] dependency set is closed"
          + ("" if ok_closed else f"  missing: {', '.join(sorted(unresolved))}"))

    if not (ok_anon and ok_closed):
        shutil.rmtree(OUT)
        print("\nA CHECK FAILED - no archive written, working copy removed.")
        return 1

    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(shipped):
            z.write(p, p.relative_to(OUT))
    size = ARCHIVE.stat().st_size / 1e6
    print(f"\nwrote {ARCHIVE} ({size:.1f} MB, {len(shipped)} files)")
    return 0


README = """# Code and data for "How Much Ranking a Benchmark Can Support Is
# Predictable Before It Is Run"

Anonymised artefact for double-blind review. It contains the modules the
paper's results depend on, closed under import; the per-item outcome matrices
those modules read; and the results files the paper's figures are parsed from.

## Reproducing the paper's tables

    python resolution_law_test.py     # Table 1, law 1
    python entropy_law_test.py        # Table 2, law 2
    python entropy_decomposition.py   # Table 3, the shape/residual split
    python tenth_board.py             # Table 4, the held-out board

Each writes a `*_results.txt` beside itself. Copies of those files as they stood
when the paper was built are in `results/`, so a reader can compare rather than
take the run on trust.

## The two constructions

Every rank set comes from `rank_sets.py`, which takes `method=` or the
environment variable `RANK_SETS_METHOD`:

    RANK_SETS_METHOD=holm      directional paired tests with Holm's FWER
                               control (the default, and what the paper uses)
    RANK_SETS_METHOD=bootstrap the multiplier bootstrap with Romano-Wolf
                               step-down, which the paper reports as
                               undercovering when systems outnumber items
    RANK_SETS_METHOD=union     the wider of the two per system

    python tie_coverage_boards.py     # the coverage measurement itself
    python holm_recompute.py          # both constructions side by side

## Self-checks

Every script runs its checks before printing anything and exits without a table
if one fails; the string `A CHECK FAILED` marks that case. `selection_sbi.py` is
not included here, but is known to fail its own calibration check and is
reported as such rather than silenced.

## Data

The matrices are systems x items, one row per submission, one column per
benchmark item, and are taken from each leaderboard's public per-item outcomes.
Sources are cited in the paper.

## Requirements

Python 3.11+, numpy, pandas, scipy.
"""

if __name__ == "__main__":
    sys.exit(main())
