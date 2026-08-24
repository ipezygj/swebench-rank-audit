"""Which of this repo's ten boards sit in the regime where the bootstrap fails?

tie_coverage.py found the multiplier bootstrap covering 0.515 at J = 181,
n = 41 - MTEB English v2's exact shape - where Holm covers 0.935. That is one
shape. This runs every board's own (J, n) under exact ties, so the blast radius
is measured rather than guessed.

    python tie_coverage_boards.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tie_coverage import boot, coverage, holm_rank_sets

REPS = 150
SEED = 20260824

MATRICES = {
    "SWE-bench Verified": "swebench_verified_matrix.csv",
    "SWE-bench Lite": "swebench_lite_matrix.csv",
    "SWE-bench Test": "swebench_test_matrix.csv",
    "MTEB English v2": "mteb_eng_v2_wide.csv",
    "HELM classic": "helm_winrate_matrix.csv",
    "ProteinGym DMS": "proteingym/matrix.csv",
    "TabArena 16 models": "tabarena/matrix_one_per_model.csv",
    "TabArena 45 variants": "tabarena/matrix_all45.csv",
    "CASP14": "casp/matrix.csv",
    "LiveBench": "livebench/matrix.csv",
    "MathArena 2025": "matharena/matrix.csv",
    "LMArena categories": "lmarena_matrix.csv",
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)
        J, n = x.shape
        print(f"  {name} (J={J}, n={n}) ...")
        ab = np.full(J, 0.5)
        cb, _ = coverage(boot, ab, n, REPS, SEED + J + n)
        ch, _ = coverage(holm_rank_sets, ab, n, REPS, SEED + J + n)
        rows.append((name, J, n, J / n, cb, ch))

    rows.sort(key=lambda r: -r[3])
    L = []
    p = L.append
    p("WHICH BOARDS SIT WHERE THE BOOTSTRAP FAILS")
    p("=" * 92)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>6} {'J/n':>6} {'bootstrap':>10} {'Holm':>7}  verdict")
    for name, J, n, r, cb, ch in rows:
        v = "BOOTSTRAP UNDERCOVERS" if cb < 0.90 else ("marginal" if cb < 0.93 else "ok")
        p(f"  {name:<22} {J:>4} {n:>6} {r:>6.2f} {cb:>10.3f} {ch:>7.3f}  {v}")
    p("")
    bad = [r for r in rows if r[4] < 0.90]
    p(f"  {len(bad)} of {len(rows)} boards undercover under exact ties at nominal 0.95.")
    if bad:
        p("  Affected: " + ", ".join(f"{r[0]} (J/n = {r[3]:.1f}, coverage {r[4]:.3f})"
                                     for r in bad))
    p("")
    p("  Simultaneous coverage is a statement about all J(J-1)/2 pairs at once,")
    p("  and the multiplier bootstrap estimates the critical value for all of")
    p("  them from n items. When J/n is large there are more statistics than")
    p("  observations to estimate their joint distribution from, and the")
    p("  critical value comes out too small: the rank sets are too NARROW, so")
    p("  every count of systems that could be first on those boards is too LOW.")
    p("")
    p("  Direction matters for reading the repo. Undercoverage does not")
    p("  manufacture ties, it hides them - so the finding that these boards")
    p("  cannot resolve their tops is, if anything, understated on exactly the")
    p("  boards affected.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("tie_coverage_boards_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                       newline=chr(10))
    print(chr(10) + "wrote tie_coverage_boards_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
