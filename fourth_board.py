"""The pre-registered test of the step-size reading on a FOURTH dated board.

Commit c9ff2df (2026-08-23) recorded, before any fourth board was sought:

    median u in [0.2, 0.6]; share u < 1 >= 60 %; A_real > A_twin; P_real < P_twin

where u = SOTA step / (1.96 sigma_p / sqrt n) and the twin is the
linear-drift Gaussian field of sota_twin.py. This runs exactly those four
checks on SWE-bench Lite (84 dated submissions, 300 instances) and, as a
secondary under-powered board, SWE-bench full test (24 submissions, 2294
instances).

CAVEAT stated before running: Lite shares 25 submission names and 93 of 300
instances with Verified. It is a different instance set and a different
frontier history, but not a different field - a pass here is weaker
evidence than a pass on an unrelated benchmark would be, and is recorded as
such. The test split shares the scaffolds too and has only 24 systems; its
counts are reported but A/P comparisons at that size are noise-dominated.

    python fourth_board.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sota_audit import parse_dates
from sota_twin import audit, fit_drift, dated_twin, sigma_p_of, SEED, TWINS
from step_sizes import steps_u
from evidence_trajectory import load

BOARDS = {
    "SWE-bench Lite (primary)": "swebench_lite_matrix.csv",
    "SWE-bench test (secondary, J=24)": "swebench_test_matrix.csv",
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    L = []
    p = L.append
    p("FOURTH DATED BOARD: THE PRE-REGISTERED STEP-SIZE CHECKS (commit c9ff2df)")
    p("=" * 80)
    for name, path in BOARDS.items():
        x, dates = load(path, None)
        J, n = x.shape
        sp = sigma_p_of(x)
        u = steps_u(x, dates, sp)
        real = audit(x, dates, SEED)
        a, beta, tau_res, si = fit_drift(x, dates, sp)
        tw = [audit(dated_twin(J, n, dates, a, beta, tau_res, si, np.random.default_rng(SEED + 50 + s)), dates, SEED + 500 + 100 * s)
              for s in range(TWINS)]
        A_t = float(np.mean([t["A"] for t in tw]))
        P_t = float(np.nanmean([t["P"] for t in tw]))
        med, share = float(np.median(u)), float(np.mean(u < 1))
        p("")
        p(f"  {name}: {J} systems x {n} items, {len(u)} frontier advances, beta {beta:+.3f}/yr")
        p(f"    median u        {med:.2f}    pre-registered [0.20, 0.60]   {'yes' if 0.2 <= med <= 0.6 else 'NO'}")
        p(f"    share u < 1     {100 * share:.0f} %   pre-registered >= 60 %       {'yes' if share >= 0.6 else 'NO'}")
        p(f"    A real vs twin  {real['A']} vs {A_t:.1f}   pre-registered A_real > A_twin   {'yes' if real['A'] > A_t else 'NO'}")
        p(f"    P real vs twin  {100 * real['P']:.0f} % vs {100 * P_t:.0f} %   pre-registered P_real < P_twin   {'yes' if real['P'] < P_t else 'NO'}")
        p(f"    (S real {100 * real['S']:.0f} %; q25 {np.percentile(u, 25):.2f} q75 {np.percentile(u, 75):.2f} max {u.max():.2f})")
    p("")
    p("  Lite shares scaffolds and 93 instances with Verified: a pass is evidence")
    p("  about the dynamics of a second board, not about a second field.")
    text = chr(10).join(L)
    print(text)
    Path("fourth_board_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote fourth_board_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
