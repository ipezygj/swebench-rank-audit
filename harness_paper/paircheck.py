"""paircheck.py - is a claimed SWE-bench improvement distinguishable from noise?

Given two per-instance result files on the same instance set, prints the discordant counts, the exact
McNemar p-value, and the smallest net gap (in points) that this pair's discordance would have let a
5 % two-sided exact test detect. The detectable-gap idea follows Card et al. (EMNLP 2020), applied here
to the pair's own discordant count rather than a benchmark-wide figure.

Accepted inputs (auto-detected):
  * a leaderboard results.json            {"resolved": [ids], ...}
  * a folder of per-instance report.json  <dir>/<instance_id>/report.json   (either SWE-bench format)
  * a plain text file of resolved ids, one per line

    python paircheck.py A B [--n 500] [--alpha 0.05]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import binomtest

from common import bashonly_rows, mcnemar


def load(path: str) -> set[str]:
    p = Path(path)
    if p.is_dir():
        return bashonly_rows(p.parent)[p.name]
    text = p.read_text(encoding="utf-8")
    try:
        d = json.loads(text)
        return set(d["resolved"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return {line.strip() for line in text.splitlines() if line.strip()}


def detectable_gap(n_disc: int, n: int, alpha: float) -> float:
    """Smallest net difference (points) whose split of n_disc discordant instances reaches p < alpha."""
    for k in range((n_disc + 1) // 2, n_disc + 1):
        if binomtest(k, n_disc, 0.5).pvalue < alpha:
            return 100 * (2 * k - n_disc) / n
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--n", type=int, default=500, help="instances in the benchmark (default 500, Verified)")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    x, y, p = mcnemar(A, B)
    gap = 100 * (len(A) - len(B)) / a.n
    print(f"A resolved {len(A)} ({100 * len(A) / a.n:.1f} %), B resolved {len(B)} ({100 * len(B) / a.n:.1f} %)")
    print(f"gap A - B: {gap:+.1f} points; discordant: A only {x}, B only {y}")
    print(f"exact McNemar p = {p:.3g} -> {'separable' if p < a.alpha else 'NOT separable'} at alpha {a.alpha} "
          f"(uncorrected; apply Holm if you test several pairs)")
    print(f"smallest net gap this pair's discordance ({x + y}) could detect: {detectable_gap(x + y, a.n, a.alpha):.1f} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
