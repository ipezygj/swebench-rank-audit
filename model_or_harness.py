"""On SWE-bench, how much of a submission's score is the model and how much the harness?

The base-model vocabulary (swebench_base_models.py) labels 62 of the 134
Verified submissions with the LLM behind them. Everything else about a
submission - the scaffold, the retrieval, the retry policy, the test-time
budget - is the harness. That makes a decomposition possible that the
leaderboard itself never prints: of the spread in resolve rates, how much
sits BETWEEN base models and how much WITHIN one base model, across the
harnesses built on it.

Two quantities, both computed only on the labelled subset and only for
families with at least three submissions:

    between  the SD of family means
    within   the pooled SD of submissions around their family mean

and one concrete question: does the best harness on an older model beat the
median harness on a newer one?

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * within is at least half of between - the harness matters on the same
    order as the model;
  * for at least one older/newer model pair, the best harness on the older
    model beats the median harness on the newer one;
  * the frontier is not a base-model ladder: at least 3 of the frontier
    advances keep the same base model as the previous leader (harness-only
    steps), among those advances where both are labelled.

CONFOUND, stated before running: submissions using the same base model
arrive at different times, and later harnesses are better. Between-family
differences are therefore inflated by whichever families arrived late, and
the decomposition below is descriptive, not causal. A date-matched version
is reported alongside: only submissions from the same calendar quarter.

SELF-CHECKS
  * the decomposition must reproduce the total variance of the labelled
    subset (between^2 + within^2 = total, within rounding);
  * on shuffled family labels, between must collapse toward zero.

    python model_or_harness.py
"""
from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sota_audit import advances, parse_dates
from swebench_base_models import base_model

MIN_FAMILY = 3
SEED = 20260823


def load():
    df = pd.read_csv("swebench_verified_matrix.csv", index_col=0)
    names = list(df.index)
    x = df.to_numpy(dtype=float)
    dates = parse_dates(names)
    fams = [base_model(n) for n in names]
    counts = Counter(f for f in fams if f)
    keep = {f for f, c in counts.items() if c >= MIN_FAMILY}
    fams = [f if f in keep else None for f in fams]
    return x, dates, names, fams


def decompose(scores, labels):
    groups = defaultdict(list)
    for s, l in zip(scores, labels):
        groups[l].append(s)
    means = {g: float(np.mean(v)) for g, v in groups.items()}
    grand = float(np.mean(scores))
    between = math.sqrt(sum(len(v) * (means[g] - grand) ** 2 for g, v in groups.items()) / len(scores))
    within = math.sqrt(sum(sum((np.array(v) - means[g]) ** 2) for g, v in groups.items()) / len(scores))
    return between, within, means, groups


def _check_variance():
    rng = np.random.default_rng(SEED)
    s = rng.normal(0, 1, 200)
    l = rng.integers(0, 5, 200)
    b, w, _, _ = decompose(s, l)
    total = float(np.var(s))
    return abs(b ** 2 + w ** 2 - total) < 1e-9, f"between^2 + within^2 = {b ** 2 + w ** 2:.6f} vs total {total:.6f}"


def _check_shuffled():
    x, dates, names, fams = load()
    sc = x.mean(axis=1)
    idx = [i for i, f in enumerate(fams) if f]
    real_b, _, _, _ = decompose(sc[idx], [fams[i] for i in idx])
    bs = []
    for s in range(30):
        sh = list(np.random.default_rng(SEED + s).permutation([fams[i] for i in idx]))
        b, _, _, _ = decompose(sc[idx], sh)
        bs.append(b)
    return float(np.mean(bs)) < 0.6 * real_b, \
        f"shuffled labels: between {np.mean(bs):.4f} vs real {real_b:.4f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_variance(), _check_shuffled()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    x, dates, names, fams = load()
    sc = x.mean(axis=1)
    idx = [i for i, f in enumerate(fams) if f]
    L = []
    p = L.append
    p("SWE-BENCH VERIFIED: THE MODEL OR THE HARNESS?")
    p("=" * 78)
    p(f"  {len(idx)} of {len(names)} submissions name their base model; "
      f"{len(set(fams[i] for i in idx))} families with >= {MIN_FAMILY} submissions")
    b, w, means, groups = decompose(sc[idx], [fams[i] for i in idx])
    p(f"  between base models: SD {100 * b:.2f} points")
    p(f"  within a base model, across harnesses: SD {100 * w:.2f} points   "
      f"(within / between = {w / b:.2f})")
    p("")
    p(f"  {'base model':<12} {'subs':>5} {'best':>7} {'median':>8} {'worst':>7} {'spread':>8} {'first seen':>11}")
    rows = []
    for g, v in sorted(groups.items(), key=lambda kv: -np.mean(kv[1])):
        gi = [i for i in idx if fams[i] == g]
        first = min(dates[i] for i in gi)
        rows.append((g, np.max(v), np.median(v), np.min(v), first))
        p(f"  {g:<12} {len(v):>5} {100 * np.max(v):>6.1f}% {100 * np.median(v):>7.1f}% {100 * np.min(v):>6.1f}% "
          f"{100 * (np.max(v) - np.min(v)):>7.1f} {str(first):>11}")
    p("")
    beats = []
    for go, bo, mo, wo, fo in rows:
        for gn, bn, mn, wn, fn_ in rows:
            if fo < fn_ and bo > mn:
                beats.append((go, gn, 100 * bo, 100 * mn))
    p(f"  best harness on an older model beats the median harness on a newer one: "
      f"{len(beats)} pairs" + (f", e.g. {beats[0][0]} best {beats[0][2]:.1f} % > {beats[0][1]} median {beats[0][3]:.1f} %" if beats else ""))
    same, both = 0, 0
    for a in advances(x, dates):
        fn_, fo = fams[int(a["new"])], fams[int(a["old"])]
        if fn_ and fo:
            both += 1
            same += fn_ == fo
    p(f"  frontier advances where both ends are labelled: {both}; same base model: {same}")
    p("")
    q = (dates // 100) % 100
    quarter = (dates // 10000) * 4 + (q - 1) // 3
    dm_b, dm_w = [], []
    for qq in sorted(set(quarter[i] for i in idx)):
        gi = [i for i in idx if quarter[i] == qq]
        if len(gi) >= 6 and len(set(fams[i] for i in gi)) >= 2:
            bb, ww, _, _ = decompose(sc[gi], [fams[i] for i in gi])
            dm_b.append(bb); dm_w.append(ww)
    if dm_b:
        p(f"  date-matched (within calendar quarter, {len(dm_b)} quarters): "
          f"between {100 * np.mean(dm_b):.2f}, within {100 * np.mean(dm_w):.2f} points, "
          f"ratio {np.mean(dm_w) / np.mean(dm_b):.2f}")
    p("")
    p("  'within' is the spread of submissions that use the SAME base model and")
    p("  differ only in harness. Where it approaches 'between', the leaderboard is")
    p("  ranking engineering as much as it is ranking models.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("model_or_harness_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote model_or_harness_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
