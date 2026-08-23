"""Is 'the harness is two thirds of the model' peculiar to agentic coding?

model_or_harness.py found that on SWE-bench Verified the spread of scores
WITHIN one base model, across the harnesses built on it, is 0.67 of the
spread BETWEEN base models (0.54 date-matched). The same decomposition
applies wherever entrants come in named families:

    MTEB        family = the HuggingFace organisation; within-family
                variation is model size, training recipe, version
    ProteinGym  family = the method family; within-family variation is
                model size and the retrieval/MSA variant

The question is whether "what you build around the model matters as much as
the model" is a property of agentic benchmarks or of leaderboards.

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * within / between is at least 0.5 on >= 2 of the 3 boards;
  * SWE-bench has the highest ratio - a scaffold is a bigger intervention
    than a size change;
  * on every board the largest within-family spread exceeds 15 points.

SELF-CHECKS
  * the decomposition reproduces the total variance (already checked in
    model_or_harness.py, repeated here on each board's own subset);
  * shuffled family labels collapse the between term.

    python within_family_spread.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from model_or_harness import decompose, MIN_FAMILY
from lineage_detection import FAMILY as NAME_FAMILY
from swebench_base_models import base_model

SEED = 20260823
BOARDS = {
    "SWE-bench Verified": ("swebench_verified_matrix.csv", base_model, "base model", "harness"),
    "MTEB English v2": ("mteb_eng_v2_wide.csv", NAME_FAMILY["MTEB English v2"], "organisation", "model variant"),
    "ProteinGym DMS": ("proteingym/matrix.csv", NAME_FAMILY["ProteinGym DMS"], "method family", "size / variant"),
}


def subset(path, fn):
    df = pd.read_csv(path, index_col=0).dropna(axis=0)
    sc = df.to_numpy(dtype=float).mean(axis=1)
    fams = [fn(n) for n in df.index]
    counts = Counter(f for f in fams if f)
    keep = {f for f, c in counts.items() if c >= MIN_FAMILY}
    idx = [i for i, f in enumerate(fams) if f in keep]
    return sc[idx], [fams[i] for i in idx], len(df), list(df.index)


def _check_variance():
    msgs, ok = [], True
    for name, (path, fn, _, _) in BOARDS.items():
        sc, fams, J, _ = subset(path, fn)
        if len(sc) < 6:
            continue
        b, w, _, _ = decompose(sc, fams)
        tot = float(np.var(sc))
        ok = ok and abs(b ** 2 + w ** 2 - tot) < 1e-9
        msgs.append(f"{name.split()[0]} ok")
    return ok, "variance identity holds: " + ", ".join(msgs)


def _check_shuffled():
    msgs, ok = [], True
    for name, (path, fn, _, _) in BOARDS.items():
        sc, fams, _, _ = subset(path, fn)
        if len(sc) < 6:
            continue
        b, _, _, _ = decompose(sc, fams)
        bs = [decompose(sc, list(np.random.default_rng(SEED + s).permutation(fams)))[0] for s in range(20)]
        ok = ok and float(np.mean(bs)) < 0.7 * b
        msgs.append(f"{name.split()[0]} {np.mean(bs) / b:.2f}")
    return ok, "shuffled/real between ratio: " + ", ".join(msgs)


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

    L = []
    p = L.append
    p("WITHIN A FAMILY OR BETWEEN FAMILIES: WHERE IS THE SPREAD?")
    p("=" * 88)
    p(f"  {'board':<20} {'family means':<16} {'labelled':>9} {'families':>9} {'between':>8} {'within':>7} "
      f"{'ratio':>6} {'largest within-family spread':>30}")
    ratios, spreads = [], []
    for name, (path, fn, what, inner) in BOARDS.items():
        sc, fams, J, names = subset(path, fn)
        if len(sc) < 6:
            p(f"  {name:<20} too few labelled systems")
            continue
        b, w, means, groups = decompose(sc, fams)
        big = max((np.max(v) - np.min(v), g) for g, v in groups.items())
        ratios.append((name, w / b))
        spreads.append((name, big[0]))
        p(f"  {name:<20} {what:<16} {len(sc):>9} {len(groups):>9} {100 * b:>7.2f}p {100 * w:>6.2f}p "
          f"{w / b:>6.2f} {f'{100 * big[0]:.1f}p in {big[1]}':>30}")
    p("")
    p(f"  ratio at least 0.5: {sum(1 for _, r in ratios if r >= 0.5)}/{len(ratios)} (pre-registered >= 2)")
    top = max(ratios, key=lambda t: t[1])[0] if ratios else "-"
    p(f"  highest ratio: {top} (pre-registered: SWE-bench)")
    p(f"  largest within-family spread above 15 points: "
      f"{sum(1 for _, s in spreads if s > 0.15)}/{len(spreads)} (pre-registered: all)")
    p("")
    p("  'between' is the SD of family means, 'within' the pooled SD of members")
    p("  around their family mean. On SWE-bench the within-family variation is a")
    p("  different harness on the same LLM; on MTEB a different model from the")
    p("  same lab; on ProteinGym a different size of the same method.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("within_family_spread_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote within_family_spread_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
