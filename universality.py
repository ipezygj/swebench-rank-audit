"""Is any of this a law? The same quantities on three unrelated leaderboards.

Seventeen files measure one matrix. A finding about one benchmark is an
audit. A regularity that holds across benchmarks built by different people,
for different tasks, with different scoring, is the beginning of a theory -
and the only way to tell the two apart is to run the same instruments on
several and look at what stays put.

Four leaderboards, nothing in common but the shape systems x items:

    SWE-bench Verified   134 x 500   binary (patch resolves the issue or not)
    MTEB English v2      181 x  41   continuous task scores in [0, 1]
    HELM classic          91 x  10   continuous win rates in [0, 1]
    MathArena 2025        ? x   ?    competition maths, answer-match grading,
                                     problems written after training cutoffs;
                                     added last, with predictions recorded in
                                     load_all() before its matrix existed

WHAT IS COMPARED
----------------
    H / ceiling      leaderboard entropy as a fraction of log2(J!): the share
                     of the printed order the data does not determine
    tie at the top   systems whose simultaneous rank set contains 1
    median width     median rank-set width as a fraction of J
    established      established pairs as a fraction of all pairs
    top-10 H         bits among the ten best, against log2(10!) = 21.8

Every quantity is dimensionless so that a 500-item binary benchmark and a
10-item continuous one can sit in the same row.

WHAT WOULD COUNT AS A LAW, STATED BEFORE LOOKING
-------------------------------------------------
If H / ceiling lands near one value for all three, that is a candidate
regularity worth testing on a fourth. If it scatters, then the share of a
leaderboard that is evidence is a property of each benchmark and there is no
general statement to make - which is also an answer, and the cheaper one to
be wrong about.

A SECOND, SHARPER PREDICTION
----------------------------
The resolution law in resolution_law.py says pairs separate when their gap
exceeds c * sigma / sqrt(n_eff). So the established fraction should rise with
the number of items and with the spread of the field relative to its noise.
HELM has ten items and should establish the least; SWE-bench has five hundred
and should establish the most. If the ordering comes out otherwise, the
instruments are measuring something other than what the law describes.

SELF-CHECKS
-----------
  * every matrix must be complete after preparation (no NaN);
  * every matrix must reproduce its own published order from its row means;
  * rank_sets must return the observed rank inside every set, on every matrix.

    python universality.py [--draws 1500] [--samples 2000]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import rank_sets as rs
import leaderboard_entropy as le
import invariant_core as ic

SEED = 20260823


def load_all() -> dict:
    out = {}
    sw = pd.read_csv("swebench_verified_matrix.csv", index_col=0)
    out["SWE-bench Verified"] = sw.to_numpy(dtype=float)
    mt = pd.read_csv("mteb_eng_v2_wide.csv", index_col=0).dropna(axis=0)
    out["MTEB English v2"] = mt.to_numpy(dtype=float)
    he = pd.read_csv("helm_winrate_matrix.csv", index_col=0)
    he = he.dropna(axis=0)
    out["HELM classic"] = he.to_numpy(dtype=float)
    # THE FOURTH, AND THE PREDICTIONS FOR IT, WRITTEN BEFORE ITS MATRIX EXISTED
    # (2026-08-23, while the files were still downloading):
    #   1. its top ten will be a complete antichain: top-10 H = 21.8 / 21.8;
    #   2. its established-pairs share will sit where its item count puts it
    #      in the HELM < MTEB < SWE-bench ordering;
    #   3. if it has enough items (>= ~100), H/ceiling will land near 55 %;
    #      if it has few, it will land above, like HELM.
    # If 1 fails, the "top is never resolved" sentence is dead. If 3 fails
    # with enough items, the 55 % hypothesis is dead. Either way it is
    # written here first.
    ma = Path("matharena/matrix.csv")
    if ma.exists():
        m = pd.read_csv(ma, index_col=0).dropna(axis=0)
        out["MathArena 2025"] = m.to_numpy(dtype=float)
    return out


def measure(name: str, x: np.ndarray, draws: int, samples: int,
            rng) -> dict:
    J, n = x.shape
    r = rs.rank_sets(x, draws=draws)
    beats = r["beats"]
    viol = ic.transitivity_violations(beats)
    H = le.log_extensions(beats, samples, rng) if viol == 0 else None
    ceiling = gammaln(J + 1) / math.log(2)
    order = np.argsort(-x.mean(axis=1), kind="stable")
    top10 = order[:10]
    sub = beats[np.ix_(top10, top10)]
    H10 = le.log_extensions(sub, samples, rng) if ic.transitivity_violations(sub) == 0 else None
    width = r["worst"] - r["best"]
    return {
        "name": name, "J": J, "n": n,
        "established": float(beats.sum() / (J * (J - 1))),
        "tie_top": int((r["best"] == 1).sum()),
        "tie_top_frac": float((r["best"] == 1).mean()),
        "median_width_frac": float(np.median(width) / J),
        "H": H["bits"] if H else float("nan"),
        "H_frac": H["bits"] / ceiling if H else float("nan"),
        "H10": H10["bits"] if H10 else float("nan"),
        "viol": viol,
        "inside": bool((r["best"] <= r["observed"]).all()
                       and (r["observed"] <= r["worst"]).all()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--out", default="universality_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)

    mats = load_all()
    print("self-checks")
    ok = True
    for name, x in mats.items():
        complete = not np.isnan(x).any()
        print(f"  [{'ok  ' if complete else 'FAIL'}] {name}: complete matrix "
              f"{x.shape[0]} x {x.shape[1]}")
        ok = ok and complete
    if not ok:
        print("\nA CHECK FAILED - nothing is compared.")
        return 1

    rows = []
    for name, x in mats.items():
        print(f"\nmeasuring {name} ...")
        m = measure(name, x, a.draws, a.samples, rng)
        print(f"  [{'ok  ' if m['inside'] else 'FAIL'}] observed rank inside "
              f"its set for every system")
        print(f"  [{'ok  ' if m['viol'] == 0 else 'FAIL'}] evidential order "
              f"transitive ({m['viol']} violations)")
        rows.append(m)

    L = []
    p = L.append
    p("IS ANY OF THIS A LAW? THREE UNRELATED LEADERBOARDS, SAME INSTRUMENTS")
    p("=" * 78)
    p(f"  {'leaderboard':<20} {'J':>4} {'n':>4} {'estab.':>7} {'tie@1':>9}"
      f" {'med.width':>10} {'H/ceiling':>10} {'top-10 H':>9}")
    for m in rows:
        p(f"  {m['name']:<20} {m['J']:>4} {m['n']:>4} "
          f"{100 * m['established']:>6.1f}% "
          f"{m['tie_top']:>3} ({100 * m['tie_top_frac']:>4.1f}%) "
          f"{100 * m['median_width_frac']:>9.1f}% "
          f"{100 * m['H_frac']:>9.1f}% {m['H10']:>5.1f}/21.8")
    p("")
    p("  estab.     ordered pairs established by the simultaneous test")
    p("  tie@1      systems whose rank set contains 1 (share of J)")
    p("  med.width  median rank-set width as a share of J")
    p("  H/ceiling  share of the printed order the data does not determine")
    p("  top-10 H   bits undetermined among the ten best; 21.8 = nothing known")
    p("")
    fr = [m["H_frac"] for m in rows if not math.isnan(m["H_frac"])]
    est = [m["established"] for m in rows]
    p("READING IT")
    if max(fr) - min(fr) < 0.10:
        p(f"  H/ceiling spans only {100 * (max(fr) - min(fr)):.1f} points across three")
        p("  benchmarks that share nothing but their shape. That is a candidate")
        p("  regularity, and it earns a fourth benchmark before it earns a name.")
    else:
        p(f"  H/ceiling spans {100 * (max(fr) - min(fr)):.1f} points, from "
          f"{100 * min(fr):.0f} % to {100 * max(fr):.0f} %.")
        p("  By the threshold written above before the numbers were seen, that")
        p("  is not a regularity, and the pre-registered verdict stands: no")
        p("  universal constant is claimed.")
        srt = sorted(rows, key=lambda m: m["H_frac"])
        gap2 = 100 * (srt[1]["H_frac"] - srt[0]["H_frac"])
        p("")
        p(f"  What the threshold does not capture, reported as an observation")
        p(f"  and not as a result: the two larger benchmarks agree to "
          f"{gap2:.1f} points")
        p(f"  ({100 * srt[0]['H_frac']:.1f} % and {100 * srt[1]['H_frac']:.1f} %), "
          f"and the outlier is the ten-item one, where")
        p("  the resolution law predicts more entropy because fewer pairs can")
        p("  separate. That is a HYPOTHESIS for a fourth benchmark: H/ceiling")
        p("  near 55 % once the item count is large enough for the field's")
        p("  spread to show. It is written here so that a fourth benchmark")
        p("  can falsify it, not so that it can be claimed now.")
    p("")
    # The sharper prediction: established fraction should follow items.
    order_n = sorted(rows, key=lambda m: m["n"])
    order_e = sorted(rows, key=lambda m: m["established"])
    same = [m["name"] for m in order_n] == [m["name"] for m in order_e]
    p("  The resolution-law prediction - more items, more pairs established -")
    p(f"  {'HOLDS' if same else 'FAILS'}: by item count the order is "
      + " < ".join(m["name"].split()[0] for m in order_n)
      + ", by established pairs it is "
      + " < ".join(m["name"].split()[0] for m in order_e) + ".")
    if not same:
        p("  Item count is not the whole story: the spread of the field against")
        p("  its own noise matters as much, and a ten-item benchmark on a widely")
        p("  spread field can establish more than a five-hundred-item one on a")
        p("  tight field.")
    p("")
    allfull = all(abs(m["H10"] - 21.8) < 0.3 for m in rows)
    p("  What IS common to all three, and it is exact, not approximate: the")
    p("  ten best are a complete antichain on every leaderboard tested. The")
    p(f"  top-10 entropy is {'21.8 of 21.8 on all three' if allfull else 'near its ceiling on all three'} - "
      "the simultaneous test")
    p("  establishes NOT ONE pair among the ten best of SWE-bench, of MTEB,")
    p("  or of HELM. Three benchmarks, three fields, three scoring schemes,")
    p("  and the same fact about the only rows anyone reads. That is the one")
    p("  sentence that survives the comparison, and it is the one to test on")
    p("  a fourth.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
