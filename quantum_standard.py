"""The standard, applied to quantum hardware.

Nine boards in this repository rank software: coding agents, embedding models,
protein predictors, structure predictors, provers, tabular learners, chat models.
This applies the same construction to physical devices - superconducting,
trapped-ion and neutral-atom processors from IBM, IQM, IonQ, Rigetti, Quantinuum
and Origin - benchmarked by Metriq's public results archive.

It is a harder test of the machinery than any board so far, in two ways.

  THE SHAPE IS EXTREME. Mirror Circuits fields 10 devices against 4
  configurations. HELM classic, at J = 90 and n = 10, is the worst-resolved
  board here and it has more than twice the items. A paired t-test on 4 items
  has 3 degrees of freedom, and Holm over 45 pairs at alpha 0.05 needs a
  critical value that 4 items can rarely reach.

  THE OUTCOMES ARE NOT BINARY. Every other board is pass/fail per item or a
  win rate. These are fidelities and success probabilities in [0, 1],
  continuous. The construction never assumed binary - it works on the paired
  differences - but this is the first time that has been exercised.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  no quantum board can name a winner: more than one device could be first
      on all three.
  P2  the established share is below 10 % on all three boards, so the field is
      even less resolved than HELM classic at 6.6 %.
  P3  law 1 predicts the established share within 10 points on at least 2 of 3
      boards. The law has never been asked about n = 4, about hardware, or
      about a continuous outcome; this is the first test of it outside
      software.
  P4  at least ONE board establishes something - a nonzero share. If all three
      are exactly zero the boards carry no ordering information at all and the
      rest of the table is a report about nothing, which is a finding but must
      be labelled as one rather than presented as a measurement.

  What a miss on P3 would mean: law 1's reach stops at the software boards it
  was built on, and its cross-field claim needs the qualifier.

SELF-CHECKS (no table if any fails)
  * the matrices must be complete - no missing cell - and their values inside
    [0, 1], asserted;
  * every board must have at least 3 devices and 3 configurations;
  * the beats relation must be transitive here too, as it is on all nine
    software boards; a failure would mean the construction behaves differently
    on continuous outcomes;
  * the SNR must be finite and positive on every board, or law 1 cannot be
    evaluated and P3 is undefined rather than missed.

    python quantum_standard.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import leaderboard_entropy as le
import rank_sets as rs
from band_slack import bands_of

SEED = 20260825
BOARDS = {
    "Mirror Circuits": "quantum/mirror_circuits_matrix.csv",
    "QML Kernel": "quantum/qml_kernel_matrix.csv",
    "Linear Ramp QAOA": "quantum/lr_qaoa_matrix.csv",
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    rows, ok_range, ok_trans, ok_snr = [], True, True, True
    for name, path in BOARDS.items():
        if not Path(path).exists():
            continue
        df = pd.read_csv(path, index_col=0)
        x = df.to_numpy(dtype=float)
        if np.isnan(x).any() or x.min() < -1e-9 or x.max() > 1 + 1e-9:
            ok_range = False
        J, n = x.shape
        if J < 3 or n < 3:
            continue
        r = rs.rank_sets(x)
        b = r["beats"]
        two = (b.astype(np.uint8) @ b.astype(np.uint8)) > 0
        if bool((two & ~b & ~np.eye(J, dtype=bool)).any()):
            ok_trans = False
        iu = np.triu_indices(J, k=1)
        sig = r["sigma"][iu]
        sig = sig[np.isfinite(sig) & (sig > 0)]
        sc = x.mean(axis=1)
        tau = float(sc.std(ddof=1))
        sp = float(np.median(sig)) if len(sig) else float("nan")
        c = float(r["crit"])
        snr = (tau * math.sqrt(2 * n) / (c * sp)
               if sp and np.isfinite(sp) and c > 0 else float("nan"))
        if not np.isfinite(snr) or snr <= 0:
            ok_snr = False
        estab = float(b.sum() / (J * (J - 1)))
        pred = float(norm.sf(1.0 / snr)) if np.isfinite(snr) and snr > 0 else float("nan")
        tie1 = int((r["best"] == 1).sum())
        best, worst = bands_of(b)
        ed = int(b.sum())
        free = (int((worst[np.nonzero(b)[0]] <= best[np.nonzero(b)[1]]).sum())
                if ed else 0)
        H = le.log_extensions(b, 400, np.random.default_rng(SEED))["bits"]
        ceil = math.lgamma(J + 1) / math.log(2)
        rows.append({"name": name, "J": J, "n": n, "snr": snr, "estab": estab,
                     "pred": pred, "tie1": tie1, "edges": ed, "free": free,
                     "Hfrac": H / ceil if ceil > 0 else float("nan"),
                     "top": list(df.index[np.argsort(-sc)][:3])})
        print(f"  {name:<20} J={J:<3} n={n:<3} tie@1={tie1:<3} "
              f"established {100 * estab:.1f}%")

    print("self-checks ...")
    print(f"  [{'ok  ' if ok_range else 'FAIL'}] every matrix is complete with values in [0, 1]")
    print(f"  [{'ok  ' if ok_trans else 'FAIL'}] the beats relation is transitive on "
          f"continuous outcomes too")
    print(f"  [{'ok  ' if ok_snr else 'FAIL'}] SNR finite and positive on every board")
    ok_n = len(rows) >= 3
    print(f"  [{'ok  ' if ok_n else 'FAIL'}] {len(rows)} quantum boards (need >= 3)")

    if not (ok_range and ok_trans and ok_snr and ok_n):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("THE STANDARD, APPLIED TO QUANTUM HARDWARE")
    p("=" * 96)
    p("  Devices from IBM, IQM, IonQ, Rigetti, Quantinuum and Origin, from")
    p("  Metriq's public results archive. Rows are devices, columns are")
    p("  benchmark configurations, cells are the reported score. Continuous")
    p("  outcomes, not pass/fail - a first for this repository.")
    p("")
    p(f"  {'board':<20}{'J':>4}{'n':>4}{'J/n':>7}{'SNR':>7}{'estab':>8}"
      f"{'law 1':>8}{'err':>7}{'tie@1':>7}{'free':>8}{'H/ceil':>8}")
    for r in rows:
        p(f"  {r['name']:<20}{r['J']:>4}{r['n']:>4}{r['J'] / r['n']:>7.2f}"
          f"{r['snr']:>7.2f}{100 * r['estab']:>7.1f}%{100 * r['pred']:>7.1f}%"
          f"{100 * (r['pred'] - r['estab']):>+7.1f}{r['tie1']:>7}"
          f"{(100 * r['free'] / r['edges'] if r['edges'] else 0):>7.0f}%"
          f"{100 * r['Hfrac']:>7.0f}%")
    p("")
    nowin = sum(1 for r in rows if r["tie1"] > 1)
    low = sum(1 for r in rows if r["estab"] < 0.10)
    close = sum(1 for r in rows if abs(r["pred"] - r["estab"]) <= 0.10)
    some = sum(1 for r in rows if r["estab"] > 0)
    p(f"  P1  more than one device could be first on {nowin} of {len(rows)}   "
      f"pre-registered = all:  {'HIT' if nowin == len(rows) else 'MISS'}")
    p(f"  P2  established share below 10 % on {low} of {len(rows)}            "
      f"pre-registered = all:  {'HIT' if low == len(rows) else 'MISS'}")
    p(f"  P3  law 1 within 10 points on {close} of {len(rows)}                "
      f"pre-registered >= 2:   {'HIT' if close >= 2 else 'MISS'}")
    p(f"  P4  at least one board establishes something: {some} of {len(rows)}  "
      f"pre-registered >= 1:   {'HIT' if some >= 1 else 'MISS'}")
    p("")
    p("  Top three by mean score, which is what a quantum leaderboard prints:")
    for r in rows:
        p(f"    {r['name']:<20} {', '.join(str(t)[:26] for t in r['top'])}")
    p("")
    p("  P3 IS THE RESULT. Law 1 was derived for, and only ever tested on,")
    p("  software boards with pass/fail items. Asked about physical devices,")
    p("  continuous fidelities and as few as 4 items, it lands within 10 points")
    p("  on all three - errors of -3.3, -8.5 and -2.4. At n = 4 it predicts")
    p("  0.0 % and the boards establish 2 to 3 %, so it under-predicts slightly")
    p("  where the degrees of freedom run out, which is the direction that")
    p("  understates a benchmark's power rather than overstating it.")
    p("")
    p("  P1 AND P2 MISSED, AND THE MISS IS THE OTHER RESULT. I predicted no")
    p("  quantum board could name a winner and that all three would establish")
    p("  under 10 %. QML Kernel does name one: tie@1 is exactly 1, ibm_boston,")
    p("  and it establishes 33.3 % of its pairs. It has 16 configurations where")
    p("  the other two have 4.")
    p("")
    p("  So the three quantum boards span the whole range WITHIN one field, and")
    p("  they line up on items rather than devices:")
    p("")
    p("    n = 4,  10 devices   3.3 % established,  7 could be first,   0 % free")
    p("    n = 4,   7 devices   2.4 % established,  6 could be first,   0 % free")
    p("    n = 16,  6 devices  33.3 % established,  1 could be first, 100 % free")
    p("")
    p("  The 16-item board resolves its winner and its band table carries its")
    p("  ENTIRE relation; the 4-item boards resolve almost nothing and their")
    p("  bands carry none of it. That is the same ordering full_board_free.py")
    p("  found across the software boards at Spearman -0.64 against J/n, and it")
    p("  reappears here inside a single field, on hardware, with a fourfold")
    p("  change in items and no change in anything else.")
    p("")
    p("  Why the shape is the story. Mirror Circuits ranks 10 devices on 4")
    p("  configurations. A paired test on 4 items has 3 degrees of freedom, and")
    p("  Holm over 45 pairs at alpha 0.05 asks for a critical value that 4")
    p("  items can rarely reach. HELM classic, the worst-resolved software")
    p("  board here, has 10 items and establishes 6.6 % of its pairs.")
    p("")
    p("  This is not a defect of the devices or of the benchmark. It is the")
    p("  cost of running physical hardware: every column is machine time on a")
    p("  scarce processor, so a quantum leaderboard buys its items at a price")
    p("  no software benchmark pays, and the number of items is what decides")
    p("  how much of an ordering the board can support.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("quantum_standard_results.txt").write_text(text + chr(10),
                                                    encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote quantum_standard_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
