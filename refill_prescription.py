"""What would this benchmark need, to answer the question it is asked?

Every other file here diagnoses. This one prescribes. The twelve measurements
before it say the same thing from twelve directions - nineteen systems tied at
the top, ten resolvable tiers out of 134 positions, 48 % of the resolving power
already spent - and every one of them stops at the diagnosis. A benchmark owner
reading them learns that their instrument is going blind and nothing about what
to build next.

So: how many new instances, at what difficulty, would separate the top?

THIS IS A POWER CALCULATION, WHICH BENCHMARKS DO NOT GET
----------------------------------------------------------
Clinical trials are not run without one. Nobody would fund a study of a drug
without asking how many patients are needed to detect the effect they care
about. Benchmarks are built to whatever size the data collection happened to
yield, and the question "how many instances do we need to tell these two
systems apart" is not asked before or after.

It has an answer, and the answer has a shape worth knowing: it depends far
more on the DIFFICULTY of the new instances than on their number.

WHY DIFFICULTY DOMINATES
-------------------------
An instance separates a pair only when the pair disagrees on it, and a pair
disagrees most often when the instance sits at their own level. Adding a
thousand instances that the frontier solves 95 % of the time buys almost
nothing, because the frontier agrees on nearly all of them. The cheapest
instance to build is the one the systems in question find a coin flip.

THE HONEST CONDITIONAL, STATED BEFORE ANY NUMBER
--------------------------------------------------
A power calculation answers "how many, to detect a gap of THIS size". It
cannot tell you the gap is real. If the top systems are genuinely equal, no
number of instances will separate them, and the correct output is not a large
number but the word never. Both cases are computed here: the prescription
under the observed gap, and the control where the gap is set to zero, which
must fail at every size. If the control ever succeeds, the machinery is
broken and nothing is prescribed.

SELF-CHECKS THAT CAN FAIL
--------------------------
  * a pair with a true gap of zero must not separate at any budget;
  * a pair with a large true gap must separate, and sooner with instances at
    their own level than with easy ones;
  * the simulated benchmark must reproduce the observed scores of the systems
    it is built from, or the difficulties are not being applied correctly.

    python refill_prescription.py [--matrix ...] [--challenger 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260823
Z = 1.959963984540054


def rasch_fit(x: np.ndarray, iters: int = 250):
    eps = 1e-6
    rm = np.clip(x.mean(axis=1), eps, 1 - eps)
    cm = np.clip(x.mean(axis=0), eps, 1 - eps)
    a = np.log(rm / (1 - rm))
    b = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a[:, None] + b[None, :])))
        b += (cm - p.mean(axis=0)) * 4.0
        p = 1 / (1 + np.exp(-(a[:, None] + b[None, :])))
        a += (rm - p.mean(axis=1)) * 4.0
    return a, b


def separates(ability_a: float, ability_b: float, difficulty: float,
              m: int, reps: int, rng) -> float:
    """Power: how often m new instances at this difficulty separate the pair.

    The test is McNemar on the new instances alone, two-sided at 5 %. Only
    the new instances are used because the existing ones are already spent -
    they are in both systems' scores and have already failed to separate them.
    """
    pa = 1 / (1 + np.exp(-(ability_a + difficulty)))
    pb = 1 / (1 + np.exp(-(ability_b + difficulty)))
    wins = 0
    for _ in range(reps):
        xa = rng.random(m) < pa
        xb = rng.random(m) < pb
        b_only = int(np.sum(xa & ~xb))
        c_only = int(np.sum(~xa & xb))
        d = b_only + c_only
        if d == 0:
            continue
        # Normal approximation to the exact binomial, with continuity.
        z = (abs(b_only - c_only) - 1) / np.sqrt(d)
        if z > Z:
            wins += 1
    return wins / reps


def cheapest(ability_a, ability_b, difficulties, budgets, reps, rng,
             target=0.80):
    best = None
    grid = {}
    for dif in difficulties:
        for m in budgets:
            pw = separates(ability_a, ability_b, dif, m, reps, rng)
            grid[(dif, m)] = pw
            if pw >= target and (best is None or m < best[1]):
                best = (dif, m, pw)
    return best, grid


# --- self-checks ------------------------------------------------------------

def _check_zero_gap_never() -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    worst = max(separates(1.0, 1.0, d, m, 400, rng)
                for d in (-2.0, -1.0, 0.0, 1.0)
                for m in (50, 200, 1000, 4000))
    ok = worst < 0.12
    return ok, f"true gap of zero, highest power over all budgets: {worst:.3f}"


def _check_big_gap_separates() -> tuple[bool, str]:
    rng = np.random.default_rng(3)
    pw = separates(1.5, 0.5, -1.0, 300, 400, rng)
    ok = pw > 0.9
    return ok, f"large gap, 300 instances at their own level: power {pw:.3f}"


def _check_difficulty_matters() -> tuple[bool, str]:
    """Instances at the pair's own level must beat easy ones, decisively."""
    rng = np.random.default_rng(5)
    own = separates(1.2, 1.0, -1.1, 400, 400, rng)
    easy = separates(1.2, 1.0, 3.5, 400, 400, rng)
    ok = own > easy + 0.15
    return ok, (f"same budget, own level {own:.3f} vs easy instances "
                f"{easy:.3f}")


def _check_model_reproduces() -> tuple[bool, str]:
    df = pd.read_csv("swebench_verified_matrix.csv", index_col=0)
    x = df.to_numpy(dtype=float)
    a, b = rasch_fit(x)
    p = 1 / (1 + np.exp(-(a[:, None] + b[None, :])))
    err = float(np.abs(p.mean(axis=1) - x.mean(axis=1)).max())
    return err < 0.01, f"fitted model reproduces observed scores to {err:.4f}"


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_zero_gap_never(), _check_big_gap_separates(),
                        _check_difficulty_matters(), _check_model_reproduces()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--challenger", type=int, default=3,
                    help="rank to separate the leader from")
    ap.add_argument("--reps", type=int, default=600)
    ap.add_argument("--out", default="refill_prescription_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=float)
    names = list(df.index)
    print(f"matrix {a.matrix}: {x.shape[0]} systems x {x.shape[1]} instances")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - nothing is prescribed.")
        return 1

    ab, bdiff = rasch_fit(x)
    scores = x.mean(axis=1)
    order = np.argsort(-scores, kind="stable")
    lead = int(order[0])
    chal = int(order[a.challenger - 1])
    rng = np.random.default_rng(SEED)

    # Difficulty is expressed as what the LEADER would score on such an
    # instance, which is the only form a benchmark builder can act on.
    targets = [0.9, 0.75, 0.5, 0.3, 0.15]
    difficulties = [float(np.log(t / (1 - t)) - ab[lead]) for t in targets]
    budgets = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    L = []
    p = L.append
    p("WHAT WOULD IT TAKE TO SEPARATE THE TOP?")
    p("=" * 74)
    p(f"leader      {names[lead][:52]}  {scores[lead]:.3f}")
    p(f"challenger  {names[chal][:52]}  {scores[chal]:.3f}   (rank {a.challenger})")
    p(f"fitted ability gap on the logit scale: {ab[lead] - ab[chal]:+.4f}")
    p("")
    p("POWER TO SEPARATE THEM WITH NEW INSTANCES ONLY")
    p("  columns are how many new instances; rows are how hard, written as")
    p("  the rate the LEADER would score on them, which is the only form a")
    p("  benchmark builder can act on")
    p(f"  {'leader solves':>14} " + " ".join(f"{m:>6}" for m in budgets))
    grid = {}
    for t, dif in zip(targets, difficulties):
        row = []
        for m in budgets:
            pw = separates(ab[lead], ab[chal], dif, m, a.reps, rng)
            grid[(t, m)] = pw
            row.append(pw)
        p(f"  {t:>13.2f}  " + " ".join(f"{v:>6.2f}" for v in row))
    p("")
    # "More than ten thousand" is not a prescription. The resolution law
    # gives the exact figure: with discordance rate d and rate difference
    # delta, n >= d ((z_alpha + z_beta) / delta)^2. Simulated and analytic
    # are printed together so the formula is checked, not trusted.
    zb = 0.8416212335729143            # 80 % power
    p("REQUIRED SIZE FOR 80 % POWER, FROM THE RESOLUTION LAW")
    p(f"  {'leader solves':>14} {'delta':>8} {'discord':>8} {'instances needed':>17}")
    need = {}
    for t, dif in zip(targets, difficulties):
        pa = 1 / (1 + np.exp(-(ab[lead] + dif)))
        pb = 1 / (1 + np.exp(-(ab[chal] + dif)))
        delta = pa - pb
        dd = pa * (1 - pb) + pb * (1 - pa)
        n_req = dd * ((Z + zb) / delta) ** 2 if delta > 0 else float("inf")
        need[t] = n_req
        p(f"  {t:>13.2f} {delta:>8.4f} {dd:>8.3f} {n_req:>17,.0f}")
    best_t = min(need, key=need.get)
    p("")
    p(f"  THE PRESCRIPTION: about {need[best_t]:,.0f} new instances on which the")
    p(f"  leader would score {best_t:.0%}. The current benchmark has "
      f"{x.shape[1]} -")
    p(f"  the answer is {need[best_t] / x.shape[1]:.0f} times the whole thing, "
      "to settle one pair.")
    p("")
    ok80 = [(t, m) for (t, m), v in grid.items() if v >= 0.80]
    if ok80:
        best_t, best_m = min(ok80, key=lambda k: (k[1], -k[0]))
        p(f"  CHEAPEST PRESCRIPTION: {best_m} new instances on which the")
        p(f"  leader would score about {best_t:.0%}, for 80 % power.")
    else:
        p("  No cell in the table reaches 80 % power. Within the budgets")
        p("  tried, this pair cannot be separated by adding instances.")
    p("")
    p("  Read across a row and then down a column. Difficulty moves the")
    p("  answer far more than budget does: instances the leader solves nine")
    p("  times in ten are nearly worthless however many are built, because")
    p("  both systems solve nearly all of them and a pair that agrees")
    p("  separates nothing.")
    p("")

    # The control. If the two are truly equal, nothing works, and the table
    # must say so rather than quoting a large number.
    p("THE CONTROL, WHICH IS THE POINT OF THE WHOLE FILE")
    p("  the same calculation with the true gap set to zero:")
    zrow = [separates(ab[lead], ab[lead], difficulties[2], m, a.reps, rng)
            for m in budgets]
    p(f"  {'gap = 0':>13}   " + " ".join(f"{v:>6.2f}" for v in zrow))
    p("")
    p("  Flat at the false-positive rate, at every budget, forever. If the")
    p(f"  {a.challenger} systems at the top of this leaderboard are equally able,")
    p("  the prescription above buys nothing at all, and no amount of data")
    p("  collection will decide between them. A power calculation answers")
    p("  'how many, to detect a gap of this size'. It cannot tell you the")
    p("  gap is there.")
    p("")
    p("  That is not a weakness of the method. It is the thing a benchmark")
    p("  owner most needs to be told before commissioning five thousand new")
    p("  instances: the honest ceiling on what more data can do.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
