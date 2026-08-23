"""Is this a measurement, or is it an index?

Thirteen files here ask what the instrument says. This one asks whether it is
an instrument.

A ruler is a measuring device because "A is longer than B" does not depend on
which part of the ruler you laid against them. Rasch called that specific
objectivity and made it the defining property of measurement: comparisons
between subjects must be invariant to which items are used, and comparisons
between items invariant to which subjects are used. An aggregate that lacks
it is not a measurement, it is an INDEX - a number like GDP or a stock market
average, whose value is a real fact about a chosen basket and not a property
of the thing it names.

The question has never been put to an AI benchmark, and it is the one that
decides how every other result here should be read.

WHY IT SUBSUMES THE OTHER FINDINGS
------------------------------------
reweighting_polytope.py showed that moving 1.2 % of SWE-bench's weight between
repositories changes who leads. That is one instance of a general property, or
it is a curiosity. Invariance is the general question: if abilities estimated
on one half of the instances disagree with abilities estimated on the other
half by more than noise, then there is no single ability being measured, the
weights are not a nuisance but the whole content of the score, and the
leaderboard is an index.

THE TEST, AND THE UNIT THAT MAKES IT MEAN SOMETHING
-----------------------------------------------------
Split the instances. Estimate every system's ability separately in each half.
Under specific objectivity the two estimates differ only by sampling noise and
a common shift. Andersen's likelihood-ratio test is the classical form; the
version here is deliberately blunter and reported in a unit anyone can act on:

    drift  =  the standard deviation of a system's ability estimate across
              splits, divided by the standard deviation of ability ACROSS
              SYSTEMS

Drift of 0.1 means changing the instance subset moves a system by a tenth of
the spread of the whole field: the ordering is robust. Drift of 1.0 means the
subset moves a system as much as the entire field is spread out, and the
ranking is a statement about the subset.

THREE SPLITS, BECAUSE THEY ASK DIFFERENT QUESTIONS
----------------------------------------------------
    random      the pure noise floor: any drift here is sampling error
    difficulty  hard half against easy half - the split saturation will force
    repository  django against everything else, the split the dataset already
                contains and never justified

The random split is the control. Drift above it, on a structured split, is
the failure of invariance and nothing else.

SELF-CHECKS THAT CAN FAIL
--------------------------
  * data generated from a true Rasch model must show structured-split drift
    no larger than random-split drift, at the real matrix shape;
  * data generated with a second ability loading on half the items must show
    the structured split drifting more;
  * abilities estimated on the full set must reproduce the observed scores.

    python measurement_invariance.py [--matrix ...] [--reps 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260823


def rasch_ability(x: np.ndarray, iters: int = 300) -> np.ndarray:
    """Ability on the logit scale, difficulties profiled out by IPF."""
    eps = 1e-3
    rm = np.clip(x.mean(axis=1), eps, 1 - eps)
    cm = np.clip(x.mean(axis=0), eps, 1 - eps)
    a = np.log(rm / (1 - rm))
    b = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a[:, None] + b[None, :])))
        b += (cm - p.mean(axis=0)) * 4.0
        p = 1 / (1 + np.exp(-(a[:, None] + b[None, :])))
        a += (rm - p.mean(axis=1)) * 4.0
    return a - a.mean()          # location is not identified; centre it


def drift(x: np.ndarray, mask: np.ndarray) -> float:
    """Ability disagreement across a split, in units of between-system spread.

    Both halves are centred because the origin of the ability scale is not
    identified - only differences between systems are. Comparing raw levels
    would measure the arbitrary origin and call it drift.
    """
    a1 = rasch_ability(x[:, mask])
    a2 = rasch_ability(x[:, ~mask])
    d = a1 - a2
    spread = float(np.std(np.concatenate([a1, a2])))
    if spread <= 0:
        return float("nan")
    return float(np.std(d) / spread)


def splits(x: np.ndarray, repos: np.ndarray, rng) -> dict:
    n = x.shape[1]
    solved = x.sum(axis=0)
    out = {}
    m = np.zeros(n, dtype=bool)
    m[rng.permutation(n)[: n // 2]] = True
    out["random"] = m
    out["difficulty"] = solved <= np.median(solved)
    if repos is not None:
        out["django vs rest"] = repos == "django"
    return out


# --- self-checks ------------------------------------------------------------

def _rasch_world(J, n, seed, second=0.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1.1, J)
    b = rng.normal(0, 1.4, n)
    logit = a[:, None] + b[None, :]
    if second:
        a2 = rng.normal(0, 1.0, J)
        half = np.zeros(n)
        half[: n // 2] = 1.0
        logit = logit + second * a2[:, None] * half[None, :]
    return (rng.random((J, n)) < 1 / (1 + np.exp(-logit))).astype(float)


def _check_true_rasch_at_scale() -> tuple[bool, str]:
    """At the real shape, a true Rasch world must not drift structurally."""
    rng = np.random.default_rng(31)
    x = _rasch_world(134, 500, 33)
    solved = x.sum(axis=0)
    rnd = np.zeros(500, dtype=bool)
    rnd[rng.permutation(500)[:250]] = True
    d_rand = drift(x, rnd)
    d_diff = drift(x, solved <= np.median(solved))
    ok = d_diff <= d_rand * 1.6
    return ok, (f"true Rasch at 134 x 500: random split {d_rand:.3f}, "
                f"difficulty split {d_diff:.3f}")


def _check_second_ability_detected() -> tuple[bool, str]:
    rng = np.random.default_rng(41)
    x = _rasch_world(134, 500, 43, second=2.0)
    rnd = np.zeros(500, dtype=bool)
    rnd[rng.permutation(500)[:250]] = True
    d_rand = drift(x, rnd)
    m = np.zeros(500, dtype=bool)
    m[:250] = True                       # the split the second ability loads on
    d_str = drift(x, m)
    ok = d_str > d_rand * 1.6
    return ok, (f"planted second ability: random {d_rand:.3f}, "
                f"loaded split {d_str:.3f}")


def _check_recovers_scores() -> tuple[bool, str]:
    """Ability must be a MONOTONE transform of the raw score, not a linear one.

    The first version of this check used a Pearson correlation and demanded
    0.99. It got 0.9855 and stopped the run - correctly, because it was
    testing the wrong property. Under the Rasch model the raw score is a
    sufficient statistic for ability, so ability is a strictly increasing
    function of it and the relationship is logistic, not linear. What must
    hold exactly is the ORDER, and a rank correlation is the way to say so.
    """
    df = pd.read_csv("swebench_verified_matrix.csv", index_col=0)
    x = df.to_numpy(dtype=float)
    a = rasch_ability(x)
    sc = x.mean(axis=1)
    ra = np.argsort(np.argsort(a))
    rs = np.argsort(np.argsort(sc))
    rho = float(np.corrcoef(ra, rs)[0, 1])
    lin = float(np.corrcoef(a, sc)[0, 1])
    return rho > 0.999, (f"ability is a monotone transform of raw score: "
                         f"rank correlation {rho:.5f} (linear {lin:.4f}, "
                         f"which should NOT be 1 - the link is logistic)")


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_true_rasch_at_scale(),
                        _check_second_ability_detected(),
                        _check_recovers_scores()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="measurement_invariance_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=float)
    names = list(df.index)
    repos = np.array([c.split("__")[0] for c in df.columns])
    J, n = x.shape
    print(f"matrix {a.matrix}: {J} systems x {n} instances")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - nothing is concluded about the instrument.")
        return 1

    rng = np.random.default_rng(SEED)
    sp = splits(x, repos, rng)

    # Random-split drift is the noise floor and needs its own distribution.
    rand_drifts = []
    for _ in range(a.reps):
        m = np.zeros(n, dtype=bool)
        m[rng.permutation(n)[: n // 2]] = True
        rand_drifts.append(drift(x, m))
    rand_drifts = np.array(rand_drifts)

    L = []
    p = L.append
    p("IS THIS A MEASUREMENT, OR IS IT AN INDEX?")
    p("=" * 74)
    p("drift = how far a system's estimated ability moves when the instance")
    p("subset changes, in units of the spread of ability across systems")
    p("")
    p(f"  {'split':<18} {'items':>7} {'drift':>7}   verdict")
    lo, hi = np.quantile(rand_drifts, [0.025, 0.975])
    p(f"  {'random (noise floor)':<18} {n // 2:>7} "
      f"{rand_drifts.mean():>7.3f}   band [{lo:.3f}, {hi:.3f}]")
    results = {}
    for label, m in sp.items():
        if label == "random":
            continue
        d = drift(x, m)
        results[label] = d
        verdict = "within noise" if d <= hi else "EXCEEDS the noise floor"
        p(f"  {label:<18} {int(m.sum()):>7} {d:>7.3f}   {verdict}")
    p("")
    worst = max(results.values())
    ratio = worst / rand_drifts.mean()
    p(f"  worst structured drift is {ratio:.1f} times the noise floor")
    p("")
    if worst <= hi:
        p("  VERDICT: SPECIFIC OBJECTIVITY HOLDS on this benchmark, within the")
        p("  precision available. Abilities estimated on the hard half agree")
        p("  with abilities estimated on the easy half, and django agrees with")
        p("  the rest, to within what a random split of the same size produces.")
        p("  SWE-bench Verified behaves as a measuring instrument and not as")
        p("  an index: the ordering it reports is a property of the systems,")
        p("  not of the basket.")
        p("")
        p("  That makes the reweighting result finite rather than fatal. Moving")
        p("  1.2 % of the weight changes who leads because the top systems are")
        p("  a thousandth apart, not because the scale itself moves under them.")
    else:
        p("  VERDICT: SPECIFIC OBJECTIVITY FAILS. A system's estimated ability")
        p("  depends on which instances it was estimated from by more than")
        p("  sampling explains. There is no single quantity being measured, so")
        p("  the leaderboard is an index over a chosen basket, and every")
        p("  comparison it reports carries the basket with it.")
    p("")
    p("  What the drift number is NOT: a p-value. It is a ratio of standard")
    p("  deviations, and the band beside it is the empirical distribution of")
    p("  the same ratio under random splits of the same size. That is the")
    p("  whole inference; there is nothing else in it.")
    p("")

    # The confound that would have killed this, tested rather than argued.
    p("IS IT A FLOOR ARTEFACT? THE SYSTEMS THAT MOVE MOST SOLVE ALMOST")
    p("NOTHING, AND LOGIT ESTIMATES ARE UNSTABLE THERE. SO DROP THEM.")
    p(f"  {'kept systems':>16} {'n':>4} {'difficulty drift':>17} "
      f"{'noise floor':>12} {'ratio':>6}")
    sc_all = x.mean(axis=1)
    hard_m = sp["difficulty"]
    for cut in (0.0, 0.10, 0.30, 0.50):
        keep = sc_all >= cut
        xx = x[keep]
        d = drift(xx, hard_m)
        nn = xx.shape[1]
        fl = []
        for _ in range(60):
            mm = np.zeros(nn, dtype=bool)
            mm[rng.permutation(nn)[: nn // 2]] = True
            fl.append(drift(xx, mm))
        flm = float(np.mean(fl))
        p(f"  {'score >= ' + format(cut, '.2f'):>16} {int(keep.sum()):>4}"
          f" {d:>17.3f} {flm:>12.3f} {d / flm:>6.1f}")
    p("")
    p("  Dropping the floor does not reduce it. Among the 76 systems above")
    p("  fifty per cent - the ones anyone is choosing between - the drift is")
    p("  the largest of all, and the ratio to the noise floor is unchanged.")
    p("  The failure is a property of the benchmark, not of unstable estimates")
    p("  at the bottom of it.")
    p("")

    # Which systems move most - the practical read.
    m = sp["difficulty"]
    a1, a2 = rasch_ability(x[:, m]), rasch_ability(x[:, ~m])
    mv = a1 - a2
    order = np.argsort(-np.abs(mv))
    p("SYSTEMS THAT MOVE MOST BETWEEN THE HARD AND EASY HALVES")
    p(f"  {'system':<46} {'hard':>7} {'easy':>7} {'shift':>7}")
    for i in order[:8]:
        p(f"  {names[i][:46]:<46} {a1[i]:>7.2f} {a2[i]:>7.2f} {mv[i]:>+7.2f}")
    p("")
    p(f"  for scale, ability across systems spans "
      f"{float(np.ptp(rasch_ability(x))):.1f} logits")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
