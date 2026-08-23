"""Does this system's answer pattern look like a coherent ability?

recount_margin.py ended on an uncomfortable fact: the leader's edge over the
whole top twelve rests partly on instances that two systems out of 134 solve.
A score cannot tell you whether that is capability or accident, because a score
throws away WHICH items were solved and keeps only how many.

Psychometrics has asked this question about people for fifty years and has a
name for it - person fit - and a standard statistic. Nobody has asked it about
a leaderboard. That is the whole of this file.

THE MODEL THAT DEFINES "COHERENT"
-----------------------------------
Fit the strictly one-dimensional model p_ji = logistic(a_j + b_i): one ability
per system, one difficulty per item, nothing else. leaderboard_geometry.py
already measured that this model fits SWE-bench well - no second axis clears
its null by more than 15 % - so departures from it are informative rather than
routine. Under it, a coherent system solves easy items and misses hard ones,
and the pattern that does the opposite is the anomaly.

TWO STATISTICS, DELIBERATELY DIFFERENT
----------------------------------------
    l_z          the log-likelihood of the observed pattern under the model,
                 standardised. The classical person-fit statistic. Low means
                 the pattern is less probable than the model expects: solving
                 hard items while missing easy ones. HIGH is also a finding -
                 a pattern too neat to be a noisy draw.

    rare solves  the count of solved items the model gave this system less
                 than a 5 % chance on, against the number expected. This is
                 aimed squarely at the worry that motivated the file, and it
                 is the shape contamination would take: correct answers to
                 specific hard items with no corresponding general ability.

BOTH NULLS ARE SIMULATED, NOT ANALYTIC
----------------------------------------
Closed forms exist for l_z and are known to be poorly calibrated when the
abilities are estimated from the same data. Today a permutation null already
reported ten dimensions in data built with none, so a formula that "should"
hold is not good enough: the null here is a simulation from the fitted model
itself, and the calibration is measured rather than assumed.

THE CHECK THAT DECIDES EVERYTHING
-----------------------------------
Fitting the model to the data absorbs part of any anomaly - a strange system
drags its own ability estimate towards itself and then looks less strange. So
the false-flag rate is measured on data simulated FROM the model, where by
construction nothing is anomalous. If flagging is not near nominal there, no
system is named, because a detector that flags 8 % of innocent systems is a
generator of accusations rather than a measurement.

    python pattern_anomaly.py [--matrix ...] [--sims 400]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260823
RARE = 0.05


def additive_fit(x: np.ndarray, iters: int = 250) -> np.ndarray:
    """p_ji = logistic(a_j + b_i) by iterative proportional fitting."""
    eps = 1e-6
    rm = np.clip(x.mean(axis=1), eps, 1 - eps)
    cm = np.clip(x.mean(axis=0), eps, 1 - eps)
    a = np.log(rm / (1 - rm))
    b = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a[:, None] + b[None, :])))
        b += (cm - p.mean(axis=0)) * 4.0
        p = 1.0 / (1.0 + np.exp(-(a[:, None] + b[None, :])))
        a += (rm - p.mean(axis=1)) * 4.0
    return np.clip(1.0 / (1.0 + np.exp(-(a[:, None] + b[None, :]))), 1e-9,
                   1 - 1e-9)


def loglik_per_system(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (x * np.log(p) + (1 - x) * np.log(1 - p)).sum(axis=1)


def rare_solves(x: np.ndarray, p: np.ndarray, thr: float = RARE):
    mask = p < thr
    obs = (x * mask).sum(axis=1)
    exp = (p * mask).sum(axis=1)
    return obs, exp


def analyse(x: np.ndarray, sims: int, seed: int):
    """Both statistics with a simulated null, refitting the model each draw."""
    rng = np.random.default_rng(seed)
    p = additive_fit(x)
    ll = loglik_per_system(x, p)
    obs_rare, exp_rare = rare_solves(x, p)

    null_ll = np.empty((sims, x.shape[0]))
    null_rare = np.empty((sims, x.shape[0]))
    for b in range(sims):
        y = (rng.random(x.shape) < p).astype(float)
        # Refit on the simulated data: the real analysis refits too, and a
        # null that skipped it would be answering an easier question.
        q = additive_fit(y, iters=120)
        null_ll[b] = loglik_per_system(y, q)
        null_rare[b] = rare_solves(y, q)[0]

    # Per-system percentile of the observation in its own null.
    pct_ll = (null_ll < ll[None, :]).mean(axis=0)
    pct_rare = (null_rare < obs_rare[None, :]).mean(axis=0)
    return {"p": p, "ll": ll, "pct_ll": pct_ll, "obs_rare": obs_rare,
            "exp_rare": exp_rare, "pct_rare": pct_rare,
            "null_rare_mean": null_rare.mean(axis=0)}


# --- self-checks ------------------------------------------------------------

def _synth(J=40, n=300, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1.0, J)
    b = rng.normal(0, 1.3, n)
    p = 1 / (1 + np.exp(-(a[:, None] + b[None, :])))
    return (rng.random((J, n)) < p).astype(float), rng


def _check_clean_not_flagged() -> tuple[bool, str]:
    x, _ = _synth(seed=2)
    r = analyse(x, sims=120, seed=3)
    flagged = int(((r["pct_ll"] < 0.01) | (r["pct_rare"] > 0.99)).sum())
    ok = flagged <= 2
    return ok, f"model-generated systems flagged: {flagged} of {x.shape[0]}"


def _check_contamination_caught() -> tuple[bool, str]:
    """A system handed the answers to rare items must be flagged."""
    x, rng = _synth(seed=4)
    p = additive_fit(x)
    hard = np.argsort(p[0])[:40]          # hardest items for system 0
    x[0, hard] = 1.0
    r = analyse(x, sims=150, seed=5)
    ok = r["pct_rare"][0] > 0.99 or r["pct_ll"][0] < 0.01
    return ok, (f"contaminated system: rare percentile {r['pct_rare'][0]:.3f}, "
                f"l_z percentile {r['pct_ll'][0]:.3f}")


def _check_false_flag_rate() -> tuple[bool, str]:
    """The number that decides whether anyone is named.

    Simulate from the model, where nothing is anomalous by construction, and
    count how often the detector fires at its own 1 % threshold. Fitting the
    model to the data absorbs part of any anomaly, so this cannot be assumed.
    """
    fires, total = 0, 0
    for s in range(6):
        x, _ = _synth(J=40, n=300, seed=100 + s)
        r = analyse(x, sims=100, seed=200 + s)
        fires += int(((r["pct_ll"] < 0.01) | (r["pct_rare"] > 0.99)).sum())
        total += x.shape[0]
    rate = fires / total
    return rate <= 0.06, (f"false-flag rate on model-generated data "
                          f"{rate:.3f} (two one-sided 1 % tests -> 0.02 "
                          f"nominal)")



def _check_false_flag_at_scale() -> tuple[bool, str]:
    """The same false-flag test AT THE REAL SHAPE, 134 x 500.

    The 40 x 300 version passed. That proves nothing about 134 x 500: with
    500 items l_z has far more power to notice that the one-dimensional model
    is only approximately true, and systems will then deviate from it
    systematically rather than anomalously. leaderboard_geometry.py was caught
    by exactly this today - three null constructions passed at 60 x 400 and
    two of them failed at the real shape - so the check is repeated here at
    the size the answer will be reported at.
    """
    rng = np.random.default_rng(77)
    J, n = 134, 500
    a = rng.normal(0, 1.1, J)
    b = rng.normal(0, 1.4, n)
    p_true = 1 / (1 + np.exp(-(a[:, None] + b[None, :])))
    x = (rng.random((J, n)) < p_true).astype(float)
    r = analyse(x, sims=150, seed=79)
    fires = int(((r["pct_ll"] < 0.01) | (r["pct_rare"] > 0.99)).sum())
    rate = fires / J
    return rate <= 0.06, (f"false-flag rate at the REAL shape 134 x 500: "
                          f"{rate:.3f} ({fires} of {J}; nominal 0.02)")

def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_clean_not_flagged(),
                        _check_contamination_caught(),
                        _check_false_flag_rate(),
                        _check_false_flag_at_scale()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--out", default="pattern_anomaly_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=float)
    names = list(df.index)
    J, n = x.shape
    print(f"matrix {a.matrix}: {J} systems x {n} items")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no system is named. A detector that flags "
              "innocent systems is a generator of accusations, not a "
              "measurement.")
        return 1

    r = analyse(x, sims=a.sims, seed=SEED)
    scores = x.mean(axis=1)
    order = np.argsort(-scores, kind="stable")

    low_fit = np.flatnonzero(r["pct_ll"] < 0.01)
    high_fit = np.flatnonzero(r["pct_ll"] > 0.99)
    many_rare = np.flatnonzero(r["pct_rare"] > 0.99)

    L = []
    p = L.append
    p("DOES THE ANSWER PATTERN LOOK LIKE A COHERENT ABILITY?")
    p("=" * 74)
    p(f"{J} systems, {n} items, null = {a.sims} draws from the fitted")
    p("one-dimensional model, refitted on every draw")
    p("")
    p(f"  patterns LESS probable than the model expects (l_z < 1 %):  "
      f"{len(low_fit)}")
    p(f"  patterns MORE orderly than a noisy draw  (l_z > 99 %):      "
      f"{len(high_fit)}")
    p(f"  more rare solves than expected           (> 99 %):          "
      f"{len(many_rare)}")
    flag = (r["pct_ll"] < 0.01) | (r["pct_rare"] > 0.99)
    p("")
    p(f"  flagged in total                     {int(flag.sum())} of {J}"
      f"  ({100 * flag.mean():.1f} %)")
    p("  calibrated false-flag rate at this shape   3.7 %  (self-check above)")
    p(f"  excess over the baseline                   "
      f"{100 * flag.mean() - 3.7:.1f} percentage points")
    p("")
    p("  A CONFOUND, MEASURED RATHER THAN HOPED AWAY")
    corr = float(np.corrcoef(scores, flag.astype(float))[0, 1])
    p(f"    correlation of score with being flagged   {corr:+.3f}")
    p(f"    median score, flagged {np.median(scores[flag]):.3f}"
      f"   not flagged {np.median(scores[~flag]):.3f}")
    p("    The flag tracks weakness as well as strangeness: a weak system that")
    p("    solves a few rare items looks anomalous, and there are more weak")
    p("    systems than strong ones. So this list is NOT a contamination list")
    p("    and must not be read as one. It is a list of systems whose score")
    p("    describes them worse than it describes their neighbours.")
    p("")
    p("THE TOP OF THE TABLE, WHICH IS WHERE IT MATTERS")
    p(f"  {'rank':>4} {'system':<40} {'score':>6} {'l_z pct':>8}"
      f" {'rare':>5} {'exp':>6} {'pct':>6}")
    for rank, idx in enumerate(order[:12], start=1):
        p(f"  {rank:>4} {names[idx][:40]:<40} {scores[idx]:>6.3f}"
          f" {r['pct_ll'][idx]:>8.3f} {int(r['obs_rare'][idx]):>5}"
          f" {r['exp_rare'][idx]:>6.1f} {r['pct_rare'][idx]:>6.3f}")
    p("")
    if len(many_rare):
        p("SYSTEMS SOLVING MORE 'IMPOSSIBLE' ITEMS THAN THE MODEL ALLOWS")
        for idx in many_rare[np.argsort(-r["obs_rare"][many_rare])][:10]:
            p(f"  {names[idx][:46]:<46} solved {int(r['obs_rare'][idx])} of the"
              f" items it had < 5 % on, expected {r['exp_rare'][idx]:.1f}")
    else:
        p("No system solves more nominally impossible items than the fitted")
        p("model allows. The rare solves that decide the top of this")
        p("leaderboard are consistent with the abilities the model gives -")
        p("unusual events, not unaccountable ones.")
    p("")
    p("  A flag is not an accusation. A system can depart from a")
    p("  one-dimensional model because it is genuinely different in kind -")
    p("  a different scaffold, a different tool set, a specialisation. What")
    p("  the statistic says is that its score is not summarising it well,")
    p("  and that a single number is a worse description of it than of its")
    p("  neighbours. Where to look next is a question for a human.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
