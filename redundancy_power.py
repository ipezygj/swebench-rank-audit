"""Was yesterday's result a finding, or a statistic running out of room?

top_redundancy.py reported that the middle of a board is significantly more
dependent than a Rasch null predicts while the top is not, and concluded that
the systems crowding the top are not doing the same work. There is an obvious
way for that to be wrong, and it is my own statistic: union(10) at the top of
SWE-bench Verified is 89.8 %, close to a ceiling of 100 %, so both the real
board and every simulated one are squeezed into the same narrow range. A test
with no room to move fails to reject whatever is true.

Two ways to find out, both of them attacks on my own conclusion.

  POWER      inject a known amount of dependence - a latent factor shared by
             the group, p = sigmoid(theta - b + lambda * z) - and find the
             lambda at which the test rejects 80 % of the time. If the top
             needs a far larger lambda than the middle, the test is weaker
             there and yesterday's asymmetry is an artefact of where the two
             groups sit, not a fact about them.

  A SECOND   mean pairwise correlation of the Rasch residuals inside the
  STATISTIC  group. It measures dependence directly, is normalised rather than
             bounded by a coverage ceiling, and should show the same pattern as
             union(10) if that pattern is real.

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  the lambda needed for 80 % power is larger at the top than in the middle
      on at least 2 of the 3 boards.
  P2  that ratio exceeds 1.5 on at least 2 of 3 - the loss of power is
      material and not a rounding difference.
  P3  at the lambda implied by the middle's own observed deviation, the test
      at the top has power below 50 % on at least 2 of 3. If so, yesterday's
      top result cannot tell "no dependence" apart from "the same dependence
      the middle has", and the conclusion has to be weakened to say so.
  P4  the residual-correlation statistic reproduces yesterday's pattern:
      significant in the middle on at least 2 of 3 boards, and not significant
      at the top on at least 2 of 3.

  P1 to P3 and P4 pull in opposite directions on purpose. If P1-P3 hit and P4
  misses, yesterday's finding was a ceiling artefact and LAWS.md is wrong. If
  P4 hits, the finding survives a statistic with no ceiling and P1-P3 only
  bound how much dependence could be hiding at the top. Both outcomes get
  written down.

SELF-CHECKS (no table if any fails)
  * calibration: at lambda = 0 the rejection rate must sit between 1 % and
    12 % for every board and both groups, or the power curve is measured off a
    broken test;
  * monotonicity: the rejection rate must not fall as lambda grows by more
    than two simulation standard errors, allowing one such reversal per curve;
  * the residual-correlation null must have spread on every board and group;
  * the null must be refitted, so that both sides carry the constraint fitting
    imposes on residuals - about -1/(J-1) of mean pairwise correlation, which
    on a 24-system board is the size of the effect being tested.

    python redundancy_power.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from top_redundancy import (BOARDS, TOPK, fit_rasch, group_indices, load,
                            sigmoid, simulate, stats)

REFIT_NULLS = 299    # each null board is refitted, so this is the slow loop

SEED = 20260824
NULLS = 599
POWER_SIMS = 400
LAMBDAS = [0.0, 0.15, 0.30, 0.50, 0.75, 1.10, 1.60]
ALPHA = 0.05


def resid_corr(x, theta, b, idx, topk=TOPK):
    """Mean pairwise correlation of Rasch residuals inside the group."""
    g = idx[:topk]
    p = sigmoid(theta[g][:, None] - b[None, :])
    r = x[g] - p
    r = r - r.mean(axis=1, keepdims=True)
    sd = r.std(axis=1)
    sd = np.where(sd > 0, sd, 1.0)
    c = (r @ r.T) / (r.shape[1] * np.outer(sd, sd))
    iu = np.triu_indices(len(g), k=1)
    return float(c[iu].mean())


def _score_preserving_shift(th_j, b, lam, z, target, iters=25):
    """Newton for the delta that restores this system's expected score."""
    d = 0.0
    for _ in range(iters):
        p = sigmoid(th_j + d - b + lam * z)
        f = p.mean() - target
        if abs(f) < 1e-10:
            break
        d -= f / max((p * (1 - p)).mean(), 1e-9)
    return d


def simulate_dep(theta, b, idx, lam, rng, topk=TOPK):
    """A board whose group shares a latent item factor of size lam.

    The factor is added to every system in the group's WINDOW, and each of them
    is then shifted so its expected score is unchanged. Without that shift the
    factor pulls scores toward 0.5, the injected systems drop out of the top,
    and the test measures whoever replaced them - which is why the first
    version of this file had flat power curves.
    """
    p = sigmoid(theta[:, None] - b[None, :])
    if lam > 0:
        z = rng.normal(0.0, 1.0, b.shape[0])
        p = p.copy()
        for j in idx:
            target = float(sigmoid(theta[j] - b).mean())
            d = _score_preserving_shift(float(theta[j]), b, lam, z, target)
            p[j] = sigmoid(theta[j] + d - b + lam * z)
    return (rng.random(p.shape) < p).astype(float)


def null_draws_both(theta, b, rng, sims=REFIT_NULLS):
    """Null distributions for both groups, off the same simulated boards.

    Each board is REFITTED and its residuals taken against its own fit, so the
    null carries the same constraint the data does: fitting forces each item's
    residual sum across systems to zero, which by itself produces a mean
    pairwise correlation near -1/(J-1). Measuring the null against the true
    parameters instead left that constraint out of one side of the comparison,
    and on a 24-system board it was the whole of the effect.
    """
    out = {"top": {"u": [], "rc": []}, "mid": {"u": [], "rc": []}}
    for _ in range(sims):
        y = simulate(theta, b, rng)
        th2, b2 = fit_rasch(y)
        for kind in ("top", "mid"):
            idx = group_indices(y, kind)
            out[kind]["u"].append(stats(y, idx)["union"])
            out[kind]["rc"].append(resid_corr(y, th2, b2, idx))
    return {k: (np.array(v["u"]), np.array(v["rc"])) for k, v in out.items()}


def power_at(theta, b, kind, lam, u_null, rc_null, rng, x_shape, sims=POWER_SIMS):
    """Share of dependent boards the union test rejects at alpha."""
    lo = float(np.quantile(u_null, ALPHA))
    order = np.argsort(-sigmoid(theta[:, None] - b[None, :]).mean(axis=1))
    idx0 = (order[:max(TOPK, 30)] if kind == "top"
            else order[max(0, len(order) // 2 - TOPK // 2):][:max(TOPK, 30)])
    hits = 0
    for _ in range(sims):
        y = simulate_dep(theta, b, idx0, lam, rng)
        idx = group_indices(y, kind)
        if stats(y, idx)["union"] < lo:
            hits += 1
    return hits / sims


def interp_lambda(curve, target=0.80):
    """Smallest lambda reaching the target power, linearly interpolated."""
    for (l0, p0), (l1, p1) in zip(curve, curve[1:]):
        if p0 < target <= p1:
            return l0 + (l1 - l0) * (target - p0) / max(p1 - p0, 1e-9)
    return float("inf") if curve[-1][1] < target else curve[0][0]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    rows = {}
    for name, path in BOARDS.items():
        if not Path(path).exists():
            continue
        print(f"  {name}: fitting ...")
        x = load(path)
        theta, b = fit_rasch(x)
        rows[name] = {"x": x, "theta": theta, "b": b, "groups": {}}
        print("    null (refitting every simulated board) ...")
        nulls = null_draws_both(theta, b, rng)
        for kind in ("top", "mid"):
            print(f"    {kind}: statistic, power curve ...")
            u_null, rc_null = nulls[kind]
            idx = group_indices(x, kind)
            obs_u = stats(x, idx)["union"]
            obs_rc = resid_corr(x, theta, b, idx)
            curve = [(lam, power_at(theta, b, kind, lam, u_null, rc_null, rng, x.shape))
                     for lam in LAMBDAS]
            rows[name]["groups"][kind] = {
                "obs_u": obs_u, "u_p": float((u_null < obs_u).mean()),
                "obs_rc": obs_rc, "rc_p": float((rc_null < obs_rc).mean()),
                "rc_sd": float(rc_null.std()), "rc_med": float(np.median(rc_null)),
                "curve": curve, "lam80": interp_lambda(curve)}

    print("self-checks ...")
    bad_cal, bad_mono, bad_spread = [], [], []
    for name, v in rows.items():
        for kind, g in v["groups"].items():
            r0 = g["curve"][0][1]
            if not (0.01 <= r0 <= 0.12):
                bad_cal.append(f"{name}/{kind} {r0:.3f}")
            ps = [p for _, p in g["curve"]]
            se2 = 2.0 * (0.25 / POWER_SIMS) ** 0.5      # two standard errors
            drops = sum(1 for a, c in zip(ps, ps[1:]) if c < a - se2)
            if drops > 1:
                bad_mono.append(f"{name}/{kind} {drops}")
            if g["rc_sd"] <= 1e-12:
                bad_spread.append(f"{name}/{kind}")
    for label, bad in (("rejection at lambda = 0 inside 1-12 %", bad_cal),
                       ("power curves monotone (one reversal allowed)", bad_mono),
                       ("residual-correlation null has spread", bad_spread)):
        print(f"  [{'ok  ' if not bad else 'FAIL'}] {label}"
              + ("" if not bad else "  off: " + ", ".join(bad)))
    if bad_cal or bad_mono or bad_spread:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("WAS THE TOP-VERSUS-MIDDLE RESULT A FINDING OR A CEILING?")
    p("=" * 100)
    p(f"  {'leaderboard':<22} {'group':>6} {'union':>7} {'p':>6} {'resid r':>9} {'null':>8} "
      f"{'p':>6} {'lambda for 80 % power':>22}")
    for name, v in rows.items():
        for kind in ("top", "mid"):
            g = v["groups"][kind]
            lam = "never" if not np.isfinite(g["lam80"]) else f"{g['lam80']:.2f}"
            p(f"  {name:<22} {kind:>6} {100 * g['obs_u']:>6.1f}% {g['u_p']:>6.3f} "
              f"{g['obs_rc']:>9.4f} {g['rc_med']:>8.4f} {g['rc_p']:>6.3f} {lam:>22}")
    p("")
    p("  power curves (rejection rate of the union test against injected lambda)")
    p(f"  {'leaderboard':<22} {'group':>6} " + " ".join(f"{l:>6.2f}" for l in LAMBDAS))
    for name, v in rows.items():
        for kind in ("top", "mid"):
            ps = [f"{q:>6.2f}" for _, q in v["groups"][kind]["curve"]]
            p(f"  {name:<22} {kind:>6} " + " ".join(ps))
    p("")
    n = len(rows)
    p1 = sum(1 for v in rows.values()
             if v["groups"]["top"]["lam80"] > v["groups"]["mid"]["lam80"])
    p2 = sum(1 for v in rows.values()
             if np.isfinite(v["groups"]["mid"]["lam80"]) and v["groups"]["mid"]["lam80"] > 0
             and v["groups"]["top"]["lam80"] / v["groups"]["mid"]["lam80"] > 1.5)
    p3 = 0
    for v in rows.values():
        lam_mid = v["groups"]["mid"]["lam80"]
        if not np.isfinite(lam_mid):
            continue
        curve = dict(v["groups"]["top"]["curve"])
        near = min(curve, key=lambda l: abs(l - lam_mid))
        if curve[near] < 0.50:
            p3 += 1
    p4_mid = sum(1 for v in rows.values() if v["groups"]["mid"]["rc_p"] > 0.95)
    p4_top = sum(1 for v in rows.values() if v["groups"]["top"]["rc_p"] <= 0.95)
    p4 = p4_mid >= 2 and p4_top >= 2
    p(f"  P1  lambda for 80 % power larger at the top on {p1} of {n}    "
      f"pre-registered >= 2:  {'HIT' if p1 >= 2 else 'MISS'}")
    p(f"  P2  that ratio above 1.5 on {p2} of {n}                     "
      f"pre-registered >= 2:  {'HIT' if p2 >= 2 else 'MISS'}")
    p(f"  P3  top power below 50 % at the middle's own lambda on {p3} of {n}  "
      f"pre-registered >= 2:  {'HIT' if p3 >= 2 else 'MISS'}")
    p(f"  P4  residual correlation: significant mid on {p4_mid} of {n}, not at top on "
      f"{p4_top} of {n}   {'HIT' if p4 else 'MISS'}")
    p("")
    p("  lambda is the size of a latent item factor shared by the group, in")
    p("  logits. The power column is the smallest one the union test would")
    p("  catch four times in five; 'never' means it does not reach 80 % even at")
    p("  the largest lambda tried.")
    p("")
    p("  The residual correlation is the same question asked without a coverage")
    p("  ceiling: correlate each system's Rasch residuals with each other's")
    p("  inside the group and average over pairs. Every simulated board is")
    p("  refitted so that its residuals carry the same constraint the data's do;")
    p("  without that, a board with few systems shows complementarity that is")
    p("  only the estimator (-1/(J-1), which is -0.04 at J = 24).")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("redundancy_power_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote redundancy_power_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
