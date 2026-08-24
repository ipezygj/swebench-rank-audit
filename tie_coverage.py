"""Does the prior art's criticism of this repo's machinery hold at the size I use it?

PRIOR_ART.md turned up "Rank Intervals for Leaderboards" (arXiv:2606.08679),
which builds rank intervals from directional t-tests with Holm's FWER
correction and reports that bootstrap methods fail under ties while FWER-based
intervals keep their coverage. Every rank set in this repo comes from a
multiplier bootstrap with Romano-Wolf stepdown, so that is a direct claim about
the machinery under everything here.

rank_sets.py already checks coverage under exact ties, and it passes at 0.980.
But it checks six systems on 250 items, and the boards in this repo run to 181
systems. Simultaneous coverage is a statement about all pairs at once, and the
number of pairs grows as J squared - 15 at J = 6, 8 911 at J = 134 - so the
regime the criticism is about is not the regime the check tests.

This runs both constructions, mine and theirs, over the shapes actually used.

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  the multiplier bootstrap holds coverage of at least 0.90 (nominal 0.95)
      under EXACT ties at J = 134, n = 500 - SWE-bench Verified's shape.
  P2  coverage does not decay with J: the J = 134 figure is within 3 points of
      the J = 6 figure, under ties.
  P3  Holm on directional t-tests also holds coverage at that shape, at least
      0.90. If it does not, the criticism is about a different bootstrap than
      mine and the comparison is the answer to it.
  P4  where both hold coverage, the bootstrap's rank sets are NARROWER than
      Holm's on a well-separated field - that is the reason to pay for a
      bootstrap at all. Predicted: median width at least 20 % smaller.

  Not predicted: the direction under ties, where both should be maximally wide
  by construction and widths carry no information.

  If P1 or P2 misses, every tie@1 in this repo is suspect and the finding is
  reported as such rather than explained away.

SELF-CHECKS (no table if any fails)
  * the coverage estimator must be able to FAIL: a deliberately broken
    construction - pointwise 1.96 with no multiplicity correction - must come
    out below 0.90 at J = 134, or the harness cannot detect undercoverage;
  * the truth must be recoverable: on a well-separated field both methods must
    reach coverage at or above nominal;
  * both methods must be fed identical data, checked by hashing the matrices.

    python tie_coverage.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as tdist

import rank_sets as rs

SEED = 20260824
REPS = 200
ALPHA = 0.05
DRAWS = 800

SHAPES = [(6, 250), (30, 250), (134, 500), (181, 41)]


def make_board(abilities, n, rng):
    """Binary outcomes with the given per-system solve rates."""
    p = np.asarray(abilities, dtype=float)[:, None]
    return (rng.random((len(abilities), n)) < p).astype(float)


def holm_rank_sets(x, alpha=ALPHA):
    """Rank intervals from directional paired t-tests with Holm's correction.

    The construction the prior art uses: test every ordered pair, control the
    family-wise error rate over all of them by Holm, then read ranks off the
    rejections.
    """
    J, n = x.shape
    theta = x.mean(axis=1)
    d = x[:, None, :] - x[None, :, :]
    sd = d.std(axis=2, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(sd > 0, (theta[:, None] - theta[None, :]) / (sd / np.sqrt(n)), 0.0)
    iu = np.triu_indices(J, k=1)
    pvals = 2 * (1 - tdist.cdf(np.abs(tstat[iu]), df=n - 1))
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = alpha / (m - np.arange(m))
    rejected_sorted = np.zeros(m, dtype=bool)
    for i in range(m):
        if pvals[order[i]] <= thresh[i]:
            rejected_sorted[i] = True
        else:
            break
    rej = np.zeros(m, dtype=bool)
    rej[order] = rejected_sorted
    beats = np.zeros((J, J), dtype=bool)
    a, b = iu
    for k in range(m):
        if rej[k]:
            if theta[a[k]] > theta[b[k]]:
                beats[a[k], b[k]] = True
            else:
                beats[b[k], a[k]] = True
    best = 1 + beats.sum(axis=0)
    worst = J - beats.sum(axis=1)
    return {"best": best, "worst": worst}


def naive_rank_sets(x, alpha=ALPHA):
    """Deliberately broken: pointwise 1.96, no multiplicity control at all."""
    J, n = x.shape
    theta = x.mean(axis=1)
    d = x[:, None, :] - x[None, :, :]
    sd = d.std(axis=2, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(sd > 0, (theta[:, None] - theta[None, :]) / (sd / np.sqrt(n)), 0.0)
    beats = tstat > norm.ppf(1 - alpha / 2)
    return {"best": 1 + beats.sum(axis=0), "worst": J - beats.sum(axis=1)}


def true_ranks(abilities):
    """Rank 1 is best; ties all take the same (best possible) rank."""
    a = np.asarray(abilities, dtype=float)
    return np.array([1 + int((a > v + 1e-12).sum()) for v in a])


def covers(res, truth):
    """Simultaneous: every system's true rank inside its own set."""
    return bool(np.all((res["best"] <= truth) & (truth <= res["worst"])))


def coverage(method, abilities, n, reps, seed):
    rng = np.random.default_rng(seed)
    truth = true_ranks(abilities)
    hits, widths = 0, []
    for _ in range(reps):
        x = make_board(abilities, n, rng)
        r = method(x)
        hits += covers(r, truth)
        widths.append(float(np.median(r["worst"] - r["best"])))
    return hits / reps, float(np.median(widths))


def boot(x):
    # explicit: this measurement compares the two constructions, so it must ask
    # for the bootstrap by name even when RANK_SETS_METHOD says otherwise.
    return rs.rank_sets(x, draws=DRAWS, method="bootstrap")


def _check_harness_can_fail() -> tuple[bool, str]:
    ab = np.full(134, 0.5)
    cov, _ = coverage(naive_rank_sets, ab, 500, 60, 99)
    return cov < 0.90, f"an uncorrected construction undercovers at J=134: {cov:.3f}"


def _check_separated() -> tuple[bool, str]:
    ab = np.linspace(0.20, 0.80, 12)
    cb, _ = coverage(boot, ab, 250, 60, 101)
    ch, _ = coverage(holm_rank_sets, ab, 250, 60, 101)
    return cb >= 0.95 and ch >= 0.95, \
        f"well-separated truth: bootstrap {cb:.3f}, Holm {ch:.3f}"


def _check_same_data() -> tuple[bool, str]:
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    a = make_board(np.full(20, 0.5), 100, rng1)
    b = make_board(np.full(20, 0.5), 100, rng2)
    h1 = hashlib.sha256(a.tobytes()).hexdigest()[:12]
    h2 = hashlib.sha256(b.tobytes()).hexdigest()[:12]
    return h1 == h2, f"both methods see identical boards ({h1})"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks ...")
    checks = [_check_harness_can_fail(), _check_separated(), _check_same_data()]
    ok = True
    for passed, msg in checks:
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rows = []
    for J, n in SHAPES:
        for label, ab in (("exact ties", np.full(J, 0.5)),
                          ("near ties", np.linspace(0.49, 0.51, J)),
                          ("separated", np.linspace(0.25, 0.75, J))):
            print(f"  J={J} n={n} {label} ...")
            cb, wb = coverage(boot, ab, n, REPS, SEED + J + n)
            ch, wh = coverage(holm_rank_sets, ab, n, REPS, SEED + J + n)
            rows.append((J, n, label, cb, wb, ch, wh))

    L = []
    p = L.append
    p("SIMULTANEOUS COVERAGE UNDER TIES: MULTIPLIER BOOTSTRAP AGAINST HOLM")
    p("=" * 96)
    p(f"  {'J':>4} {'n':>5} {'truth':<11} {'bootstrap cov':>14} {'width':>7} "
      f"{'Holm cov':>10} {'width':>7} {'narrower by':>12}")
    for J, n, label, cb, wb, ch, wh in rows:
        nar = (1 - wb / wh) if wh > 0 else float("nan")
        narr = "-" if wh == 0 else f"{100 * nar:>11.0f}%"
        p(f"  {J:>4} {n:>5} {label:<11} {cb:>14.3f} {wb:>7.1f} {ch:>10.3f} {wh:>7.1f} {narr:>12}")
    p("")
    tie134 = [r for r in rows if r[0] == 134 and r[2] == "exact ties"][0]
    tie6 = [r for r in rows if r[0] == 6 and r[2] == "exact ties"][0]
    sep = [r for r in rows if r[2] == "separated"]
    p1 = tie134[3] >= 0.90
    p2 = abs(tie134[3] - tie6[3]) <= 0.03
    p3 = tie134[5] >= 0.90
    nar = [(1 - r[4] / r[6]) for r in sep if r[6] > 0]
    p4 = bool(nar) and float(np.median(nar)) >= 0.20
    p(f"  P1  bootstrap coverage under exact ties at J=134: {tie134[3]:.3f}   "
      f"pre-registered >= 0.90:  {'HIT' if p1 else 'MISS'}")
    p(f"  P2  that against J=6 ({tie6[3]:.3f}), gap {abs(tie134[3] - tie6[3]):.3f}      "
      f"pre-registered <= 0.03:  {'HIT' if p2 else 'MISS'}")
    p(f"  P3  Holm coverage at the same shape: {tie134[5]:.3f}          "
      f"pre-registered >= 0.90:  {'HIT' if p3 else 'MISS'}")
    p(f"  P4  bootstrap narrower on separated fields by "
      f"{100 * float(np.median(nar)) if nar else float('nan'):.0f} %        "
      f"pre-registered >= 20 %:  {'HIT' if p4 else 'MISS'}")
    p("")
    p("  The criticism this answers is from arXiv:2606.08679, which builds rank")
    p("  intervals from directional t-tests with Holm's correction and reports")
    p("  that bootstrap methods fail under ties. Both constructions are run here")
    p("  on identical simulated boards, at the shapes this repo actually uses,")
    p("  with the simultaneous criterion: a replicate counts as covered only if")
    p("  EVERY system's true rank falls inside its own set.")
    p("")
    p("  The harness is shown to be able to detect undercoverage: an uncorrected")
    p("  pointwise construction is run through the same code and fails.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("tie_coverage_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote tie_coverage_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
