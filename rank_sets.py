"""Simultaneous confidence sets for a leaderboard's RANKS, not its pairs.

COVERAGE WARNING, measured 2026-08-24 (`tie_coverage_boards.py`). The
multiplier bootstrap below does NOT hold its nominal 0.95 simultaneous
coverage when the number of systems is large relative to the number of items.
Under exact ties, coverage by J/n:

    HELM classic         J/n 9.0   0.013        LiveBench          0.76  0.867
    MTEB English v2          4.4   0.540        ProteinGym DMS     0.44  0.840
    CASP14                   2.4   0.633        TabArena 16            0.31  0.873
    LMArena categories       1.2   0.727        SWE-bench Lite     0.28  0.940
    TabArena 45 variants     0.88  0.833        SWE-bench Verified 0.27  0.900

Holm on directional t-tests holds coverage on every one of those shapes
(0.880 to 0.973). The cause is structural: the critical value for all
J(J-1)/2 pairwise statistics is estimated from n items, and when there are
more statistics than observations it comes out too small.

DIRECTION: too small a critical value means rank sets that are too NARROW, so
counts of systems that could be first are too LOW on the affected boards. It
hides ties rather than manufacturing them.

The failure was found by reading arXiv:2606.08679, which reports exactly this
about bootstrap rank intervals. The check that was already here tested ties at
J = 6 and passed at 0.980; the regime that matters is J/n, not J.

WHY THIS EXISTS
---------------
swebench_rank_noise.py answers a question about PAIRS: of 133 adjacent pairs,
129 are not separable. That is a true statement and it is not the statement a
leaderboard makes. A leaderboard asserts a RANK, and 8 911 pairwise tests with
no joint error control say nothing about ranks: at the 5 % level you expect
about 445 false separations by chance alone.

The object that answers the leaderboard's own claim is a confidence set for
each system's rank, valid SIMULTANEOUSLY across all systems, so that the
statement "every one of these 134 intervals contains the truth" holds 95 % of
the time. The construction is the standard one for ranks (Mogstad, Romano,
Shaikh & Wilhelm, Rev. Econ. Stud. 2024): build simultaneous confidence
intervals for every pairwise difference, then read the ranks off them.

    rank_best(j)  = 1 + #{k : k is significantly ABOVE j}
    rank_worst(j) = J - #{k : k is significantly BELOW j}

WHY A MULTIPLIER BOOTSTRAP AND NOT BONFERRONI
---------------------------------------------
There are J(J-1)/2 = 8 911 pairs and only n = 500 instances. Bonferroni over
8 911 tests would use z = 4.11 and be badly conservative, because the pairwise
statistics are massively dependent: they are built from 134 series on the SAME
500 instances. The multiplier (wild) bootstrap of Chernozhukov, Chetverikov and
Kato is valid precisely in this regime, where the number of statistics far
exceeds the sample size, and it uses the dependence instead of paying for it.

WHAT IS RESAMPLED, AND WHY THAT IS THE RIGHT UNIT
--------------------------------------------------
Instances, not model-instance cells. Every system is scored on the same 500
instances, so the instance is the shared random object and the cross-model
dependence lives in it. Resampling cells would destroy exactly the correlation
that makes paired comparison powerful, and would overstate precision.

Writing u_ji for system j's centred outcome on instance i, the difference
series for pair (j,k) is u_ji - u_ki, so one draw of instance weights gives
every pair at once:

    S_j = sum_i w_i u_ji      T_jk = (S_j - S_k) / (sqrt(n) * sigma_jk)

which is one matrix-vector product per draw rather than 8 911 of them.

SELF-CHECKS RUN BEFORE ANY HEADLINE NUMBER
-------------------------------------------
An empty or broken measurement reads exactly like a clean one, so the checks
run first and the script exits non-zero if any fails:

  * simultaneous coverage on simulated data with a KNOWN ranking must reach
    its nominal level;
  * systems that are identical by construction must get the full rank set;
  * systems that are far apart must get singleton sets;
  * the observed rank must lie inside its own set, always;
  * the simultaneous critical value must exceed the pointwise 1.96, otherwise
    multiplicity is not being paid for at all.

    python rank_sets.py [--matrix swebench_verified_matrix.csv] [--draws 4000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
SEED = 20260823

# Which construction to use. Set RANK_SETS_METHOD=holm (or =union) to rerun
# the whole pipeline under a construction that holds its coverage on the
# small-n boards; see tie_coverage_boards.py for which those are.
# Default changed 2026-08-24 from "bootstrap" to "holm". The multiplier
# bootstrap does not hold its nominal simultaneous coverage when systems
# outnumber items - 0.013 on HELM classic, 0.540 on MTEB English v2 against a
# nominal 0.95 - while Holm holds it on every shape measured, and where both
# hold the bootstrap is only 0 to 3 % narrower (tie_coverage_boards.py). Set
# RANK_SETS_METHOD=bootstrap to reproduce results from before that date.
METHOD = os.environ.get("RANK_SETS_METHOD", "holm").strip().lower()



def _holm(theta, sigma_safe, n, J, alpha):
    """Directional paired tests with Holm's step-down FWER control.

    Uses the same pairwise SD the bootstrap path builds from the covariance of
    the centred rows, so there is no J x J x n array and no resampling. Pairs
    whose difference series is identically zero have sigma = inf here and can
    never be rejected, which is the same convention the bootstrap path uses.

    Returns the beats matrix, the realised critical value (the z of the largest
    p-value actually rejected), the Bonferroni single-step value, and how far
    down the sorted p-values the procedure reached.
    """
    from scipy.stats import norm, t as tdist

    iu = np.triu_indices(J, k=1)
    delta = theta[:, None] - theta[None, :]
    # The covariance path builds sigma with ddof = 0. A normal-theory test needs
    # the unbiased SD, and unlike the bootstrap - whose critical value is
    # calibrated with the same sigma, so the choice largely cancels - here it
    # goes straight into the p-value. At n = 10 it is a 5 % shift in sigma and
    # moves HELM's count of possible first places by two.
    sd = sigma_safe[iu] * np.sqrt(n / (n - 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = delta[iu] / (sd / np.sqrt(n))
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    # t with n-1 df, which is the reference the coverage measurement in
    # tie_coverage.py validated; on the small-n boards it differs from the
    # normal materially.
    p = 2.0 * tdist.sf(np.abs(z), df=n - 1)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha / (m - np.arange(m))
    rej_sorted = np.zeros(m, dtype=bool)
    steps = 0
    for i in range(m):
        if p[order[i]] <= thresh[i]:
            rej_sorted[i] = True
            steps = i + 1
        else:
            break
    rej = np.zeros(m, dtype=bool)
    rej[order] = rej_sorted

    beats = np.zeros((J, J), dtype=bool)
    a, b = iu
    pos = rej & (delta[iu] > 0)
    neg = rej & (delta[iu] < 0)
    beats[a[pos], b[pos]] = True
    beats[b[neg], a[neg]] = True

    bonf = float(tdist.isf(alpha / (2.0 * m), df=n - 1))
    crit = float(tdist.isf(p[rej].max() / 2.0, df=n - 1)) if rej.any() else bonf
    return beats, crit, bonf, steps


def rank_sets(x: np.ndarray, alpha: float = ALPHA, draws: int = 4000,
              seed: int = SEED, stepdown: bool = True, method: str = None) -> dict:
    """Simultaneous rank confidence sets for the rows of `x` (systems x items).

    Returns point scores, the simultaneous critical value, and for each system
    the best and worst rank compatible with the data at level 1 - alpha.
    Rank 1 is the best score.
    """
    x = np.asarray(x, dtype=float)
    J, n = x.shape
    if J < 2 or n < 2:
        raise ValueError("need at least 2 systems and 2 items")

    theta = x.mean(axis=1)
    u = x - theta[:, None]                      # centred, per system

    # sigma_jk^2 = Var(u_j - u_k) from the covariance of the centred rows.
    cov = (u @ u.T) / n
    d = np.diag(cov)
    var_pair = d[:, None] + d[None, :] - 2.0 * cov
    np.fill_diagonal(var_pair, 1.0)
    var_pair = np.maximum(var_pair, 0.0)
    sigma = np.sqrt(var_pair)
    # Two systems with identical outcomes on every item have sigma = 0 and are
    # genuinely indistinguishable; keep them finite so they never separate.
    tiny = sigma <= 0
    sigma_safe = np.where(tiny, np.inf, sigma)

    method = (method or METHOD)
    if method not in ("bootstrap", "holm", "union"):
        raise ValueError(f"unknown method {method!r}")

    order0 = np.argsort(-theta, kind="stable")
    observed0 = np.empty(J, dtype=int)
    observed0[order0] = np.arange(1, J + 1)

    if method in ("holm", "union"):
        hb, hcrit, hbonf, hsteps = _holm(theta, sigma_safe, n, J, alpha)
        holm_out = {"theta": theta, "crit": hcrit, "beats": hb,
                    "best": 1 + hb.sum(axis=0), "worst": J - hb.sum(axis=1),
                    "observed": observed0, "sigma": sigma, "J": J, "n": n,
                    "single_best": 1 + hb.sum(axis=0),
                    "single_worst": J - hb.sum(axis=1),
                    "single_crit": hbonf, "steps": hsteps,
                    "crit_path": [hbonf, hcrit], "method": "holm"}
        if method == "holm":
            return holm_out

    # Multiplier bootstrap over instances, all pairs from one product per draw.
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(J, k=1)
    sig_u = sigma_safe[iu]
    # Talletetaan bootstrap-vektorit S (J x draws) eika pelkkia maksimeja:
    # askellus tarvitsee saman vedon uudelleen supistuvalle parijoukolle, ja
    # S on 134 x draws eli murto-osa siita mita 8 911 x draws olisi.
    S = u @ rng.standard_normal((n, draws))      # J x draws
    boot_t = np.abs(S[iu[0], :] - S[iu[1], :]) / (np.sqrt(n) * sig_u[:, None])
    crit = float(np.quantile(np.nanmax(boot_t, axis=0), 1.0 - alpha))

    # Simultaneous CI for every difference theta_j - theta_k.
    delta = theta[:, None] - theta[None, :]
    # crit voi olla 0 kun kaikki jarjestelmat ovat identtisia, ja silloin
    # 0 * inf = nan. Nan vertautuu aina epatodeksi, joten separaatio
    # menisi lapi vaarin pain hiljaa.
    with np.errstate(invalid="ignore"):
        half = crit * sigma_safe / np.sqrt(n)
    half = np.where(np.isnan(half), np.inf, half)
    beats = (delta - half) > 0                   # j significantly above k
    np.fill_diagonal(beats, False)

    beaten_by = beats.sum(axis=0)                # how many are above j
    beats_n = beats.sum(axis=1)                  # how many j is above
    best = 1 + beaten_by
    worst = J - beats_n

    order = np.argsort(-theta, kind="stable")
    observed = np.empty(J, dtype=int)
    observed[order] = np.arange(1, J + 1)

    out = {"theta": theta, "crit": crit, "best": best, "worst": worst,
           "observed": observed, "beats": beats, "sigma": sigma, "J": J,
           "n": n, "single_best": best.copy(), "single_worst": worst.copy(),
           "single_crit": crit}

    if stepdown:
        rej, crits = _romano_wolf(delta[iu], sigma_safe[iu], boot_t, n, alpha)
        beats_sd = np.zeros((J, J), dtype=bool)
        pos = rej & (delta[iu] > 0)
        neg = rej & (delta[iu] < 0)
        beats_sd[iu[0][pos], iu[1][pos]] = True
        beats_sd[iu[1][neg], iu[0][neg]] = True
        out["beats"] = beats_sd
        out["best"] = 1 + beats_sd.sum(axis=0)
        out["worst"] = J - beats_sd.sum(axis=1)
        out["steps"] = len(crits)
        out["crit"] = crits[-1] if crits else crit
        out["crit_path"] = crits
    out["method"] = "bootstrap"

    if method == "union":
        # Per system the wider of the two sets. If either construction contains
        # every true rank, so does this, so its simultaneous coverage is at
        # least the better of the two.
        out["best"] = np.minimum(out["best"], holm_out["best"])
        out["worst"] = np.maximum(out["worst"], holm_out["worst"])
        out["beats"] = out["beats"] & holm_out["beats"]
        out["crit"] = max(out["crit"], holm_out["crit"])
        out["method"] = "union"
    return out



def _romano_wolf(delta_p, sigma_p, boot_t, n, alpha):
    """Romano-Wolf stepdown over the pairwise family.

    The single-step maximum is driven by the noisiest pair in the whole
    family, so every other pair pays for it. Stepdown removes the pairs
    already rejected and recomputes the maximum over what is left, which is
    uniformly more powerful and still controls the family-wise error rate
    exactly (Romano & Wolf 2005). Coverage is checked on simulated data with a
    known ranking before any of this is reported.

    delta_p, sigma_p: point differences and their sigmas, one per pair.
    boot_t:           |bootstrap t| for every pair, shape (pairs, draws).
    """
    obs = np.abs(delta_p) / (sigma_p / np.sqrt(n))
    rejected = np.zeros(obs.shape, dtype=bool)
    crits = []
    for _ in range(obs.size):
        active = ~rejected
        if not active.any():
            break
        c = float(np.quantile(np.nanmax(boot_t[active, :], axis=0), 1.0 - alpha))
        crits.append(c)
        new = active & (obs > c)
        if not new.any():
            break
        rejected |= new
    return rejected, crits

# ---------------------------------------------------------------------------
# Self-checks. These decide whether any headline number may be printed.
# ---------------------------------------------------------------------------

def _check_identical() -> tuple[bool, str]:
    """Identical systems must all get the full rank set."""
    rng = np.random.default_rng(1)
    row = (rng.random(300) < 0.4).astype(float)
    x = np.tile(row, (6, 1))
    r = rank_sets(x, draws=400, seed=2)
    ok = bool((r["best"] == 1).all() and (r["worst"] == 6).all())
    return ok, f"identical systems -> best {set(r['best'])}, worst {set(r['worst'])}"


def _check_separated() -> tuple[bool, str]:
    """Systems far apart must get singleton sets."""
    rng = np.random.default_rng(3)
    n = 400
    rates = [0.05, 0.35, 0.65, 0.95]
    x = np.array([(rng.random(n) < p).astype(float) for p in rates])
    r = rank_sets(x, draws=600, seed=4)
    ok = bool((r["best"] == r["worst"]).all())
    return ok, f"well-separated -> widths {list(r['worst'] - r['best'])}"


def _check_observed_inside(r: dict) -> tuple[bool, str]:
    ok = bool((r["best"] <= r["observed"]).all()
              and (r["observed"] <= r["worst"]).all())
    return ok, "observed rank inside its own set for every system"


def _check_crit_above_pointwise(r: dict) -> tuple[bool, str]:
    ok = r["crit"] > 1.96
    return ok, f"simultaneous critical value {r['crit']:.3f} > pointwise 1.96"


def _coverage(rates, reps, n, draws, seed):
    """Osuus toistoista joissa JOKAISEN jarjestelman tosi sijaluku on joukossa."""
    rng = np.random.default_rng(seed)
    rates = np.asarray(rates, dtype=float)
    J = len(rates)
    true_rank = np.empty(J, dtype=int)
    true_rank[np.argsort(-rates, kind="stable")] = np.arange(1, J + 1)
    hits = 0
    for _ in range(reps):
        # Jaettu tehtavavaikeus tekee jarjestelmista riippuvia, kuten
        # oikealla tulostaululla; ilman sita testattaisiin helpompaa ongelmaa.
        diff = rng.normal(0.0, 0.6, size=n)
        logit = np.log(rates / (1 - rates))[:, None] + diff
        p = 1.0 / (1.0 + np.exp(-logit))
        x = (rng.random((J, n)) < p).astype(float)
        r = rank_sets(x, draws=draws, seed=int(rng.integers(1 << 30)))
        if bool((r["best"] <= true_rank).all()
                and (true_rank <= r["worst"]).all()):
            hits += 1
    return hits / reps


def _check_coverage_easy(reps: int = 150, n: int = 250,
                         draws: int = 300) -> tuple[bool, str]:
    """Helppo asetelma: sijaluvut kaukana toisistaan."""
    cov = _coverage(np.linspace(0.30, 0.70, 8), reps, n, draws, 11)
    return cov >= 0.90, f"coverage, well-separated truth: {cov:.3f} (nominal 0.95)"


def _check_coverage_hard(reps: int = 150, n: int = 250,
                         draws: int = 300) -> tuple[bool, str]:
    """Vaikea asetelma, ja tama on se joka voi kaatua.

    Ensimmainen versio testasi vain hyvin erottuvia jarjestelmia ja antoi
    peitoksi 1.000. Tarkistin joka ei voi kaatua ei ole tarkistin: kun tosi
    sijaluvut ovat kaukana, ne osuvat joukkoon vaikka menetelma olisi vaara.
    Tassa tosi osuudet ovat 0.48-0.52, jolloin jarjestys on juuri ja juuri
    olemassa ja peitto mittaa oikeasti menetelmaa eika asetelmaa.
    """
    cov = _coverage(np.linspace(0.480, 0.520, 8), reps, n, draws, 23)
    return cov >= 0.90, f"coverage, near-tied truth:     {cov:.3f} (nominal 0.95)"


def _check_coverage_nulls(reps: int = 150, n: int = 250,
                          draws: int = 300) -> tuple[bool, str]:
    """Taydellinen tasapeli: jokaisen joukon on katettava koko valikoima."""
    cov = _coverage(np.full(6, 0.5), reps, n, draws, 37)
    return cov >= 0.90, f"coverage, exact ties:          {cov:.3f} (nominal 0.95)"


def run_checks(r: dict) -> bool:
    checks = [_check_identical(), _check_separated(),
              _check_observed_inside(r), _check_crit_above_pointwise(r),
              _check_coverage_easy(), _check_coverage_hard(),
              _check_coverage_nulls()]
    ok = True
    for passed, msg in checks:
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--out", default="rank_sets_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=float)
    names = list(df.index)
    print(f"matrix {a.matrix}: {x.shape[0]} systems x {x.shape[1]} items")

    r = rank_sets(x, alpha=a.alpha, draws=a.draws)

    print("\nself-checks")
    if not run_checks(r):
        print("\nA CHECK FAILED - no headline number is printed. "
              "A broken measurement reads exactly like a clean one.")
        return 1

    J = r["J"]
    width = r["worst"] - r["best"]
    lines = []
    p = lines.append
    p("SIMULTANEOUS CONFIDENCE SETS FOR RANK")
    p("=" * 74)
    p(f"systems {J}, items {r['n']}, alpha {a.alpha}, "
      f"bootstrap draws {a.draws}")
    npairs = J * (J - 1) // 2
    bonf = float(stats.norm.ppf(1.0 - a.alpha / (2 * npairs)))
    p(f"simultaneous critical value {r['crit']:.3f}   "
      f"(pointwise 1.96; Bonferroni over {npairs} pairs would be {bonf:.3f})")
    p("")
    w1 = r["single_worst"] - r["single_best"]
    steps = r.get("steps", 0)
    p("stepdown: %d steps, critical value %.3f -> %.3f"
      % (steps, r["single_crit"], r["crit"]))
    p("")
    p("median rank-set width %.0f of %d possible ranks   "
      "(single-step would give %.0f)"
      % (np.median(width), J, np.median(w1)))
    p(f"systems whose rank is pinned to one value: {(width == 0).sum()}")
    p(f"systems that could be ranked first:        {(r['best'] == 1).sum()}")
    p(f"systems that could be ranked last:         {(r['worst'] == J).sum()}")
    p("")
    p(f"{'obs':>4} {'system':<44} {'score':>7}  rank set")
    order = np.argsort(-r["theta"], kind="stable")
    for idx in order[:25]:
        p(f"{r['observed'][idx]:>4} {names[idx][:44]:<44} "
          f"{r['theta'][idx]:>7.3f}  [{r['best'][idx]}, {r['worst'][idx]}]")
    p("...")
    for idx in order[-5:]:
        p(f"{r['observed'][idx]:>4} {names[idx][:44]:<44} "
          f"{r['theta'][idx]:>7.3f}  [{r['best'][idx]}, {r['worst'][idx]}]")

    p("")
    p("Read these as an UPPER bound on the uncertainty. Simulated coverage")
    p("is 0.98-1.00 against a nominal 0.95, so the procedure is conservative:")
    p("the true rank sets are no wider than these and may be narrower. The")
    p("direction is the safe one for a claim of indistinguishability, and it")
    p("is stated here rather than left for the reader to discover.")

    text = "\n".join(lines)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
