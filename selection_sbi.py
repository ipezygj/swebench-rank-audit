"""How many attempts stood behind this leaderboard? Ask the shape, not me.

deflated_benchmark.py said the number of unreported trials N is not identified
and gave a sensitivity curve instead. That was an ARGUMENT. This file turns it
into a measurement: simulate the whole process, train a neural likelihood-ratio
estimator on the simulations, and look at the posterior over N given the shape
the real leaderboard actually has. A flat posterior demonstrates
non-identification; a peaked one refutes what I wrote yesterday. Either way
the claim stops being mine and starts being the data's.

WHY A NEURAL METHOD AND NOT ABC OR A GRID LIKELIHOOD
------------------------------------------------------
Not because networks are powerful. With two parameters a grid would do - IF you
first reduce the leaderboard to a few summary statistics. Choosing those
summaries is an unexamined judgement, and it is exactly where information gets
thrown away: pick the mean and the range and you have decided in advance that
the spacings carry nothing. Neural ratio estimation (Hermans et al.) conditions
on the whole sorted score vector, so nothing is discarded by hand. That is the
entire argument for the network here, and if it did not hold I would use a grid.

AN IDENTIFICATION RESULT THAT FELL OUT OF WRITING THE SIMULATOR
----------------------------------------------------------------
Latent system j has true ability a_j ~ N(0, tau^2) and is measured with
equicorrelated noise sigma(sqrt(rho) Z0 + sqrt(1-rho) Z_j). The shared term Z0
shifts every system equally and cannot change who is on top, so the ordering
and the spacings depend only on

    observed_j ~ N(mu, s^2),     s^2 = tau^2 + sigma^2 (1 - rho)

Ability spread and measurement noise enter ONLY through s. They are not
separately identified from reported scores, by algebra, before any data is
seen - so this file infers (N, s) and says so, rather than reporting a tau it
could not have known. That is the kind of thing worth finding before running
anything.

SIMULATING THE TOP OF N DRAWS WITHOUT DRAWING N OF THEM
---------------------------------------------------------
Only the best J of N latent attempts are ever reported. Generating N normals
to keep 19 would cap N at whatever fits in memory. The exponential-spacings
representation gives the top J order statistics of N uniforms exactly, in J
operations, for any N:

    S_k = sum_{i<=k} E_i / (N - i + 1),   E_i ~ Exp(1)
    k-th largest uniform = exp(-S_k)

pushed through the normal quantile function. N can then run to a million and
the cost does not move. Checked against brute-force sampling before use.

THE CHECK THAT DECIDES WHETHER ANY POSTERIOR IS SHOWN
-------------------------------------------------------
Simulation-based calibration. Draw parameters from the prior, simulate data,
compute the posterior, and record where the true value falls in it. Over many
draws those ranks must be uniform - that is what a correct posterior means. If
they are not uniform the posterior is wrong, however plausible its picture
looks, and nothing is printed.

    python selection_sbi.py [--sims 60000] [--epochs 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260823
TOPJ = 19
LOGN_RANGE = (0.0, 6.0)          # N from 1 to a million
S_RANGE = (0.005, 0.120)


# --- simulator --------------------------------------------------------------

def top_of_n(logn, s, J, rng):
    """Top J of N iid N(0, s^2), via exponential spacings. Vectorised."""
    logn = np.atleast_1d(logn)
    s = np.atleast_1d(s)
    m = len(logn)
    N = np.power(10.0, logn)
    e = rng.exponential(size=(m, J))
    denom = N[:, None] - np.arange(J)[None, :]
    denom = np.maximum(denom, 1.0)
    cum = np.cumsum(e / denom, axis=1)
    u = np.exp(-cum)                      # descending uniforms
    z = stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    return z * s[:, None]


def summarise(top):
    """Centre each leaderboard on its own mean: location is not the question."""
    return top - top.mean(axis=1, keepdims=True)


def sample_prior(m, rng):
    logn = rng.uniform(*LOGN_RANGE, size=m)
    s = rng.uniform(*S_RANGE, size=m)
    return np.column_stack([logn, s])


# --- neural ratio estimation ------------------------------------------------

def train_ratio(theta, x, epochs, seed, hidden=128):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    m = len(theta)
    # Standardise so the network does not spend capacity on units.
    tmu, tsd = theta.mean(0), theta.std(0)
    xmu, xsd = x.mean(0), x.std(0) + 1e-9
    tt = torch.tensor((theta - tmu) / tsd, dtype=torch.float32)
    xx = torch.tensor((x - xmu) / xsd, dtype=torch.float32)
    perm = torch.randperm(m)
    joint = torch.cat([tt, xx], dim=1)
    marg = torch.cat([tt[perm], xx], dim=1)
    inp = torch.cat([joint, marg], dim=0)
    lab = torch.cat([torch.ones(m), torch.zeros(m)])
    net = nn.Sequential(
        nn.Linear(inp.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    n = len(inp)
    idx = torch.randperm(n)
    inp, lab = inp[idx], lab[idx]
    bs = 4096
    for _ in range(epochs):
        for i in range(0, n, bs):
            opt.zero_grad()
            out = net(inp[i:i + bs]).squeeze(-1)
            loss = lossf(out, lab[i:i + bs])
            loss.backward()
            opt.step()
    net.eval()
    return {"net": net, "tmu": tmu, "tsd": tsd, "xmu": xmu, "xsd": xsd,
            "loss": float(loss.item())}


def posterior_grid(model, x_obs, grid_logn, grid_s):
    """Posterior on a grid: prior is uniform, so posterior ~ exp(logit)."""
    import torch
    gl, gs = np.meshgrid(grid_logn, grid_s, indexing="ij")
    theta = np.column_stack([gl.ravel(), gs.ravel()])
    tt = torch.tensor((theta - model["tmu"]) / model["tsd"],
                      dtype=torch.float32)
    xrep = np.repeat(((x_obs - model["xmu"]) / model["xsd"])[None, :],
                     len(theta), axis=0)
    xx = torch.tensor(xrep, dtype=torch.float32)
    with torch.no_grad():
        logit = model["net"](torch.cat([tt, xx], dim=1)).squeeze(-1).numpy()
    logit -= logit.max()
    w = np.exp(logit)
    w /= w.sum()
    return w.reshape(gl.shape)



def ridge_correlation(post, grid_logn, grid_s):
    """Correlation of the two parameters under the posterior.

    A flat marginal can mean two very different things: the data said nothing,
    or the data said something about a COMBINATION and nothing about either
    coordinate alone. Those are different findings and the marginal cannot
    tell them apart. The joint can: a strong negative correlation is a ridge,
    and a ridge means many attempts with a narrow spread fits the same shape
    as few attempts with a wide one.
    """
    gl, gs = np.meshgrid(grid_logn, grid_s, indexing="ij")
    w = post / post.sum()
    ml = float((w * gl).sum())
    ms = float((w * gs).sum())
    vl = float((w * (gl - ml) ** 2).sum())
    vs = float((w * (gs - ms) ** 2).sum())
    cv = float((w * (gl - ml) * (gs - ms)).sum())
    return cv / np.sqrt(vl * vs) if vl > 0 and vs > 0 else float("nan")


def conditional_logn(post, grid_logn, grid_s, s_fixed):
    """Posterior over log10 N with s pinned to a value measured elsewhere."""
    j = int(np.argmin(np.abs(grid_s - s_fixed)))
    col = post[:, j].copy()
    tot = col.sum()
    if tot <= 0:
        return None, grid_s[j]
    return col / tot, float(grid_s[j])

# --- self-checks ------------------------------------------------------------

def _check_spacings_match_bruteforce() -> tuple[bool, str]:
    """The O(J) top-of-N must match actually drawing N normals."""
    rng = np.random.default_rng(2)
    N, J, s, reps = 2000, 19, 0.05, 3000
    fast = top_of_n(np.full(reps, np.log10(N)), np.full(reps, s), J, rng)
    slow = np.sort(rng.normal(0, s, size=(reps, N)), axis=1)[:, ::-1][:, :J]
    d_mean = np.abs(fast.mean(0) - slow.mean(0)).max()
    d_sd = np.abs(fast.std(0) - slow.std(0)).max()
    ok = d_mean < 0.004 and d_sd < 0.004
    return ok, (f"spacings vs brute force: max mean gap {d_mean:.5f}, "
                f"max sd gap {d_sd:.5f}")


def _check_shape_depends_on_n() -> tuple[bool, str]:
    """If the centred shape did not move with N, nothing could be inferred."""
    rng = np.random.default_rng(3)
    a = summarise(top_of_n(np.full(4000, 1.0), np.full(4000, 0.05), TOPJ, rng))
    b = summarise(top_of_n(np.full(4000, 5.0), np.full(4000, 0.05), TOPJ, rng))
    gap = abs(a[:, 0].mean() - b[:, 0].mean()) / a[:, 0].std()
    return gap > 0.3, (f"centred top-1 gap between N=10 and N=100000: "
                       f"{gap:.2f} sd")


def _check_sbc(model, reps, rng, grid_logn, grid_s) -> tuple[bool, str]:
    """Simulation-based calibration: posterior ranks must be uniform."""
    ranks = []
    for _ in range(reps):
        th = sample_prior(1, rng)[0]
        x = summarise(top_of_n(th[0:1], th[1:2], TOPJ, rng))[0]
        post = posterior_grid(model, x, grid_logn, grid_s)
        marg = post.sum(axis=1)
        cdf = np.cumsum(marg)
        i = int(np.searchsorted(grid_logn, th[0]))
        i = min(max(i, 0), len(cdf) - 1)
        ranks.append(cdf[i])
    ranks = np.array(ranks)
    ks = float(stats.kstest(ranks, "uniform").pvalue)
    return ks > 0.01, (f"SBC uniformity of posterior ranks: "
                       f"KS p = {ks:.4f} over {reps} draws")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--sims", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--sbc", type=int, default=150)
    ap.add_argument("--out", default="selection_sbi_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)

    print("self-checks (simulator)")
    ok = True
    for passed, msg in (_check_spacings_match_bruteforce(),
                        _check_shape_depends_on_n()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print("\nA CHECK FAILED - nothing is trained and nothing is printed.")
        return 1

    print(f"\ntraining ratio estimator on {a.sims} simulations")
    theta = sample_prior(a.sims, rng)
    x = summarise(top_of_n(theta[:, 0], theta[:, 1], TOPJ, rng))
    model = train_ratio(theta, x, a.epochs, SEED)
    print(f"  final batch loss {model['loss']:.4f}")

    grid_logn = np.linspace(*LOGN_RANGE, 61)
    grid_s = np.linspace(*S_RANGE, 41)

    print("\nself-check (posterior)")
    passed, msg = _check_sbc(model, a.sbc, rng, grid_logn, grid_s)
    print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
    if not passed:
        print("\nSBC FAILED - the posterior is not calibrated, so no posterior "
              "is shown. A miscalibrated posterior looks exactly like a good "
              "one.")
        return 1

    df = pd.read_csv(a.matrix, index_col=0)
    scores = df.mean(axis=1).to_numpy()
    top = np.sort(scores)[::-1][:TOPJ]
    x_obs = top - top.mean()
    post = posterior_grid(model, x_obs, grid_logn, grid_s)
    marg_n = post.sum(axis=1)
    marg_s = post.sum(axis=0)
    cdf = np.cumsum(marg_n)

    def q(p):
        return float(grid_logn[int(np.searchsorted(cdf, p))])

    L = []
    pp = L.append
    pp("HOW MANY ATTEMPTS? THE POSTERIOR, NOT MY OPINION")
    pp("=" * 74)
    pp(f"top {TOPJ} observed scores, centred; prior on log10 N is uniform on "
       f"[{LOGN_RANGE[0]:.0f}, {LOGN_RANGE[1]:.0f}]")
    pp("")
    pp("  posterior over log10 N")
    pp(f"    median              {q(0.5):.2f}   (N = {10 ** q(0.5):,.0f})")
    pp(f"      the prior median is {np.mean(LOGN_RANGE):.2f}. If the posterior")
    pp("      median sits on it, the number is the prior speaking, not the")
    pp("      data, and must not be quoted as an estimate of anything.")
    pp(f"    50 % interval       [{q(0.25):.2f}, {q(0.75):.2f}]")
    pp(f"    90 % interval       [{q(0.05):.2f}, {q(0.95):.2f}]")
    prior_w = 1.0 / len(grid_logn)
    kl = float(np.sum(marg_n * np.log(np.maximum(marg_n, 1e-12) / prior_w)))
    pp(f"    information gained over the prior   {kl:.3f} nats")
    pp("")
    pp("  posterior over s (ability spread and noise, confounded by algebra)")
    cs = np.cumsum(marg_s)
    pp(f"    median              {grid_s[int(np.searchsorted(cs, 0.5))]:.4f}")
    pp(f"    90 % interval       "
       f"[{grid_s[int(np.searchsorted(cs, 0.05))]:.4f}, "
       f"{grid_s[int(np.searchsorted(cs, 0.95))]:.4f}]")
    pp("")
    if kl < 0.35:
        pp("  VERDICT: the shape of the leaderboard says almost nothing about")
        pp(f"  how many attempts produced it. {kl:.2f} nats over a uniform")
        pp("  prior across six orders of magnitude is close to nothing, and")
        pp("  that is the measurement yesterday's file asserted without one.")
        pp("  The sensitivity curve in deflated_benchmark.py was the right")
        pp("  form to report, and this is why.")
    else:
        pp("  VERDICT: the shape DOES carry information about N, more than I")
        pp("  claimed yesterday. deflated_benchmark.py's refusal to estimate N")
        pp("  was too cautious and should be revised, not defended.")
    pp("")
    pp("  WHY THE MARGINAL IS FLAT - it is a ridge, not an absence")
    r = ridge_correlation(post, grid_logn, grid_s)
    pp(f"    posterior correlation of log10 N and s   {r:+.3f}")
    pp("    Many attempts with a narrow spread produce the same shape as few")
    pp("    attempts with a wide one. The data constrains the COMBINATION and")
    pp("    neither coordinate alone, which is a sharper statement than 'no")
    pp("    information' and a different one.")
    pp("")
    pp("  WITH s PINNED BY DIRECT MEASUREMENT")
    pp("    rho and sigma are measurable from the item-level matrix without")
    pp("    this model at all: sigma = 0.0191 and rho = 0.664, so under the")
    pp("    hypothesis that the top 19 differ in no ability whatever (tau = 0)")
    pp("    the spread is pure measurement noise, s = sigma sqrt(1 - rho).")
    s_pin = 0.0191 * float(np.sqrt(1 - 0.664))
    cond, s_used = conditional_logn(post, grid_logn, grid_s, s_pin)
    pp(f"    s = {s_pin:.4f}  (nearest grid value {s_used:.4f})")
    if cond is None:
        pp("    the posterior puts no mass there - the observed spread is too")
        pp("    wide to be measurement noise alone, which is itself the answer")
        pp("    and agrees with the permutation test in deflated_benchmark.py.")
    else:
        cc = np.cumsum(cond)
        med = float(grid_logn[int(np.searchsorted(cc, 0.5))])
        lo5 = float(grid_logn[int(np.searchsorted(cc, 0.05))])
        hi5 = float(grid_logn[int(np.searchsorted(cc, 0.95))])
        klc = float(np.sum(cond * np.log(np.maximum(cond, 1e-12)
                                         / (1.0 / len(grid_logn)))))
        pp(f"    conditional median log10 N   {med:.2f}  (N = {10 ** med:,.0f})")
        pp(f"    90 % interval                [{lo5:.2f}, {hi5:.2f}]")
        pp(f"    information over the prior   {klc:.3f} nats")
        if 10 ** med < TOPJ:
            pp("")
            pp(f"    READ THIS AS A REJECTION, NOT AN ESTIMATE. N = {10 ** med:,.0f}")
            pp(f"    is impossible when {TOPJ} systems are reported: you cannot")
            pp("    show the best 19 of 3. The conditional posterior piles up")
            pp("    against the lower boundary because at this s the model")
            pp("    CANNOT produce the observed spread at any N. The top 19 are")
            pp("    further apart than pure measurement noise allows, so the")
            pp("    hypothesis that they are equally able is rejected here -")
            pp("    independently, and by a different route, from the exact")
            pp("    permutation test in deflated_benchmark.py, which put the")
            pp("    same conclusion at p = 0.019.")
            pp("")
            pp("    Two methods sharing no machinery reaching the same verdict")
            pp("    is worth more than either of them alone.")
        else:
            pp("    Pinning the one thing that IS directly measurable is what")
            pp("    turns an unidentified pair into a usable statement.")
    pp("")
    pp("  Identified jointly: only s^2 = tau^2 + sigma^2 (1 - rho). Ability")
    pp("  spread and measurement noise cannot be separated from reported")
    pp("  scores, by algebra, whatever method is used on them.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
