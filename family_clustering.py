"""Is the crowded top a base-model cluster, or just a crowded calendar?

top_compression.py showed that what makes a leaderboard's top undecidable is
the SHAPE of the field there and not the correlation between systems, and left
open where the shape comes from. SWE-bench Verified offers a candidate in
plain sight: its four highest submissions all run claude-4.5 and span 4.4
points, while the board as a whole spans seventy.

The candidate has an obvious confound, and it is the whole difficulty of the
question. Base-model families are also calendar cohorts - every claude-3
submission is old and low, every claude-4.5 submission is new and high - so
same-family pairs are close in score partly because they are close in time.
Two nulls separate the two readings:

  free       family labels permuted across all labelled systems. Rejects if
             family membership carries ANY information about proximity.
  by date    labels permuted only within a calendar quarter, so each shuffled
             board keeps the real relation between date and score and moves
             only which family a submission of that quarter belongs to.

An effect that survives the second is about the model. An effect that appears
only in the first is about the calendar, and the honest conclusion would be
that "base-model cluster" is a restatement of "recent systems are close".

Two statistics per pair, both already defined elsewhere in this repo and
imported rather than reimplemented: the score gap, and kappa from
pair_sharpness.py, which is scale-free and says how sharp the paired
comparison is rather than how far apart the two sit.

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  same-family pairs are closer in score than the FREE null, p < 0.05.
  P2  same-family pairs have lower kappa - sharper comparisons, correlated
      errors - than the FREE null, p < 0.05.
  P3  same-family pairs are still closer in score than the BY-DATE null,
      p < 0.05. This is the one I am genuinely unsure of; a miss here says
      the cluster is a cohort and the family label adds nothing.
  P4  among the top 20 systems, the share of labelled pairs that share a
      family exceeds the by-date null's 95th percentile.

  Not predicted: any direction for kappa under the by-date null, and nothing
  about the 72 unlabelled submissions.

  Note against my own interest: 72 of 134 submissions do not name a base
  model and are dropped, not guessed. Dropping them removes pairs that may
  well be same-family, which can only weaken every effect above.

SELF-CHECKS (no table if any fails)
  * kappa must come from pair_sharpness.kappa_matrix and match a direct
    recomputation on twenty random pairs to 1e-12;
  * the permutation machinery must be centred: fed a RANDOM family labelling
    of the same sizes, its p-values must be roughly uniform - mean between
    0.3 and 0.7 over 20 relabellings - or every p below is a reading of the
    machinery;
  * the date strata must be non-degenerate: at least three quarters must
    contain two or more families, otherwise the by-date null cannot permute
    anything and P3 is vacuous;
  * P4's null must have spread. Permuting labels inside the top group alone
    leaves its same-family count exactly invariant - 500 draws gave one value,
    sd 1e-17 - so the null runs over the whole labelled board and the check
    refuses the table if it still cannot move.

    python family_clustering.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pair_sharpness import kappa_matrix
from sota_audit import parse_dates
from swebench_base_models import base_model

MATRIX = "swebench_verified_matrix.csv"
SEED = 20260824
PERMS = 999
TOPK = 20


def load():
    df = pd.read_csv(MATRIX, index_col=0)
    x = df.to_numpy(dtype=float)
    names = list(df.index)
    fam = np.array([base_model(n) or "" for n in names], dtype=object)
    dates = parse_dates(names)
    return x, names, fam, dates


def quarter(d) -> str:
    """parse_dates returns YYYYMMDD as an int, not a timestamp.

    Reading it as one put every submission in 1970Q1, which the strata check
    caught: a by-date null with one stratum is the free null wearing a label,
    and P3 would have silently repeated P1.
    """
    d = int(d)
    return f"{d // 10000}Q{((d // 100 % 100) - 1) // 3 + 1}"


def pair_stats(x, keep):
    """Score gaps and kappa for every pair among the kept systems."""
    sub = x[keep]
    sc = sub.mean(axis=1)
    K = kappa_matrix(sub)
    m = len(keep)
    iu = np.triu_indices(m, k=1)
    gap = np.abs(sc[iu[0]] - sc[iu[1]])
    return iu, gap, K[iu]


def same_mask(fam_kept, iu):
    return fam_kept[iu[0]] == fam_kept[iu[1]]


def permute_free(fam_kept, rng):
    p = fam_kept.copy()
    rng.shuffle(p)
    return p


def permute_by_stratum(fam_kept, strata, rng):
    p = fam_kept.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if len(idx) > 1:
            p[idx] = rng.permutation(p[idx])
    return p


def perm_p(observed, draws, lower_is_extreme=True):
    d = np.asarray(draws, dtype=float)
    hits = (d <= observed).sum() if lower_is_extreme else (d >= observed).sum()
    return float((hits + 1) / (len(d) + 1))


def run(fam_kept, iu, gap, kap, strata, rng, perms=PERMS):
    """Observed same-family means and their two permutation distributions."""
    sm = same_mask(fam_kept, iu)
    if sm.sum() == 0:
        return None
    obs = {"n_same": int(sm.sum()), "gap": float(gap[sm].mean()),
           "kap": float(kap[sm].mean()),
           "gap_diff": float(gap[~sm].mean()), "kap_diff": float(kap[~sm].mean())}
    out = {"obs": obs}
    for tag, maker in (("free", lambda: permute_free(fam_kept, rng)),
                       ("date", lambda: permute_by_stratum(fam_kept, strata, rng))):
        g, k = [], []
        for _ in range(perms):
            pm = maker()
            m = same_mask(pm, iu)
            if m.sum() == 0:
                continue
            g.append(gap[m].mean())
            k.append(kap[m].mean())
        out[tag] = {"gap_p": perm_p(obs["gap"], g), "kap_p": perm_p(obs["kap"], k),
                    "gap_med": float(np.median(g)), "kap_med": float(np.median(k))}
    return out


def _check_kappa_source(x, keep) -> tuple[bool, str]:
    sub = x[keep]
    K = kappa_matrix(sub)
    c = sub - sub.mean(axis=0, keepdims=True)
    sd = c.std(axis=1, ddof=1)
    rng = np.random.default_rng(1)
    worst = 0.0
    for a, b in rng.integers(0, len(keep), (20, 2)):
        if a == b:
            continue
        direct = c[a] - c[b]
        want = direct.std(ddof=1) / np.sqrt(sd[a] ** 2 + sd[b] ** 2)
        worst = max(worst, abs(want - K[a, b]))
    return worst < 1e-12, f"kappa matches a direct recomputation, worst gap {worst:.1e}"


def _check_null_centred(fam_kept, iu, gap, kap, strata) -> tuple[bool, str]:
    """A random labelling of the same shape must not look clustered."""
    rng = np.random.default_rng(2)
    ps = []
    for _ in range(20):
        fake = fam_kept.copy()
        rng.shuffle(fake)
        r = run(fake, iu, gap, kap, strata, rng, perms=199)
        if r:
            ps.append(r["free"]["gap_p"])
    m = float(np.mean(ps))
    return 0.30 <= m <= 0.70, f"random labellings give mean p = {m:.2f} over {len(ps)} draws"


def _check_p4_null_moves(fam_kept, strata, tidx, tiu, rng) -> tuple[bool, str]:
    """A null with no spread prints a verdict it has not earned."""
    def share_of(labels):
        t = labels[tidx]
        return float((t[tiu[0]] == t[tiu[1]]).mean())
    draws = [share_of(permute_by_stratum(fam_kept, strata, rng)) for _ in range(200)]
    sd = float(np.std(draws))
    uniq = len(set(np.round(draws, 12)))
    return sd > 1e-9, (f"the top-group null moves: {uniq} distinct values, sd "
                       f"{100 * sd:.2f} points over 200 draws")


def _check_strata(strata, fam_kept) -> tuple[bool, str]:
    good = 0
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if len(set(fam_kept[idx])) >= 2:
            good += 1
    return good >= 3, f"{good} quarters contain two or more families (needs >= 3)"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    x, names, fam, dates = load()
    keep = np.flatnonzero(fam != "")
    fam_kept = fam[keep]
    strata = np.array([quarter(dates[i]) for i in keep])
    iu, gap, kap = pair_stats(x, keep)

    sc0 = x.mean(axis=1)
    order0 = np.argsort(-sc0)
    top_lab0 = [int(i) for i in order0[:TOPK] if fam[i] != ""]
    pos0 = {int(i): j for j, i in enumerate(keep)}
    tidx0 = np.array([pos0[i] for i in top_lab0])
    tiu0 = np.triu_indices(len(tidx0), k=1)

    print("self-checks ...")
    checks = [_check_kappa_source(x, keep),
              _check_null_centred(fam_kept, iu, gap, kap, strata),
              _check_strata(strata, fam_kept),
              _check_p4_null_moves(fam_kept, strata, tidx0, tiu0,
                                   np.random.default_rng(9))]
    ok = True
    for passed, msg in checks:
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rng = np.random.default_rng(SEED)
    r = run(fam_kept, iu, gap, kap, strata, rng)

    # the top-K concentration question, with the null taken over the WHOLE
    # labelled board: permuting inside the top group alone cannot move it.
    sc = x.mean(axis=1)
    order = np.argsort(-sc)
    top_lab = [int(i) for i in order[:TOPK] if fam[i] != ""]
    pos = {int(i): j for j, i in enumerate(keep)}
    tidx = np.array([pos[i] for i in top_lab])
    tiu = np.triu_indices(len(tidx), k=1)

    def share_of(labels):
        t = labels[tidx]
        return float((t[tiu[0]] == t[tiu[1]]).mean())

    obs_share = share_of(fam_kept)
    shares = [share_of(permute_by_stratum(fam_kept, strata, rng)) for _ in range(PERMS)]
    share_sd = float(np.std(shares))
    share_p = perm_p(obs_share, shares, lower_is_extreme=False)
    p4_live = share_sd > 1e-9

    L = []
    p = L.append
    p("IS THE CROWDED TOP A BASE-MODEL CLUSTER, OR A CROWDED CALENDAR?")
    p("=" * 92)
    p(f"  SWE-bench Verified: {len(names)} submissions, {len(keep)} name a base model, "
      f"{len(set(fam_kept))} families.")
    p(f"  Pairs among the labelled: {len(gap)}, of which {r['obs']['n_same']} share a family.")
    p("")
    p(f"  {'statistic':<26} {'same family':>12} {'different':>11} "
      f"{'free null':>11} {'p':>7} {'by-date null':>13} {'p':>7}")
    p(f"  {'score gap':<26} {r['obs']['gap']:>11.4f} {r['obs']['gap_diff']:>11.4f} "
      f"{r['free']['gap_med']:>11.4f} {r['free']['gap_p']:>7.3f} "
      f"{r['date']['gap_med']:>13.4f} {r['date']['gap_p']:>7.3f}")
    p(f"  {'kappa (pair sharpness)':<26} {r['obs']['kap']:>11.4f} {r['obs']['kap_diff']:>11.4f} "
      f"{r['free']['kap_med']:>11.4f} {r['free']['kap_p']:>7.3f} "
      f"{r['date']['kap_med']:>13.4f} {r['date']['kap_p']:>7.3f}")
    p("")
    p(f"  Top {TOPK} by score: {len(top_lab)} labelled, same-family share of their pairs "
      f"{100 * obs_share:.0f} %,")
    p(f"  against a by-date null median of {100 * float(np.median(shares)):.0f} % "
      f"(sd {100 * share_sd:.1f} points, p = {share_p:.3f}).")
    p("")
    p(f"  P1  gap below the free null            p = {r['free']['gap_p']:.3f}   "
      f"{'HIT' if r['free']['gap_p'] < 0.05 else 'MISS'}")
    p(f"  P2  kappa below the free null          p = {r['free']['kap_p']:.3f}   "
      f"{'HIT' if r['free']['kap_p'] < 0.05 else 'MISS'}")
    p(f"  P3  gap below the BY-DATE null         p = {r['date']['gap_p']:.3f}   "
      f"{'HIT' if r['date']['gap_p'] < 0.05 else 'MISS'}")
    p(f"  P4  top-{TOPK} family share above its by-date null   p = {share_p:.3f}   "
      f"{('HIT' if share_p < 0.05 else 'MISS') if p4_live else 'VACUOUS - null cannot move'}")
    p("")
    p("  The free null asks whether the family label carries any information")
    p("  about proximity. The by-date null keeps every submission in its own")
    p("  calendar quarter and permutes only which family it belongs to, so it")
    p("  answers the question that matters: is the cluster the model, or is it")
    p("  that the systems near the top are all recent?")
    p("")
    p("  72 of 134 submissions do not name a base model and are dropped rather")
    p("  than guessed, which removes same-family pairs and weakens every effect")
    p("  reported here.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("family_clustering_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote family_clustering_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
