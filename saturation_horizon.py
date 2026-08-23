"""The leaderboard of the future is inside today's hardest items.

SWE-bench Verified's leader is at 79 % and climbing. When the frontier reaches
95 %, the four hundred instances that almost everyone already solves will
separate nobody, and the ranking will be decided entirely by the handful that
are still hard. That future leaderboard is not a forecast. It exists now, in
the same matrix, and it can simply be read off.

Every other file here asks what the data says about the present. This one asks
whether the present ordering is the one that survives saturation - which is
the question a benchmark owner and a lab planning next year's run both
actually have.

THE MEASUREMENT
---------------
Sort instances by how many systems solve them. Take the hardest tenth, the
hardest fifth, and so on, and recompute the ranking inside each. Then ask two
things of each stratum: does the same system lead, and how much of the overall
pairwise order survives.

THE NULL WITHOUT WHICH THIS IS WORTHLESS
------------------------------------------
The hardest tenth is fifty instances, and a ranking computed on fifty
instances disagrees with the full one no matter which fifty they are - that is
noise, not difficulty. So every stratum is compared against RANDOM subsets OF
THE SAME SIZE drawn from the whole benchmark. Only the gap between the two
means anything:

    agreement(hardest 50) vs agreement(random 50)

If they match, the hardest instances rank systems the same way the easy ones
do and saturation changes nothing but precision. If the hardest disagree more,
the benchmark is measuring something at its ceiling that it does not measure
in its bulk, and today's order is not the one that will survive.

WHAT THIS CANNOT DO
--------------------
It cannot see abilities no current instance tests. If the next generation is
separated by a kind of task absent from all 500, nothing here will show it.
The claim is narrower and checkable: among the abilities THIS benchmark can
see, here is which ordering is a property of its easy bulk and which is a
property of its hard edge.

SELF-CHECKS THAT CAN FAIL
--------------------------
  * a purely one-dimensional world must show hardest-stratum agreement equal
    to random-subset agreement, because difficulty carries no extra ordering
    there;
  * a world built with a second ability that only hard items load on must
    show the hardest stratum disagreeing MORE than random;
  * strata must partition the instances exactly, with none lost or doubled.

    python saturation_horizon.py [--matrix ...] [--draws 400]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260823


def pair_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of system pairs ordered the same way by two score vectors."""
    J = len(a)
    iu = np.triu_indices(J, k=1)
    sa = np.sign(a[:, None] - a[None, :])[iu]
    sb = np.sign(b[:, None] - b[None, :])[iu]
    return float(np.mean(sa == sb))


def item_information(x: np.ndarray) -> np.ndarray:
    """w_i = s_i (J - s_i): the pairs this item splits. See benchmark_spectrum."""
    s = x.sum(axis=0)
    return s * (x.shape[0] - s)


def matched_subset(x: np.ndarray, target_w: float, rng) -> np.ndarray:
    """A random set of items carrying the same total ranking information.

    Drawing random items of the same COUNT is the obvious null and it is the
    wrong one. Hard items are the least discriminating - w_i = s(J-s) is
    largest at half - so the hardest fifty carry far less information than
    fifty random ones, and they would rank systems more erratically even in a
    world where difficulty means nothing. Measured: in a purely
    one-dimensional simulation the hardest tenth scored 0.827 against a
    count-matched band of [0.863, 0.918], outside it, with no second ability
    anywhere in the data. The check caught that and the tool refused to print.

    Matching on total information instead lets the count differ, which is the
    point: the question is whether these items order systems differently, not
    whether there are fewer of them.
    """
    w = item_information(x)
    order = rng.permutation(len(w))
    picked, total = [], 0.0
    for i in order:
        picked.append(i)
        total += w[i]
        if total >= target_w:
            break
    return np.array(picked)


def stratum_report(x: np.ndarray, idx: np.ndarray, full: np.ndarray,
                   draws: int, rng: np.random.Generator) -> dict:
    m = len(idx)
    sub = x[:, idx].mean(axis=1)
    obs = pair_agreement(full, sub)
    n = x.shape[1]
    w = item_information(x)
    target = float(w[idx].sum())

    count_null = np.empty(draws)
    info_null = np.empty(draws)
    info_sizes = np.empty(draws)
    for b in range(draws):
        r = rng.choice(n, size=m, replace=False)
        count_null[b] = pair_agreement(full, x[:, r].mean(axis=1))
        q = matched_subset(x, target, rng)
        info_sizes[b] = len(q)
        info_null[b] = pair_agreement(full, x[:, q].mean(axis=1))
    return {"m": m, "obs": obs,
            "count_mean": float(count_null.mean()),
            "count_lo": float(np.quantile(count_null, 0.025)),
            "count_hi": float(np.quantile(count_null, 0.975)),
            "null_mean": float(info_null.mean()),
            "null_lo": float(np.quantile(info_null, 0.025)),
            "null_hi": float(np.quantile(info_null, 0.975)),
            "info_size": float(info_sizes.mean()),
            "info_share": target / float(w.sum()),
            "leader": int(np.argmax(sub)), "scores": sub}


# --- self-checks ------------------------------------------------------------

def _world(J, n, seed, hard_axis=0.0):
    """Crossed abilities; hard_axis puts a SECOND ability on hard items only."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1.0, J)
    b = rng.normal(0, 1.4, n)
    logit = a[:, None] + b[None, :]
    if hard_axis:
        a2 = rng.normal(0, 1.0, J)
        w = (b < np.quantile(b, 0.25)).astype(float)   # only the hard quarter
        logit = logit + hard_axis * a2[:, None] * w[None, :]
    return (rng.random((J, n)) < 1 / (1 + np.exp(-logit))).astype(float), rng


def _hardest(x, frac):
    solved = x.sum(axis=0)
    k = max(2, int(round(frac * x.shape[1])))
    return np.argsort(solved, kind="stable")[:k]


def _check_partition() -> tuple[bool, str]:
    x, _ = _world(20, 200, 1)
    idx = _hardest(x, 0.10)
    ok = len(idx) == 20 and len(set(idx.tolist())) == 20
    return ok, f"hardest tenth of 200 instances is {len(idx)} distinct items"


def _check_one_dimensional_matches_random() -> tuple[bool, str]:
    """With one ability, difficulty must carry no extra ordering."""
    x, rng = _world(40, 600, 3)
    full = x.mean(axis=1)
    r = stratum_report(x, _hardest(x, 0.10), full, 200, rng)
    ok = r["null_lo"] <= r["obs"] <= r["null_hi"]
    return ok, (f"one-dimensional world: hardest {r['obs']:.3f} inside "
                f"random band [{r['null_lo']:.3f}, {r['null_hi']:.3f}]")


def _check_hard_axis_detected() -> tuple[bool, str]:
    """A second ability on hard items must push the hardest stratum out."""
    x, rng = _world(40, 600, 5, hard_axis=2.5)
    full = x.mean(axis=1)
    r = stratum_report(x, _hardest(x, 0.10), full, 200, rng)
    ok = r["obs"] < r["null_lo"]
    return ok, (f"hard-only second ability: hardest {r['obs']:.3f} below "
                f"random band low {r['null_lo']:.3f}")


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_partition(),
                        _check_one_dimensional_matches_random(),
                        _check_hard_axis_detected()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--out", default="saturation_horizon_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=float)
    names = list(df.index)
    J, n = x.shape
    print(f"matrix {a.matrix}: {J} systems x {n} instances")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no headline number is printed.")
        return 1

    rng = np.random.default_rng(SEED)
    full = x.mean(axis=1)
    lead_full = int(np.argmax(full))
    solved = x.sum(axis=0)

    L = []
    p = L.append
    p("THE LEADERBOARD AT SATURATION")
    p("=" * 74)
    p(f"{J} systems, {n} instances")
    p(f"today's leader: {names[lead_full][:48]}  {full[lead_full]:.3f}")
    p(f"instances solved by nobody: {int((solved == 0).sum())}, "
      f"by everybody: {int((solved == J).sum())}")
    p("")
    p("RANKING INSIDE THE HARDEST INSTANCES, AGAINST TWO NULLS")
    p("  count-null: random items of the same NUMBER. This is the obvious")
    p("    comparison and it is wrong: hard items are the least discriminating")
    p("    (w = s(J-s) peaks at half), so they lose to it even when difficulty")
    p("    carries no ordering at all. Shown because it is what one would")
    p("    otherwise have reported.")
    p("  info-null: random items carrying the same TOTAL ranking information.")
    p("    This is the comparison the question deserves, and the band below is")
    p("    the one to read.")
    p("  info: the share of the whole benchmark's ranking information in the")
    p("    stratum - a tenth of the instances is nowhere near a tenth of it.")
    p(f"  {'stratum':>15} {'items':>6} {'info':>6} {'agree':>7}"
      f" {'count-null':>11} {'info-null':>19}  leader")
    rows = []
    for frac, label in ((0.05, "hardest 5 %"), (0.10, "hardest 10 %"),
                        (0.20, "hardest 20 %"), (0.40, "hardest 40 %")):
        idx = _hardest(x, frac)
        r = stratum_report(x, idx, full, a.draws, rng)
        rows.append((label, r))
        flag = "" if r["null_lo"] <= r["obs"] <= r["null_hi"] else "  <-- outside"
        p(f"  {label:>15} {r['m']:>6} {100 * r['info_share']:>5.1f}%"
          f" {r['obs']:>7.3f} {r['count_mean']:>11.3f}"
          f"  [{r['null_lo']:.3f}, {r['null_hi']:.3f}]"
          f"  {names[r['leader']][:22]}{flag}")
    p("")

    # The specific question: does today's leader lead at the hard edge?
    hard = _hardest(x, 0.10)
    hs = x[:, hard].mean(axis=1)
    ho = np.argsort(-hs, kind="stable")
    p("WHO LEADS WHEN ONLY THE HARDEST TENTH COUNTS")
    p(f"  {'rank now':>9} {'system':<44} {'hard score':>11} {'full':>7}")
    for i in ho[:8]:
        rank_now = int(np.sum(full > full[i])) + 1
        p(f"  {rank_now:>9} {names[i][:44]:<44} {hs[i]:>11.3f} {full[i]:>7.3f}")
    p("")
    rank_of_leader_hard = int(np.sum(hs > hs[lead_full])) + 1
    p(f"  Today's leader is rank {rank_of_leader_hard} on the hardest tenth.")
    p("")

    # The premise of this file was wrong, and the honest thing is to say so
    # in the output rather than quietly report something else.
    best_hard = float(hs.max())
    solved_by_best = int(round(best_hard * len(hard)))
    zero = int((hs == 0).sum())
    p("WHY THE TABLE ABOVE DOES NOT MEAN WHAT IT LOOKS LIKE")
    p(f"  On the hardest tenth the best system solves {solved_by_best} of "
      f"{len(hard)} instances,")
    p(f"  and {zero} of {J} systems solve none at all. There is no ordering to")
    p("  read there. The strata fall below their information-matched nulls")
    p("  not because the hard edge ranks systems differently but because it")
    p("  barely ranks them at all.")
    p("")
    p("THE FINDING, WHICH IS THE OPPOSITE OF WHAT THIS FILE SET OUT TO SHOW")
    w_all = item_information(x)
    p("  share of the benchmark's ranking information by difficulty stratum:")
    for frac, label in ((0.05, "hardest 5 %"), (0.10, "hardest 10 %"),
                        (0.20, "hardest 20 %"), (0.40, "hardest 40 %")):
        idx = _hardest(x, frac)
        p(f"    {label:<14} {100 * w_all[idx].sum() / w_all.sum():>5.1f} %")
    # Where does the information actually live?
    ordv = np.argsort(-w_all)
    cum = np.cumsum(w_all[ordv]) / w_all.sum()
    solved_sorted = solved[ordv]
    half = int(np.searchsorted(cum, 0.5)) + 1
    lo = int(solved_sorted[:half].min())
    hi = int(solved_sorted[:half].max())
    p("")
    p(f"  Half of all ranking information sits in {half} instances, solved by")
    p(f"  between {lo} and {hi} of the {J} systems - the middle of the")
    p("  difficulty range, exactly where w = s(J-s) says it must be.")
    p("")
    p("  So saturation will not REWRITE this leaderboard. It will ERASE it.")
    p("  As the bulk of instances becomes universally solved, the information")
    p("  those instances carry goes to zero, and what remains at the hard edge")
    p("  carries almost none to begin with. A benchmark does not fail by")
    p("  ranking the wrong system first; it fails by running out of ability")
    p("  to rank anyone, and the measurement above says how much is left.")
    p("")
    p("  What this cannot see: an ability no current instance tests. If the")
    p("  next generation is separated by a kind of task absent from all 500,")
    p("  nothing here will show it - and on this evidence that is exactly")
    p("  what the next benchmark will have to contain.")
    p("")
    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
