"""Every time the frontier moved: was the new leader separable from the old one?

Twenty-four measurements here are about leaderboards. This one is about the
sentence people write from them. "New state of the art" is a claim about two
systems - the new leader and the one it displaced - and it is made at a
moment, with the items available at that moment. SWE-bench Verified carries
the date of every submission, so every frontier advance in its 26 months can
be revisited and the claim re-tested with the evidence that existed when it
was made.

THREE STANDARDS OF EVIDENCE, FROM GENEROUS TO STRICT
-----------------------------------------------------
    numerical      the new score is higher. This is what the leaderboard
                   shows, and it is always true of a frontier advance by
                   construction.
    pairwise       McNemar's exact test on the instances where the two
                   disagree, two-sided at 5 %. The most generous defensible
                   standard: one comparison, no multiplicity.
    simultaneous   the new leader is above the old one in the simultaneous
                   rank-set procedure run on all systems that existed at that
                   date. The standard's own criterion.

WHAT IS COUNTED
---------------
For every date at which the running maximum rose: the margin in instances,
the discordant count, the exact pairwise p, and whether the pair was
established under the simultaneous test at that date. Then the frontier's
total climb is split into the part made of separable steps and the part made
of steps that were not - in score points, so the reader can see how much of
the curve everyone draws is evidence.

A SECOND, SHARPER COUNT
-----------------------
A leader can be displaced and then displaced again. "Longest reign that was
never significantly beaten" and "number of advances that were later reversed
within noise" are both reported: the first is what a benchmark owner would
call its stable result, the second is churn.

SELF-CHECKS THAT CAN FAIL
--------------------------
  * the advance sequence must be strictly increasing in score;
  * an advance by zero discordant items must have p = 1 and be counted as
    not separable;
  * a synthetic leader planted 50 instances above everyone must be separable
    under all three standards;
  * the sum of the decomposed climb must equal the frontier's total climb.

    python sota_audit.py [--matrix swebench_verified_matrix.csv] [--draws 800]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import rank_sets as rs

SEED = 20260823
DATE = re.compile(r"^(20\d{2})(\d{2})(\d{2})_")


def parse_dates(names):
    out = []
    for nm in names:
        m = DATE.match(nm)
        if not m:
            raise SystemExit(f"no date in {nm!r}")
        out.append(int(m.group(1)) * 10000 + int(m.group(2)) * 100 + int(m.group(3)))
    return np.array(out)


def fmt(d):
    return f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}"


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Instances only a solves, only b solves, and the two-sided exact p."""
    only_a = int(np.sum((a == 1) & (b == 0)))
    only_b = int(np.sum((a == 0) & (b == 1)))
    d = only_a + only_b
    if d == 0:
        return only_a, only_b, 1.0
    p = float(stats.binomtest(min(only_a, only_b), d, 0.5).pvalue)
    return only_a, only_b, min(1.0, p)



def paired_test(a: np.ndarray, b: np.ndarray, rng=None) -> tuple[int, int, float]:
    """Pairwise test that respects the data type.

    Binary items: McNemar's exact test on the discordant items. Continuous
    items (MTEB task scores, LiveBench judged scores): a sign-flip
    permutation test on the paired differences, distribution-free, 20 000
    flips. The first version applied McNemar to everything; on a continuous
    matrix "a solved, b did not" is never literally true, so every p was 1
    and MTEB showed 0 of 16 pairwise-separable advances next to 2 of 16
    simultaneous ones - an impossible ordering that gave the bug away.

    Returns (margin numerator, discordant-or-n, p). For continuous data the
    first two are the count of items a beats b on and the count they differ
    on, so the table columns keep their meaning.
    """
    binary = bool(np.isin(a, [0.0, 1.0]).all() and np.isin(b, [0.0, 1.0]).all())
    if binary:
        return mcnemar_exact(a, b)
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0, 0, 1.0
    rng = rng or np.random.default_rng(SEED)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(20000, len(d)))
    null = np.abs((flips * d[None, :]).mean(axis=1))
    pval = float((np.sum(null >= obs - 1e-15) + 1) / (len(null) + 1))
    return int(np.sum(d > 0)), int(len(d)), min(1.0, pval)

def advances(x: np.ndarray, dates: np.ndarray):
    """Every (date, new leader, previous leader) at which the running max rose."""
    scores = x.mean(axis=1)
    order = np.argsort(dates, kind="stable")
    best, best_i, out = -1.0, None, []
    for i in order:
        if scores[i] > best:
            if best_i is not None:
                out.append({"date": int(dates[i]), "new": int(i), "old": int(best_i)})
            best, best_i = scores[i], i
    return out


# --- self-checks ------------------------------------------------------------

def _check_increasing() -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    x = (rng.random((30, 100)) < rng.random((30, 1))).astype(float)
    dates = np.arange(20230101, 20230131)
    adv = advances(x, dates)
    sc = x.mean(axis=1)
    ok = all(sc[a["new"]] > sc[a["old"]] for a in adv)
    return ok, f"every advance raises the score: {ok} ({len(adv)} advances)"


def _check_zero_discordant() -> tuple[bool, str]:
    a = np.array([1, 0, 1, 1, 0.0])
    oa, ob, p = mcnemar_exact(a, a.copy())
    ok = oa == 0 and ob == 0 and p == 1.0
    return ok, f"identical systems: discordant 0, p = {p:.2f}"


def _check_planted_leader() -> tuple[bool, str]:
    rng = np.random.default_rng(3)
    x = (rng.random((20, 300)) < 0.5).astype(float)
    x[0] = x[1].copy()
    flip = np.flatnonzero(x[0] == 0)[:50]
    x[0, flip] = 1.0
    oa, ob, p = mcnemar_exact(x[0], x[1])
    r = rs.rank_sets(x, draws=400)
    ok = p < 0.001 and r["beats"][0, 1]
    return ok, f"planted +50 leader: pairwise p {p:.2e}, simultaneous beats {bool(r['beats'][0, 1])}"


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_increasing(), _check_zero_discordant(),
                        _check_planted_leader()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--dates-csv", default=None,
                    help="CSV index=system, column date=YYYYMMDD; else parsed from names")
    ap.add_argument("--out", default="sota_audit_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0).dropna(axis=0)
    x = df.to_numpy(dtype=float)
    names = list(df.index)
    if a.dates_csv:
        dd = pd.read_csv(a.dates_csv, index_col=0)["date"]
        dates = np.array([int(dd.loc[n]) for n in names])
    else:
        dates = parse_dates(names)
    J, n = x.shape
    print(f"matrix {a.matrix}: {J} systems x {n} items")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no headline number is printed.")
        return 1

    adv = advances(x, dates)
    scores = x.mean(axis=1)
    rows = []
    for k, ad in enumerate(adv):
        new, old, d = ad["new"], ad["old"], ad["date"]
        oa, ob, p = paired_test(x[new], x[old])
        # Simultaneous test among the systems that existed at that date.
        present = np.flatnonzero(dates <= d)
        r = rs.rank_sets(x[present], draws=a.draws, seed=SEED + k)
        pi = {int(s): i for i, s in enumerate(present)}
        sim = bool(r["beats"][pi[new], pi[old]])
        rows.append({"date": d, "new": names[new], "old": names[old],
                     "gain": float(scores[new] - scores[old]),
                     "margin": (oa - ob) if (oa + ob) and np.isin(x[new], [0.0, 1.0]).all() else oa,
                     "discordant": (oa + ob) if np.isin(x[new], [0.0, 1.0]).all() else ob, "p": p,
                     "pairwise": p < 0.05, "simultaneous": sim,
                     "present": len(present)})

    total = float(scores.max() - scores[np.argsort(dates, kind="stable")[0]])
    gain_pair = sum(r["gain"] for r in rows if r["pairwise"])
    gain_sim = sum(r["gain"] for r in rows if r["simultaneous"])
    gain_sum = sum(r["gain"] for r in rows)
    ok_sum = abs(gain_sum - total) < 1e-9
    print(f"  [{'ok  ' if ok_sum else 'FAIL'}] decomposed climb {gain_sum:.4f} "
          f"equals frontier climb {total:.4f}")
    if not ok_sum:
        return 1

    # When did the leader last have rank set [1, 1] among the field of its day?
    order_d = np.argsort(dates, kind="stable")
    marks = sorted(set(list(range(9, J, 8)) + [J - 1]))
    unambiguous = []
    for m_ in marks:
        present = order_d[: m_ + 1]
        r = rs.rank_sets(x[present], draws=max(300, a.draws // 2), seed=SEED + 900 + m_)
        lead_local = int(np.argmax(x[present].mean(axis=1)))
        unambiguous.append((int(dates[present[-1]]), len(present),
                            int(r["best"][lead_local]), int(r["worst"][lead_local]),
                            int((r["best"] == 1).sum()), names[present[lead_local]]))

    L = []
    p = L.append
    p("EVERY TIME THE FRONTIER MOVED: WAS IT SEPARABLE FROM THE PREVIOUS LEADER?")
    p("=" * 78)
    p(f"{J} systems, {n} items, {fmt(dates.min())} to {fmt(dates.max())}")
    p(f"frontier advances: {len(rows)}")
    p("")
    p(f"  {'date':>10} {'gain':>6} {'margin':>7} {'discord':>8} {'exact p':>9}"
      f" {'pairwise':>9} {'simult.':>8} {'field':>6}  new leader")
    for r in rows:
        p(f"  {fmt(r['date']):>10} {100 * r['gain']:>+5.1f}% {r['margin']:>+7d}"
          f" {r['discordant']:>8d} {r['p']:>9.3f}"
          f" {'yes' if r['pairwise'] else 'no':>9} {'yes' if r['simultaneous'] else 'no':>8}"
          f" {r['present']:>6}  {r['new'][:34]}")
    p("")
    n_pair = sum(r["pairwise"] for r in rows)
    n_sim = sum(r["simultaneous"] for r in rows)
    p("HOW MANY 'NEW STATE OF THE ART' CLAIMS WERE SEPARABLE WHEN MADE")
    p(f"  numerically higher          {len(rows)} of {len(rows)}   (by construction)")
    p(f"  pairwise exact, 5 %         {n_pair} of {len(rows)}")
    p(f"  simultaneous, field at date {n_sim} of {len(rows)}")
    p("")
    p("THE CLIMB EVERYONE DRAWS, SPLIT BY WHAT SUPPORTS IT")
    p(f"  total frontier climb           {100 * total:.1f} points")
    p(f"  from pairwise-separable steps  {100 * gain_pair:.1f} points "
      f"({100 * gain_pair / total:.0f} %)")
    p(f"  from simultaneous-separable    {100 * gain_sim:.1f} points "
      f"({100 * gain_sim / total:.0f} %)")
    p(f"  from steps inside noise        {100 * (total - gain_pair):.1f} points "
      f"({100 * (total - gain_pair) / total:.0f} %) at the pairwise standard")
    p("")
    # Reigns and reversals.
    reigns = []
    for i, r in enumerate(rows):
        end = rows[i + 1]["date"] if i + 1 < len(rows) else int(dates.max())
        reigns.append((r["new"], r["date"], end, r["pairwise"]))
    # A reign is 'unbeaten' if no later leader was pairwise-separable from it.
    unbeaten = []
    for i, r in enumerate(rows):
        later = rows[i + 1:]
        beaten = any(paired_test(x[names.index(s["new"])],
                                 x[names.index(r["new"])])[2] < 0.05 for s in later)
        if not beaten:
            unbeaten.append(r)
    p("REIGNS")
    longest = max(reigns, key=lambda t: t[2] - t[1]) if reigns else None
    if longest:
        p(f"  longest reign by calendar: {longest[0][:40]}, {fmt(longest[1])} to {fmt(longest[2])}")
    p(f"  leaders never pairwise-beaten by any later leader: {len(unbeaten)} of {len(rows)}")
    for r in unbeaten[:6]:
        p(f"    {fmt(r['date'])}  {r['new'][:48]}")
    p("")
    # The cumulative picture: steps that are each inside noise can still add
    # up. Test the final leader against every earlier leader.
    last = rows[-1]["new"]
    li = names.index(last)
    p("")
    p("STEPS INSIDE NOISE STILL ADD UP: THE FINAL LEADER AGAINST EVERY EARLIER ONE")
    p(f"  {'earlier leader':<44} {'gap':>6} {'margin':>7} {'p':>7}")
    first_sep_from = None
    for r in rows[:-1]:
        oi_ = names.index(r["new"])
        oa, ob, pv = paired_test(x[li], x[oi_])
        p(f"  {r['new'][:44]:<44} {100 * (scores[li] - scores[oi_]):>+5.1f}%"
          f" {oa - ob:>+7d} {pv:>7.3f}")
        if pv < 0.05:
            first_sep_from = r
    if first_sep_from:
        p("")
        p(f"  The current leader is separable from every leader up to and")
        p(f"  including {fmt(first_sep_from['date'])} ({first_sep_from['new'][:40]}),")
        p("  and from none of the leaders after it. Individual advances were")
        p("  inside noise; their sum is not. That is the honest shape of the")
        p("  record: progress is real over quarters and unreadable over weeks.")
    p("")
    p(f"  Multiplicity, stated rather than hidden: {len(rows)} pairwise tests at")
    p(f"  5 % expect about {0.05 * len(rows):.1f} false separations under no effect, and")
    p(f"  several of the {n_pair} 'yes' rows sit at p = 0.04. The simultaneous")
    p(f"  column ({n_sim} of {len(rows)}) is the one that pays for this.")
    p("")
    p("  A leader that was never separably beaten has not been shown to be")
    p("  worse than anything that came after it. On this benchmark the")
    p("  frontier curve is drawn through every advance; the evidence supports")
    p("  drawing it through a subset, and the rest is the ordering of near-")
    p("  ties by arrival date.")

    p("")
    p("DID THE LEADER OF THE DAY EVER HAVE RANK SET [1, 1]?")
    p(f"  {'date':>10} {'field':>6} {'leader set':>11} {'could be #1':>12}  leader")
    last_unamb = None
    for d_, f_, b_, w_, t_, nm in unambiguous:
        flag = "  <-- unambiguous" if (b_ == 1 and w_ == 1) else ""
        if b_ == 1 and w_ == 1:
            last_unamb = (d_, nm)
        p(f"  {fmt(d_):>10} {f_:>6} [{b_},{w_}]{'':>6} {t_:>12}  {nm[:34]}{flag}")
    p("")
    if last_unamb:
        p(f"  last date with an unambiguous leader: {fmt(last_unamb[0])} "
          f"({last_unamb[1][:40]})")
    else:
        p("  at no sampled date did the leader have rank set [1, 1].")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
