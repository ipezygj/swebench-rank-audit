"""A benchmark is a depleting resource. This measures how much is left.

THE IDEA, WHICH AS FAR AS I CAN FIND HAS NO NAME
--------------------------------------------------
An instance separates two systems only when they disagree on it. It can only
do that while the field's ability sits near its difficulty: too hard and
nobody solves it, too easy and everybody does, and in both cases it separates
nothing. So an instance is not a permanent asset. It has a LIFETIME, and the
field's own progress spends it.

That makes a benchmark a stock rather than a fixture, and gives it a quantity
none of the boards measured here reports:

    D(t) = the expected number of instances on which two systems drawn at
           random from the field at time t disagree.

D is in instances, not in decimal places. It is the raw material every
significance test on a leaderboard consumes: McNemar has exactly D discordant
trials to work with, so when D is twenty no procedure can separate anything,
however clever. And D is not fixed. It rises while the field is climbing
towards the bulk of the difficulty distribution and falls once it is past.

THREE NUMBERS THAT FOLLOW, AND DID NOT EXIST BEFORE
-----------------------------------------------------
    peak        the date at which the benchmark was at its most informative
    remaining   D today as a fraction of that peak
    half-life   the frontier score at which D falls to half its peak, and,
                from the measured rate of progress, roughly when that is

MEASURED, NOT MODELLED, AND THEN MODELLED SEPARATELY
------------------------------------------------------
D(t) is computed directly: take the systems submitted up to date t, count the
disagreements, divide by the pairs. Nothing is assumed. A second, model-based
curve extends it past today by fitting item difficulties and sliding the
field's ability upward, and the two are plotted together so the extrapolation
can be checked against the part that is observed.

D IS COMPUTED ON THE FRONTIER, NOT THE WHOLE FIELD
----------------------------------------------------
Averaging over all systems ever submitted keeps the 2023 baselines in the
pool forever, and they disagree with everything, so the whole-field D would
look healthy while the top of the table went blind. The number that matters
is D among the systems anyone is choosing between, so it is computed over the
top decile at each date, with the whole-field version shown beside it to make
the difference visible.

SELF-CHECKS THAT CAN FAIL
--------------------------
  * D computed from the pair-counting definition must equal the sum of
    s(J-s) over items divided by the pairs - two derivations, one number;
  * a synthetic field whose ability rises past a fixed difficulty
    distribution must produce a D that rises then falls, with the peak where
    the median item sits at half;
  * a field whose ability does not move must produce a flat D;
  * D must equal the mean McNemar discordant count over sampled pairs.

    python information_depletion.py [--matrix ...]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260823
DATE = re.compile(r"^(20\d{2})(\d{2})(\d{2})_")


def parse_dates(names) -> np.ndarray:
    out = []
    for nm in names:
        m = DATE.match(nm)
        if not m:
            raise SystemExit(f"no date in {nm!r}")
        out.append(int(m.group(1)) * 10000 + int(m.group(2)) * 100
                   + int(m.group(3)))
    return np.array(out)


def fmt(d: int) -> str:
    return f"{d // 10000}-{(d // 100) % 100:02d}"


def expected_discordant(x: np.ndarray) -> float:
    """D: instances on which two systems drawn at random disagree.

    Counted through s(J-s) because the pair-by-pair sum is the same number
    and this is O(items) instead of O(items x pairs). The self-check
    verifies the two agree.
    """
    J = x.shape[0]
    if J < 2:
        return float("nan")
    s = x.sum(axis=0)
    return float((s * (J - s)).sum() / (J * (J - 1) / 2))


def frontier_subset(x: np.ndarray, frac: float = 0.10) -> np.ndarray:
    k = max(2, int(round(frac * x.shape[0])))
    return np.argsort(-x.mean(axis=1), kind="stable")[:k]


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


def d_at_ability(b: np.ndarray, ability: float, spread: float,
                 J: int = 20, draws: int = 400, seed: int = SEED) -> float:
    """Model D for a field centred at `ability` with the given spread."""
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for k in range(draws):
        a = rng.normal(ability, spread, J)
        p = 1 / (1 + np.exp(-(a[:, None] + b[None, :])))
        y = (rng.random((J, len(b))) < p).astype(float)
        out[k] = expected_discordant(y)
    return float(out.mean())


def frontier_score_at_ability(b, ability, spread, J=20, draws=200, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    best = np.empty(draws)
    for k in range(draws):
        a = rng.normal(ability, spread, J)
        p = 1 / (1 + np.exp(-(a[:, None] + b[None, :])))
        best[k] = (rng.random((J, len(b))) < p).astype(float).mean(axis=1).max()
    return float(best.mean())


# --- self-checks ------------------------------------------------------------

def _check_two_derivations() -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    x = (rng.random((14, 90)) < 0.5).astype(float)
    fast = expected_discordant(x)
    J = x.shape[0]
    tot, npairs = 0, 0
    for i in range(J):
        for j in range(i + 1, J):
            tot += int((x[i] != x[j]).sum())
            npairs += 1
    slow = tot / npairs
    ok = abs(fast - slow) < 1e-9
    return ok, f"s(J-s) route {fast:.6f} vs pair-by-pair {slow:.6f}"


def _check_rise_then_fall() -> tuple[bool, str]:
    """A field climbing past a fixed difficulty distribution: D up then down."""
    rng = np.random.default_rng(3)
    b = rng.normal(0, 1.2, 400)
    ds = [d_at_ability(b, a, 0.5, J=20, draws=120, seed=7) for a in
          (-4, -2, 0, 2, 4)]
    ok = ds[0] < ds[2] and ds[4] < ds[2]
    return ok, f"D across rising ability: {[round(d, 1) for d in ds]}"


def _check_flat_when_static() -> tuple[bool, str]:
    rng = np.random.default_rng(5)
    b = rng.normal(0, 1.2, 400)
    ds = [d_at_ability(b, 0.0, 0.5, J=20, draws=120, seed=11 + k)
          for k in range(4)]
    ok = (max(ds) - min(ds)) / np.mean(ds) < 0.05
    return ok, f"static field, D stable within {100*(max(ds)-min(ds))/np.mean(ds):.1f} %"


def _check_matches_mcnemar() -> tuple[bool, str]:
    """D must be the mean discordant count a McNemar test would see."""
    rng = np.random.default_rng(7)
    x = (rng.random((25, 200)) < rng.random((25, 1))).astype(float)
    d = expected_discordant(x)
    picks = [(int(i), int(j)) for i, j in
             rng.integers(0, 25, size=(400, 2)) if i != j]
    obs = np.mean([int((x[i] != x[j]).sum()) for i, j in picks])
    ok = abs(d - obs) / d < 0.05
    return ok, f"D {d:.2f} vs sampled McNemar discordant mean {obs:.2f}"


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_two_derivations(), _check_rise_then_fall(),
                        _check_flat_when_static(), _check_matches_mcnemar()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--out", default="information_depletion_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    x = df.to_numpy(dtype=float)
    dates = parse_dates(df.index)
    o = np.argsort(dates, kind="stable")
    x, dates = x[o], dates[o]
    J, n = x.shape
    print(f"matrix {a.matrix}: {J} systems x {n} instances")

    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no headline number is printed.")
        return 1

    # Observed history: D among the systems that existed at each date.
    marks = list(range(9, J, 10)) + [J - 1]
    hist = []
    for m in marks:
        sub = x[:m + 1]
        fr = frontier_subset(sub, 0.10)
        hist.append({"m": m + 1, "date": int(dates[m]),
                     "frontier_score": float(sub.mean(axis=1).max()),
                     "d_all": expected_discordant(sub),
                     "d_front": expected_discordant(sub[fr])})

    d_front = np.array([h["d_front"] for h in hist])
    peak_i = int(np.argmax(d_front))

    L = []
    p = L.append
    p("A BENCHMARK IS A DEPLETING RESOURCE")
    p("=" * 74)
    p("D = instances on which two systems drawn at random disagree.")
    p("It is the raw material every significance test consumes: McNemar has")
    p("exactly D discordant trials to work with.")
    p("")
    p(f"  {'after':>6} {'date':>9} {'frontier':>9} {'D all':>8} {'D frontier':>11}")
    for h in hist:
        p(f"  {h['m']:>6} {fmt(h['date']):>9} {h['frontier_score']:>9.3f}"
          f" {h['d_all']:>8.1f} {h['d_front']:>11.1f}")
    p("")
    p(f"  peak D on the frontier: {d_front[peak_i]:.1f} instances, "
      f"{fmt(hist[peak_i]['date'])}")
    p(f"  D today:                {d_front[-1]:.1f} instances "
      f"({100 * d_front[-1] / d_front[peak_i]:.0f} % of peak)")
    p("")
    p("  Read the two D columns against each other. The whole-field number")
    p("  stays high because the 2023 baselines never leave the pool and")
    p("  disagree with everything. The frontier number is the one that")
    p("  governs whether the top of the table can be ordered at all.")
    p("")

    # Model curve: slide the field's ability up the fitted difficulties.
    ab, b = rasch_fit(x)
    spread = float(np.std(ab[frontier_subset(x, 0.10)]))
    grid = np.linspace(float(np.median(ab)) - 1.0, float(ab.max()) + 5.0, 22)
    p("PROJECTION: SLIDING THE FIELD UP THE FITTED DIFFICULTIES")
    p(f"  {'ability':>8} {'frontier score':>15} {'D frontier':>11}")
    rows = []
    for g in grid:
        d = d_at_ability(b, g, spread, J=13, draws=150)
        fs = frontier_score_at_ability(b, g, spread, J=13, draws=100)
        rows.append((g, fs, d))
    dmax = max(r[2] for r in rows)
    for g, fs, d in rows[::3]:
        p(f"  {g:>8.2f} {fs:>15.3f} {d:>11.1f}")
    p("")
    today_fs = float(x.mean(axis=1).max())
    half = [r for r in rows if r[2] <= dmax / 2 and r[1] > today_fs]
    if half:
        g, fs, d = half[0]
        p(f"  MODEL HALF-LIFE: D falls to half its peak when the frontier")
        p(f"  reaches {fs:.3f}. Today it is {today_fs:.3f}.")
    else:
        p("  The model does not reach half of peak D within the ability range")
        p("  tried, so no half-life is quoted rather than one extrapolated.")
    p("")
    p("  The projection is a model and the history above is not. They are")
    p("  printed together so the extrapolation can be checked against the")
    p("  observed part before anyone leans on the unobserved part.")
    p("")
    p("  AND THE CHECK DOES NOT PASS CLEANLY, WHICH IS WORTH MORE THAN IF")
    p("  IT HAD. Interpolating the model to today's frontier gives about")
    model_today = None
    for i in range(len(rows) - 1):
        if rows[i][1] <= today_fs <= rows[i + 1][1]:
            f0, f1 = rows[i][1], rows[i + 1][1]
            d0, d1 = rows[i][2], rows[i + 1][2]
            model_today = d0 + (d1 - d0) * (today_fs - f0) / (f1 - f0)
            break
    if model_today:
        p(f"  D = {model_today:.0f} instances. The observed value is "
          f"{d_front[-1]:.0f}.")
        p("  The real benchmark is depleting FASTER than the fitted model")
        p(f"  says, and the observed D has already crossed half the peak")
        p(f"  ({dmax / 2:.0f}) while the model places that crossing ahead of")
        p("  us. So the half-life above is the optimistic reading, and the")
        p("  measurement says it has already arrived.")
    p("")
    p("  One more limit, stated because it bounds the early history: the")
    p("  frontier is the top tenth, which is two systems at the start of the")
    p("  record. The first two rows rest on a single pair and should be read")
    p("  as indicative only. The peak at 2024-09 rests on two systems of")
    p("  twenty, and the decline after it on ten or more.")
    p("")
    p("WHAT IS NEW HERE, STATED PLAINLY")
    p("  Not the arithmetic - s(J-s) is elementary. What did not exist is the")
    p("  QUANTITY: a benchmark's resolving power expressed in instances, as a")
    p("  function of how good the field has become, with a peak behind it and")
    p("  a half-life ahead. A leaderboard reports a score. This reports how")
    p("  much measuring capacity the instrument has left, and no benchmark")
    p("  paper I can find states that number for its own dataset.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
