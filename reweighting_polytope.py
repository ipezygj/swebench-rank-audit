"""Who could be number one if the benchmark had been assembled differently?

SWE-bench Verified is 46.2 % django. Not by design - by how the dataset was
collected. The headline score is a weighted average over twelve repositories
and the weights are instance counts, which is to say the weights are an
accident that nobody has ever had to defend. Change them and the ranking
changes, and the interesting question is not whether it changes but by HOW
LITTLE.

TWO QUESTIONS, BOTH EXACT
--------------------------
Score of system j under repository weights w is linear: score_j(w) = sum_r
w_r p_jr, with p_jr the solve rate of j on repo r. So everything here is a
linear program and the answers are exact, not statistical. This sits beside
the confidence sets rather than inside them: one says which systems the noise
cannot separate, this says which systems the composition cannot separate.

  1. THE CHAMPION SET. For which systems does there exist a weighting under
     which they lead? Feasibility of

         (p_j - p_t) . w <= 0  for all j,   w >= 0,   sum w = 1

     Note that a champion need not win at any single repository: a system
     second everywhere can still lead a mixture, so checking the corners is
     not enough and the LP is not decoration.

  2. THE PRICE OF THE CROWN. If a system can lead, what is the SMALLEST move
     from the current weights that gets it there? Measured as total variation
     distance, so the answer reads as a percentage of the benchmark's mass
     that would have to be reassigned. Minimising an L1 deviation is again a
     linear program, with d_r >= |w_r - w0_r| split into two constraints.

That number is the one worth having. "Reweight the benchmark and the ranking
changes" is a truism. "System number seven leads if four per cent of the
benchmark's weight moves between repositories it already contains" is a fact
about how firmly the published order is held.

SELF-CHECKS THAT CAN FAIL
--------------------------
  * a system that beats everyone on every repository must be a champion at
    zero cost;
  * a system that loses to someone on every repository need not be excluded -
    that is the interior-champion case above - so the exclusion check uses a
    system dominated by a single rival everywhere, which genuinely cannot
    lead;
  * the LP champion set must agree with a brute-force grid over the simplex
    on a small synthetic case;
  * the current weights must reproduce the published scores exactly.

    python reweighting_polytope.py [--matrix ...] [--top 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

SEED = 20260823


def repo_of(instance: str) -> str:
    return instance.split("__")[0]


def repo_matrix(df: pd.DataFrame):
    repos = np.array([repo_of(c) for c in df.columns])
    names = sorted(set(repos))
    x = df.to_numpy(dtype=float)
    p = np.column_stack([x[:, repos == r].mean(axis=1) for r in names])
    counts = np.array([(repos == r).sum() for r in names], dtype=float)
    return p, names, counts / counts.sum()


def can_lead(p: np.ndarray, t: int) -> bool:
    """Is there any weighting in the simplex under which t leads?"""
    J, R = p.shape
    a_ub = p - p[t][None, :]              # (p_j - p_t) . w <= 0
    a_ub = np.delete(a_ub, t, axis=0)
    res = linprog(c=np.zeros(R), A_ub=a_ub, b_ub=np.zeros(len(a_ub)),
                  A_eq=np.ones((1, R)), b_eq=[1.0],
                  bounds=[(0, 1)] * R, method="highs")
    return bool(res.status == 0)


def price_of_crown(p: np.ndarray, t: int, w0: np.ndarray):
    """Smallest total-variation move from w0 that puts t in the lead."""
    J, R = p.shape
    # variables: w (R), d (R); minimise sum d
    c = np.concatenate([np.zeros(R), np.ones(R)])
    rows, rhs = [], []
    dom = np.delete(p - p[t][None, :], t, axis=0)
    for r in dom:
        rows.append(np.concatenate([r, np.zeros(R)]))
        rhs.append(0.0)
    for i in range(R):                     # w_i - d_i <= w0_i
        row = np.zeros(2 * R); row[i] = 1.0; row[R + i] = -1.0
        rows.append(row); rhs.append(w0[i])
    for i in range(R):                     # -w_i - d_i <= -w0_i
        row = np.zeros(2 * R); row[i] = -1.0; row[R + i] = -1.0
        rows.append(row); rhs.append(-w0[i])
    a_eq = np.zeros((1, 2 * R)); a_eq[0, :R] = 1.0
    res = linprog(c=c, A_ub=np.array(rows), b_ub=np.array(rhs),
                  A_eq=a_eq, b_eq=[1.0],
                  bounds=[(0, 1)] * R + [(0, None)] * R, method="highs")
    if res.status != 0:
        return None, None
    w = res.x[:R]
    return float(res.x[R:].sum() / 2.0), w


# --- self-checks ------------------------------------------------------------

def _check_dominant_is_free() -> tuple[bool, str]:
    p = np.array([[0.9, 0.9, 0.9], [0.5, 0.6, 0.4], [0.3, 0.2, 0.7]])
    w0 = np.array([0.5, 0.3, 0.2])
    ok1 = can_lead(p, 0)
    cost, _ = price_of_crown(p, 0, w0)
    ok = ok1 and cost is not None and cost < 1e-9
    return ok, f"system dominant everywhere: champion={ok1}, cost={cost}"


def _check_dominated_excluded() -> tuple[bool, str]:
    """Dominated by ONE rival on every repo -> can never lead."""
    p = np.array([[0.9, 0.8, 0.7], [0.4, 0.3, 0.2], [0.5, 0.6, 0.5]])
    ok = not can_lead(p, 1)
    return ok, f"system dominated by a single rival everywhere: champion={not ok}"


def _check_interior_champion() -> tuple[bool, str]:
    """A system that wins no repository can still lead a mixture."""
    p = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.6]])
    ok = can_lead(p, 2) and not (p[2] >= p.max(axis=0)).any()
    return ok, "a system winning no single repo is still a champion of mixtures"


def _check_against_grid() -> tuple[bool, str]:
    """LP champion set must equal a brute-force grid on the simplex."""
    rng = np.random.default_rng(11)
    p = rng.random((6, 3))
    lp = {t for t in range(6) if can_lead(p, t)}
    grid = set()
    step = 0.01
    for a in np.arange(0, 1 + step, step):
        for b in np.arange(0, 1 - a + step, step):
            w = np.array([a, b, max(0.0, 1 - a - b)])
            grid.add(int(np.argmax(p @ w)))
    ok = lp == grid
    return ok, f"LP champions {sorted(lp)} vs grid {sorted(grid)}"


def _check_weights_reproduce() -> tuple[bool, str]:
    """Current weights must give back the published scores exactly."""
    df = pd.read_csv("swebench_verified_matrix.csv", index_col=0)
    p, _, w0 = repo_matrix(df)
    direct = df.mean(axis=1).to_numpy()
    ok = bool(np.allclose(p @ w0, direct, atol=1e-12))
    return ok, f"instance-count weights reproduce the published scores: {ok}"


def run_checks() -> bool:
    ok = True
    for passed, msg in (_check_dominant_is_free(), _check_dominated_excluded(),
                        _check_interior_champion(), _check_against_grid(),
                        _check_weights_reproduce()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="swebench_verified_matrix.csv")
    ap.add_argument("--top", type=int, default=30,
                    help="how many systems to test for the crown")
    ap.add_argument("--out", default="reweighting_polytope_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(a.matrix, index_col=0)
    p, repos, w0 = repo_matrix(df)
    names = list(df.index)
    J, R = p.shape
    scores = p @ w0
    order = np.argsort(-scores, kind="stable")

    print(f"matrix {a.matrix}: {J} systems, {R} repositories")
    print("\nself-checks")
    if not run_checks():
        print("\nA CHECK FAILED - no headline number is printed.")
        return 1

    equal = np.full(R, 1.0 / R)
    s_eq = p @ equal
    order_eq = np.argsort(-s_eq, kind="stable")

    L = []
    pp = L.append
    pp("WHO COULD BE NUMBER ONE IF THE BENCHMARK WERE WEIGHTED DIFFERENTLY?")
    pp("=" * 74)
    pp(f"{J} systems, {R} repositories, 500 instances")
    pp("")
    pp("  the weights nobody chose")
    for i in np.argsort(-w0):
        pp(f"    {repos[i]:<16} {100 * w0[i]:>5.1f} %")
    pp("")
    pp("EQUAL WEIGHT PER REPOSITORY INSTEAD OF PER INSTANCE")
    pp(f"  leader as published      {names[order[0]][:46]}  {scores[order[0]]:.3f}")
    pp(f"  leader at equal weights  {names[order_eq[0]][:46]}  {s_eq[order_eq[0]]:.3f}")
    moved = int((order[:10] != order_eq[:10]).sum())
    iu = np.triu_indices(J, k=1)
    ag = float(np.mean(np.sign(scores[:, None] - scores[None, :])[iu]
                       == np.sign(s_eq[:, None] - s_eq[None, :])[iu]))
    pp(f"  positions changed in the top ten   {moved}")
    pp(f"  pairwise order preserved           {100 * ag:.1f} %")
    pp("")
    pp(f"THE PRICE OF THE CROWN (top {a.top} systems as published)")
    pp(f"  {'rank':>4} {'system':<44} {'score':>7} {'TV move':>9}")
    champions = 0
    cheap = []
    for rank, idx in enumerate(order[:a.top], start=1):
        cost, w = price_of_crown(p, int(idx), w0)
        if cost is None:
            pp(f"  {rank:>4} {names[idx][:44]:<44} {scores[idx]:>7.3f}"
               f" {'never':>9}")
            continue
        champions += 1
        pp(f"  {rank:>4} {names[idx][:44]:<44} {scores[idx]:>7.3f}"
           f" {100 * cost:>8.1f} %")
        if cost > 1e-9:
            cheap.append((rank, names[idx], cost, w))
    pp("")
    pp(f"  {champions} of the top {a.top} can be made leader by some weighting.")
    costs = np.array([c for _, _, c, _ in cheap])
    for lim in (0.05, 0.10, 0.25):
        pp(f"    with a move of at most {100 * lim:>4.0f} %: "
           f"{int((costs <= lim).sum())} systems")
    pp("    A cost near 100 % means the weight has to collapse onto a single")
    pp("    repository. Technically a champion, practically not an argument.")
    pp("    Systems tied at the published top are excluded here: they lead at")
    pp("    zero cost because they already do.")
    if cheap:
        cheap.sort(key=lambda t: t[2])
        r0, n0, c0, w0b = cheap[0]
        pp("")
        pp(f"  The cheapest usurper is rank {r0}: moving {100 * c0:.1f} % of the")
        pp("  benchmark's weight between repositories it already contains puts")
        pp(f"  {n0[:44]} on top.")
        pp("  weights that do it:")
        for i in np.argsort(-w0b):
            if w0b[i] > 0.005:
                pp(f"    {repos[i]:<16} {100 * w0[i]:>5.1f} % -> "
                   f"{100 * w0b[i]:>5.1f} %")
    pp("")
    pp("  This is exact arithmetic, not a test. No noise, no null, no p-value:")
    pp("  these weightings exist. What it adds to the confidence sets is a")
    pp("  second reason the published order is soft. The sets say the noise")
    pp("  cannot separate 19 systems; this says the composition cannot either,")
    pp("  and the composition was never argued for in the first place.")

    text = "\n".join(L)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
