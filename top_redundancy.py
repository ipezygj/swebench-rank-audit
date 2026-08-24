"""Are the systems crowding the top doing the same work, or different work?

The chain so far: the top of a board is undecidable because of the shape of the
field there (`top_compression.py`), the headline comparison is carried by a
small fraction of the items (`effective_items.py`), the field's top is enriched
in one base-model family on five boards (`family_clustering.py`,
`family_generalises.py`) - and collapsing that family does not uncrowd it,
because 71 % of top pairs are different families the board still cannot
separate.

That leaves one question. Those cross-family systems at the top: are they one
ability measured with noise, or several different competences that a single
ranking is the wrong instrument for? The difference is visible in what they
solve, not in what they score.

THE NULL

Fit each binary board a Rasch model - system ability, item difficulty, and
nothing else - and simulate boards from it. That null keeps every system's
score and every item's solve rate and assumes only that, given ability and
difficulty, outcomes are independent. Comparing the real board with it isolates
the dependence between systems and nothing else.

  union(k)   share of items solved by at least one of the top k
  unique     items solved by exactly one of the top ten
  n90        how many of the top systems it takes to cover 90 % of the items

DIRECTION, STATED BEFORE THE RUN

I expect REDUNDANCY, not complementarity: pair sharpness on these boards is
below 1, which is already a statement that systems fail the same items, so the
real union should fall SHORT of what independence predicts. I record here that
the opposite result would be the more interesting one - it would say the top of
a leaderboard is a set of specialists and that ranking them at all is a
category error - and that I would report it as such rather than as a surprise
to be explained away.

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  union(10) below the null's 5th percentile on at least 2 of the 3 boards.
  P2  n90 above the null's 95th percentile on at least 2 of 3.
  P3  the uniquely-solved count below the null's 5th percentile on at least
      2 of 3.
  P4  the same redundancy appears for a middle-of-board group of the same
      size, on at least 2 of 3 - that is, it is a property of the field and
      not of its top. This one I am unsure of.

  Not predicted: anything about SWE-bench Test, which has 24 systems, so its
  "top ten" is nearly half the board.

SELF-CHECKS (no table if any fails)
  * the fitted model must reproduce the board's own margins: mean absolute
    error under 0.02 on both system scores and item solve rates;
  * calibration - ten boards simulated FROM the fitted model, run through the
    whole pipeline, must reject at most once between them. A null that rejects
    its own data measures the fit, not the field;
  * the null distribution of union(10) must have spread on every board.

    python top_redundancy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260824
SIMS = 199
TOPK = 10
BOARDS = {
    "SWE-bench Verified": "swebench_verified_matrix.csv",
    "SWE-bench Lite": "swebench_lite_matrix.csv",
    "SWE-bench Test": "swebench_test_matrix.csv",
}


def load(path):
    return pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_rasch(x, iters=400, tol=1e-9):
    """Joint MLE for ability and difficulty, alternating Newton steps.

    Perfect rows and columns have no finite estimate; they are clipped to a
    wide but finite range, which keeps their margins right in simulation.
    """
    J, n = x.shape
    r = x.sum(axis=1)
    c = x.sum(axis=0)
    theta = np.zeros(J)
    b = np.zeros(n)
    for _ in range(iters):
        p = sigmoid(theta[:, None] - b[None, :])
        g = r - p.sum(axis=1)
        h = np.maximum((p * (1 - p)).sum(axis=1), 1e-9)
        theta_new = np.clip(theta + g / h, -12, 12)
        p = sigmoid(theta_new[:, None] - b[None, :])
        gb = p.sum(axis=0) - c
        hb = np.maximum((p * (1 - p)).sum(axis=0), 1e-9)
        b_new = np.clip(b + gb / hb, -12, 12)
        b_new -= b_new.mean()
        if max(np.abs(theta_new - theta).max(), np.abs(b_new - b).max()) < tol:
            theta, b = theta_new, b_new
            break
        theta, b = theta_new, b_new
    return theta, b


def simulate(theta, b, rng):
    p = sigmoid(theta[:, None] - b[None, :])
    return (rng.random(p.shape) < p).astype(float)


def stats(x, idx, topk=TOPK):
    """union curve, uniquely solved, and coverage depth for one group."""
    sub = x[idx][:topk]
    solved = sub > 0.5
    union = np.maximum.accumulate(solved, axis=0).mean(axis=1)
    hits = solved.sum(axis=0)
    unique = int(((hits == 1)).sum())
    full = union[-1]
    target = 0.90
    n90 = int(np.argmax(union >= target) + 1) if (union >= target).any() else len(union) + 1
    return {"union": float(full), "unique": unique, "n90": n90, "curve": union}


def group_indices(x, kind, topk=TOPK):
    order = np.argsort(-x.mean(axis=1))
    if kind == "top":
        return order[:topk]
    mid = len(order) // 2
    lo = max(0, mid - topk // 2)
    return order[lo:lo + topk]


def pct_of(observed, draws):
    d = np.asarray(draws, dtype=float)
    return float((d < observed).mean())


def analyse(x, theta, b, rng, kind, sims=SIMS):
    idx = group_indices(x, kind)
    obs = stats(x, idx)
    u, q, m = [], [], []
    for _ in range(sims):
        y = simulate(theta, b, rng)
        s = stats(y, group_indices(y, kind))
        u.append(s["union"])
        q.append(s["unique"])
        m.append(s["n90"])
    return {"obs": obs,
            "u_p": pct_of(obs["union"], u), "u_med": float(np.median(u)), "u_sd": float(np.std(u)),
            "q_p": pct_of(obs["unique"], q), "q_med": float(np.median(q)),
            "m_p": pct_of(obs["n90"], m), "m_med": float(np.median(m))}


def _check_margins(x, theta, b) -> tuple[bool, float, float]:
    p = sigmoid(theta[:, None] - b[None, :])
    er = float(np.abs(p.mean(axis=1) - x.mean(axis=1)).mean())
    ec = float(np.abs(p.mean(axis=0) - x.mean(axis=0)).mean())
    return er < 0.02 and ec < 0.02, er, ec


def _check_calibration(theta, b, rng) -> tuple[bool, str]:
    """A board generated from the fitted model must not look dependent."""
    rejects = 0
    for _ in range(10):
        y = simulate(theta, b, rng)
        th2, b2 = fit_rasch(y)
        a = analyse(y, th2, b2, rng, "top", sims=49)
        if a["u_p"] < 0.05:
            rejects += 1
    return rejects <= 1, f"{rejects} of 10 simulated boards reject their own null (allows <= 1)"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    fits, rows = {}, {}
    ok_margin, margin_msgs = True, []
    for name, path in BOARDS.items():
        if not Path(path).exists():
            continue
        print(f"  fitting {name} ...")
        x = load(path)
        theta, b = fit_rasch(x)
        good, er, ec = _check_margins(x, theta, b)
        ok_margin = ok_margin and good
        margin_msgs.append(f"{name}: system {er:.4f}, item {ec:.4f}")
        fits[name] = (x, theta, b)

    print("self-checks ...")
    print(f"  [{'ok  ' if ok_margin else 'FAIL'}] fitted margins reproduce the board  "
          + "; ".join(margin_msgs))
    cal_board = "SWE-bench Lite" if "SWE-bench Lite" in fits else list(fits)[0]
    ok_cal, msg_cal = _check_calibration(fits[cal_board][1], fits[cal_board][2], rng)
    print(f"  [{'ok  ' if ok_cal else 'FAIL'}] {msg_cal} (on {cal_board})")

    for name, (x, theta, b) in fits.items():
        print(f"  measuring {name} ...")
        rows[name] = {"top": analyse(x, theta, b, rng, "top"),
                      "mid": analyse(x, theta, b, rng, "mid"),
                      "J": x.shape[0], "n": x.shape[1]}

    flat = [k for k, v in rows.items() if v["top"]["u_sd"] <= 1e-12]
    ok_spread = not flat
    print(f"  [{'ok  ' if ok_spread else 'FAIL'}] the null has spread on "
          f"{len(rows) - len(flat)} of {len(rows)} boards")

    if not (ok_margin and ok_cal and ok_spread):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("ARE THE SYSTEMS AT THE TOP DOING THE SAME WORK?")
    p("=" * 104)
    p(f"  {'leaderboard':<22} {'J':>4} {'n':>6} {'group':>6} {'union10':>8} {'null':>7} "
      f"{'pct':>6} {'unique':>7} {'null':>6} {'pct':>6} {'n90':>5} {'null':>5} {'pct':>6}")
    for name, v in rows.items():
        for kind in ("top", "mid"):
            a = v[kind]
            o = a["obs"]
            p(f"  {name:<22} {v['J']:>4} {v['n']:>6} {kind:>6} "
              f"{100 * o['union']:>7.1f}% {100 * a['u_med']:>6.1f}% {a['u_p']:>6.3f} "
              f"{o['unique']:>7} {a['q_med']:>6.0f} {a['q_p']:>6.3f} "
              f"{o['n90']:>5} {a['m_med']:>5.0f} {a['m_p']:>6.3f}")
    p("")
    tops = [v["top"] for v in rows.values()]
    mids = [v["mid"] for v in rows.values()]
    p1 = sum(1 for a in tops if a["u_p"] < 0.05)
    p2 = sum(1 for a in tops if a["m_p"] > 0.95)
    p3 = sum(1 for a in tops if a["q_p"] < 0.05)
    p4 = sum(1 for a in mids if a["u_p"] < 0.05)
    p(f"  P1  union(10) below the null on {p1} of {len(tops)}        "
      f"pre-registered >= 2:  {'HIT' if p1 >= 2 else 'MISS'}")
    p(f"  P2  n90 above the null on {p2} of {len(tops)}              "
      f"pre-registered >= 2:  {'HIT' if p2 >= 2 else 'MISS'}")
    p(f"  P3  uniquely solved below the null on {p3} of {len(tops)}  "
      f"pre-registered >= 2:  {'HIT' if p3 >= 2 else 'MISS'}")
    p(f"  P4  the same holds mid-board on {p4} of {len(mids)}        "
      f"pre-registered >= 2:  {'HIT' if p4 >= 2 else 'MISS'}")
    p("")
    p("  pct is where the real board falls in its own null: 0.000 means every")
    p("  simulated board covered more items with its top ten than the real one")
    p("  does. The null keeps each system's score and each item's solve rate and")
    p("  assumes only that outcomes are independent given the two, so the whole")
    p("  distance is dependence between systems.")
    p("")
    p("  A top ten that covers less than independence predicts is a top ten")
    p("  failing the same instances. That is redundancy, and it is the reason no")
    p("  quantity of items of the same kind decides the top: the systems are not")
    p("  disagreeing about which problems are hard.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("top_redundancy_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote top_redundancy_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
