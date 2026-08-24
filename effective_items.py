"""How many of a benchmark's items actually separate the two systems at its top.

top_compression.py found that a leaderboard's top is unresolvable because the
field is compressed there, not because the items are correlated. Compressed
means the top systems agree, and agreement has a direct cost in evidence: an
item both systems get right, or both get wrong, contributes nothing to the
comparison the headline makes. Only the items where they differ carry it.

That count has a name on binary boards - McNemar's discordant pairs, b + c -
and no name on continuous ones. The participation ratio generalises it:

    n_eff = (sum_i |d_i|)^2 / (sum_i d_i^2),      d_i = x_ji - x_ki

On binary outcomes |d_i| is 0 or 1 and this is exactly b + c, so one formula
covers both kinds of board and reduces to the standard quantity where the
standard quantity exists. On continuous data with no concentration - iid
Gaussian differences - it is 2/pi = 0.637 of n, which is the baseline every
continuous board is read against rather than 1.

The constructive question follows. With m discordant items and a share p of
them favouring the leader, the paired statistic is about (2p - 1) * sqrt(m), so
reaching t = 2 needs

    m* = (2 / (2p - 1))^2

DISCORDANT items - not items. A benchmark whose top pair splits 52/48 needs
2 500 of them, and adding instances that both systems already solve adds none.
That is the number a benchmark owner needs before deciding to grow an item set,
and no board publishes it.

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  the top pair's effective item count is below half the nominal n on at
      least 7 of the 9 boards.
  P2  the top pair's n_eff is below the board's median pair n_eff on at least
      7 of 9 - the systems a board argues about agree more than two systems
      picked at random from it.
  P3  the implied m* exceeds ten times the board's current n on at least 5 of
      9 boards.
  P4  across the 9 boards, Spearman(tie@1, n_eff/n at the top) is negative:
      boards whose top pair is separated by a larger share of their items have
      fewer systems that could be first.

  Not predicted: anything about CASP14, whose top pair is separated at t = 9.89
  and whose m* is therefore small by construction; it is reported and excluded
  from no count, but a miss there is not surprising.

  Note against my own interest: n_eff counts items with ANY difference, however
  small, so on continuous boards it is generous - a board can look well served
  by items that each move the difference by a thousandth. P1 is therefore
  tested against a statistic biased towards saying the items are sufficient.

SELF-CHECKS (no table if any fails)
  * on the one binary board, n_eff must equal McNemar's b + c exactly, for the
    top pair and for twenty random pairs;
  * iid Gaussian differences must give n_eff / n = 0.637 within 0.03;
  * the m* formula must round-trip: fed the observed discordant count and
    split, it must reproduce the observed t within 10 %.

    python effective_items.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import rank_sets as rs
from sota_audit import mcnemar_exact

SEED = 20260824
DRAWS = 800

MATRICES = {
    "SWE-bench Verified": "swebench_verified_matrix.csv",
    "MTEB English v2": "mteb_eng_v2_wide.csv",
    "HELM classic": "helm_winrate_matrix.csv",
    "ProteinGym DMS": "proteingym/matrix.csv",
    "TabArena 16 models": "tabarena/matrix_one_per_model.csv",
    "TabArena 45 variants": "tabarena/matrix_all45.csv",
    "CASP14": "casp/matrix.csv",
    "LiveBench": "livebench/matrix.csv",
    "MathArena 2025": "matharena/matrix.csv",
}


def load(path: str) -> np.ndarray:
    return pd.read_csv(path, index_col=0).dropna(axis=0).to_numpy(dtype=float)


def n_eff(d: np.ndarray) -> float:
    """Participation ratio of a difference series; b + c on binary outcomes."""
    a = np.abs(d)
    s2 = float((d ** 2).sum())
    return float(a.sum() ** 2 / s2) if s2 > 0 else 0.0


def split_share(d: np.ndarray) -> float:
    """Share of the moved evidence that favours the leader."""
    pos = float(np.abs(d[d > 0]).sum())
    tot = float(np.abs(d).sum())
    return pos / tot if tot > 0 else 0.5


def m_star(p: float, target_t: float = 2.0) -> float:
    """Discordant items needed to reach target_t at this split."""
    edge = 2.0 * p - 1.0
    return float("inf") if abs(edge) < 1e-9 else (target_t / edge) ** 2


def paired_t(d: np.ndarray) -> float:
    sd = float(d.std(ddof=1))
    return float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0


def _check_binary_identity() -> tuple[bool, str]:
    x = load(MATRICES["SWE-bench Verified"])
    sc = x.mean(axis=1)
    order = np.argsort(-sc)
    rng = np.random.default_rng(1)
    pairs = [(int(order[0]), int(order[1]))]
    pairs += [(int(a), int(b)) for a, b in rng.integers(0, x.shape[0], (20, 2)) if a != b]
    worst = 0.0
    for j, k in pairs:
        a, b, _ = mcnemar_exact(x[j], x[k])
        worst = max(worst, abs(n_eff(x[j] - x[k]) - (a + b)))
    return worst < 1e-9, f"n_eff equals McNemar b+c on {len(pairs)} pairs, worst gap {worst:.1e}"


def _check_gaussian_baseline() -> tuple[bool, str]:
    rng = np.random.default_rng(2)
    vals = [n_eff(rng.normal(0, 1, 4000)) / 4000 for _ in range(30)]
    m = float(np.mean(vals))
    return abs(m - 2 / math.pi) < 0.03, f"iid Gaussian gives n_eff/n = {m:.3f} (2/pi = {2 / math.pi:.3f})"


def _check_mstar_roundtrip() -> tuple[bool, str]:
    """Fed the observed split and count, the formula must reproduce observed t."""
    x = load(MATRICES["SWE-bench Verified"])
    sc = x.mean(axis=1)
    order = np.argsort(-sc)
    d = x[order[0]] - x[order[1]]
    m, p = n_eff(d), split_share(d)
    implied = (2 * p - 1) * math.sqrt(m)
    obs = paired_t(d)
    rel = abs(implied - obs) / max(abs(obs), 1e-9)
    ok = rel < 0.10 or (abs(obs) < 0.2 and abs(implied) < 0.2)
    return ok, f"round trip on the top pair: implied t {implied:+.3f}, observed {obs:+.3f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks ...")
    checks = [_check_binary_identity(), _check_gaussian_baseline(), _check_mstar_roundtrip()]
    ok_all = True
    for passed, msg in checks:
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok_all = ok_all and passed
    if not ok_all:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rng = np.random.default_rng(SEED)
    rows = {}
    for name, path in MATRICES.items():
        if not Path(path).exists():
            continue
        x = load(path)
        J, n = x.shape
        sc = x.mean(axis=1)
        order = np.argsort(-sc)
        d = x[order[0]] - x[order[1]]
        med = float(np.median([n_eff(x[a] - x[b]) for a, b in
                               rng.integers(0, J, (400, 2)) if a != b]))
        tie = int((rs.rank_sets(x, draws=DRAWS)["best"] == 1).sum())
        p = split_share(d)
        rows[name] = {"J": J, "n": n, "tie": tie, "t": paired_t(d),
                      "neff": n_eff(d), "share": n_eff(d) / n, "med": med,
                      "p": p, "mstar": m_star(p)}

    L = []
    out = L.append
    out("HOW MANY ITEMS SEPARATE THE TOP TWO")
    out("=" * 104)
    out(f"  {'leaderboard':<22} {'J':>4} {'n':>5} {'tie@1':>6} {'top t':>7} {'n_eff':>7} "
        f"{'share':>7} {'median pair':>12} {'split':>7} {'m* for t=2':>11}")
    p1 = p2 = p3 = 0
    for name, v in rows.items():
        if v["share"] < 0.50:
            p1 += 1
        if v["neff"] < v["med"]:
            p2 += 1
        if v["mstar"] > 10 * v["n"]:
            p3 += 1
        ms = "inf" if not np.isfinite(v["mstar"]) else f"{v['mstar']:,.0f}"
        out(f"  {name:<22} {v['J']:>4} {v['n']:>5} {v['tie']:>6} {v['t']:>7.2f} "
            f"{v['neff']:>7.1f} {100 * v['share']:>6.0f}% {v['med']:>12.1f} "
            f"{100 * v['p']:>6.1f}% {ms:>11}")
    out("")
    shares = [v["share"] for v in rows.values()]
    ties = [v["tie"] for v in rows.values()]
    rho, pv = spearmanr(ties, shares)
    out(f"  P1  top pair separated by under half the items on {p1} of {len(rows)}   "
        f"pre-registered >= 7:  {'HIT' if p1 >= 7 else 'MISS'}")
    out(f"  P2  top pair below the median pair on {p2} of {len(rows)}              "
        f"pre-registered >= 7:  {'HIT' if p2 >= 7 else 'MISS'}")
    out(f"  P3  m* above ten times the current n on {p3} of {len(rows)}            "
        f"pre-registered >= 5:  {'HIT' if p3 >= 5 else 'MISS'}")
    out(f"  P4  Spearman(tie@1, share) = {rho:+.2f} (p {pv:.2f})               "
        f"pre-registered negative:  {'HIT' if rho < 0 else 'MISS'}")
    out("")
    out("  n_eff is the participation ratio of the difference series between the")
    out("  printed first and second: on the binary board it is exactly McNemar's")
    out("  discordant count, on the others it is its continuous analogue, whose")
    out("  no-concentration baseline is 0.637 of n and not 1. share is n_eff / n.")
    out("  median pair is the same statistic over 400 random pairs of the board.")
    out("")
    out("  m* is the number of DISCORDANT items the top comparison would need to")
    out("  reach t = 2 at the split it currently shows. It is not a number of")
    out("  items: instances both systems already solve, or both already fail,")
    out("  move it not at all. A board whose m* is far above its n cannot buy")
    out("  a decidable top by growing the way it grew.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("effective_items_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote effective_items_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
