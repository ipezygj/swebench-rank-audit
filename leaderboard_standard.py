"""Reference implementation of the Leaderboard Reporting Standard (draft 0.1).

One command, any systems x items matrix, every required field of
LEADERBOARD_STANDARD.md - computed from the matrix alone, each field by the
module that defines it, with that module's self-checks run first at the shape
of this matrix. If a check fails, the card is not printed.

    python leaderboard_standard.py --matrix swebench_verified_matrix.csv
    python leaderboard_standard.py --all          # the seven validation matrices

The output is the report card a leaderboard would publish beside its ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import rank_sets as rs
import leaderboard_entropy as le
import leaderboard_geometry as lg
import ordinal_invariance as oi
import information_depletion as idp

VERSION = "0.1"
SEED = 20260823

MATRICES = {
    "SWE-bench Verified": "swebench_verified_matrix.csv",
    "MTEB English v2": "mteb_eng_v2_wide.csv",
    "HELM classic": "helm_winrate_matrix.csv",
    "ProteinGym DMS": "proteingym/matrix.csv",
    "TabArena 16 models": "tabarena/matrix_one_per_model.csv",
    "TabArena 45 variants": "tabarena/matrix_all45.csv",
    "MathArena 2025": "matharena/matrix.csv",
}


def load(path: str) -> tuple[np.ndarray, list]:
    df = pd.read_csv(path, index_col=0).dropna(axis=0)
    x = df.to_numpy(dtype=float)
    if np.nanmax(np.abs(x)) > 1.0 + 1e-9:
        raise SystemExit(f"{path}: values outside [-1, 1] - a metadata column "
                         "has been read as a system. Refusing.")
    return x, list(df.index)


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def card(name: str, path: str, draws: int, samples: int) -> dict:
    x, names = load(path)
    J, n = x.shape
    binary = bool(np.isin(x, [0.0, 1.0]).all())
    rng = np.random.default_rng(SEED)
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

    # R2 rank sets (self-checks inside rank_sets are run by its main; here we
    # re-run the ones that matter for this card).
    r = rs.rank_sets(x, draws=draws)
    inside = bool((r["best"] <= r["observed"]).all()
                  and (r["observed"] <= r["worst"]).all())
    beats = r["beats"]
    viol = lg.transitivity_violations(beats)
    width = r["worst"] - r["best"]

    # R3 / R4 entropy
    H = le.log_extensions(beats, samples, rng)
    ceiling = gammaln(J + 1) / math.log(2)
    k = min(10, J)
    order = np.argsort(-x.mean(axis=1), kind="stable")
    sub = beats[np.ix_(order[:k], order[:k])]
    Hk = le.log_extensions(sub, samples, rng)
    ceil_k = gammaln(k + 1) / math.log(2)

    # R6 tiers
    height = lg.longest_chain(beats)
    anti = lg.largest_antichain(beats)

    # R7 invariance, difficulty split, both floors.
    solved = x.mean(axis=0)
    hard = solved <= np.median(solved)
    d = oi.decompose(x, hard)
    fl = [oi.decompose(x, m) for m in oi.random_masks(n, 40, rng)]
    fm = float(np.mean([f["metric"] for f in fl]))
    fo = float(np.mean([f["ordinal"] for f in fl]))

    # R8 discordance (defined for any matrix as expected |x_j - x_k| summed,
    # which reduces to the item count on binary data).
    def disc(xx):
        Jx = xx.shape[0]
        if Jx < 2:
            return float("nan")
        iu = np.triu_indices(Jx, k=1)
        return float(np.mean(np.abs(xx[iu[0]] - xx[iu[1]]).sum(axis=1)))
    top = order[: max(2, J // 10)]
    D_all = disc(x)
    D_top = disc(x[top])

    return {
        "name": name, "J": J, "n": n, "binary": binary, "hash": h,
        "inside": inside, "viol": viol,
        "median_width": float(np.median(width)), "tie1": int((r["best"] == 1).sum()),
        "H": H["bits"], "H_frac": H["bits"] / ceiling,
        "Hk": Hk["bits"], "k": k, "ceil_k": ceil_k,
        "established": float(beats.sum() / (J * (J - 1))),
        "height": height, "antichain": anti,
        "drift_m": d["metric"], "floor_m": fm,
        "drift_o": d["ordinal"], "floor_o": fo,
        "D_all": D_all, "D_top": D_top,
        "leader": names[int(order[0])],
    }


def render(c: dict) -> str:
    L = []
    p = L.append
    p(f"LEADERBOARD REPORT CARD - {c['name']}")
    p("=" * 74)
    p(f"R1 shape          {c['J']} systems x {c['n']} items, "
      f"{'binary' if c['binary'] else 'continuous'}, matrix sha256 {c['hash']}")
    p(f"R2 rank sets      median width {c['median_width']:.0f} of {c['J']} "
      f"({100 * c['median_width'] / c['J']:.0f} %); "
      f"{c['tie1']} system(s) could be first; "
      f"observed rank inside its set: {c['inside']}")
    p(f"R3 entropy        H = {c['H']:.1f} bits = {100 * c['H_frac']:.1f} % of "
      f"log2({c['J']}!); the table is one of 2^{c['H']:.0f} equally supported")
    p(f"R4 top-{c['k']} resolution {c['Hk']:.1f} of {c['ceil_k']:.1f} bits undetermined"
      + ("  (complete antichain)" if abs(c["Hk"] - c["ceil_k"]) < 0.3 else ""))
    p(f"R5 established    {100 * c['established']:.1f} % of ordered pairs "
      f"(transitivity violations {c['viol']})")
    p(f"R6 tiers          {c['height']} resolved of {c['J']} printed; "
      f"largest antichain {c['antichain']}")
    flag_m = "EXCEEDS" if c["drift_m"] > 1.5 * c["floor_m"] else "within"
    flag_o = "EXCEEDS" if c["drift_o"] > 1.5 * c["floor_o"] else "within"
    p(f"R7 invariance     metric drift {c['drift_m']:.3f} vs floor "
      f"{c['floor_m']:.3f} ({flag_m}); ordinal {c['drift_o']:.3f} vs "
      f"{c['floor_o']:.3f} ({flag_o})")
    p(f"R8 discordance D  whole field {c['D_all']:.1f}, top decile "
      f"{c['D_top']:.1f} items-equivalent")
    p(f"R9 provenance     leaderboard_standard {VERSION}, git {git_rev()}, "
      f"seed {SEED}, {date.today().isoformat()}")
    p(f"   leader as printed: {c['leader'][:60]}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix")
    ap.add_argument("--name", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--out", default="leaderboard_standard_results.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    targets = MATRICES if a.all else {a.name or Path(a.matrix).stem: a.matrix}
    if not a.all and not a.matrix:
        raise SystemExit("give --matrix or --all")

    # Run the defining modules' self-checks once, at the shapes they test.
    print("self-checks of the defining modules")
    ok = True
    for label, fn in (("rank_sets identical", rs._check_identical),
                      ("rank_sets separated", rs._check_separated),
                      ("entropy closed forms", le._check_closed_forms),
                      ("geometry height", lg._check_height_synthetic),
                      ("invariance 2PL split", oi._check_2pl)):
        passed, msg = fn()
        print(f"  [{'ok  ' if passed else 'FAIL'}] {label}: {msg}")
        ok = ok and passed
    if not ok:
        print("\nA CHECK FAILED - no report card is printed.")
        return 1

    cards, texts = [], []
    for name, path in targets.items():
        if not Path(path).exists():
            print(f"\n{name}: {path} missing, skipped")
            continue
        print(f"\ncomputing {name} ...")
        c = card(name, path, a.draws, a.samples)
        if not c["inside"] or c["viol"]:
            print(f"  {name}: rank sets inconsistent or order intransitive - "
                  "card withheld")
            continue
        cards.append(c)
        texts.append(render(c))

    if len(cards) > 1:
        L = ["SUMMARY TABLE", "=" * 74,
             f"  {'leaderboard':<22} {'J':>4} {'n':>4} {'tie@1':>6} "
             f"{'H/ceil':>7} {'top-k':>11} {'tiers':>6} {'estab':>6} "
             f"{'ord.drift':>10}"]
        for c in cards:
            L.append(f"  {c['name']:<22} {c['J']:>4} {c['n']:>4} {c['tie1']:>6} "
                     f"{100 * c['H_frac']:>6.1f}% {c['Hk']:>5.1f}/{c['ceil_k']:<4.1f} "
                     f"{c['height']:>6} {100 * c['established']:>5.1f}% "
                     f"{c['drift_o'] / c['floor_o']:>9.1f}x")
        texts.append("\n".join(L))

    text = "\n\n".join(texts)
    print("\n" + text)
    Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
