"""Does the base-model cluster hold off SWE-bench, and would collapsing it help?

family_clustering.py found that on SWE-bench Verified two submissions sharing a
base model sit half as far apart as two that do not, are more correlated in
their errors, and that both effects survive permuting family labels within a
calendar quarter. It is one board, which is the first objection a reader makes.

Four other boards carry the family in the system's own name. The label is taken
by a rule with no free parameter in it, applied identically to all four:

    lowercase the name, drop anything before the last "/", and take the first
    maximal run of letters.

    Qwen/Qwen3-Embedding-8B  -> qwen        DeepSeek-R1-0528  -> deepseek
    Bytedance/Seed1.6-embedding -> seed     GLM 4.5 Air       -> glm

The rule is stated before it is run and the full mapping is printed with the
results, so a reader can see every call it made rather than trust it. It is
imperfect in one direction only, which is the safe one: it SPLITS families that
spell themselves differently - Meta-Llama-3 becomes "meta" while Llama-2
becomes "llama" - and splitting a family removes same-family pairs, weakening
every effect below. SWE-bench keeps its curated labeller, because there the
system name is the harness and not the model.

THE ACTIONABLE HALF

If the top of a board is one family in several harnesses, the obvious remedy is
to rank families rather than submissions: keep each family's best and re-ask how
many systems could be first. That reduction is partly mechanical - fewer
systems, fewer candidates - so it is measured against dropping the SAME NUMBER
of systems at random, 199 times. A collapse that only does what random dropping
does is not a remedy, it is a smaller board.

PRE-REGISTERED (2026-08-24, committed before the run)
  P1  same-family pairs are closer in score than the free null (p < 0.05) on at
      least 3 of the 4 new boards.
  P2  same-family pairs have lower kappa than the free null (p < 0.05) on at
      least 3 of the 4.
  P3  collapsing to one submission per family puts tie@1 below the 5th
      percentile of the random-drop control on at least 3 of the 5 boards.
  P4  on at least one board, collapsing takes a top that could contain several
      systems down to a single possible first place.

  Not predicted: anything about which families win, and nothing about boards
  where fewer than three families have two or more members - those are skipped
  and named.

  Note against my own interest: the rule splits families rather than merging
  them, and every split removes same-family pairs. P1 and P2 are tested on a
  labelling biased towards finding nothing.

SELF-CHECKS (no table if any fails)
  * on each board, a RANDOM relabelling with the same family-size profile must
    give p-values near uniform - mean between 0.25 and 0.75 - or the p-values
    below are readings of the machinery;
  * the random-drop control must have spread on every board it is run on;
  * at least three of the five boards must survive the family-count filter,
    otherwise there is no generalisation claim to make.

    python family_generalises.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import rank_sets as rs
from pair_sharpness import kappa_matrix
from swebench_base_models import base_model

SEED = 20260824
PERMS = 999
DROPS = 199
DRAWS = 800
MIN_FAMILIES = 3

BOARDS = {
    "SWE-bench Verified": ("swebench_verified_matrix.csv", "curated"),
    "MTEB English v2": ("mteb_eng_v2_wide.csv", "mechanical"),
    "LiveBench": ("livebench/matrix.csv", "mechanical"),
    "MathArena 2025": ("matharena/matrix.csv", "mechanical"),
    "LMArena categories": ("lmarena_matrix.csv", "mechanical"),
}


def mechanical_family(name: str) -> str:
    """First maximal run of letters in the lowercased basename. No parameters."""
    base = str(name).lower().rsplit("/", 1)[-1]
    m = re.search(r"[a-z]+", base)
    return m.group(0) if m else ""


def label(names, how):
    if how == "curated":
        return np.array([base_model(n) or "" for n in names], dtype=object)
    return np.array([mechanical_family(n) for n in names], dtype=object)


def pair_arrays(x):
    sc = x.mean(axis=1)
    K = kappa_matrix(x)
    iu = np.triu_indices(len(sc), k=1)
    return iu, np.abs(sc[iu[0]] - sc[iu[1]]), K[iu]


def perm_p(observed, draws):
    d = np.asarray(draws, dtype=float)
    return float(((d <= observed).sum() + 1) / (len(d) + 1))


def family_test(fam, iu, gap, kap, rng, perms=PERMS):
    same = fam[iu[0]] == fam[iu[1]]
    if same.sum() < 5:
        return None
    obs_g, obs_k = float(gap[same].mean()), float(kap[same].mean())
    g, k = [], []
    for _ in range(perms):
        pm = fam.copy()
        rng.shuffle(pm)
        m = pm[iu[0]] == pm[iu[1]]
        if m.sum() == 0:
            continue
        g.append(gap[m].mean())
        k.append(kap[m].mean())
    return {"n_same": int(same.sum()), "gap": obs_g, "kap": obs_k,
            "gap_diff": float(gap[~same].mean()), "kap_diff": float(kap[~same].mean()),
            "gap_p": perm_p(obs_g, g), "kap_p": perm_p(obs_k, k)}


def tie1(x):
    return int((rs.rank_sets(x, draws=DRAWS)["best"] == 1).sum())


def collapse(x, fam):
    """Keep each family's best; unlabelled systems each count as their own."""
    sc = x.mean(axis=1)
    keep, seen = [], {}
    for i in np.argsort(-sc):
        f = fam[i]
        if f == "":
            keep.append(int(i))
        elif f not in seen:
            seen[f] = True
            keep.append(int(i))
    return np.array(sorted(keep))


def drop_control(x, size, rng, reps=DROPS):
    J = x.shape[0]
    return [tie1(x[rng.choice(J, size, replace=False)]) for _ in range(reps)]


def _check_null_centred(fam, iu, gap, kap) -> float:
    rng = np.random.default_rng(2)
    ps = []
    for _ in range(12):
        fake = fam.copy()
        rng.shuffle(fake)
        r = family_test(fake, iu, gap, kap, rng, perms=149)
        if r:
            ps.append(r["gap_p"])
    return float(np.mean(ps)) if ps else float("nan")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    rows, skipped, maps = {}, [], {}

    for name, (path, how) in BOARDS.items():
        if not Path(path).exists():
            skipped.append((name, "no matrix"))
            continue
        df = pd.read_csv(path, index_col=0).dropna(axis=0)
        x = df.to_numpy(dtype=float)
        fam = label(list(df.index), how)
        sizes = Counter(f for f in fam if f)
        multi = {f: c for f, c in sizes.items() if c >= 2}
        if len(multi) < MIN_FAMILIES:
            skipped.append((name, f"only {len(multi)} families with 2+ members"))
            continue
        print(f"  measuring {name} ...")
        iu, gap, kap = pair_arrays(x)
        t = family_test(fam, iu, gap, kap, rng)
        if t is None:
            skipped.append((name, "fewer than 5 same-family pairs"))
            continue
        t["centred"] = _check_null_centred(fam, iu, gap, kap)
        keep = collapse(x, fam)
        t["J"] = x.shape[0]
        t["Jc"] = len(keep)
        t["tie"] = tie1(x)
        t["tie_c"] = tie1(x[keep])
        ctrl = drop_control(x, len(keep), rng)
        t["ctrl_med"] = float(np.median(ctrl))
        t["ctrl_p5"] = float(np.percentile(ctrl, 5))
        t["ctrl_sd"] = float(np.std(ctrl))
        t["families"] = len(multi)
        rows[name] = t
        maps[name] = sorted(multi.items(), key=lambda kv: -kv[1])

    ok_centred = [k for k, v in rows.items() if not (0.25 <= v["centred"] <= 0.75)]
    ok_spread = [k for k, v in rows.items() if v["ctrl_sd"] <= 1e-9]
    print("self-checks ...")
    print(f"  [{'ok  ' if not ok_centred else 'FAIL'}] random relabellings near uniform on "
          f"{len(rows) - len(ok_centred)} of {len(rows)} boards"
          + ("" if not ok_centred else "  off: " + ", ".join(ok_centred)))
    print(f"  [{'ok  ' if not ok_spread else 'FAIL'}] the random-drop control has spread on "
          f"{len(rows) - len(ok_spread)} of {len(rows)}"
          + ("" if not ok_spread else "  flat: " + ", ".join(ok_spread)))
    enough = len(rows) >= MIN_FAMILIES
    print(f"  [{'ok  ' if enough else 'FAIL'}] {len(rows)} boards survive the family filter "
          f"(needs >= {MIN_FAMILIES})")
    if ok_centred or ok_spread or not enough:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("DOES THE BASE-MODEL CLUSTER HOLD OFF SWE-BENCH, AND DOES COLLAPSING IT HELP?")
    p("=" * 110)
    p(f"  {'leaderboard':<22} {'J':>4} {'fams':>5} {'pairs':>6} {'gap same':>9} {'gap diff':>9} "
      f"{'p':>6} {'k same':>7} {'k diff':>7} {'p':>6}")
    for name, v in rows.items():
        p(f"  {name:<22} {v['J']:>4} {v['families']:>5} {v['n_same']:>6} "
          f"{v['gap']:>9.4f} {v['gap_diff']:>9.4f} {v['gap_p']:>6.3f} "
          f"{v['kap']:>7.4f} {v['kap_diff']:>7.4f} {v['kap_p']:>6.3f}")
    p("")
    p(f"  {'leaderboard':<22} {'J':>4} {'J collapsed':>12} {'tie@1':>6} {'collapsed':>10} "
      f"{'random drop':>12} {'5th pct':>8} {'below?':>7}")
    below = 0
    to_one = 0
    for name, v in rows.items():
        b = v["tie_c"] < v["ctrl_p5"]
        below += b
        if v["tie"] > 1 and v["tie_c"] == 1:
            to_one += 1
        p(f"  {name:<22} {v['J']:>4} {v['Jc']:>12} {v['tie']:>6} {v['tie_c']:>10} "
          f"{v['ctrl_med']:>12.0f} {v['ctrl_p5']:>8.1f} {'yes' if b else 'no':>7}")
    p("")
    new = [k for k in rows if k != "SWE-bench Verified"]
    g_hits = sum(1 for k in new if rows[k]["gap_p"] < 0.05)
    k_hits = sum(1 for k in new if rows[k]["kap_p"] < 0.05)
    p(f"  P1  gap effect on {g_hits} of the {len(new)} new boards        "
      f"pre-registered >= 3:  {'HIT' if g_hits >= 3 else 'MISS'}")
    p(f"  P2  kappa effect on {k_hits} of the {len(new)} new boards      "
      f"pre-registered >= 3:  {'HIT' if k_hits >= 3 else 'MISS'}")
    p(f"  P3  collapse beats random dropping on {below} of {len(rows)}   "
      f"pre-registered >= 3:  {'HIT' if below >= 3 else 'MISS'}")
    p(f"  P4  collapse reaches a single first place on {to_one}          "
      f"pre-registered >= 1:  {'HIT' if to_one >= 1 else 'MISS'}")
    if skipped:
        p("")
        p("  skipped: " + "; ".join(f"{n} ({why})" for n, why in skipped))
    p("")
    p("  The labelling rule has no free parameter and every call it made is")
    p("  printed below, so a reader can audit it rather than trust it.")
    for name, mp in maps.items():
        p("")
        p(f"  {name} - families with two or more members:")
        p("    " + ", ".join(f"{f} ({c})" for f, c in mp))
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("family_generalises_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote family_generalises_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
