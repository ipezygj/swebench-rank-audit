"""EXPLORATORY (not in PREREG): symmetric comparisons between the two designs.
  * all pairs on the fixed-harness board (B) separable after Holm, vs within-model harness pairs (A, = H1);
  * exact-label variance decomposition on A: between models vs within model (harness);
  * how many LATER models, run on the reference harness, score below each model's best harness.

    python explore.py -> results/explore.txt
"""
from __future__ import annotations

import itertools
import math
import sys

import numpy as np
import pandas as pd

from common import HERE, bashonly_rows, holm, mcnemar

OUT = HERE / "results"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    R = pd.read_csv(OUT / "b_runs.csv")
    R = R[R.usable]
    B = {k: v for k, v in bashonly_rows().items() if k in set(R.run)}
    L = []
    ps, d = [], []
    for a, b in itertools.combinations(B, 2):
        _, _, p = mcnemar(B[a], B[b]); ps.append(p); d.append(abs(len(B[a]) - len(B[b])) / 5)
    sep = np.array(holm(ps)); d = np.array(d)
    L.append(f"B all pairs (fixed harness, different models): {sep.sum()}/{len(ps)} = {100 * sep.mean():.1f} % "
             f"separable (Holm); median |diff| {np.median(d):.1f} pp; largest non-separable {d[~sep].max():.1f} pp")
    # same-size comparison: B pairs with |diff| in the A range
    A1 = pd.read_csv(OUT / "a_pairs_all.csv")
    L.append(f"A within-model pairs (= H1): {A1.holm_sep.sum()}/{len(A1)} = {100 * A1.holm_sep.mean():.1f} %; "
             f"median |diff| {A1.diff_pp.median():.1f} pp")
    # smallest separable difference in each design
    L.append(f"smallest separable |diff|: A {A1[A1.holm_sep].diff_pp.min():.1f} pp, B {d[sep].min():.1f} pp")

    I0 = pd.read_csv(OUT / "a_submissions.csv")

    def decomp(D, label):
        """SD between group means and pooled SD within groups; only groups with >= 2 members (after filtering)."""
        D = D[D.groupby("grp").score.transform("size") >= 2]
        if D.grp.nunique() < 2:
            L.append(f"  {label}: too few groups"); return
        s = D.score.to_numpy(); mu = D.groupby("grp").score.transform("mean").to_numpy()
        b = math.sqrt(np.mean((mu - s.mean()) ** 2)); w = math.sqrt(np.mean((s - mu) ** 2))
        L.append(f"  {label} ({len(D)} subs, {D.grp.nunique()} groups): between SD {100 * b:.2f} pp, "
                 f"within SD {100 * w:.2f} pp, ratio {w / b:.2f}")

    L.append("variance decomposition (groups with >= 2 submissions after each filter):")
    I = I0[I0.model.notna()].assign(grp=lambda d: d.model)
    decomp(I, "exact tag, all")
    decomp(I[I.attempts.astype(str) == "1"], "exact tag, attempts == 1")
    decomp(I[~I.flag_any], "exact tag, flagged rows removed")
    decomp(I[~I.flag_any & (I.attempts.astype(str) == "1")], "exact tag, flagged removed, attempts == 1")
    sys.path.insert(0, str(HERE.parent))
    from swebench_base_models import base_model           # the August regex families, on today's data
    F = I0.assign(grp=I0["sub"].map(base_model)).dropna(subset=["grp"])
    decomp(F, "regex FAMILY (August method), all labelled")
    decomp(F[F.model.notna()], "regex FAMILY, restricted to single-model-tag submissions")

    H2 = pd.read_csv(OUT / "h2.csv")
    L.append("later reference-harness runs / distinct models scoring below each model's best leaderboard harness:")
    for _, r in H2.iterrows():
        L.append(f"  {r.model:<28} best {100 * r.best_score:.1f} % (attempts {r.best_attempts}, n_harnesses {r.n_harnesses}) "
                 f"runs {r.newer_runs_below_best}/{r.newer_runs_total}, distinct models {r.newer_models_below_best}/"
                 f"{r.newer_models_total}; best single-attempt {100 * r.best1_score:.1f} % gap {r.gap1_pp:.1f} p {r.p_best1:.1e}")
    txt = "\n".join(L)
    print(txt)
    (OUT / "explore.txt").write_text(txt + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
