"""PREREG H1 + H3: within one exact base model, are harness differences separable?

    python harness_a.py  -> results/a_groups.csv, results/a_pairs.csv, results/a_summary.txt
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from common import HERE, holm, mcnemar, meta, norm_model, self_checks, verified_rows

OUT = HERE / "results"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    subs, universe, rows = verified_rows()
    n_inst = len(universe)
    L = [f"Verified submissions with per-instance results: {len(subs)}; instance universe {n_inst}"]
    ok = True
    for passed, msg in self_checks(rows, universe):
        L.append(f"  [{'ok' if passed else 'FAIL'}] {msg}")
        ok &= passed
    lbf = HERE / "leaderboards.json"
    if lbf.exists():
        V = [b for b in json.loads(lbf.read_text(encoding="utf-8"))["leaderboards"] if b["name"] == "Verified"][0]
        lb = {r["folder"]: float(r["resolved"]) for r in V["results"]}
        on = [s for s in subs if s in lb]
        mism = [s for s in on if abs(len(rows[s]) / 5 - lb[s]) > 0.2 + 1e-9]
        L.append(f"  [{'ok' if not mism else 'FAIL'}] parity vs official Verified leaderboard: "
                 f"{len(on) - len(mism)}/{len(on)} match within 0.2 pp; not on the board: {[s for s in subs if s not in lb]}")
        ok &= not mism
    if not ok:
        print("\n".join(L)); print("SELF-CHECK FAILED"); return 1

    info = []
    for s in subs:
        t = (meta(s).get("tags") or {})
        models = t.get("model") or []
        att = str((t.get("system") or {}).get("attempts", "?"))
        info.append(dict(sub=s, n_models=len(models) if isinstance(models, list) else 0,
                         model=norm_model(models[0]) if isinstance(models, list) and len(models) == 1 else None,
                         raw_model=models[0] if isinstance(models, list) and len(models) == 1 else None,
                         attempts=att, agent=t.get("agent"), score=len(rows[s]) / 500))
    I = pd.DataFrame(info)
    # data-quality flags (post hoc, from the independent review; PREREG Deviation 2)
    def impossible(r):          # submission dated before the model snapshot named in its tag
        d = re.findall(r"(20\d{6})", str(r["model"] or ""))
        return bool(d) and r["sub"][:8] < d[0]
    I["flag_date"] = I.apply(impossible, axis=1)
    I["flag_rag"] = I.agent.astype(str).str.contains("RAG", case=False)
    I["flag_mixed"] = I.model.eq("2-5-qwen")   # tag "Qwen 2.5" covers different models (SWE-Fixer 7B+72B, SWE-agent-LM-32B)
    I["flag_any"] = I.flag_date | I.flag_rag | I.flag_mixed
    L.append(f"flags: date-impossible tag {list(I['sub'][I.flag_date])}; RAG baselines {list(I['sub'][I.flag_rag & I.model.notna()])}; "
             f"mixed tag {list(I['sub'][I.flag_mixed])}")
    I.to_csv(OUT / "a_submissions.csv", index=False)
    L.append(f"single-model tag: {I.model.notna().sum()}; multi-model: {(I.n_models > 1).sum()}; "
             f"no tag: {(I.n_models == 0).sum()}")
    G = I[I.model.notna()].groupby("model")
    grp = {m: g for m, g in G if len(g) >= 2}
    L.append(f"model groups with >= 2 submissions: {len(grp)}, covering {sum(len(g) for g in grp.values())} submissions")

    def pairs(filter_attempts1: bool, clean: bool = False):
        P = []
        for m, g in grp.items():
            gg = g[g.attempts == "1"] if filter_attempts1 else g
            if clean:
                gg = gg[~gg.flag_any]
            for (_, r1), (_, r2) in itertools.combinations(gg.iterrows(), 2):
                x, y, p = mcnemar(rows[r1["sub"]], rows[r2["sub"]])
                P.append(dict(model=m, a=r1["sub"], b=r2["sub"], score_a=r1.score, score_b=r2.score,
                              diff_pp=100 * abs(r1.score - r2.score), a_only=x, b_only=y, p=p,
                              attempts=f"{r1.attempts}/{r2.attempts}"))
        P = pd.DataFrame(P)
        if len(P):
            P["holm_sep"] = holm(P.p.tolist())
        return P

    for tag, f1, cl in (("H1 all attempts", False, False), ("H3 attempts == 1", True, False),
                        ("H1 sensitivity, flagged rows removed", False, True),
                        ("H3 sensitivity, flagged rows removed", True, True)):
        P = pairs(f1, cl)
        P.to_csv(OUT / f"a_pairs_{'att1' if f1 else 'all'}{'_clean' if cl else ''}.csv", index=False)
        if not len(P):
            L.append(f"{tag}: no pairs"); continue
        sep = P.holm_sep.mean()
        L.append(f"{tag}: {len(P)} within-model pairs in {P.model.nunique()} models; "
                 f"separable (Holm 5 %) {P.holm_sep.sum()}/{len(P)} = {100 * sep:.1f} %; "
                 f"raw p<0.05 {(P.p < 0.05).sum()}; median |diff| {P.diff_pp.median():.1f} pp; "
                 f"largest NON-separable diff {P[~P.holm_sep].diff_pp.max() if (~P.holm_sep).any() else float('nan'):.1f} pp; "
                 f"smallest separable {P[P.holm_sep].diff_pp.min() if P.holm_sep.any() else float('nan'):.1f} pp; "
                 f"pairs < 3 pp separable {int((P.holm_sep & (P.diff_pp < 3)).sum())}/{int((P.diff_pp < 3).sum())}")

    rowsG = []
    for m, g in grp.items():
        rowsG.append(dict(model=m, n=len(g), best=g.score.max(), median=g.score.median(), worst=g.score.min(),
                          spread_pp=100 * (g.score.max() - g.score.min()), best_sub=g.loc[g.score.idxmax(), "sub"],
                          worst_sub=g.loc[g.score.idxmin(), "sub"]))
    Gt = pd.DataFrame(rowsG).sort_values("best", ascending=False)
    Gt.to_csv(OUT / "a_groups.csv", index=False)
    pd.set_option("display.width", 220)
    L.append(Gt.drop(columns=["best_sub", "worst_sub"]).round(3).to_string(index=False))
    txt = "\n".join(L)
    print(txt)
    (OUT / "a_summary.txt").write_text(txt + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
