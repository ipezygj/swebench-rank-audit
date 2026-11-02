"""PREREG H2 + H4 (+ git-peek sensitivity): the fixed-harness board (mini-SWE-agent) against the
leaderboard's harnesses on the same model.

    python harness_b.py -> results/b_runs.csv, results/b_adjacent.csv, results/h2.csv, results/b_summary.txt
"""
from __future__ import annotations

import re
import subprocess
import sys

import numpy as np
import pandas as pd

from common import HERE, bashonly_rows, holm, mcnemar, meta, norm_model, verified_rows

OUT = HERE / "results"
EXPREPO = HERE.parent / "swebench_exp"


def git_peek_flags(runs: list[str]) -> dict[str, set[str]]:
    """Instance ids listed in each run's git_peek_suspicious_commits.md (trajectories reading future commits)."""
    out = {}
    for r in runs:
        p = f"HEAD:evaluation/verified/{r}/git_peek_suspicious_commits.md"
        t = subprocess.run(["git", "-C", str(EXPREPO), "show", p], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if t.returncode == 0:
            m = re.search(r"All iids:\s*```(.*?)```", t.stdout, re.S)
            out[r] = set(m.group(1).split()) if m else set()
    return out


def adjacent(rows: dict[str, set[str]], order: list[str]) -> pd.DataFrame:
    P = []
    for a, b in zip(order, order[1:]):
        x, y, p = mcnemar(rows[a], rows[b])
        P.append(dict(upper=a, lower=b, diff_pp=(len(rows[a]) - len(rows[b])) / 5, a_only=x, b_only=y, p=p))
    P = pd.DataFrame(P)
    P["holm_sep"] = holm(P.p.tolist())
    return P


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    B = bashonly_rows()
    subs, universe, A = verified_rows()
    L = [f"bash-only runs downloaded: {len(B)}"]

    info, bad = [], []
    for r, res in B.items():
        m = meta(r)
        t, i = (m.get("tags") or {}), (m.get("info") or {})
        models = t.get("model") or []
        score = len(res) / 500
        pub = i.get("resolved")
        if pub is not None and abs(100 * score - float(pub)) > 0.2 + 1e-9:
            bad.append((r, round(100 * score, 2), pub))
        info.append(dict(run=r, model=norm_model(models[0]) if len(models) == 1 else None,
                         raw_model=models[0] if models else None, release=i.get("model_release_date"),
                         version=i.get("mini-swe-agent_version"), score=score, published=pub,
                         n_reports=sum(1 for _ in (HERE / "bashonly" / r).glob("*/report.json"))))
    R = pd.DataFrame(info).sort_values("score", ascending=False)
    badset = {b[0] for b in bad}
    R["usable"] = (R.n_reports >= 490) & R.published.notna() & ~R.run.isin(badset)
    R.to_csv(OUT / "b_runs.csv", index=False)
    L.append(f"usable runs (>= 490 per-instance reports AND parity within 0.2 pp): {R.usable.sum()}/{len(R)}; "
             f"incomplete in bucket: {(R.n_reports < 490).sum()}")
    R_all = R.copy()
    R = R[R.usable].copy()
    B = {k: v for k, v in B.items() if k in set(R.run)}
    with_pub = int((R_all.published.notna() & (R_all.n_reports >= 490)).sum())
    bad_complete = [b for b in bad if int(R_all.loc[R_all.run == b[0], "n_reports"].iloc[0]) >= 490]
    L.append(f"parity on complete runs: {with_pub - len(bad_complete)}/{with_pub} within 0.2 pp; complete-run mismatches: {bad_complete}")
    L.append(f"runs with no metadata: {R.release.isna().sum()} (model tag missing: {R.model.isna().sum()})")
    if bad:
        L.append("  (mismatched runs are EXCLUDED from H2/H4 and listed above; PREREG Deviation 1)")

    flags = git_peek_flags(list(B))
    flagged = set().union(*flags.values()) if flags else set()
    L.append(f"git-peek flag files: {len(flags)} runs, {len(flagged)} distinct flagged instances")

    # H4: adjacent ranks on the fixed-harness board
    order = R.run.tolist()
    P4 = adjacent(B, order)
    P4.to_csv(OUT / "b_adjacent.csv", index=False)
    L.append(f"H4 fixed-harness board: adjacent pairs separable (Holm) {P4.holm_sep.sum()}/{len(P4)}; "
             f"raw p<0.05 {(P4.p < 0.05).sum()}")
    Bx = {k: v - flagged for k, v in B.items()}
    P4x = adjacent(Bx, sorted(Bx, key=lambda k: -len(Bx[k])))
    L.append(f"H4 excluding git-peek-flagged instances: {P4x.holm_sep.sum()}/{len(P4x)}")

    # H2: best leaderboard harness vs mini-SWE-agent on the same model string
    ainfo = pd.read_csv(OUT / "a_submissions.csv")
    rows = []
    for m, g in ainfo[ainfo.model.notna()].groupby("model"):
        rb = R[R.model == m]
        if not len(rb):
            continue
        best = g.loc[g.score.idxmax()]
        mini = rb.iloc[0]                       # highest mini score for that model (conservative for H2)
        x, y, p = mcnemar(A[best["sub"]], B[mini.run])
        x2, y2, p2 = mcnemar(A[best["sub"]] - flagged, B[mini.run] - flagged)
        n_keep = 500 - len(flagged)
        gap_noflag = 100 * (len(A[best["sub"]] - flagged) - len(B[mini.run] - flagged)) / n_keep
        g1 = g[g.attempts.astype(str) == "1"]
        if len(g1):
            b1 = g1.loc[g1.score.idxmax()]
            _, _, p1 = mcnemar(A[b1["sub"]], B[mini.run])
            best1_sub, best1_score, p_best1 = b1["sub"], b1.score, p1
        else:
            best1_sub, best1_score, p_best1 = None, np.nan, np.nan
        rel = mini.release
        if pd.notna(rel):
            newer = R[R.release.notna() & (R.release.astype(float) > float(rel))]
            n_newer, n_newer_below = len(newer), int((newer.score < best.score).sum())
            nb = newer.groupby("model").score.max()          # one entry per distinct model: its best run
            n_newer_m, n_newer_m_below = len(nb), int((nb < best.score).sum())
        else:
            n_newer = n_newer_below = n_newer_m = n_newer_m_below = None
        rows.append(dict(model=m, n_harnesses=len(g), best_sub=best["sub"], best_score=best.score,
                         best_attempts=best.attempts, mini_run=mini.run, mini_score=mini.score,
                         gap_pp=100 * (best.score - mini.score), best_only=x, mini_only=y, p=p,
                         p_noflag=p2, gap_noflag_pp=gap_noflag, best1_sub=best1_sub, best1_score=best1_score,
                         gap1_pp=100 * (best1_score - mini.score), p_best1=p_best1,
                         newer_runs_below_best=n_newer_below, newer_runs_total=n_newer,
                         newer_models_below_best=n_newer_m_below, newer_models_total=n_newer_m))
    H2 = pd.DataFrame(rows)
    if len(H2):
        H2["holm_sep"] = holm(H2.p.tolist())
        H2.to_csv(OUT / "h2.csv", index=False)
        n5 = ((H2.gap_pp >= 5) & H2.holm_sep).sum()
        L.append(f"H2: {len(H2)} models in both A and B; best harness >= 5 pp above mini AND separable: "
                 f"{n5}/{len(H2)}")
        H2["holm_noflag"] = holm(H2.p_noflag.tolist())
        n5x = ((H2.gap_noflag_pp >= 5) & H2.holm_noflag).sum()
        L.append(f"H2 sensitivity, git-peek-flagged instances removed ({500 - len(flagged)} instances): {n5x}/{len(H2)}")
        L.append(f"H2 leaderboard side screened for git-peek: no (no flag files exist for leaderboard submissions)")
        pd.set_option("display.width", 250)
        H2.to_csv(OUT / "h2.csv", index=False)
        L.append(H2.drop(columns=["best_sub", "mini_run", "best1_sub"]).round(4).to_string(index=False))
    else:
        L.append("H2: no model appears in both A and B")
    L.append("")
    L.append(R.drop(columns=["raw_model"]).head(60).round(4).to_string(index=False))
    txt = "\n".join(L)
    print(txt)
    (OUT / "b_summary.txt").write_text(txt + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
