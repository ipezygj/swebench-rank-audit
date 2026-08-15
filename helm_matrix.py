"""
Build a HELM Lite per-scenario WIN-RATE matrix from the public leaderboard JSON,
reproducing HELM's own ranking metric (mean win rate) so the resolution analysis
tests the ORDER THE BOARD ACTUALLY PUBLISHES, not a substitute aggregate.

HELM ranks models on `core_scenarios.json` table "Accuracy" by "Mean win rate".
Win rate of model m on scenario s = fraction of the other models m' with
score(m,s) > score(m',s) (ties count 0.5). Mean over scenarios = the leaderboard
number. We recompute it and PARITY-CHECK against the published column before
trusting anything (same discipline as the SWE-bench parity gate).

Output: helm_winrate_matrix.csv  (rows = models, cols = scenarios, cells = win rate)
        feed it to leaderboard_resolution.py (continuous mode).
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "helm_core.json"

def load_accuracy_table():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    tables = d if isinstance(d, list) else d["tables"]
    tbl = next(t for t in tables if t.get("title") == "Accuracy")
    cols = [c.get("value") or c.get("name") or "" for c in tbl["header"]]
    # col0 = Model, col1 = Mean win rate, col2.. = scenarios
    scen_cols = cols[2:]
    models, published_mwr, scores = [], [], []
    for row in tbl["rows"]:
        name = row[0].get("value")
        mwr = row[1].get("value")
        vals = [c.get("value") for c in row[2:]]
        if name is None:
            continue
        models.append(name)
        published_mwr.append(mwr)
        scores.append(vals)
    return models, scen_cols, np.array(published_mwr, dtype=float), scores

def to_winrate(scores):
    """scores: list (models) of list (scenarios) of float-or-None.
    Returns win-rate matrix (models x scenarios); NaN where a model lacks that scenario."""
    n = len(scores); k = len(scores[0])
    S = np.full((n, k), np.nan)
    for i in range(n):
        for j in range(k):
            v = scores[i][j]
            if v is not None:
                S[i, j] = float(v)
    W = np.full((n, k), np.nan)
    for j in range(k):
        col = S[:, j]
        have = ~np.isnan(col)
        idx = np.where(have)[0]
        m = len(idx)
        if m < 2:
            continue
        for i in idx:
            wins = np.sum(col[i] > col[idx]) + 0.5 * (np.sum(col[i] == col[idx]) - 1)
            W[i, j] = wins / (m - 1)   # exclude self
    return W

def main():
    models, scen_cols, pub_mwr, scores = load_accuracy_table()
    W = to_winrate(scores)
    recomputed = np.nanmean(W, axis=1)

    # PARITY GATE against the published Mean win rate column
    diff = np.abs(recomputed - pub_mwr)
    order = np.argsort(-pub_mwr)
    max_diff = float(np.nanmax(diff))
    print(f"models={len(models)}  scenarios={len(scen_cols)}")
    print(f"parity vs published Mean win rate: max |Δ| = {max_diff:.4f}  "
          f"(mean {np.nanmean(diff):.4f})")
    if max_diff > 0.02:
        worst = int(np.nanargmax(diff))
        print(f"  WARN worst: {models[worst]} recomputed {recomputed[worst]:.4f} "
              f"vs published {pub_mwr[worst]:.4f}")

    df = pd.DataFrame(W, index=models, columns=scen_cols)
    df = df.loc[[models[i] for i in order]]      # published rank order
    out = HERE / "helm_winrate_matrix.csv"
    df.to_csv(out)
    print(f"wrote {out.name}  ({df.shape[0]} x {df.shape[1]})")
    print("top 6 by published Mean win rate:")
    for i in order[:6]:
        print(f"  {pub_mwr[i]:.4f}  {models[i]}")

if __name__ == "__main__":
    main()
