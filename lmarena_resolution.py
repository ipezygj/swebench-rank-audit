"""
LMArena (Chatbot Arena) is the POSITIVE control for this whole repo: it ranks on a
Bradley-Terry score with bootstrap 95% CIs and its published `final_ranking` already
assigns TIES to models whose intervals overlap. This script quantifies that from
LMArena's own public numbers — no re-derivation, just reading their CIs.

Data: elo_results_YYYYMMDD.pkl from the public HF space
  huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard
The pkl embeds plotly Figures (visualisations) that fail to unpickle across plotly
versions; a stub unpickler skips them so the leaderboard DataFrame loads cleanly.

Finding (elo_results_20250829, text/full, 242 models): 238/241 adjacent-by-rating
pairs have overlapping 95% CIs, and ranks 2-9 are an eight-way tie. The #1 IS
separated. The point is not that Arena is broken — it is that Arena SHOWS this, and
its rank column is honest because of it. The other boards in this repo do not.
"""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
PKL = HERE / "lmarena_elo.pkl"   # fetch from the HF space (see module docstring)

class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass
    def __reduce__(self): return (_Stub, ())

class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("plotly", "_plotly")):
            return _Stub
        return super().find_class(module, name)

def main(category="text", subset="full"):
    if not PKL.exists():
        sys.exit(f"missing {PKL.name} — download elo_results_<date>.pkl from the HF space "
                 "(see docstring) and save it here.")
    d = _StubUnpickler(open(PKL, "rb")).load()
    cell = d[category][subset]
    df = cell["leaderboard_table_df"].sort_values("rating", ascending=False)
    n = len(df)
    up, lo, fr = df["rating_upper"].values, df["rating_lower"].values, df["final_ranking"].values

    overlap = sum(1 for i in range(n - 1) if lo[i] <= up[i + 1])
    n_ranks = df["final_ranking"].nunique()
    top_tie = int((fr == fr[0]).sum())
    second_tie = int((fr == 2).sum())

    print(f"LMArena {category}/{subset}  updated {cell.get('last_updated_datetime')}")
    print(f"models: {n}")
    print(f"adjacent-by-rating pairs with overlapping 95% CI: {overlap}/{n-1} "
          f"({100*overlap/(n-1):.0f}%)")
    print(f"distinct published ranks: {n_ranks}  (avg {n/n_ranks:.1f} models per rank)")
    print(f"models sharing rank 1: {top_tie}   models sharing rank 2: {second_tie}")
    print(f"#1: {df.index[0]}  rating {df['rating'].iloc[0]:.0f} "
          f"[{lo[0]:.0f}, {up[0]:.0f}] — separated from the pack")
    print("Arena's own `final_ranking` already encodes these ties; that is the point.")

if __name__ == "__main__":
    main()
