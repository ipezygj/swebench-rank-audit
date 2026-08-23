"""Export MTEB English v2 as standard-0.2 report-card data, R10 included.

The SWE-bench certified page was built from standard 0.1. MTEB is the board
where the standard's numbers are most extreme - 181 systems, 41 tasks, 15
of which could be first once the comparison is paired and 131 if it is not
- and it is the second field, so it tests whether the page generalises past
the board it was designed on.

Adds, beyond card_data.json: kappa for every pair on the frontier and for
the top pair (R10), the pairing dividend (R2's justification as a number),
and the lineage groups from kappa clustering.

    python export_card_mteb.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import rank_sets as rs
import leaderboard_entropy as le
import leaderboard_geometry as lg
from pair_sharpness import kappa_matrix
from independence_flag import threshold, lineages
from sota_audit import advances

sys.stdout.reconfigure(encoding="utf-8")
SEED = 20260823

df = pd.read_csv("mteb_dated_matrix.csv", index_col=0).dropna(axis=0)
dates = pd.read_csv("mteb_dates.csv", index_col=0)["date"]
x = df.to_numpy(dtype=float)
names = list(df.index)
J, n = x.shape
rng = np.random.default_rng(SEED)

r = rs.rank_sets(x, draws=1500)
beats = r["beats"]
tiers = lg.tiers(beats)
tier_of = {j: t for t, members in enumerate(tiers, start=1) for j in members}
H = le.log_extensions(beats, 2000, rng)
order = np.argsort(-r["theta"], kind="stable")
top10 = order[:10]
H10 = le.log_extensions(beats[np.ix_(top10, top10)], 2000, rng)

K = kappa_matrix(x)
thr = threshold(x, np.random.default_rng(SEED + 3))
iu = np.triu_indices(J, k=1)

# lineage id for each of the top 20, from complete-linkage clustering
top20 = [int(i) for i in order[:20]]
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
sub = np.nan_to_num((K[np.ix_(top20, top20)] + K[np.ix_(top20, top20)].T) / 2, nan=1.0)
np.fill_diagonal(sub, 0.0)
sub[sub < 0] = 0.0
lin = fcluster(linkage(squareform(sub, checks=False), method="complete"), t=thr, criterion="distance")
lineage_of = {j: int(l) for j, l in zip(top20, lin)}

systems = []
for rank, j in enumerate(order, start=1):
    j = int(j)
    systems.append({
        "rank": rank, "name": names[j],
        "date": str(int(dates.loc[names[j]])) if names[j] in dates.index else "",
        "score": round(float(r["theta"][j]), 4),
        "best": int(r["best"][j]), "worst": int(r["worst"][j]),
        "tier": tier_of[j],
        "lineage": lineage_of.get(j, 0),
    })

# R10 for the pair the board argues about
i1, i2 = int(order[0]), int(order[1])
d12 = x[i1] - x[i2]
se12 = float(d12.std(ddof=1) / math.sqrt(n))
xc = x - x.mean(axis=0, keepdims=True)
sd = xc.std(axis=1, ddof=1)

# frontier advances with their own kappa and paired test
dts = np.array([int(dates.loc[nm]) for nm in names])
adv = []
for a in advances(x, dts):
    i, k = int(a["new"]), int(a["old"])
    d = x[i] - x[k]
    nz = d[d != 0]
    rg = np.random.default_rng(SEED + len(adv))
    flips = rg.choice([-1.0, 1.0], size=(20000, len(nz)))
    null = np.abs((flips * nz[None, :]).mean(axis=1))
    pv = float((np.sum(null >= abs(nz.mean()) - 1e-15) + 1) / (len(null) + 1))
    present = np.flatnonzero(dts <= a["date"])
    rr = rs.rank_sets(x[present], draws=600, seed=SEED + 100 + len(adv))
    pi = {int(s): t for t, s in enumerate(present)}
    adv.append({
        "date": str(int(a["date"])), "name": names[i], "prev": names[k],
        "gain": round(float(x[i].mean() - x[k].mean()), 4),
        "kappa": round(float(K[i, k]), 3), "p": round(pv, 4),
        "pair": pv < 0.05, "sim": bool(rr["beats"][pi[i], pi[k]]),
        "field": int(len(present)),
    })

out = {
    "board": "MTEB English v2", "J": J, "n": n,
    "H": round(H["bits"], 1), "ceiling": round(gammaln(J + 1) / math.log(2), 1),
    "H10": round(H10["bits"], 1),
    "established": round(float(beats.sum() / (J * (J - 1))), 4),
    "tiers": len(tiers), "antichain": lg.largest_antichain(beats),
    "tie1": int((r["best"] == 1).sum()),
    "median_width": int(np.median(r["worst"] - r["best"])),
    "kappa_all": round(float(np.nanmedian(K[iu])), 3),
    "kappa_top": round(float(K[i1, i2]), 3),
    "kappa_frontier": round(float(np.nanmedian([a["kappa"] for a in adv])), 3),
    "top_gap": round(float(x[i1].mean() - x[i2].mean()), 4),
    "top_se": round(se12, 4),
    "top_t": round(float((x[i1].mean() - x[i2].mean()) / se12), 2),
    "lineage_thr": round(float(thr), 3),
    "lineages_top10": int(len(set(lineage_of[j] for j in top20[:10]))),
    "systems": systems, "advances": adv,
}
Path("card_data_mteb.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"wrote card_data_mteb.json: {J} systems, {len(adv)} advances, tie@1 {out['tie1']}, "
      f"kappa top {out['kappa_top']}, lineages in top 10 {out['lineages_top10']}")
