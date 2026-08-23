"""Export the SWE-bench report-card data as JSON for the certified-leaderboard page."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import rank_sets as rs
import leaderboard_entropy as le
import leaderboard_geometry as lg

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_csv("swebench_verified_matrix.csv", index_col=0)
x = df.to_numpy(dtype=float)
names = list(df.index)
J, n = x.shape
rng = np.random.default_rng(20260823)

r = rs.rank_sets(x, draws=1500)
beats = r["beats"]
tiers = lg.tiers(beats)
tier_of = {}
for t, members in enumerate(tiers, start=1):
    for j in members:
        tier_of[j] = t
H = le.log_extensions(beats, 2000, rng)
order = np.argsort(-r["theta"], kind="stable")
top10 = order[:10]
H10 = le.log_extensions(beats[np.ix_(top10, top10)], 2000, rng)

systems = []
for rank, j in enumerate(order, start=1):
    m = __import__("re").match(r"^(\d{4})(\d{2})(\d{2})_(.*)$", names[j])
    systems.append({
        "rank": rank, "name": m.group(4) if m else names[j],
        "date": f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "",
        "score": round(float(r["theta"][j]), 4),
        "best": int(r["best"][j]), "worst": int(r["worst"][j]),
        "tier": tier_of[int(j)],
    })

out = {
    "J": J, "n": n,
    "H": round(H["bits"], 1), "ceiling": round(gammaln(J + 1) / math.log(2), 1),
    "H10": round(H10["bits"], 1),
    "established": round(float(beats.sum() / (J * (J - 1))), 4),
    "tiers": len(tiers), "antichain": lg.largest_antichain(beats),
    "tie1": int((r["best"] == 1).sum()),
    "median_width": int(np.median(r["worst"] - r["best"])),
    "systems": systems,
}
sota = Path("sota_audit_results.txt").read_text(encoding="utf-8") if Path("sota_audit_results.txt").exists() else ""
adv = []
for line in sota.splitlines():
    parts = line.split()
    if len(parts) > 8 and parts[0].count("-") == 2 and parts[0][:2] == "20" and parts[1].endswith("%"):
        try:
            adv.append({"date": parts[0], "gain": float(parts[1].rstrip("%")),
                        "margin": int(parts[2]), "disc": int(parts[3]), "p": float(parts[4]),
                        "pair": parts[5] == "yes", "sim": parts[6] == "yes",
                        "field": int(parts[7]), "name": " ".join(parts[8:])})
        except ValueError:
            pass
out["advances"] = adv
Path("card_data.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"wrote card_data.json: {J} systems, {len(adv)} advances, H {out['H']}, tiers {out['tiers']}")
