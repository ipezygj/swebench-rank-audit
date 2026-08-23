"""Fetch HF model creation dates for the MTEB matrix rows -> mteb_dates.csv.

createdAt of the model repo is the earliest public date of the model and a
reasonable proxy for when it entered the leaderboard. Models without a public
repo (API-only, e.g. gemini-embedding-001 hosted under google/) still have a
repo card on HF in most cases; those that 404 are recorded as missing and the
audit drops them rather than guessing.
"""
import json
import sys
import time
import urllib.request

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
m = pd.read_csv("mteb_eng_v2_wide.csv", index_col=0).dropna(axis=0)
out, miss = {}, []
for i, mid in enumerate(m.index):
    url = f"https://huggingface.co/api/models/{mid}?expand[]=createdAt"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
            d = json.loads(r.read().decode("utf-8"))
        ts = d.get("createdAt", "")
        if ts:
            out[mid] = int(ts[:10].replace("-", ""))
        else:
            miss.append(mid)
    except Exception as e:
        miss.append(mid)
    if i % 30 == 0:
        print(f"  {i}/{len(m.index)} ...", flush=True)
    time.sleep(0.15)
pd.Series(out, name="date").to_csv("mteb_dates.csv", header=True)
print(f"dated {len(out)}, missing {len(miss)}: {miss[:8]}")
