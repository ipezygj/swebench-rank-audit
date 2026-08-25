"""Build leaderboard matrices from Metriq's quantum hardware benchmark data.

Every board in this repository so far measures software: agents, embedding
models, protein predictors, provers, tabular learners. This adds hardware, and a
field where the entrants are physical devices rather than programs.

The source is metriq-data (Unitary Foundation), the results archive behind the
Metriq platform: one JSON per benchmark run, laid out as
{source}/{version}/{provider}/{device}/{timestamp}_{benchmark}_{hash}.json. A
run reports a single score for one device on one parameter configuration, so
the natural matrix is

    rows    = devices (IBM, IQM, IonQ, Rigetti, Quantinuum, Origin Wukong)
    columns = benchmark configurations (circuit width, layers, gate, shots)
    cell    = the run's reported score

which is exactly the systems-by-items shape the rest of this repository reads.
Repeat runs of the same device on the same configuration are collapsed by
median, and the count of collapsed duplicates is reported rather than hidden.

The matrices are SMALL and that is the point. A quantum benchmark fields on the
order of ten devices against under twenty configurations - a regime past HELM
classic's J=90, n=10, which is already the worst-resolved board here. Whether
the standard says anything at all at this size is the question, not a caveat.

    python quantum_matrix.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

SRC = ("C:/Users/ipezy.DESKTOP-GD1DJED/AppData/Local/Temp/claude/"
       "C--Users-ipezy-DESKTOP-GD1DJED/0bfba4dc-942d-4499-89e2-d2373e687ea2/"
       "scratchpad/metriq-data")
OUT = "quantum"
WANT = {"QML Kernel": "qml_kernel", "Mirror Circuits": "mirror_circuits",
        "Linear Ramp QAOA": "lr_qaoa"}
SCORE_KEYS = ("success_probability", "accuracy_score", "score")


def froze(v):
    if isinstance(v, list):
        return tuple(froze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, froze(x)) for k, x in v.items()))
    return v


def load(src: str):
    """(benchmark, device, config, value) for every run in the archive."""
    rows = []
    for f in glob.glob(os.path.join(src, "**", "*.json"), recursive=True):
        if os.path.basename(f) == "results.json":
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for rec in (d if isinstance(d, list) else [d]):
            if not isinstance(rec, dict) or "params" not in rec:
                continue
            dev = (rec.get("platform") or {}).get("device")
            res = rec.get("results") or {}
            b = rec["params"].get("benchmark_name")
            val = None
            for k in SCORE_KEYS:
                if isinstance(res.get(k), dict) and "value" in res[k]:
                    val = float(res[k]["value"])
                    break
            if dev and b and val is not None and np.isfinite(val):
                cfg = froze({k: v for k, v in rec["params"].items()
                             if k != "benchmark_name"})
                rows.append((b, dev, cfg, val))
    return rows


def dense_block(cells: dict, devices: list, configs: list):
    """Largest complete device-by-config block, by greedy removal.

    Drops whichever row or column is currently missing the most cells until
    nothing is missing. Greedy, so it is a lower bound on the best block, and
    the printed shape says which.
    """
    D, C = list(devices), list(configs)
    while D and C:
        miss = [(sum(1 for c in C if (d, c) not in cells), "d", d) for d in D]
        miss += [(sum(1 for d in D if (d, c) not in cells), "c", c) for c in C]
        worst = max(miss, key=lambda t: t[0])
        if worst[0] == 0:
            break
        if worst[1] == "d":
            D.remove(worst[2])
        else:
            C.remove(worst[2])
    return D, C


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not os.path.isdir(SRC):
        print(f"source archive not found: {SRC}")
        return 2
    rows = load(SRC)
    print(f"  {len(rows)} runs read from the archive")

    os.makedirs(OUT, exist_ok=True)
    vals = defaultdict(list)
    for b, dev, cfg, v in rows:
        vals[(b, dev, cfg)].append(v)

    for bench, stem in WANT.items():
        cells = {(d, c): float(np.median(vs))
                 for (b, d, c), vs in vals.items() if b == bench}
        dups = sum(len(vs) - 1 for (b, d, c), vs in vals.items() if b == bench)
        if not cells:
            print(f"  {bench}: no runs")
            continue
        devices = sorted({d for d, _ in cells})
        configs = sorted({c for _, c in cells}, key=str)
        D, C = dense_block(cells, devices, configs)
        if len(D) < 3 or len(C) < 3:
            print(f"  {bench}: dense block too small ({len(D)}x{len(C)})")
            continue
        lab = []
        for c in C:
            dd = dict(c)
            key = [f"{k}{dd[k]}" for k in ("width", "num_qubits", "num_layers",
                                           "max_qubits", "min_qubits")
                   if k in dd]
            lab.append("_".join(key) if key else str(abs(hash(c)) % 100000))
        seen, uniq = {}, []
        for t in lab:
            seen[t] = seen.get(t, 0) + 1
            uniq.append(t if seen[t] == 1 else f"{t}#{seen[t]}")
        M = pd.DataFrame([[cells[(d, c)] for c in C] for d in D],
                         index=D, columns=uniq)
        path = os.path.join(OUT, f"{stem}_matrix.csv")
        M.to_csv(path)
        print(f"  {bench:<20} {len(devices)}x{len(configs)} raw -> dense "
              f"{M.shape[0]}x{M.shape[1]}, {dups} duplicate runs collapsed by "
              f"median  ->  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
