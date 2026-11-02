"""Shared loaders for the harness paper (see PREREG.md)."""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest

HERE = Path(__file__).parent
EXP = HERE.parent / "swebench_exp" / "evaluation" / "verified"


def norm_model(m: str) -> str:
    """Normalise word order and separators only; never merge different dated snapshots."""
    s = str(m).lower().strip()
    s = s.rsplit("/", 1)[-1]                               # huggingface urls -> repo name
    s = re.sub(r"(20\d\d)[-_.](\d\d)[-_.](\d\d)", r"\1\2\3", s)  # 2024-11-20 -> 20241120
    toks = [t for t in re.split(r"[-_ .]+", s) if t]
    return "-".join(sorted(toks))


def meta(sub: str) -> dict:
    p = HERE / "meta" / f"{sub}.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def verified_rows() -> tuple[list[str], list[str], dict[str, set[str]]]:
    """Leaderboard (A) rows: submission -> resolved instance set, plus the 500-instance universe."""
    rows, universe = {}, set()
    for f in sorted(EXP.glob("*/results/results.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("resolved") is None:
            continue
        rows[f.parents[1].name] = set(d["resolved"])
        universe |= set(d.get("generated") or []) | set(d["resolved"])
    return sorted(rows), sorted(universe), rows


def bashonly_rows(root: Path = HERE / "bashonly") -> dict[str, set[str]]:
    out = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        res = set()
        for rp in d.glob("*/report.json"):
            try:
                r = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "resolved" in r:                      # newer format: the report IS the instance record
                if r.get("resolved") is True:
                    res.add(rp.parent.name)
                continue
            for iid, v in r.items():                 # older format: {instance_id: record}
                if isinstance(v, dict) and v.get("resolved") is True:
                    res.add(iid)
        out[d.name] = res
    return out


def mcnemar(a: set[str], b: set[str]) -> tuple[int, int, float]:
    """Exact McNemar on discordant instances: (a-only, b-only, two-sided p)."""
    x, y = len(a - b), len(b - a)
    if x + y == 0:
        return x, y, 1.0
    return x, y, float(binomtest(x, x + y, 0.5).pvalue)


def holm(p: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(p)
    m = len(p)
    rej = [False] * m
    for k, i in enumerate(order):
        if p[i] <= alpha / (m - k):
            rej[i] = True
        else:
            break
    return rej


def self_checks(rows: dict[str, set[str]], universe: list[str]) -> list[tuple[bool, str]]:
    out = []
    top = max(rows, key=lambda s: len(rows[s]))
    a = rows[top]
    _, _, p = mcnemar(a, a)
    out.append((p == 1.0, f"identity: McNemar({top}, itself) p = {p}"))
    rng = np.random.default_rng(20261102)
    unsolved = sorted(set(universe) - a)
    flip = set(rng.choice(sorted(a), 60, replace=False))
    b = a - flip
    _, _, p2 = mcnemar(a, b)
    out.append((p2 < 1e-6, f"planted 60 flips into {top}: p = {p2:.2e}"))
    return out
