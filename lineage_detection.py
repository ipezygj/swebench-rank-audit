"""Can the matrix alone tell which entrants are relatives?

kappa(j,k) measures how much two systems move together item by item. If it
is picking up shared lineage, then clustering on kappa should recover
families that are visible in the NAMES - which the matrix never sees.

Ground truth from names, defined before looking at any kappa:
  SWE-bench Verified  the base model in the submission id, the token after
                      the date and the scaffold: 20250416_openhands_
                      claude-sonnet-4 -> "claude-sonnet-4"; families with
                      fewer than 3 members are dropped
  MTEB English v2     the HuggingFace organisation before the slash
  ProteinGym DMS      the method family before the first size or variant
                      marker: "ESM2 (650M)" -> ESM2, "Tranception L no
                      retrieval" -> Tranception

Score: adjusted Rand index between the kappa clustering (average linkage on
1 - kappa, cut at the number of name families) and the name families,
against a permutation null (labels shuffled, 200 draws).

PRE-REGISTERED EXPECTATION (2026-08-23, before running)
  * ARI above the 95th percentile of the permutation null on >= 2 of the 3
    boards;
  * ARI >= 0.2 on >= 2 of 3;
  * ProteinGym is where it should work best: its families are method
    families with genuinely shared machinery (MSA transformers, ESM
    variants), while SWE-bench families share only a base LLM behind
    different scaffolds.

SELF-CHECKS
  * planted families with known labels must give ARI > 0.8;
  * shuffled labels must give ARI near 0 (|ARI| < 0.05 on average).

    python lineage_detection.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from pair_sharpness import kappa_matrix

BOARDS = {
    "SWE-bench Verified": "swebench_verified_matrix.csv",
    "MTEB English v2": "mteb_eng_v2_wide.csv",
    "ProteinGym DMS": "proteingym/matrix.csv",
}
MIN_FAMILY = 3
SEED = 20260823


def family_swebench(name):
    m = re.match(r"^\d{8}_[^_]+_(.+)$", name)
    return m.group(1).lower() if m else None


def family_mteb(name):
    return name.split("/")[0].lower() if "/" in name else None


def family_proteingym(name):
    base = re.split(r"[ (]", name)[0]
    base = re.sub(r"[-_]?\d+[BM]?$", "", base)
    return base.lower() or None


FAMILY = {"SWE-bench Verified": family_swebench, "MTEB English v2": family_mteb,
          "ProteinGym DMS": family_proteingym}


def labels_of(names, fn):
    fams = [fn(n) for n in names]
    counts = Counter(f for f in fams if f)
    keep = {f for f, c in counts.items() if c >= MIN_FAMILY}
    idx = [i for i, f in enumerate(fams) if f in keep]
    lab = [fams[i] for i in idx]
    return np.array(idx), np.array(lab), len(keep)


def cluster_ari(K, idx, lab, k):
    D = 1.0 - K[np.ix_(idx, idx)]
    D = np.nan_to_num((D + D.T) / 2)
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    Z = linkage(squareform(D, checks=False), method="average")
    pred = fcluster(Z, t=k, criterion="maxclust")
    return float(adjusted_rand_score(lab, pred)), pred


def _check_planted():
    rng = np.random.default_rng(SEED)
    J, n, G = 60, 300, 4
    lab = np.repeat(np.arange(G), J // G)
    base = rng.normal(0, 0.45, (G, n))
    x = rng.normal(0.4, 0.05, J)[:, None] + 0.85 * base[lab] + np.sqrt(1 - 0.85 ** 2) * rng.normal(0, 0.45, (J, n))
    K = kappa_matrix(x)
    ari, _ = cluster_ari(K, np.arange(J), lab.astype(str), G)
    return ari > 0.8, f"planted 4 families: ARI {ari:.2f}"


def _check_shuffled():
    rng = np.random.default_rng(SEED + 1)
    J, n, G = 60, 300, 4
    lab = np.repeat(np.arange(G), J // G)
    base = rng.normal(0, 0.45, (G, n))
    x = rng.normal(0.4, 0.05, J)[:, None] + 0.85 * base[lab] + np.sqrt(1 - 0.85 ** 2) * rng.normal(0, 0.45, (J, n))
    K = kappa_matrix(x)
    aris = []
    for s in range(20):
        sh = np.random.default_rng(100 + s).permutation(lab).astype(str)
        aris.append(cluster_ari(K, np.arange(J), sh, G)[0])
    m = float(np.mean(aris))
    return abs(m) < 0.05, f"shuffled labels: mean ARI {m:+.3f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("self-checks")
    ok = True
    for passed, msg in (_check_planted(), _check_shuffled()):
        print(f"  [{'ok  ' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    if not ok:
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("DOES PAIR SHARPNESS RECOVER LINEAGE THAT ONLY THE NAMES KNOW?")
    p("=" * 84)
    p(f"  {'board':<22} {'systems':>8} {'families':>9} {'in families':>12} {'ARI':>6} "
      f"{'null 95th':>10} {'p':>6} {'kappa in':>9} {'kappa across':>13}")
    above, strong = 0, 0
    for name, path in BOARDS.items():
        df = pd.read_csv(path, index_col=0).dropna(axis=0)
        x = df.to_numpy(dtype=float)
        K = kappa_matrix(x)
        idx, lab, k = labels_of(list(df.index), FAMILY[name])
        if k < 2 or len(idx) < 8:
            p(f"  {name:<22} too few families to test")
            continue
        ari, pred = cluster_ari(K, idx, lab, k)
        null = []
        for s in range(200):
            sh = np.random.default_rng(SEED + s).permutation(lab)
            null.append(cluster_ari(K, idx, sh, k)[0])
        q95 = float(np.percentile(null, 95))
        pv = float((np.sum(np.array(null) >= ari) + 1) / (len(null) + 1))
        sub = K[np.ix_(idx, idx)]
        same = np.equal.outer(lab, lab) & ~np.eye(len(idx), dtype=bool)
        kin = float(np.nanmedian(sub[same]))
        kout = float(np.nanmedian(sub[~same & ~np.eye(len(idx), dtype=bool)]))
        above += ari > q95
        strong += ari >= 0.2
        p(f"  {name:<22} {x.shape[0]:>8} {k:>9} {len(idx):>12} {ari:>6.2f} {q95:>10.2f} {pv:>6.3f} "
          f"{kin:>9.2f} {kout:>13.2f}")
    p("")
    p(f"  ARI above the permutation 95th percentile: {above}/3 (pre-registered >= 2)")
    p(f"  ARI at least 0.20: {strong}/3 (pre-registered >= 2)")
    p("")
    p("  Families come from the names only: the base model in a SWE-bench")
    p("  submission id, the HuggingFace organisation for MTEB, the method family")
    p("  for ProteinGym. kappa never sees them. 'kappa in' and 'kappa across'")
    p("  are the median pair sharpness within and between name families.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("lineage_detection_results.txt").write_text(text + chr(10), encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote lineage_detection_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
