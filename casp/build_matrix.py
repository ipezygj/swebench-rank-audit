"""Groups x domains GDT_TS matrix from CASP14 result tables (model 1 only).

CASP convention: each group submits up to five models per target; the
assessors rank on model 1. A row "T1024TS427_1-D1" is group 427, model 1,
domain D1. GDT_TS is on [0, 100]; divided by 100 to sit in [0, 1] with the
other matrices. Only domain-level tables (T####-D#) are used; whole-target
tables duplicate them.

    python casp/build_matrix.py --out casp/matrix.csv
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="casp/tables")
    ap.add_argument("--out", default="casp/matrix.csv")
    ap.add_argument("--min-cover", type=float, default=0.8,
                    help="keep groups that submitted on at least this share of domains")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    rows = []
    files = sorted(glob.glob(str(Path(a.tables) / "T1???-D?.txt")))
    for f in files:
        dom = Path(f).stem
        for line in open(f, encoding="utf-8", errors="replace"):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            m = re.match(r"T1\d{3}TS(\d+)_(\d)-D\d", parts[1])
            if not m or m.group(2) != "1":
                continue
            try:
                gdt = float(parts[3])
            except ValueError:
                continue
            rows.append((f"G{m.group(1)}", dom, gdt / 100.0))
    d = pd.DataFrame(rows, columns=["group", "domain", "gdt"])
    print(f"{len(files)} domain tables, {d['group'].nunique()} groups, "
          f"{len(d)} model-1 entries")
    m = d.pivot_table(index="group", columns="domain", values="gdt", aggfunc="max")
    keep = m.notna().mean(axis=1) >= a.min_cover
    m = m[keep].dropna(axis=1)
    assert float(m.values.max()) <= 1.0 and float(m.values.min()) >= 0.0
    m.to_csv(a.out)
    print(f"matrix {m.shape[0]} groups x {m.shape[1]} domains -> {a.out}")
    sc = m.mean(axis=1).sort_values(ascending=False)
    print("top-5:", {k: round(v, 3) for k, v in sc.head(5).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
