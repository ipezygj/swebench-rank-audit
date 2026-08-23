"""Build a systems x problems correctness matrix from MathArena output files.

MathArena (Balunovic et al., NeurIPS D&B 2025) runs models on fresh
competition problems and publishes every answer. Each `*_outputs` parquet
holds one row per (model, problem, run) with the model's final answer and
whether it was graded correct. Several runs per problem are averaged to a
correctness rate in [0, 1]; with a single run it is binary.

The fourth benchmark for universality.py. It shares nothing with the other
three: mathematics, not code or embeddings or QA; fresh problems written after
the models' training cutoffs; grading by exact answer match.

    python matharena/build_matrix.py matharena/*.parquet --out matharena/matrix.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="matharena/matrix.csv")
    ap.add_argument("--min-problems", type=float, default=0.9,
                    help="keep models answering at least this share of problems")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    frames = []
    for f in a.files:
        d = pd.read_parquet(f)
        comp = Path(f).stem.replace("_outputs", "")
        cols = {c.lower(): c for c in d.columns}
        # Column names vary slightly between releases; resolve defensively.
        model = cols.get("model_name") or cols.get("model")
        prob = cols.get("problem_idx") or cols.get("problem_id") or cols.get("problem")
        corr = cols.get("correct") or cols.get("is_correct") or cols.get("score")
        if not (model and prob and corr):
            print(f"  {f}: columns not recognised: {list(d.columns)[:12]}")
            continue
        sub = d[[model, prob, corr]].copy()
        sub.columns = ["model", "problem", "correct"]
        sub["correct"] = sub["correct"].astype(float)
        sub["problem"] = comp + "__" + sub["problem"].astype(str)
        frames.append(sub)
        print(f"  {f}: {len(sub)} rows, {sub['model'].nunique()} models, "
              f"{sub['problem'].nunique()} problems")
    if not frames:
        raise SystemExit("no usable files")
    allrows = pd.concat(frames, ignore_index=True)
    mat = allrows.groupby(["model", "problem"])["correct"].mean().unstack()
    keep = mat.notna().mean(axis=1) >= a.min_problems
    mat = mat[keep].dropna(axis=1)
    mat.to_csv(a.out)
    print(f"\nmatrix {mat.shape[0]} models x {mat.shape[1]} problems -> {a.out}")
    print(f"mean correctness {mat.values.mean():.3f}; binary cells: "
          f"{(mat.isin([0.0, 1.0])).values.mean():.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
