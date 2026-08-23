"""Base-model families for SWE-bench submissions, from a fixed vocabulary.

frontier_lineage.py could only place 6 % of SWE-bench frontier advances in a
family, because the family was taken as the last underscore-separated token
of the submission id. Real ids do not obey that: 20241029_OpenHands-CodeAct-
2.1-sonnet-20241022 names its base model in the middle, 20240824_gru names
none at all, and 20250603_Refact_Agent_claude-4-sonnet has three tokens.

This matches a FIXED vocabulary of base-model markers anywhere in the id.
The vocabulary is written down here before any coverage number is looked at,
and a submission that mentions none of them gets no family - it is not
guessed from the scaffold name.

    python swebench_base_models.py        # coverage report
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# family -> patterns that identify it, matched case-insensitively anywhere in
# the submission id. Ordered: the first family that matches wins, so specific
# model names come before the vendor that made them.
VOCAB = [
    ("claude-4.5", [r"opus-?4-?5", r"sonnet-?4-?5", r"claude-?4[.-]?5"]),
    ("claude-4",   [r"opus-?4", r"sonnet-?4", r"claude-?4"]),
    ("claude-3.7", [r"3-?7-?sonnet", r"sonnet-?3-?7", r"claude-?3-?7", r"claude37"]),
    ("claude-3.5", [r"3-?5-?sonnet", r"sonnet-?3-?5", r"claude-?3-?5", r"claude35"]),
    ("claude-3",   [r"claude-?3", r"claude3", r"sonnet", r"opus", r"haiku"]),
    ("claude-2",   [r"claude-?2", r"claude2"]),
    ("gpt-5",      [r"gpt-?5"]),
    ("gpt-4.1",    [r"gpt-?4[.-]?1"]),
    ("gpt-4o",     [r"gpt-?4o", r"4o\b"]),
    ("gpt-4",      [r"gpt-?4"]),
    ("gpt-3.5",    [r"gpt-?3[.-]?5", r"gpt35"]),
    ("o-series",   [r"\bo1\b", r"\bo3\b", r"\bo4\b", r"o1-?mini", r"o3-?mini"]),
    ("gemini",     [r"gemini"]),
    ("qwen",       [r"qwen"]),
    ("deepseek",   [r"deepseek", r"\br1\b"]),
    ("llama",      [r"llama", r"swellama"]),
    ("kimi",       [r"kimi"]),
    ("glm",        [r"\bglm"]),
    ("mistral",    [r"mistral", r"devstral", r"codestral"]),
    ("grok",       [r"grok"]),
]


def base_model(name: str):
    low = name.lower()
    for fam, pats in VOCAB:
        for pat in pats:
            if re.search(pat, low):
                return fam
    return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    idx = pd.read_csv("swebench_verified_matrix.csv", index_col=0).index
    fams = [base_model(n) for n in idx]
    c = Counter(f for f in fams if f)
    named = sum(1 for f in fams if f)
    print(f"{len(idx)} submissions, {named} placed in a base-model family ({100 * named / len(idx):.0f} %), "
          f"{len(c)} families")
    for fam, k in c.most_common():
        print(f"  {fam:<12} {k}")
    unmatched = [n for n, f in zip(idx, fams) if not f]
    print(f"unmatched ({len(unmatched)}), first 12:")
    for n in unmatched[:12]:
        print("   ", n)
    Path("swebench_families.csv").write_text(
        "system,family\n" + "\n".join(f"{n},{f or ''}" for n, f in zip(idx, fams)) + "\n",
        encoding="utf-8", newline="\n")
    print("wrote swebench_families.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
