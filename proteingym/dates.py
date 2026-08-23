"""Date the ProteinGym models from their reference URLs -> proteingym/dates.csv.

Sources, in order of precision:
  biorxiv DOI 10.1101/YYYY.MM.DD.*        day        (39 models)
  arxiv id YYMM.*                          month      (10)
  NeurIPS / ICML proceedings year          month (Dec / Jul)
  Nature-family DOI s4xxxx-YYYY-*          year (set to 1 July)
  six manual entries from venues without a date in the URL, listed below
  with the venue and month they were published (publication, not preprint):
    ESM2 family      Science 2023-03  (ade2574)
    ESM3 open        Science 2025-01  (ads0018)
    ProteinMPNN      Science 2022-09  (add2187)
    GEMME            MBE     2019-08  (pubmed 31406981)
    ESM-C family     blog    2024-12  (esm-cambrian)
    EVmutation / Site-Independent  Nat Biotech 2017-01 (nbt.3769)

Two kinds of date are mixed: preprint (biorxiv/arxiv, 49 models) and
publication (the rest), and publication lags preprint by about a year.
The column `source` records which; the fifth-board test runs a robustness
variant that shifts the preprint dates +12 months.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
d = pd.read_csv(HERE.parent.parent / "bio-eval" / "Summary_performance_DMS_substitutions_Spearman.csv")
m = pd.read_csv(HERE / "matrix.csv", index_col=0)

biorxiv = re.compile(r"10[.]1101/(20\d{2})[.](\d{2})[.](\d{2})")
arxiv = re.compile(r"arxiv[.]org/(?:abs|pdf)/(\d{2})(\d{2})[.]")
nips = re.compile(r"(?:papers[.]nips[.]cc|proceedings[.]neurips[.]cc)/paper(?:_files/paper)?/(20\d{2})/")
mlr = re.compile(r"proceedings[.]mlr[.]press/v(\d+)/")
nature = re.compile(r"nature[.]com/articles/s\d+-0?(\d{2})-")   # s41586-021-... -> 2021
springer = re.compile(r"springer[.]com/article/10[.]1007/s\d+-0?(\d{2})-")
MLR_YEAR = {"162": 2022, "139": 2021}
MANUAL = {
    "ade2574": (2023, 3, 16, "manual:Science"),
    "ads0018": (2025, 1, 16, "manual:Science"),
    "add2187": (2022, 9, 15, "manual:Science"),
    "31406981": (2019, 8, 15, "manual:MBE"),
    "esm-cambrian": (2024, 12, 4, "manual:blog"),
    "nbt.3769": (2017, 1, 16, "manual:NatBiotech"),
}

rows = []
for _, r in d.iterrows():
    name, ref = r["Model_name"], str(r["References"])
    if name not in m.index:
        continue
    cands = []
    for mm in biorxiv.finditer(ref):
        cands.append((int(mm.group(1)), int(mm.group(2)), int(mm.group(3)), "biorxiv"))
    for mm in arxiv.finditer(ref):
        cands.append((2000 + int(mm.group(1)), int(mm.group(2)), 15, "arxiv"))
    for mm in nips.finditer(ref):
        cands.append((int(mm.group(1)), 12, 10, "neurips"))
    for mm in mlr.finditer(ref):
        y = MLR_YEAR.get(mm.group(1))
        if y:
            cands.append((y, 7, 15, "icml"))
    for mm in nature.finditer(ref):
        cands.append((2000 + int(mm.group(1)), 7, 1, "nature-year"))
    for mm in springer.finditer(ref):
        cands.append((2000 + int(mm.group(1)), 7, 1, "springer-year"))
    for key, v in MANUAL.items():
        if key in ref:
            cands.append(v)
    if not cands:
        print(f"UNDATED {name}: {ref[:120]}")
        continue
    y, mo, da, src = min(cands)
    rows.append({"model": name, "date": y * 10000 + mo * 100 + da, "source": src})

out = pd.DataFrame(rows).set_index("model")
out.to_csv(HERE / "dates.csv")
print(f"dated {len(out)} of {m.shape[0]}; sources: {out['source'].value_counts().to_dict()}")
print(f"range {out['date'].min()} .. {out['date'].max()}")
