# Supplementary material: What Is a Harness Worth?

Anonymized for double-blind review.

1. Data (all public):
   - sparse clone of github.com/SWE-bench/experiments at commit 40f164d (`evaluation/verified/*/results/results.json`, `metadata.yaml`);
   - `leaderboards.json` from the swe-bench.github.io site repository;
   - `sh fetch_bashonly.sh` downloads per-instance reports from the public bucket s3://swe-bench-submissions/bash-only/ (~280 MB, `--no-sign-request`).
   `common.py` expects the clone at `../swebench_exp` (next to this folder), and `leaderboards.json` and `bashonly/` inside this folder; edit the paths at the top of `common.py` otherwise.
2. `python harness_a.py` (H1, H3, parity, flags) -> `python harness_b.py` (H2, H4, git-peek sensitivity) -> `python explore.py` -> `python fig.py`. Outputs go to `results/` (the files shipped here are the outputs used in the paper).
3. `python paircheck.py A B`: exact paired p-value and smallest detectable gap for any two result files.

Pre-registration: `PREREG_original.md` (SHA-256 in `PREREG.sha256`); deviations are appended in `PREREG.md`.
`meta/` holds the metadata.yaml files used for model tags. `results/b_runs.csv` lists every bash-only run with its inclusion status.
Environment: Python 3.14, numpy, pandas, scipy, PyYAML, matplotlib.
