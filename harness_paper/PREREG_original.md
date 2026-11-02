# PREREG — How much of a SWE-bench score is the harness? (Kaggle Gemma 4 paper track)

Written 2026-11-02, BEFORE any outcome in this file's analysis was computed. What had been seen
before writing: the 2026-08-23 `model_or_harness_results.txt` (coarse regex families: within/between
SD 0.67, date-matched 0.54) and label COUNTS from `metadata.yaml` (no resolve rates).
The sha256 of this file is recorded in `PREREG.sha256` at writing time.

## Data
- **A. fixed model, varying harness.** SWE-bench Verified submissions (SWE-bench/experiments, pulled
  2026-11-02, commit 40f164d) whose `tags.model` lists exactly ONE model. Model strings normalised only
  for word order and separators (e.g. `claude-4-sonnet-20250514` = `claude-sonnet-4-20250514`); a dated
  snapshot is never merged with a different date. Groups with >= 2 submissions.
- **B. fixed harness, varying model.** The bash-only board: mini-SWE-agent runs, per-instance
  `report.json` from the public bucket `s3://swe-bench-submissions/bash-only/`.
- Outcome per instance: resolved yes/no on the same 500 Verified instances.

## Tests and expectations
- **H1 (the harness effect is real, not noise).** Every pair of submissions within a model group
  (A): exact McNemar on discordant instances; Holm across all within-model pairs, alpha 0.05.
  *Expect: >= 50 % of within-model pairs separable.*
- **H2 (how many model generations is a harness worth).** For each model present in both A and B:
  best harness score in A minus the model's own mini-SWE-agent score in B (same model string).
  *Expect: the best harness exceeds mini-SWE-agent by >= 5 points for >= half of those models,
  with the gap separable by paired McNemar.*
- **H3 (the selection / attempts confound).** Repeat H1 restricted to `attempts == 1`.
  *Expect: the share of separable pairs falls but stays >= 1/3.*
- **H4 (the fixed-harness board resolves models).** On B: adjacent-rank exact McNemar, Holm.
  No directional expectation; reported as the counterpart to A.

## Self-checks (analysis aborts if any fails)
1. McNemar of a submission against itself: p = 1.
2. 60 planted flips into the top-scoring row are detected at p < 1e-6.
3. Parity: the resolve rate rebuilt from per-instance files matches the `resolved`/leaderboard figure in
   metadata within 0.2 points for every B run that reports one; mismatches are listed, not dropped silently.
4. A-row resolve rates match the published Verified leaderboard (as in the August census).

## Confounds stated now
- Harness submissions arrive later than the model's release; later harnesses are better. Not causal.
- `attempts: 2+` submissions select among several runs; H3 addresses this.
- Flagged `git_peek_suspicious_commits.md` files (trajectories that read future commits) exist for some
  B runs; their count is reported; H2/H4 rerun excluding flagged instances as a sensitivity check.
- Self-reported model tags; a wrong tag moves a row to the wrong group. Not verified against trajectories.
