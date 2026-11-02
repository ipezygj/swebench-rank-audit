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

## Deviation 1 (2026-11-02, after a crash-test run on 15 partially downloaded runs, before the full B data)
The public bucket is incomplete for some bash-only runs (e.g. claude-3-7-sonnet v0.0.0: 98 of 500
`report.json`; gpt-4.1 v1.0.0: no logs at all). A run enters H2/H4 only if it has >= 490 per-instance
reports AND its rebuilt score matches the published `resolved` within 0.2 pp. Excluded runs are listed in
`results/b_runs.csv` (`usable` = False) and counted in the summary. Seen at the time of this decision:
the crash-test output (3 H2 rows, 2 of them from incomplete runs).

## Deviation 2 (2026-11-02, after ALL pre-registered results had been computed; prompted by an independent review)
The recorded sha256 (PREREG.sha256) covers lines 1-42 only, i.e. the text before Deviation 1; that text is
kept verbatim as `PREREG_original.md`. Deviation 1 was written after the H1/H3 results existed (it concerns B only).
Added after the results, therefore EXPLORATORY / sensitivity only, never replacing a pre-registered result:
1. Data-quality flags on A rows: (a) a tag naming a model snapshot dated AFTER the submission
   (20240612_MASAI_gpt4o, 20240620_sweagent_claude3.5sonnet); (b) RAG baselines (not agents);
   (c) the tag "Qwen 2.5", which covers different models (SWE-Fixer 7B+72B, SWE-agent-LM-32B). H1/H3 are
   reported both as pre-registered and with flagged rows removed.
2. H2 used model tags with ONE leaderboard submission, although Data A says groups of >= 2 - unlogged
   until now. H2's 'best harness' is therefore 'the only leaderboard harness' for 5 of 7 models.
3. H2 paper text adds "after Holm" (stricter than the pre-registered wording) and reports a single-attempt
   variant for Claude Sonnet 4 and the git-peek sensitivity (4/7 with a different membership).
4. Parity tolerance 0.2 pp now has a 1e-9 float tolerance (one run, 71.2 vs 71.4, had been excluded by a
   float artefact): B usable runs 37 -> 38.
