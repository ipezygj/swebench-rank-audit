# Leaderboard measurement toolkit

*What a systems x items matrix supports, and what a printed ranking claims.*

Two documents state the results:

- **[LEADERBOARD_STANDARD.md](LEADERBOARD_STANDARD.md)** - what a leaderboard must
  publish beside its ranking (draft 0.2, 10 required fields and
  6 prohibited presentations), with `leaderboard_standard.py` as the reference
  implementation.
- **[LAWS.md](LAWS.md)** - two relations that predict how much ranking a benchmark
  supports and how much of a printed order is evidence, tested on ten boards from
  five fields, including one held out until after the relations were fixed.

Everything is computed from the score matrix alone. No tool needs metadata, and
the ones that use names (lineage detection, base-model families) say so and are
validated against a permutation null.

## Matrices

| board | shape | notes |
|---|---|---|
| SWE-bench Verified | 134 x 500 | binary, per-instance resolve lists |
| SWE-bench Lite | 84 x 300 | binary |
| SWE-bench test | 24 x 2294 | binary, pre-selected entrants |
| MTEB English v2 | 181 x 41 | continuous task scores; 174 dated from HF createdAt |
| HELM classic | 90 x 10 | win rates |
| ProteinGym DMS | 96 x 217 | Spearman per assay; dated from reference URLs |
| TabArena | 16 x 51 and 45 x 51 | min-max normalised within dataset |
| CASP14 | 101 x 42 | GDT_TS / 100 |
| LiveBench | 152 x 200 | judged scores, largest complete block |
| MathArena 2025 | 35 x 183 | per-problem correctness |
| LMArena categories | 35 x 28 | category win rates; held out during the 2026-08-23 loop |

## Tools

### The standard and its reference implementation

- `leaderboard_standard.py` - Reference implementation of the Leaderboard Reporting Standard (draft 0.1)
- `export_card_data.py` - Export the SWE-bench report-card data as JSON for the certified-leaderboard page
- `build_certified_page.py` - Render card_data.json as the certified SWE-bench leaderboard page (HTML)
- `export_card_mteb.py` - Export MTEB English v2 as standard-0.2 report-card data, R10 included
- `build_certified_mteb.py` - Render card_data_mteb.json as a standard-0.2 certified leaderboard page
- `build_laws_md.py` - Assemble LAWS.md from the results files, so no number is retyped by hand
- `build_readme.py` - Generate README.md: the map of this repo, with the tool index read from the code

### Core measurement

- `rank_sets.py` - Simultaneous confidence sets for a leaderboard's RANKS, not its pairs
- `two_way_bootstrap.py` - Which bootstrap? The answer depends on what you claim to generalise to
- `leaderboard_entropy.py` - How many bits of the published ranking does the data not determine?
- `leaderboard_geometry.py` - A leaderboard is a line. The data underneath it is not, and this measures how
- `invariant_core.py` - If the ranking depends on the basket, which comparisons do not?
- `measurement_invariance.py` - Is this a measurement, or is it an index?
- `ordinal_invariance.py` - Two of my own results contradict each other. Resolving it sharpens the theorem
- `benchmark_spectrum.py` - The benchmark as an instrument: resolution spectrum, aperture, dead pixels
- `information_depletion.py` - A benchmark is a depleting resource. This measures how much is left
- `recount_margin.py` - How many test cases would have to be re-graded for someone else to lead?
- `pattern_anomaly.py` - Does this system's answer pattern look like a coherent ability?
- `reweighting_polytope.py` - Who could be number one if the benchmark had been assembled differently?
- `saturation_horizon.py` - The leaderboard of the future is inside today's hardest items
- `deflated_benchmark.py` - How much of the top score is just the fact that it is a maximum?
- `progress_or_selection.py` - Was it progress, or just more attempts? A leaderboard is a time series
- `selection_sbi.py` - How many attempts stood behind this leaderboard? Ask the shape, not me
- `leaderboard_resolution.py` - What is this leaderboard's resolution, and which of its rows are ordered?
- `resolution_law.py` - How many instances does a benchmark need before its ranking means anything?
- `swebench_rank_noise.py` - How much of the SWE-bench Verified ranking survives its own sampling error?

### The two laws

- `resolution_law_test.py` - Is the established share of a leaderboard determined by one number?
- `entropy_law_test.py` - Is a leaderboard's entropy also determined by (J, n, SNR)?
- `evidence_trajectory.py` - Do the two laws hold THROUGH TIME inside one leaderboard?
- `entropy_law_twin2.py` - Entropy law, second attempt: a twin that keeps the NOISE PROFILE
- `entropy_decomposition.py` - Where does the entropy law's residual live? A two-term decomposition
- `residual_correlation.py` - Is the universal negative residual the CORRELATION of entrants' residuals?
- `law1_pairwise.py` - Law 1 with pair-specific resolution instead of one sigma_p
- `tenth_board.py` - A tenth board, untouched all evening: LMArena categories x models
- `universality.py` - Is any of this a law? The same quantities on three unrelated leaderboards

### Pair resolution (R10) and lineage

- `pair_sharpness.py` - Pair sharpness: resolution is a property of the PAIR, not the benchmark
- `kappa_reliability.py` - Is pair sharpness a property of the pair, or of the items it was measured on?
- `kappa_generality.py` - Does the sharp-pair finding hold at ranks other than the top?
- `kappa_trend.py` - Is the frontier getting sharper? Pair sharpness over time
- `kappa_predicts_future.py` - Does pair sharpness predict the FUTURE, or only describe the past?
- `lineage_detection.py` - Can the matrix alone tell which entrants are relatives?
- `independence_flag.py` - How many independent lineages are in the top ten?
- `effective_entrants.py` - How many INDEPENDENT entrants does a leaderboard really have?
- `isotonic_families.py` - Families, conditioned on ability: isotonic residuals
- `pairing_dividend.py` - What does pairing buy? Rank sets with and without the pair covariance
- `prescription_pairwise.py` - What does pair-specific resolution cost, or save, in items?
- `incentive_asymmetry.py` - Does pair-specific resolution reward derivative work?
- `swebench_base_models.py` - Base-model families for SWE-bench submissions, from a fixed vocabulary

### SOTA claims through time

- `sota_audit.py` - Every time the frontier moved: was the new leader separable from the old one?
- `sota_twin.py` - Is the share of 'real' SOTA advances predictable from the field's drift?
- `step_sizes.py` - How big is a SOTA step, in units of what the benchmark can resolve?
- `fourth_board.py` - The pre-registered test of the step-size reading on a FOURTH dated board
- `fifth_board.py` - Fifth dated board, from a DIFFERENT FIELD: ProteinGym DMS (96 x 217)
- `sota_luck_law.py` - How large is a SOTA step compared with luck among equals? (corrected)
- `sota_families.py` - Does the frontier move within families?
- `frontier_lineage.py` - The open question, asked of the names instead of the residuals
- `leader_luck.py` - The winner's curse, seen in the residuals
- `chase_model.py` - A generative account of the frontier: entrants CHASE the record
- `chase_correlated.py` - Chase model with the board's own noise: does correlation fix P?
- `chase_refit.py` - The fair version: chase parameters fitted INSIDE the board's own noise
- `sibling_chase.py` - Chase model, final form: the chaser inherits the record-holder's noise
- `lineage_tree.py` - The chase model with a family tree, which is what sibling_chase was missing
- `broad_or_deep.py` - Why are real SOTA steps more separable than simulated steps of the same size?
- `decisiveness_history.py` - Was the top of the board ever decisive, and when did it stop being so?
- `crown_stability.py` - How often does the crown change hands if the items are resampled?
- `model_or_harness.py` - On SWE-bench, how much of a submission's score is the model and how much the harness?
- `within_family_spread.py` - Is 'the harness is two thirds of the model' peculiar to agentic coding?

### Prescription

- `refill_prescription.py` - What would this benchmark need, to answer the question it is asked?
- `refill_all.py` - How many new items would each leaderboard need to settle its top pair?

### Matrix builders

- `swebench_matrix.py` - Build the SWE-bench Verified system x instance outcome matrix
- `all_splits.py` - Run the Verified analysis on every SWE-bench split that publishes per-instance
- `helm_matrix.py` - Build a HELM Lite per-scenario WIN-RATE matrix from the public leaderboard JSON,
- `mteb_dates.py` - Fetch HF model creation dates for the MTEB matrix rows -> mteb_dates.csv
- `lmarena_resolution.py` - LMArena (Chatbot Arena) is the POSITIVE control for this whole repo: it ranks on a
- `matharena/build_matrix.py` - Build a systems x problems correctness matrix from MathArena output files
- `casp/build_matrix.py` - Groups x domains GDT_TS matrix from CASP14 result tables (model 1 only)
- `proteingym/dates.py` - Date the ProteinGym models from their reference URLs -> proteingym/dates.csv

### Unsorted (add to a group in build_readme.py)

- `alpha_sensitivity.py` - Do the standard's headline numbers depend on the confidence level?
- `casp/probe.py` - Probe predictioncenter.org for a groups x targets table
- `casp/probe2.py` - (no docstring)
- `cluster_bootstrap.py` - Are the items independent? A cluster bootstrap says how much that assumption buys
- `merge_boards.py` - Does merging two benchmarks resolve what neither resolves alone?
- `method_independence.py` - Does the headline survive a different statistical method?
- `minimal_benchmark.py` - How small could this benchmark be and still say the same thing?

## How the results were produced

Each tool prints its own self-checks first and refuses to print a table if one
fails. Thresholds and expectations are written into the module docstring and
committed to git BEFORE the run that tests them; when an expectation fails, the
failure is recorded in the same file rather than removed. Where a construction
was changed after a failed check, the change and its reason are in the docstring.

Results live next to each tool as `*_results.txt` and are regenerated by running
the tool. `build_laws_md.py` and `build_readme.py` assemble the documents from
those files and from the code, so neither document can drift silently.
