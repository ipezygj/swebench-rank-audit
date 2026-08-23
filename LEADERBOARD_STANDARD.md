# Leaderboard Reporting Standard — draft 0.2 (2026-08-23)

*What a leaderboard must publish beside its ranking, so that a reader can tell
evidence from typography. Modelled on the GUM (JCGM 100:2008) for physical
measurement, the AERA/APA/NCME Standards for tests, and CONSORT for trials —
none of which have an equivalent for machine-learning benchmarks.*

## 0. Scope and principle

A leaderboard is a total order printed from a systems × items matrix. The
matrix supports a partial order; the difference is a choice the reader is not
shown. This standard requires that difference to be reported, in quantities
that are computable from the matrix alone, with no model beyond what the
leaderboard itself already assumes.

Two empirical facts from nine leaderboards in seven fields shape what is
required. First, the aggregate quantities (established share, entropy) are
well predicted by four numbers — J, n, the spread of scores τ, and the
median pairwise σ — so a board's overall resolving power is not a mystery
and can be computed before any run. Second, the resolution of an INDIVIDUAL
comparison is not that median: systems that share a base model, a scaffold
or a method move together item by item, and their comparison is sharper
than the board average. Reporting therefore splits into board-level fields
(R2–R8) and claim-level fields (R10).

A leaderboard **conforms** if it publishes the required fields below with the
matrix they were computed from. A reference implementation
(`leaderboard_standard.py`) computes every field from the matrix; conformance
is therefore a matter of publishing, not of effort.

## 1. Required fields

| # | field | definition | why it is required | defined in |
|---|---|---|---|---|
| R1 | **shape** | J systems, n items, data type (binary / continuous), matrix hash | nothing below is interpretable without it | — |
| R2 | **simultaneous rank sets** | for every system, [best, worst] rank at 95 % simultaneous coverage; summary: median width / J, number of systems whose set contains 1 | the per-system error bar overstates precision 8× because it ignores pairing; only the simultaneous set answers "what rank" | `rank_sets.py`, `two_way_bootstrap.py` |
| R3 | **leaderboard entropy** | H = log₂(number of total orders consistent with every established pair), and H / log₂(J!) | the single number for how much of the printed order is unsupported; a leaderboard is one of 2^H | `leaderboard_entropy.py` |
| R4 | **top-k resolution** | bits undetermined among the k best (k = 10 unless J < 10), against log₂(k!) | the rows anyone reads; on five of seven leaderboards tested this is the full 21.8 bits | `leaderboard_entropy.py` |
| R5 | **established share** | fraction of ordered pairs separated by the simultaneous test | the raw material of the ranking | `rank_sets.py` |
| R6 | **tiers resolved** | height of the partial order of established pairs; largest antichain | how many levels the instrument distinguishes versus how many positions it prints | `leaderboard_geometry.py` |
| R7 | **invariance drift** | ability drift across the item halves split by difficulty, metric and ordinal, each against its random-split floor | whether the instrument measures one thing; if the order moves with the subset, the table is an index | `measurement_invariance.py`, `ordinal_invariance.py` |
| R8 | **discordance D** | expected number of items on which two random systems disagree, on the top decile and on the whole field | the resource every test consumes; zero D means no method can separate anything | `information_depletion.py` |
| R10 | **pair resolution of every claim** | for each comparison the leaderboard asserts (a "new SOTA", a "beats", a highlighted row), the pair's own difference SD, its ratio κ to what independence gives, and the paired test on that pair | the board's global σ is not the resolution of the claim: on five dated boards the frontier pairs have κ 0.53–0.94 while all pairs average 1.00, so quoting one number for the board understates the evidence for exactly the comparisons people read | `pair_sharpness.py`, `sota_audit.py` |
| R9 | **provenance** | software version, seed, date, and the exact command | reproducibility of every number above | — |

## 2. Conditional fields

| # | field | when | defined in |
|---|---|---|---|
| C1 | effective item count n_eff, dead items, top-group live items | binary matrices | `benchmark_spectrum.py` |
| C2 | depletion history and half-life | submission dates available | `information_depletion.py` |
| C3 | composition sensitivity: price of the crown | items carry group labels | `reweighting_polytope.py` |
| C4 | invariant core and its height | items carry group labels | `invariant_core.py` |
| C5 | recount margin in items, swing items | binary matrices | `recount_margin.py` |
| C6 | pattern-fit flags | binary matrices, with the confound statement | `pattern_anomaly.py` |

## 3. Prohibited presentations

A conforming leaderboard does not:

- print a per-system error bar derived from the system's own items alone, as
  if it were a comparison (it destroys the pairing and understates width by
  up to 8×);
- print positions beyond the number of tiers resolved without marking ties;
- report a "new state of the art" whose gap to the previous is inside the
  simultaneous rank set of either;
- quote a single board-wide resolution as the precision of a specific
  comparison; the pair's own difference SD is the resolution of that claim
  (R10). The board-wide number misstates the items needed for the claim by
  the pair's kappa squared TIMES the ratio of the pair's own variance to
  the board's typical one - measured at 1.0x to 45x across nine boards,
  in both directions (`prescription_pairwise.py`);
- add items to a benchmark and report the new ranking without stating the
  power calculation that motivated the number and difficulty of the items
  added (`refill_prescription.py`).

## 4. Validation requirements on the implementation

Every field's implementation must ship self-checks that can fail, run before
any number is printed, **at the shape of the matrix being reported**. Three
times in one day a null passed at 60 × 400 and failed at 134 × 500. A
conforming implementation refuses to print when a check fails.

## 5. What this standard does not cover

It does not say whether a benchmark measures anything worth measuring, whether
its items are correct, whether its grading is sound, or whether its systems
were trained on its items. It covers exactly one thing: whether the **order**
the leaderboard prints is supported by the **matrix** it was printed from, and
by how much. That is the part that has had no standard, and it is the part
that can be checked by anyone with the matrix.

## 6. Status

Draft. Validated on seven matrices from five fields (code, embeddings, QA,
competition mathematics, protein fitness, tabular prediction). Two
regularities proposed along the way were falsified by pre-registered tests;
one mechanism hypothesis has passed one discriminating test. None of the
fields above depends on any of those hypotheses being true.
