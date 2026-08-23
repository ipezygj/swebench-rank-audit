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
| R10 | **pair resolution of every claim** (report the split-half reliability of kappa with it; below roughly 40 items it is not measurable — HELM with 10 items gives r = 0.15) | for each comparison the leaderboard asserts (a "new SOTA", a "beats", a highlighted row), the pair's own difference SD, its ratio κ to what independence gives, and the paired test on that pair | the board's global σ is not the resolution of the claim: on five dated boards the frontier pairs have κ 0.53–0.94 while all pairs average 1.00, so quoting one number for the board understates the evidence for exactly the comparisons people read | `pair_sharpness.py`, `sota_audit.py` |
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
| C7 | **cluster-resampled rank sets** | whenever items come in groups (repositories, competitions, protein targets, task families) — report R2 again with the bootstrap resampling groups, and state the grouping | `cluster_bootstrap.py` |

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
- present item-resampled intervals as the board's uncertainty when the items
  come in groups without also reporting the cluster-resampled ones (C7): on
  SWE-bench Verified the rank sets widen 46 % when repositories are resampled
  instead of instances, and on MTEB 88 % when task families are, taking the
  number of models that could be first from 12 to 26;
- add items to a benchmark and report the new ranking without stating the
  power calculation that motivated the number and difficulty of the items
  added (`refill_prescription.py`).

## 4. Validation requirements on the implementation

Every field's implementation must ship self-checks that can fail, run before
any number is printed, **at the shape of the matrix being reported**. Three
times in one day a null passed at 60 × 400 and failed at 134 × 500. A
conforming implementation refuses to print when a check fails.

## 4b. A consequence of R10, stated rather than discovered later

Reporting the pair's own resolution makes a claim by a RELATIVE of the
current leader easier to establish than the same improvement by an unrelated
entrant, because the relative's difference vector has the smaller SD. The
saving is exactly the ratio of the two kappas, measured at 1 % (HELM, where
every entrant is a stranger) to 47 % (MTEB) across nine boards
(`incentive_asymmetry.py`). On MTEB an unrelated entrant would need 4.90
points of improvement to separate from the leader, and the largest frontier
step in the board's whole history is 3.81 - a stranger has never been able to
make a separable claim there, while a variant of the leader can.

This asymmetry is not created by the standard. It is present in every paired
test anyone runs, including the ones leaderboards already use informally; R10
makes it visible. A conforming leaderboard therefore reports kappa next to the
claim, so a reader can see whether a claim was cheap because the systems are
relatives.

## 5. What this standard does not cover

It does not say whether a benchmark measures anything worth measuring, whether
its items are correct, whether its grading is sound, or whether its systems
were trained on its items. It covers exactly one thing: whether the **order**
the leaderboard prints is supported by the **matrix** it was printed from, and
by how much. That is the part that has had no standard, and it is the part
that can be checked by anyone with the matrix.

## Appendix A. Two design findings, measured while testing the standard

These are not requirements. They are what the same instrument says about
choices a benchmark makes before it has anything to report, and they are
here because a standard that only tells owners what to print is less useful
than one that also tells them what the printing will be worth.

**Score on more than two levels.** Binarising a continuous board - scoring
each item above or below its median across systems - costs more resolution
than any other single choice measured here. CASP14's decisive top pair
(t = 9.89) falls to t = 1.78; the number of models that could be first goes
from 1 to 24 on CASP14, 12 to 66 on MTEB, 3 to 4 on ProteinGym, 8 to 12 on
LiveBench (`granularity.py`). The recovery curve is short: three levels
restore CASP14 to a single possible first place and eight levels are
indistinguishable from continuous on three of four boards
(`quantisation_curve.py`). The practical rule is not "use a continuous
metric" but "do not score an item pass/fail if a three-point rubric is
possible". SWE-bench cannot avoid it - a patch passes the tests or does not -
and that is one reason its top is unresolvable.

**Cheap entry costs little.** A new system that runs a quarter of the items
gets a rank set 1.13x to 1.97x the width it would get from running all of
them, on nine boards of nine; on the smaller boards the factor is 1.1 to 1.3
(`cheap_entry.py`). A submission policy can therefore offer a cheap tier
without giving up the ability to place its entrants.

## 6. Status

Draft. Validated on seven matrices from five fields (code, embeddings, QA,
competition mathematics, protein fitness, tabular prediction). Two
regularities proposed along the way were falsified by pre-registered tests;
one mechanism hypothesis has passed one discriminating test. None of the
fields above depends on any of those hypotheses being true.
