# Two laws of leaderboard resolution — draft 1 (2026-08-23)

*How much ranking a benchmark supports, and how much of a printed order is
evidence, are both computable before the benchmark is run. This states the
two relations, the evidence for them across ten leaderboards in five fields,
and the conditions under which they fail.*

> **Coverage warning (2026-08-24).** Every rank set in this document comes
> from a multiplier bootstrap that does not hold its nominal coverage when
> systems outnumber items. Measured under exact ties: HELM classic 0.013,
> MTEB English v2 0.540, CASP14 0.633, LMArena 0.727, TabArena 45 0.833,
> LiveBench 0.867, ProteinGym 0.840, TabArena 16 0.873, against a nominal
> 0.95; SWE-bench Verified 0.900, Lite 0.940, Test 0.947 and MathArena
> 0.933 are the sound ones. Holm on directional t-tests holds coverage on
> all of them (`tie_coverage_boards.py`). The bias narrows rank sets, so
> counts of possible first places are too LOW on the eight affected boards.
> Found by reading arXiv:2606.08679, not by the checks here.
>
> **Recomputed** (`holm_recompute.py`). Under Holm the established share
> falls on all eight affected boards and tie@1 rises on 5 of them - HELM
> classic from 15 possible first places to 50, MTEB from 16 to 24,
> TabArena's 45 variants from 12 to 18. The realised critical value is
> larger under Holm on every board (HELM 3.54 to 8.45), which is the
> diagnosis confirmed directly. On the four sound boards the two
> constructions agree exactly - 19, 11, 6 and 1 - so the differences are
> the construction and not the data.
>
> **Law 1 survives the correction**: mean absolute error 4.4 points under
> the bootstrap, 5.1 under Holm, a change of +0.7. It was not an artefact
> of the biased construction. The law-1 table below is the CORRECTED one:
> every results file in the repository has since been regenerated under Holm
> and the full pipeline reruns clean, so the observed column here and the
> Holm column of `holm_recompute_results.txt` are the same numbers.
>
> Every figure in this box is now read out of `holm_recompute_results.txt`
> rather than typed. Six of them had been typed and all six were wrong: they
> came from the Holm implementation with a normal reference, discarded when
> the two implementations were reconciled onto t with n-1 degrees of freedom.
> HELM's 21 first places and its critical value of 4.26 were that discarded
> run. The file said 50 and 8.45 for eight days.

## Law 1 — the established share

For a field whose scores have spread `tau`, measured on `n` items whose
pairwise difference has per-item SD `sigma_p`, with simultaneous critical
value `c`, the share of ORDERED pairs the benchmark can separate is

```
    established  =  Phibar( c * sigma_p / ( sqrt(2 n) * tau ) )  =  Phibar(1 / SNR)
```

One dimensionless argument, the field's signal-to-noise ratio
`SNR = tau sqrt(2n) / (c sigma_p)`. The unordered share is twice it; the
ceiling of the ordered share is 0.5.

| leaderboard | J | n | SNR | observed | law | error (points) |
|---|---|---|---|---|---|---|
| SWE-bench Verified | 134 | 500 | 3.00 | 37.4 % | 36.9 % | -0.4 |
| MTEB English v2 | 181 | 41 | 1.96 | 28.5 % | 30.5 % | +1.9 |
| HELM classic | 90 | 10 | 0.58 | 6.6 % | 4.3 % | -2.2 |
| ProteinGym DMS | 96 | 217 | 2.70 | 34.4 % | 35.5 % | +1.1 |
| TabArena 16 models | 16 | 51 | 3.12 | 19.6 % | 37.4 % | +17.8 |
| TabArena 45 variants | 45 | 51 | 2.99 | 19.7 % | 36.9 % | +17.2 |
| CASP14 | 101 | 42 | 1.51 | 21.0 % | 25.4 % | +4.4 |
| LiveBench | 152 | 200 | 1.69 | 28.8 % | 27.7 % | -1.2 |
| MathArena 2025 | 35 | 183 | 1.94 | 30.3 % | 30.3 % | +0.0 |

Held-out test, run blind on a board untouched while the law was developed (LMArena, 35 models x 28 category win rates): observed 36.6 %, law 35.6 %.

**Where it fails.** TabArena, both versions: the law over-predicts by 12 to 17
points. Its field is 16 models with a few far below the rest, so `tau` is
inflated by outliers that separate from everyone without making the dense top
separable. Replacing the SD with an IQR-based spread does not rescue it
(`resolution_law_test.py` reports both), which places the failure in the
Gaussian shape assumption, not in the law's form.

**What it is for.** A benchmark owner with a planned item count and an expected
spread of entrants can compute, before running anything, what share of pairs
the instrument will resolve. `refill_prescription.py` inverts it for the other
direction: how many items, at what difficulty, to separate a given pair.

## Law 2 — the entropy of the printed order

A leaderboard prints one total order; the data supports many. The number of
orders consistent with every established pair, in bits, is
`H = log2 e(P)`, and `H / log2(J!)` is the share of the printed order that is
unsupported. The law: `H / log2(J!)` is reproduced by a Gaussian field with the
same `J`, `n`, `tau`, `sigma_p` and nothing else of the real field.

| leaderboard | J | n | H/ceiling real | Gaussian twin | difference |
|---|---|---|---|---|---|
| SWE-bench Verified | 134 | 500 | 54.6 % | 53.0 % | +1.6 |
| MTEB English v2 | 181 | 41 | 61.2 % | 61.8 % | -0.6 |
| HELM classic | 90 | 10 | 85.9 % | 82.8 % | +3.1 |
| ProteinGym DMS | 96 | 217 | 53.4 % | 53.2 % | +0.2 |
| TabArena 16 models | 16 | 51 | 63.9 % | 35.7 % | +28.2 |
| TabArena 45 variants | 45 | 51 | 67.2 % | 44.4 % | +22.8 |
| CASP14 | 101 | 42 | 75.0 % | 64.6 % | +10.5 |
| LiveBench | 152 | 200 | 65.3 % | 65.3 % | +0.0 |
| MathArena 2025 | 35 | 183 | 57.7 % | 54.3 % | +3.4 |

Held-out test on the same blind board: real 47.5 %, twin 46.1 %.

**The full accounting.** The agreement is not simple; it is two effects that
partly cancel, and `entropy_decomposition.py` separates them:

```
    H(real) - H(Gaussian twin)  =  SHAPE  +  CORRELATION
```

SHAPE is what the ability distribution's form adds beyond four numbers, and it
is positive on every board tested (outliers and clusters, +1.4 to +40.0 points;
a smooth skew does nothing). CORRELATION is what entrants sharing base models,
scaffolds or methods take away, and it is negative on all nine (-1.7 to -10.6),
recovered almost exactly by permuting each system's residuals
(`residual_correlation.py`: the permuted board's entropy lands within 3 points
of the independent-noise level on 8 of 9). Where the two happen to be of
similar size, the four-number law looks exact.

## What the laws do not do

They describe aggregates. The resolution of an INDIVIDUAL comparison is not the
median `sigma_p` that enters them: the pair a leaderboard argues about has its
own difference SD, and on five dated boards those pairs are 6 to 47 per cent
sharper than the board average (`pair_sharpness.py`). Substituting the
board-wide number into a power calculation misstates the items needed by 1.0x
to 45x in both directions (`prescription_pairwise.py`). Law 1 is not improved
by integrating over the pairwise sigma distribution (`law1_pairwise.py`: mean
error 4.4 -> 4.1 points), which is the same statement from the other side - the
heterogeneity averages out of the aggregate and matters entirely for the claim.

Neither do they predict the TOP. A simulated board with SWE-bench Verified's
shape and its own SNR of 3.1 reproduces the established share (38.3 % against
37.9 %) and the entropy (52.7 % against 54.2 %) and misses the number of
systems that could be first by a factor of six (`target_board.py`). That miss
survives being stated properly, which at first it was not: tie@1 moves from one
ability draw of a field spec to the next, so the twin's prediction is an
interval and not the single number `target_board.py` prints. Over 99 draws
SWE-bench Verified's twin gives 2 [1-7] against a real 19, and the real value falls outside
the twin's central 90 % on 7 of 9 boards (`top_compression.py`).

What causes it is the SHAPE of the field, not the correlation between systems.
Two twins with the same J, n and latent spread separate the two: one keeps the
real score shape and gives it synthetic independent item noise, the other keeps
the real residual matrix and hangs it on a Gaussian ability vector. The shape
twin contains the real tie@1 on 7 of 9 boards and is closer to it than the
correlation twin on 8 of 8 boards where the two differ, while the correlation
twin is indistinguishable from the plain Gaussian one on every row. The
item-level dependence that makes paired comparison powerful does not change how
many systems can be first; the spacing of the field does.

The spacing is not always a cluster at the top, and the earlier wording here -
"every real board has a dense cluster at the top" - was wrong twice over. The
real gap between first and second sits below the twin's median on 7 of 9 boards,
but CASP14 and LiveBench go the other way, and where compression appears it is
often board-wide: MTEB (0.002 at the top, 0.001 in the bulk) and TabArena's 45
variants (0.070, 0.002) are compressed everywhere. SWE-bench Verified is the
clean case of a crowded top: 0.000 at the top against 0.992 in the bulk,
compressed at the front and stretched in the middle. Top-specific compression
was pre-registered for at least 6 of 9 boards and holds on 3; the prediction is
recorded as a miss in `top_compression_results.txt`.

Where the shape itself comes from is answerable on the one board that names
base models. On SWE-bench Verified, 62 of 134 submissions name theirs, and
two submissions sharing a base model sit 0.1337 apart against 0.2656 for two that
do not, with a pair sharpness of 0.9011 against 1.0098 - closer together AND
more correlated in their errors. Families are also calendar cohorts, so the
null permutes family labels only WITHIN a quarter, keeping each submission's
date and score: the gap effect survives it at p = 0.011 and the sharpness
effect at p = 0.001 (`family_clustering.py`). Among the top twenty, 29 % of
labelled pairs share a base model against 11 % for that null (p = 0.002).
The crowded top is a base-model cluster and not a crowded calendar.

It is not a SWE-bench peculiarity. Applying a labelling rule with no free
parameter - the first run of letters in the lowercased name - to four other
boards, the gap effect appears on 3 of the 4 and the sharpness effect on 4 of the 4
(`family_generalises.py`), MTEB most sharply of all, where two models from one
family compare at kappa 0.51 against 0.96 for two from different families.

The obvious remedy does not work, and an earlier version of this paragraph
asserted it would. Ranking families rather than submissions - keeping each
family's best - takes SWE-bench from 19 possible first places to 13, which is
what dropping the same number of systems AT RANDOM gives (13); across five
boards the collapse beats random dropping on 0 of 5. The reason is visible in the
top-twenty figure above: a same-family share of 29 % against a null of 11 % is
an enriched top, not a top made of duplicates, and the other 71 % of top pairs
are different families that the board still cannot separate. The cluster
explains why the top is crowded; removing it does not uncrowd the top.

They ARE redundant, and establishing that took two passes because the first
statistic could not see it. Asking how much of each top ten's coverage
exceeds a Rasch null - system ability, item difficulty, nothing else - found
no dependence at the top and some in the middle, and I wrote that down as a
finding. It was not one: injecting a latent factor of known size shows that
the coverage statistic rejects a factor of 1.6 logits only a quarter of the
time on SWE-bench Verified, so its silence was blindness
(`redundancy_power.py`). Mean pairwise correlation of the Rasch residuals,
with the null refitted so both sides carry the constraint fitting imposes,
puts the top of SWE-bench Verified at 0.1209 against a null of -0.0147 - above
every one of 299 simulated boards - and its middle at 0.0056. The excess at
the top is roughly eight times the excess in the middle, and every group on
every board is more dependent than independence predicts.

That dependence does not, however, change how many systems could be first.
The twin carrying the real residual matrix is indistinguishable from the
plain Gaussian one on tie@1 (above), because the simultaneous rank sets are
built on a multiplier bootstrap that already uses the dependence rather than
paying for it. So the two findings sit together without contradiction: the
systems at the top fail the same instances, and correcting for that fact is
already in the machinery, which is why the top stays crowded either way.

What remains, after a family explanation that does not survive collapsing
and a redundancy explanation that does not change the rank sets, is the
plain reading: the systems are close, their difference is carried by a small
and evenly split set of items, and the benchmark is being asked a question
its item set cannot answer.

Neither law predicts the future. Pair sharpness at entry does not predict being
overtaken later once score is held fixed (`kappa_predicts_future.py`, partial
correlations -0.03 to +0.18).

## Reproducing

```
python resolution_law_test.py     # law 1 across nine boards
python entropy_law_test.py        # law 2 across nine boards
python evidence_trajectory.py     # both laws replayed through time on three boards
python entropy_decomposition.py   # shape and correlation terms
python tenth_board.py             # the blind board
```

Every threshold in those files was committed to git before the run that tested
it. Failures are recorded in the same files, not removed.
