# Two laws of leaderboard resolution — draft 1 (2026-08-23)

*How much ranking a benchmark supports, and how much of a printed order is
evidence, are both computable before the benchmark is run. This states the
two relations, the evidence for them across ten leaderboards in five fields,
and the conditions under which they fail.*

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
| SWE-bench Verified | 134 | 500 | 3.12 | 37.9 % | 37.4 % | -0.4 |
| MTEB English v2 | 181 | 41 | 2.51 | 34.2 % | 34.5 % | +0.3 |
| HELM classic | 90 | 10 | 1.39 | 24.4 % | 23.7 % | -0.7 |
| ProteinGym DMS | 96 | 217 | 2.98 | 36.0 % | 36.9 % | +0.9 |
| TabArena 16 models | 16 | 51 | 3.64 | 22.1 % | 39.2 % | +17.1 |
| TabArena 45 variants | 45 | 51 | 3.70 | 25.3 % | 39.3 % | +14.0 |
| CASP14 | 101 | 42 | 1.88 | 24.9 % | 29.7 % | +4.8 |
| LiveBench | 152 | 200 | 1.81 | 30.4 % | 29.0 % | -1.4 |
| MathArena 2025 | 35 | 183 | 2.08 | 31.6 % | 31.5 % | -0.1 |

Held-out test, run blind on a board untouched while the law was developed (LMArena, 35 models x 28 category win rates): observed 39.3 %, law 38.3 %.

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
| SWE-bench Verified | 134 | 500 | 54.5 % | 52.9 % | +1.6 |
| MTEB English v2 | 181 | 41 | 54.7 % | 58.2 % | -3.4 |
| HELM classic | 90 | 10 | 64.7 % | 64.1 % | +0.6 |
| ProteinGym DMS | 96 | 217 | 50.4 % | 52.1 % | -1.7 |
| TabArena 16 models | 16 | 51 | 61.9 % | 32.1 % | +29.8 |
| TabArena 45 variants | 45 | 51 | 58.9 % | 42.1 % | +16.8 |
| CASP14 | 101 | 42 | 68.2 % | 61.1 % | +7.1 |
| LiveBench | 152 | 200 | 63.6 % | 63.8 % | -0.2 |
| MathArena 2025 | 35 | 183 | 56.1 % | 52.2 % | +3.9 |

Held-out test on the same blind board: real 41.9 %, twin 41.7 %.

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
the twin's central 90 % on 6 of 9 boards (`top_compression.py`).

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
