# How much of the SWE-bench Verified ranking survives its own sampling error?

**Measured 2026-08-09.** 134 systems, 500 instances, exact McNemar on paired outcomes.
Not yet published anywhere.

## The short version

The leaderboard's top eight systems cannot be told apart from each other. Of the 133
adjacent-rank comparisons on the board, **129 are not statistically distinguishable**
at the 5% level. The number a reader takes from the top of that table — which system
is best — is the part the data supports least.

This is not a claim that the benchmark is broken. Across all 8 911 pairs, 87.9% *are*
separated. The benchmark discriminates fine between systems that are far apart. It is
the neighbours that it cannot order, and neighbours are what a ranking is made of.

## Numbers

| | |
|---|---:|
| systems | 134 |
| instances | 500 |
| adjacent-rank pairs **not** separated at 5% | **129 / 133** |
| all pairs separated at 5% | 7 835 / 8 911 (87.9%) |
| systems not separable from the #1 | **8** (ranks 1–8, 76.4%–79.2%) |
| largest resolve-rate gap that was still undecided | **4.60 pp** |
| smallest gap that was decided | 2.40 pp |
| median 95% rank interval width | **13 places** of 134 |
| systems whose rank interval spans ≥ 10 places | 102 / 134 |
| instances solved by no system | 32 |
| instances solved by every system | 0 |

The top of the board, with exact McNemar on the instances where each pair disagrees:

```
  1 vs 2   diff +0.00pp  discordant 18/18  p=1.000  not separated
  2 vs 3   diff +0.40pp  discordant 33/31  p=0.901  not separated
  3 vs 4   diff +1.20pp  discordant 37/31  p=0.545  not separated
  4 vs 5   diff +0.20pp  discordant 26/25  p=1.000  not separated
  5 vs 6   diff +0.60pp  discordant 31/28  p=0.795  not separated
  6 vs 7   diff +0.00pp  discordant 28/28  p=1.000  not separated
  7 vs 8   diff +0.40pp  discordant 28/26  p=0.892  not separated
  8 vs 9   diff +0.80pp  discordant 39/35  p=0.728  not separated
  9 vs 10  diff +0.40pp  discordant 34/32  p=0.902  not separated
 10 vs 11  diff +0.40pp  discordant 31/29  p=0.897  not separated
```

Ranks 1 and 2 resolve the same number of instances. They disagree on 36 of them,
18 each way. That is a coin flip, and the order between them is presentation, not
measurement.

Under a paired bootstrap over instances (2 000 draws, the same instance draw applied
to every system so the pairing survives), the system displayed at #1 occupies ranks
1–6; the one at #5 occupies ranks 1–11.

## Method

Every submission is scored on the same 500 instances and publishes the list of
instance IDs it resolved, so two systems can be compared as paired binary data.
**Exact McNemar** uses only the instances where they disagree: the ones both solve
and both miss carry no information about which is better. An unpaired test on the
same data throws that structure away and is much less powerful.

Rate = `|resolved| / 500`, which is the leaderboard's own definition.

## Why the numbers can be trusted

**Parity against the published leaderboard: 133 systems compared, zero mismatches.**
Every resolve rate recomputed here from the raw per-instance lists reproduces the
percentage published on swebench.com exactly.

That check matters more than it looks. 121 of the 134 `results.json` files have an
**empty `generated` list**, so the files cannot themselves confirm that every system
was run on all 500 instances. The parity check resolves it from the other direction:
whatever convention the leaderboard uses to reach its published percentage, this
reproduces it for all 133 systems that appear in both.

Self-checks run before any headline number is produced, and the analysis aborts if
they fail:

- a system compared against itself returns p = 1 (a test that cannot return "same"
  cannot return "different" meaningfully either);
- a copy with 60 outcomes flipped to failure is detected at p < 1e-6, so the
  identity check above is not passing merely because nothing is ever detected;
- the Wilson interval reproduces the published reference value for 5/10,
  [0.2366, 0.7634].

## Scope and limits, stated rather than buried

- The official Verified board lists 180 entries; 134 have a `results.json` in
  `SWE-bench/experiments`, and 133 of those matched a board entry by folder name.
  **The census covers those 133, not all 180.**
- This measures sampling error on a fixed 500-instance set. It says nothing about
  whether those 500 instances represent real software work, about contamination, or
  about cost. Those are different audits.
- "Not separated" means the data does not establish an ordering. It is not evidence
  that two systems are equally good.
- 32 instances are solved by no system and contribute nothing to any comparison;
  no instance is solved by all.

## Reproducing

```bash
git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/SWE-bench/experiments.git
cd experiments && git sparse-checkout init --no-cone
printf '/evaluation/verified/*/results/results.json\n' > .git/info/sparse-checkout
git checkout
python swebench_matrix.py      # builds swebench_verified_matrix.csv
python swebench_rank_noise.py  # self-checks, then the table above
```

`swebench_verified_matrix.csv` (134 × 500, binary) is included so the analysis can be
re-run without the clone.


---

# Part 2 — the same measurement across four splits (2026-08-09)

The Verified result raised an obvious question: is that band a property of SWE-bench
Verified, or of any leaderboard scored on a finite instance set? A prediction was
written into `PREREGISTERED.md` **before** the other splits were fetched, so it could
miss. It did, once.

| split | systems | instances | parity vs published board | adjacent pairs undecided | all pairs separated | largest undecided gap |
|---|---:|---:|---|---:|---:|---:|
| Verified | 134 | 500 | 133 compared, **0 mismatches** (max dev 0.0000) | **129 / 133** | 87.9% | 4.60 pp |
| Lite | 84 | 300 | 84 compared, **0 mismatches** (max dev 0.0033) | **82 / 83** | 77.7% | 6.67 pp |
| Test | 24 | 2294 | 24 compared, 1 mismatch (see below) | **9 / 23** | 95.7% | 1.61 pp |
| Multimodal | 12 | 517 | 12 compared, **0 mismatches** (max dev 0.0046) | **11 / 11** | 63.6% | 2.71 pp |

Multimodal drops 10 further submissions that publish `resolved` as a count rather than
a list of instance ids; a count cannot be paired, so they are excluded and counted.

## The pre-registered prediction: 2 of 3

Anchored on Verified (n = 500 → 4.60 pp) and scaled as `1/sqrt(n)`:

| split | n | predicted | envelope | actual | |
|---|---:|---:|---|---:|---|
| Lite | 300 | 5.94 pp | 4.5–7.5 | **6.67 pp** | HIT |
| Test | 2294 | 2.15 pp | 1.5–3.0 | **1.61 pp** | HIT |
| Multimodal | 517 | 4.52 pp | 3.4–5.9 | **2.71 pp** | **MISS** |

The direction holds where it matters most: quadrupling the instance count from 500 to
2294 cuts the undecided band from 4.60 pp to 1.61 pp, and **Test is the only split
where most adjacent pairs are separated** (9 of 23 undecided, versus 129 of 133 on
Verified). More instances buy resolution, roughly as the square root says.

The miss is real and is not being explained away. Post-hoc — and labelled as post-hoc,
because it was not predicted — the simple `1/sqrt(n)` model ignores two things: the
binomial term `p(1-p)` differs between a split where the top system resolves 79% and
one where it resolves 36%, and with only 12 systems there are 66 pairs to draw a
"largest undecided gap" from rather than 8 911. A statistic defined as a maximum
shrinks when you take fewer draws.

## An earlier finding, reproduced without looking for it

Test is the one split with a genuine parity mismatch: `20240402_sweagent_claude3opus`,
published 10.51%, recomputed **9.29%**, a gap of **1.2249 pp**.

That is the defect reported as [SWE-bench/experiments#463](https://github.com/SWE-bench/experiments/issues/463)
on 2026-07-30: the submission's `resolved` list contains duplicate instance ids — 241
entries over 213 unique ids. This analysis counts unique ids, the board counts entries,
and the difference is exactly the 1.22 pp figure in that report. It was found here by a
parity gate that had no idea the issue existed, and it shows **the board still carries
the inflated number**.

## What the parity gate caught in this analysis itself

Two of the four splits initially failed parity, and both failures were mine:

1. **Lite reported 55 "mismatches" where every value in fact agreed.** The board
   publishes two decimals, so `k/300` (a repeating decimal) can only ever agree to
   0.005 — and the gate was set at 0.001. Fixed by matching the gate to the published
   rounding and, so that a looser gate cannot hide a real disagreement, reporting the
   **maximum observed deviation** alongside it. Lite's is 0.0033.
2. **Multimodal was scored on the wrong denominator.** The union of instance ids across
   its submissions is 248, but the benchmark has 517 — the other 269 were solved by
   nobody and `generated` is empty in those files, so they left no trace. Every rate was
   therefore doubled. Padding with the unsolved instances (concordant zeros, so no
   McNemar result changes) restored parity on all 12 systems from one number.

The second correction is the one worth noticing: **fixing it turned the Multimodal
prediction from a hit into a miss.** Had the parity gate been skipped, this write-up
would have claimed 3 for 3.


---

# Part 3 — a resolution law, derived and then tested (2026-08-09)

Parts 1 and 2 describe. This asks the designer's question: **how many instances does
a benchmark need before its ranking means anything?**

## The derivation

Two systems on `n` paired instances. Let `b` and `c` be the instances only A or only
B solves, `d = (b+c)/n` the **discordance rate**, and `delta = (b-c)/n` the difference
in resolve rate — the number a leaderboard shows. McNemar tests `b/(b+c)` against a
half, and `b/(b+c) - 1/2 = delta / 2d`. A binomial test on `dn` trials detects that
shift at significance `alpha` and power `1-beta` when

> **delta ≥ (z(alpha/2) + z(beta)) · sqrt(d / n)**  — constant 2.80 at alpha 0.05, power 0.8

Two things fall out, and the second is the one nobody says out loud:

- resolution improves only as `sqrt(n)`: to halve the gap you can resolve, quadruple
  the benchmark;
- it *degrades* as `sqrt(d)`: **the more two systems disagree instance by instance, the
  harder they are to separate at a given headline gap.** Systems that differ in style
  rather than strength are the expensive case, and `d` is measurable from published
  per-instance outcomes yet is never reported.

No constant here was chosen by looking at the data.

## First test: it failed, 0 of 4

Plugging each split's median discordance over *all* pairs over-predicted the observed
decision boundary by about 1.4× everywhere — 6.49 pp predicted against a 2.40–4.60 pp
boundary on Verified, and similarly on the other three.

The stated hypothesis for the failure was that the wrong `d` was being used: the pairs
that sit at the decision boundary are *similar* systems, and similar systems should
disagree on fewer instances than distant ones. That was then measured rather than
assumed:

| split | median d, pairs < 2 pp apart | median d, pairs further apart |
|---|---:|---:|
| Verified | **0.180** | 0.276 |
| Lite | **0.192** | 0.247 |
| Test | **0.073** | 0.192 |
| Multimodal | **0.091** | 0.101 |

Close pairs do disagree less, on every split. The law was not wrong; the quantity fed
to it was.

## Second test: per pair, no free parameters

Every pair predicted from its own `(gap, d, n)` and scored against what McNemar
actually said:

| split | agreement | false positives | false negatives |
|---|---:|---:|---:|
| Verified | 95.6% | **0** | 393 |
| Lite | 92.4% | **0** | 266 |
| Test | 98.2% | **0** | 5 |
| Multimodal | 89.4% | **0** | 7 |

**Zero false positives across all 12 739 pairs on four splits.** The formula never
claims a pair is resolvable when the test cannot resolve it. Every disagreement runs
the other way — pairs it calls too close that McNemar decided anyway, which is exactly
what an 80%-power threshold should produce.

So (*) is validated as a **conservative bound**: if the gap clears it, the ordering is
real; below it, sometimes yes and sometimes no.

## What that costs, in instances

Using each split's own close-pair discordance:

| split | instances today | to resolve 3 pp | 2 pp | **1 pp** | 0.5 pp |
|---|---:|---:|---:|---:|---:|
| Verified | 500 | 1 570 | 3 532 | **14 128** | 56 512 |
| Lite | 300 | 1 672 | 3 761 | **15 044** | 60 175 |
| Test | 2 294 | 641 | 1 441 | **5 765** | 23 061 |
| Multimodal | 517 | 793 | 1 784 | **7 135** | 28 541 |

SWE-bench Verified has 500 instances. Ordering two systems one percentage point apart
would take roughly **14 000** — twenty-eight times the benchmark. At its actual size,
with a close-pair discordance of 0.180, its resolution is **5.32 pp**: the top eight
systems are spread over 2.8 pp, comfortably inside it, which is why they come out as
one undifferentiated block.

*(An earlier draft of this paragraph said "about 3 pp". That was wrong — 3 pp is the gap
that 1 570 instances would buy, not what 500 do. The error was caught by running
`leaderboard_resolution.py` and noticing it disagreed with the prose. Recording it
because a write-up that quietly self-corrects is not one you can check.)*

This is not an argument for building a 14 000-instance benchmark. It is an argument for
reporting the resolution alongside the ranking, so a 0.4 pp difference is not read as
progress. The number costs nothing to compute: it is already in the submission files.


---

# Part 4 — the instrument (2026-08-09)

`leaderboard_resolution.py` takes any matrix of per-item outcomes — one row per
system, one column per evaluation item — and answers the three questions this
write-up has been asking, on any benchmark, without knowing what the benchmark is.

Both outcome types go through the same report because the question is the same and
only the test changes: **exact McNemar** on discordant items for pass/fail data,
**paired bootstrap over items** for a score per item. Both keep the pairing.

Run on the two benchmark families measured here, from one code path:

| | SWE-bench Verified | MTEB(eng, v2) |
|---|---|---|
| shape | 134 systems × 500 items | 181 systems × 41 items |
| outcome type | binary | continuous |
| adjacent pairs **not** ordered | **129 / 133** | **176 / 180** |
| all pairs separated | 87.9% | 85.4% |
| indistinguishable from the top | 8 | 4 |
| resolution | 5.32 pp | — (binary only) |

Two benchmarks that share no data, no scoring, no maintainers and no outcome type,
and the shape of the answer is the same: the board separates distant systems and
cannot order neighbours.

The tool ships its own `--selftest`, which must pass before any number is believed:

- identical systems are never ordered, in **both** modes (the negative control);
- 60 planted losses and a uniform +0.5 shift are both detected, so the control above
  is not passing merely because nothing is ever detected;
- on random data, no pair whose gap clears the resolution bound is left unresolved by
  the test — the bound stays conservative, which is the property Part 3 measured.

```
python leaderboard_resolution.py --selftest
python leaderboard_resolution.py matrix.csv          # type auto-detected
```

It is about 200 lines and depends on numpy, pandas and scipy. The input it needs —
per-item outcomes — is already published by both benchmarks and by most others; it is
simply never used this way.


---

**Correction, 2026-08-09.** Part 4 first gave the pair total as 11 807. The correct
figure is **12 739** (8 911 + 3 486 + 276 + 66 for Verified, Lite, Test and Multimodal).
The error was arithmetic on my side, not in the analysis: the per-split counts and the
zero-false-positive result were right throughout, and the corrected total makes the
claim stronger rather than weaker. It was caught by recomputing the sum from
`all_splits_results.json` while preparing a separate document, and it had already gone
out in a letter before being found.
