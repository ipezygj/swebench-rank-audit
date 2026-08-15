# How much of a leaderboard ranking survives its own sampling error?

A benchmark leaderboard arrives already sorted, a crisp number beside every name. But
every entrant is scored on the *same* items and publishes which it solved, so the board
is a paired experiment, not a list of independent numbers — and you can ask it whether
its order is real.

This repo does that for **SWE-bench** (four splits) and **MTEB**, from public data.

## The finding

On **SWE-bench Verified** (134 systems × 500 instances), exact McNemar on the instances
where each pair disagrees:

- **129 of 133 adjacent-rank pairs are not separable** at the 5% level.
- **8 systems are statistically inseparable from the #1** (ranks 1–8, 76.4–79.2%).
- #1 vs #2 resolve the same count; of 36 disagreements the split is 18/18, p = 1.000.

It is **not** that the benchmark is broken: 87.9% of all 8,911 pairs *are* separated. The
board discriminates distant systems cleanly and can't order neighbours — and neighbours
are what the top of a ranking is made of.

The same test across four splits, and across a different benchmark family:

| split | tasks | tied for first | "#1" survives a 50-way task reshuffle |
|---|---:|---:|---|
| SWE-bench Test | 2,294 | 1 | every time — a real champion |
| SWE-bench Verified | 500 | 3 | never |
| SWE-bench Lite | 300 | 4 | never |
| SWE-bench Multimodal | 517 | 5 | never |
| MTEB (eng, v2) | 41 | 4 tied w/ top | 176/180 adjacent pairs not ordered |

Whether a board can order its leaders comes down to task count and spread, not benchmark
quality. Full write-up and derivation in [`RESULTS.md`](RESULTS.md); the fixed test set
and prediction written down first in [`PREREGISTERED.md`](PREREGISTERED.md).

## The tool

`leaderboard_resolution.py` takes any matrix of per-item outcomes — one row per system,
one column per item — and reports which adjacent ranks are separated, the tie-groups, and
how many more items it would take to resolve a given gap. Pass/fail data goes through exact
McNemar on discordant items; a score per item goes through a paired bootstrap over items.
One code path, ~200 lines (numpy / pandas / scipy).

```bash
python leaderboard_resolution.py --selftest                 # run this first
python leaderboard_resolution.py swebench_verified_matrix.csv   # binary, auto-detected
python leaderboard_resolution.py mteb_eng_v2_wide.csv          # continuous scores
python swebench_rank_noise.py                                 # the Verified deep-dive
```

The `--selftest` must pass before any number is believed: identical systems are never
ordered (in both modes); 60 planted losses and a uniform shift are both detected, so the
control isn't passing merely because nothing is ever caught.

## Reproducing from raw data

`swebench_verified_matrix.csv` and `mteb_eng_v2_wide.csv` are the processed per-item
matrices the analysis runs on, included here so the results above reproduce directly.
`swebench_matrix.py` and `all_splits.py` rebuild those matrices from a local checkout of
[SWE-bench/experiments](https://github.com/SWE-bench/experiments) (the `evaluation/<split>`
result files); point them at that checkout to regenerate from scratch.

Parity: the recomputed resolve rates match the official leaderboard for all 133 systems
with a result file, exact to the digits it prints.

## Related

Filed against SWE-bench/experiments: the duplicate instance IDs under `evaluation/test/20240402`
([#463](https://github.com/SWE-bench/experiments/issues/463), fix in
[#465](https://github.com/SWE-bench/experiments/pull/465)) and the rank-resolution point
([#466](https://github.com/SWE-bench/experiments/issues/466)).

— Ilpo Väätäinen · [Measured, Not Believed](https://leanpub.com/measurednotbelieved)
