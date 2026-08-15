# Pre-registration — written before the other splits were fetched

**Date: 2026-08-09.** Recorded before running anything beyond SWE-bench Verified,
so the prediction can miss.

## The claim being tested

The Verified result (129 of 133 adjacent-rank pairs not separable; gaps up to
**4.60 pp** still undecided) is caused by **sampling error on a finite instance
set**, not by something specific to Verified.

If that is right, the resolution of a benchmark is set by its instance count, and
the largest-undecided-gap should shrink roughly as `1 / sqrt(n)`.

## Anchor

SWE-bench **Verified**: n = 500 instances → largest undecided gap **4.60 pp**
(measured, exact McNemar, 134 systems).

## Predictions

Scaling from the anchor by `4.60 * sqrt(500 / n)`:

| split | instances (n) | predicted largest undecided gap | envelope I will accept |
|---|---:|---:|---|
| **Lite** | 300 | **5.94 pp** | 4.5 – 7.5 pp |
| **Test** | 2294 | **2.15 pp** | 1.5 – 3.0 pp |
| **Multimodal** | 517 | **4.52 pp** | 3.4 – 5.9 pp |

Secondary prediction: the *share* of adjacent pairs that are undecided depends on
how tightly systems are packed, not only on n, so I do **not** predict it goes down
on Test. If Test still has most adjacent pairs undecided despite 2294 instances,
that means the systems there are simply closer together, and the finding becomes
"the leaderboards are packed tighter than they can resolve" rather than "500 is
too few".

## What would refute the sampling-error explanation

- Test (n = 2294) showing a largest-undecided-gap **above 3.0 pp**. With 4.6× the
  instances, if the undecided band does not shrink at all, the cause is not
  sampling error on the instance set and the Verified write-up needs rewriting.
- Lite (n = 300) showing a gap **below 4.5 pp**, i.e. a *smaller* undecided band
  on *fewer* instances, which the mechanism cannot produce.

## Method, fixed in advance

Identical to the Verified run: exact McNemar on paired per-instance outcomes,
alpha = 0.05, rate = |resolved| / n. Same self-checks (identity p = 1, planted
60-flip detected, Wilson 5/10 reference), and the analysis aborts if they fail.
Instance count `n` is taken as the size of the union of instance IDs appearing in
the split's own submissions, and reported rather than assumed.

Parity against the published board is required before any number from a split is
quoted, as it was for Verified (133/133 exact there).

## What I am not predicting

Nothing about multilingual or bash-only, which I have not looked at. Adding them
later does not get to count as a hit.
