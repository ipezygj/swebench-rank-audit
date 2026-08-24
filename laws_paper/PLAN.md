# The two laws — scope, claim, and what this paper is not

Written 2026-08-24, before any of the paper. The point of writing it first is
that this project already has a paper on the adjacent question
(`../paper.tex`, "How Much of a Leaderboard Ranking Survives Its Own Sampling
Error?"), arXiv moderation declined it, and `../PRIOR_ART.md` then found that
its territory is occupied by concurrent 2026 work. Writing the same paper again
in a new coat is the failure mode to avoid, so the boundary is set here and the
draft is held to it.

## The claim, in one sentence

**How much ranking a leaderboard can support is predictable from four numbers
that are measurable before any system is run**, and the prediction is a closed
form rather than a simulation.

## The two laws

**Law 1 — the established share.** For a field whose scores have spread `tau`,
measured on `n` items whose paired difference SD is `sigma_p`, with a
simultaneous critical value `c`, the share of ordered pairs the evidence
establishes is

    established = Phibar(1 / SNR),    SNR = tau * sqrt(2n) / (c * sigma_p)

**Law 2 — the ordering entropy.** The entropy of the set of orderings the
evidence permits, `H / log2(J!)`, is reproduced by a Gaussian field with the
same `(J, n, tau, sigma_p)` and nothing else of the real board - no skew, no
clusters, no outliers.

## What makes this publishable, if anything does

1. It is a **closed form**, not an index fitted to leaderboards. Nothing in it
   was tuned: the constant is the normal quantile the construction itself uses.
2. It is validated across **five fields** - code, embeddings, competition
   mathematics, protein fitness, tabular prediction, plus a structural-biology
   board - where the prior art is almost entirely LLM benchmarks.
3. A **tenth board was held out** until the thresholds were committed to git,
   and it passes both laws.
4. It survives a **change of estimator**. The rank-set construction underneath
   was found to undercover on eight of twelve boards and was replaced; the
   realised critical value moved from 3.14 to 8.45 across boards, and the
   prediction followed. A law that tracks a threefold change in its own input
   is about resolution, not about an estimator.

## What this paper is NOT, and must not drift into

- **Not** "benchmark comparisons are underpowered". That is Card et al. (2020)
  and saying it again adds nothing.
- **Not** a method for rank confidence sets. That is Mogstad et al. (2024) and
  two concurrent 2026 papers.
- **Not** the leaderboard standard, the 99 tools, or the conformance costs.
  Those go in a separate artefact if anywhere.
- **Not** the top-of-board chain (compression, family clustering, redundancy).
  It is a good story and it is a different paper.
- **Not** a claim that leaderboards are bad. The laws are neutral: CASP14
  resolves its top at t = 9.89 and the same formula says so.

## The honest limits, stated in the paper and not buried

- **TabArena fails both laws** by 17 points on law 1 and 23-28 on law 2, in
  the pre-registered direction (skewed field, the Gaussian twin spreads
  established pairs evenly where the real field concentrates them). Two boards
  of nine.
- **The laws do not predict the top.** Aggregate quantities are reproduced;
  the number of systems that could be first is not, by a factor of six on
  SWE-bench Verified.
- **`c` is not free.** It is the construction's own critical value, so the law
  predicts resolution *given* a multiplicity procedure, not in the abstract.
  This is a real restriction and the paper states it as one.
- **Sample of boards is opportunistic**, not a random sample of benchmarks:
  they are the boards whose per-item outcomes are public.

## Target

TMLR. No endorsement gate, no affiliation gate, rolling submission, open
review. arXiv moderation already declined this project once on submitter
grounds, so a venue with an actual review process is the route, and a TMLR
acceptance is also what would make an arXiv appeal viable later.

## Discipline for the draft

Every number in the paper is parsed out of a `*_results.txt` by
`build_paper.py`, exactly as `../build_laws_md.py` does for LAWS.md, and any
figure it cannot find prints as MISSING. Nothing is retyped by hand. This
project has put a wrong number in a commit message four times in one evening by
writing from memory, and a paper is not the place to do it a fifth.
