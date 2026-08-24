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
fixed before the field of entrants exists**, and for the first of the two the
prediction is a closed form rather than a simulation. Three of the four are
design choices; the fourth needs a two-system pilot on the item set, which is
the correction the limitations section forced on the abstract.

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

## TMLR requirements, checked 2026-08-24

From the author guide (jmlr.org/tmlr/author-guide.html):

- **Double blind, enforced.** "Non-anonymous submissions will be rejected
  without review." The `tmlr` package hides the author block by itself when
  loaded without `[accepted]` or `[preprint]`, so the work is in the body: the
  repository URL carried the author's username and is gone, and the
  reproducibility statement no longer points at a named repository.
- **A submission must not link to a non-anonymous version of itself.** So the
  code link becomes a statement until an anonymised mirror exists. Note this
  cuts against the repo being public under the author's name — an anonymised
  mirror has to be made before submitting, not after.
- **Style file mandatory**, from the tmlr-style-file repository. Applied:
  `tmlr.sty`, `fancyhdr.sty`, `tmlr.bst` sit beside the paper and the
  bibliography style is `tmlr`.
- **No page limit**, but "a paper's length should be justified by its content"
  and unusually long papers delay review. Currently seven pages plus
  references, which is short for TMLR and appropriate for two results.
- **Broader Impact Statement is conditional** — required only if the work
  "carries a significant risk of harm". This does not, and none is included.
  If a reviewer asks, the answer is that the laws are neutral about whether a
  benchmark is good.
- **arXiv preprints are allowed at any time**, anonymously or not, as long as
  the submission itself is not linked to a named version.

### The blocker, and it is not the paper

All authors must have "complete and active OpenReview profiles". Ours is not
complete: OpenReview's moderation rejects a profile whose only history entry is
"Independent Researcher", and the account has no institution and no registered
business to supply a second entry. What passes moderation is a second history
record (a former employer with an end year is enough) plus a homepage showing
name, affiliation and email. That is an administrative task with a moderation
queue of up to two weeks, and it has to be finished before the paper can be
submitted at all.

### Visibility: the submission goes public, not just the acceptance

Checked 2026-08-24 against the editorial policies, because the assumption in
the other direction would have changed the plan.

- An action editor is assigned **within a week**. Once they assign at least
  three reviewers, "**the paper will become public**". Public visibility
  therefore arrives in weeks, not at acceptance.
- **Reviews** are the part held back: visible to the authors as they arrive,
  but "not visible to the public nor to the other reviewers until all the
  reviews are submitted".
- Final recommendations come no earlier than two weeks after all three reviews
  are public, so a decision is months away even though visibility is not.
- A rejected paper may be revised and resubmitted, but "it will need to be
  entered as a new submission and a link provided to the previously rejected
  submission" — not a dead end, but it stays on the record.
- Certifications exist (Outstanding, Featured, Reproducibility, Survey) and are
  awarded on acceptance, not requested.

**Why this changes the order of work.** The route gives a public, timestamped
version of the work within weeks and without an endorsement gate, which is
precisely what arXiv moderation denied this project. But the public version is
the *anonymous* one, so it builds no name until acceptance, and the
anonymisation is not a formality — it is what the world sees. It also means the
OpenReview profile is not merely an administrative box: for as long as it is
incomplete, nothing is public at all. Profile first, then submit; everything
else in this directory is already done.
