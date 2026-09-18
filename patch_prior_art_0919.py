"""Corrections to the toolkit's prose files, 19 September 2026.

PRIOR_ART.md is a file whose entire job is to describe other people's papers
accurately, so it is the file in this repository where getting them wrong costs
most. Twelve statements did not survive being read against the sources. The paper
built on these files was corrected yesterday; these were not.
"""
import io, sys

E = []


def sub(path, old, new, label):
    E.append((path, old, new, label))


P = "PRIOR_ART.md"
L = "LEADERBOARD_STANDARD.md"

# --- 1. numbers credited to a paper that does not contain them ---------------
sub(P,
    """**Occupied, with better data.** *Stop Comparing LLM Agents Without Disclosing
the Harness* (arXiv:2605.23950) and *The Scaffold Effect in Coding Agents*
(arXiv:2607.22585) report cross-scaffold gaps of 34 points for Claude Sonnet
4.5 and 48 for o4-mini on the Holistic Agent Leaderboard - larger than the
cross-model spread under either scaffold.""",
    """**Occupied, with better data.** *Stop Comparing LLM Agents Without Disclosing
the Harness* (arXiv:2605.23950) reports cross-scaffold gaps under SWE-Agent
versus HAL Generalist of 34 points for Claude Sonnet 4.5, 34 for GPT-5 Medium and
nearly 48 for o4-mini - larger than the cross-model spread under either scaffold.
Those figures are the Holistic Agent Leaderboard's, which that paper cites
(Kapoor et al., arXiv:2510.11977); this entry previously presented them as its
own and also credited them to *The Scaffold Effect in Coding Agents*
(arXiv:2607.22585), which contains none of them - it measures up to a 40x
difference in tokens per solved task across three harnesses, with paired
within-model pass-rate differences of 0 to 8 points.""",
    "scaffold attribution")

# --- 2. GLUE scope and hedge -------------------------------------------------
sub(P,
    """does exactly this for NLP: most attempted comparisons to state of the art on
GLUE tasks are not adequately powered, and a 2000-sentence MT test set has""",
    """does exactly this for NLP: for several tasks in GLUE, small test sets mean most
attempted comparisons to state of the art will not be adequately powered, and a
2000-sentence MT test set has""",
    "glue scope")

# --- 3. the correlation hedge ------------------------------------------------
sub(P,
    """SE with the correlation term and makes the point that eval scores are
positively correlated across models so pairing reduces variance.""",
    """SE with the correlation term and makes the point that eval question scores are
likely to be positively correlated across models, so pairing buys a reduction in
estimator variance.""",
    "correlation hedge")

# --- 4. the IRT distribution hedge ------------------------------------------
sub(P,
    """benchmark data has few models and many items, capability distributions that are
skewed or clustered, and scalable estimators that produce unreliable item-level""",
    """benchmark data has few models and many items, capability distributions that *may
be* skewed, clustered or multimodal, and scalable estimators that produce
unreliable item-level""",
    "irt hedge")

# --- 5. the saturation index's own stated reason -----------------------------
sub(P,
    """uses an unpaired binomial SE with an ad-hoc effective size `n_eff = n^0.5`
   - an exponent standing in for exactly the pair correlation `pair_sharpness.py`
   measures.""",
    """uses an unpaired binomial SE with an effective size `n_eff = n^alpha`, default
   `alpha = 0.5`. Their stated reason is to keep the saturation calculation from
   depending too strongly on test-set size, not to stand in for pair correlation;
   an earlier version of this file asserted the latter. It remains true that the
   quantity `pair_sharpness.py` measures is absent from it.""",
    "n_eff purpose")

# --- 6. the plateau paper's criterion ---------------------------------------
sub(P,
    """from leaderboard data and makes the point that discriminative power is lost
when the top models sit within a couple of points.""",
    """from leaderboard data and makes the point that discriminative power is lost when
the top models are statistically similar, which it defines as a gap no larger
than z times its standard error rather than as a fixed number of points.""",
    "plateau criterion")

# --- 7. the IRT title -------------------------------------------------------
sub(P,
    """**Heavily occupied.** *Lost in Benchmarks? Rethinking LLM Benchmarking with
Item Response Theory* (arXiv:2505.15055)""",
    """**Heavily occupied.** *Lost in Benchmarks? Rethinking Large Language Model
Benchmarking with Item Response Theory* (arXiv:2505.15055)""",
    "irt title")


NOTE_P = """

---

## Corrections, 19 September 2026

This file exists to describe other people's work accurately, so it is the worst
place in the repository to get it wrong. Every claim in it was read against its
source. Twelve did not survive; seven changed the text here and the rest are
recorded below.

- **The cross-scaffold numbers were credited to two papers.** They are in
  arXiv:2605.23950 only, which itself takes them from the Holistic Agent
  Leaderboard (Kapoor et al., arXiv:2510.11977, not previously cited here).
  arXiv:2607.22585 contains no HAL, no o4-mini, no Claude Sonnet 4.5 and none of
  those figures; it reports a token-efficiency result instead. "Nearly 48" also
  lost its hedge.
- **Scope and hedges** were dropped from three entries: "several tasks in GLUE"
  became "GLUE tasks" and "will not be" became "are not"; "likely to be
  positively correlated" became "are positively correlated"; "may be skewed,
  clustered, or multimodal" became "are skewed or clustered".
- **A purpose was attributed that the source assigns otherwise.** The saturation
  index's `n_eff = n^alpha` is there to limit the calculation's dependence on
  test-set size, by its authors' own statement, not as a stand-in for pair
  correlation.
- **A statistical criterion was rendered as a number of points.** The plateau
  paper's test is a gap against its standard error.
- **Two titles** were given with a truncation or an abbreviation not in the
  original.
- **Two novelty verdicts have closer neighbours than this file names.** The
  separable-share quantity is published as "separability with confidence" in
  arXiv:2406.11939, measured by bootstrap rather than predicted in closed form;
  and arXiv:2508.13144 defines a signal-to-noise ratio for benchmarks that is the
  nearest thing in the literature to law 1. Neither falsifies the closed-form
  claim, and both are now part of the neighbourhood this file is supposed to map.
"""

NOTE_L = """

**Correction, 19 September 2026.** The claim that the reference standards this
one is modelled on have no equivalent for machine-learning benchmarks was too
broad. BetterBench (Reuel et al., NeurIPS 2024, arXiv:2411.12990) assesses 24
benchmarks against 46 best practices across a benchmark's lifecycle and publishes
a minimum-quality checklist, and it reports that most benchmarks do not give
statistical significance for their results. The narrower claim this document
actually needs - that no standard specifies what a published *ordering* must
report about its own resolution - survives, and is what it now says.
"""


def main():
    by = {}
    for path, old, new, label in E:
        by.setdefault(path, []).append((old, new, label))
    for path, edits in by.items():
        s = io.open(path, encoding="utf-8").read()
        applied, missed = 0, []
        for old, new, label in edits:
            n = s.count(old)
            if n == 1:
                s = s.replace(old, new)
                applied += 1
            else:
                missed.append("%s(%d)" % (label, n))
        if path == "PRIOR_ART.md" and "Corrections, 19 September 2026" not in s:
            s = s.rstrip() + "\n" + NOTE_P
        io.open(path, "w", encoding="utf-8", newline="\n").write(s)
        print("%-28s %d/%d %s" % (path, applied, len(edits),
                                  ("MISSED: " + ", ".join(missed)) if missed else ""))
    s = io.open(L, encoding="utf-8").read()
    if "Correction, 19 September 2026" not in s:
        io.open(L, "w", encoding="utf-8", newline="\n").write(s.rstrip() + "\n" + NOTE_L)
        print("%-28s correction appended" % L)
    return 0


if __name__ == "__main__":
    sys.exit(main())
