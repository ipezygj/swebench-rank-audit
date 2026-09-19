"""The workshop manuscript never got the corrections the main paper did.

paper_tae.tex is the double-blind workshop version of the same work. It still
carries three things corrected elsewhere: the Arena vote total, a citation under
two of its four authors, and a truncated title. The main paper still had the
truncated title too.
"""
import io, sys

E = []


def sub(path, old, new, label):
    E.append((path, old, new, label))


TITLE_OLD = (r"\bibitem[Huang et al.(2025)]{huang2025} J.~Y.~Huang et al. Dropping Just a "
             r"Handful of Preferences Can Change Top LLM Rankings. \emph{arXiv:2508.11847}, 2025.")
TITLE_NEW = (r"\bibitem[Huang et al.(2025)]{huang2025} J.~Y.~Huang, Y.~Shen, D.~Wei and "
             r"T.~Broderick. Dropping Just a Handful of Preferences Can Change Top Large "
             r"Language Model Rankings. \emph{arXiv:2508.11847}, 2025.")

sub("paper.tex", TITLE_OLD, TITLE_NEW, "huang title (paper)")
sub("paper_tae.tex", TITLE_OLD, TITLE_NEW, "huang title (tae)")

sub("paper_tae.tex",
    r"""LMArena \emph{(positive control)} & $242$ & 3.5M votes & $238/241$ CIs overlap & ---\\""",
    r"""LMArena \emph{(positive control)} & $242$ & 3.0M votes & $238/241$ CIs overlap & ---\\""",
    "arena vote total")

sub("paper_tae.tex",
    r"""\bibitem[Mizzaro \& Roitero(2025)]{mizzaro2025} S.~Mizzaro and K.~Roitero. On Robustness and Reliability of Benchmark-Based Evaluation of LLMs. 2025.""",
    r"""\bibitem[Lunardi et al.(2025)]{mizzaro2025} R.~Lunardi, V.~Della~Mea, S.~Mizzaro and K.~Roitero. On Robustness and Reliability of Benchmark-Based Evaluation of LLMs. \emph{arXiv:2509.04013}, ECAI 2025.""",
    "lunardi authors")


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
        io.open(path, "w", encoding="utf-8", newline="\n").write(s)
        print("%-16s %d/%d %s" % (path, applied, len(edits),
                                  ("MISSED: " + ", ".join(missed)) if missed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
