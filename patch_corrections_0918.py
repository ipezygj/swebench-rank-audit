"""Corrections after the 2026-09-18 source check of the deposited paper.

Every edit below is a statement about someone else's work, or a checkable claim
about my own artefacts, that did not survive being read against the source. No
measurement changes: every quantitative result in the paper reproduced when it
was rerun.

Written as a file, not a heredoc: heredocs eat LaTeX escapes.
"""
import io, sys

P = "paper.tex"
src = io.open(P, encoding="utf-8").read()
done = []


def sub(old, new, label):
    global src
    n = src.count(old)
    if n != 1:
        print("FAIL %-30s count=%d" % (label, n))
        print("     " + old[:120].replace("\n", "\\n"))
        sys.exit(1)
    src = src.replace(old, new)
    done.append(label)


# --- A1: the dropping result is Huang et al.'s, not Singh et al.'s ----------
sub(r"""Recent work has circled this point from several sides. \citet{singh2025} show that dropping a small
share of Chatbot Arena's data reshuffles its top ranks; \citet{huang2025} reach the same conclusion
via influence functions; \citet{mizzaro2025} find that leaderboard \emph{rankings} stay stable under
paraphrase even as absolute scores collapse;""",
    r"""Recent work has circled this point from several sides. \citet{huang2025} show that dropping
$0.003\,\%$ of Chatbot Arena's human preferences can change its top-ranked model, by a method built
on influence functions; \citet{singh2025} document undisclosed private testing, selective
disclosure and unequal data access at the top of the same board; \citet{lunardi2025} find that
model rankings stay \emph{relatively} stable under paraphrase even as absolute scores decline
significantly;""",
    "singh/huang inversion")

# --- A4: Zhang and Hardt measured a trade-off, they did not prove it --------
sub(r"""\citet{zhang2024}
prove that no aggregation of a multi-task board can be both diverse and stable under such perturbations---a""",
    r"""\citet{zhang2024}
apply Arrow's theorem to ordinal benchmarks and measure a trade-off between a multi-task board's
diversity and its sensitivity to irrelevant changes---a""",
    "zhang prove -> measure")

# --- A7: the prior art this repo had already identified ---------------------
sub(r"""items actually carry a benchmark's signal. We contribute a direct, uniform, and reproducible test of
the object those threads all bear on---the \emph{rank}---and a bound that says, from a board's size
alone, which ranks it can and cannot resolve. Our contributions:""",
    r"""items actually carry a benchmark's signal. The requirement itself is not new:
\citet{card2020} established for NLP that comparisons need the power to support them, and
\citet{mogstad2024} give simultaneous confidence sets for ranks. We apply that requirement
uniformly to the rank itself, on five public boards, and give a closed form for it from a board's
size and its near-pair discordance. Our contributions:""",
    "prior art card/mogstad")

# --- B5: the bound needs d as well as n ------------------------------------
sub(r"""  \item A resolution bound $\delta \ge 2.80\sqrt{d/n}$, derived from test power and validated on""",
    r"""  \item A resolution bound $\delta \ge 2.80\sqrt{d/n}$ in the board's size $n$ and its near-pair
  discordance $d$, derived from test power and validated on""",
    "bound needs d")

# --- B2: the third self-test is a different check, in a different script ----
sub(r"""is ever detected; (iii) a published Wilson interval is reproduced ($5/10 \to [0.2366, 0.7634]$). The
tool is $\sim\!200$ lines (\texttt{numpy}/\texttt{pandas}/\texttt{scipy}) and ships all three.""",
    r"""is ever detected; (iii) on random data, no pair whose gap clears the bound is left unresolved by
the test. The tool is $\sim\!200$ lines (\texttt{numpy}/\texttt{pandas}/\texttt{scipy}) and ships all
three; a fourth check, that a published Wilson interval is reproduced ($5/10 \to [0.2366, 0.7634]$),
runs in the SWE-bench builder.""",
    "self-test iii")

# --- B6: the tie set includes the top system itself -------------------------
sub(r"""$129$ of the $133$ adjacent-rank pairs are not separable at $5\%$; eight systems are statistically""",
    r"""$129$ of the $133$ adjacent-rank pairs are not separable at $5\%$; the top eight systems are statistically""",
    "tie count wording")

# --- B3: the vote count -----------------------------------------------------
sub(r"""LMArena \emph{(positive control)} & $242$ & 3.5M votes""",
    r"""LMArena \emph{(positive control)} & $242$ & 3.0M votes""",
    "lmarena votes")

# --- B7: the analysed subset is not the board ------------------------------
sub(r"""SWE-bench Verified & $134$ & $500$ & $129/133$ & $87.9\%$\\""",
    r"""SWE-bench Verified\footnotemark[1] & $134$ & $500$ & $129/133$ & $87.9\%$\\""",
    "verified subset marker")

# --- B1: delta is the number every board shows ------------------------------
sub(r"""Both $d$ and $\delta$ are readable from published files, and no board reports either""",
    r"""Both are readable from published files; $\delta$ is the difference a board already shows, and $d$ is the one no board reports""",
    "no board reports either")

# --- A2, A3: what LMArena's rule actually is, and who ties with whom --------
sub(r"""confidence intervals, and its published \texttt{final\_ranking} assigns the \emph{same} rank to models
whose intervals overlap. From its own public numbers ($242$ models), $238$ of $241$ adjacent-by-rating
pairs have overlapping intervals and ranks $2$--$11$ are a ten-way tie---yet the \#1 is genuinely""",
    r"""confidence intervals, and its published \texttt{final\_ranking} gives each model one plus the number
of models significantly above it, so models it cannot separate share a rank. From its own public
numbers ($242$ models), $238$ of $241$ adjacent-by-rating pairs have overlapping intervals and ten
models share the published rank $2$---yet the \#1 is genuinely""",
    "lmarena rule and tie")

# --- A5, A7: the bibliography ----------------------------------------------
sub(r"""\bibitem[Mizzaro \& Roitero(2025)]{mizzaro2025} S.~Mizzaro and K.~Roitero. On Robustness and Reliability of Benchmark-Based Evaluation of LLMs. 2025.""",
    r"""\bibitem[Lunardi et~al.(2025)]{lunardi2025} R.~Lunardi, V.~Della Mea, S.~Mizzaro, and K.~Roitero. On Robustness and Reliability of Benchmark-Based Evaluation of LLMs. \emph{ECAI}, 2025. arXiv:2509.04013.
\bibitem[Card et~al.(2020)]{card2020} D.~Card, P.~Henderson, U.~Khandelwal, R.~Jia, K.~Mahowald, and D.~Jurafsky. With little power comes great responsibility. \emph{EMNLP}, 2020. arXiv:2010.06595.
\bibitem[Mogstad et~al.(2024)]{mogstad2024} M.~Mogstad, J.~P. Romano, A.~M. Shaikh, and D.~Wilhelm. Inference for rankings. \emph{Review of Economic Studies}, 2024.""",
    "bibliography fixes")

# --- the corrections note ---------------------------------------------------
NOTE = r"""
\section*{Corrections in this version}
Version~2. No measurement changed: every quantitative result in this paper was rerun
from the deposited data and reproduced, including the three parity gates, the
preregistration scoring and all four figure coordinates. What changed are sentences
about other people's work, and four checkable claims about my own artefacts, none of
which survived being read against the source on 2026-09-18.

\begin{itemize}
\item \textbf{The dropping result belongs to \citet{huang2025}}, whose method is built on
influence functions and who report that $0.003\,\%$ of preferences can change Chatbot
Arena's top model. Version~1 credited it to \citet{singh2025} and demoted Huang et~al.\ to
a confirmation. \citet{singh2025} document private testing, selective disclosure and
unequal data access instead.
\item \textbf{\citet{zhang2024} measure a trade-off; they do not prove an impossibility.}
Arrow's theorem is applied there to ordinal benchmarks; the diversity--sensitivity
trade-off is an empirical result over seven cardinal and eleven ordinal benchmarks.
\item \textbf{Prior art is now cited.} \citet{card2020} established the power requirement
for benchmark comparisons, and \citet{mogstad2024} give simultaneous confidence sets for
ranks. The repository's own prior-art note had recorded both before version~1 was
deposited, and the paper failed to carry them.
\item \textbf{The citation to \citet{lunardi2025}} named its third and fourth authors and
dropped the first two; and their ``relatively stable'' and ``decline significantly'' had
been strengthened to ``stable'' and ``collapse''.
\item \textbf{LMArena's rule was misdescribed.} It gives a model one plus the number of
models significantly above it, which reproduces its published ranking for all 242 models;
version~1 said equal ranks mean overlapping intervals, which the same file refutes. The
ten models sharing rank~2 sit at rating positions 2--9, 11 and 20, so ``ranks 2--11'' was
wrong; and its vote count is $3.0$M, not $3.5$M.
\item \textbf{Four claims about my own artefacts:} the bound needs the discordance $d$ as
well as the size $n$; the third self-test is the conservatism check, with the Wilson
reproduction running in the SWE-bench builder; $\delta$ is the number boards already
publish, so only $d$ is unreported; and the eight statistically indistinguishable systems
include the \#1 itself.
\end{itemize}

\footnotetext[1]{$134$ of the board's $180$ entries publish per-instance results; the
census covers those. MTEB likewise: $181$ of $331$ models report all $41$ tasks.}

"""
sub(r"""\begin{thebibliography}{9}\small""", NOTE + r"""\begin{thebibliography}{9}\small""",
    "corrections note")

io.open(P, "w", encoding="utf-8", newline="\n").write(src)
print("patched %d:" % len(done))
for d in done:
    print("  - " + d)
