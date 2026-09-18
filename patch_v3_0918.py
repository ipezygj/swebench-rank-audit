"""Version 3: what the version-2 correction missed, and what it broke.

A re-audit of the corrected paper found that three findings from the first pass
had not been applied to the PDF at all (only to the Zenodo description), that a
table column had never had a computation behind it, and that the corrected
bibliography entry introduced a new error of exactly the class being corrected.
"""
import io, sys

P = "paper.tex"
E = []


def sub(old, new, label):
    E.append((old, new, label))


# --- B4: the abstract still claimed exact parity --------------------------------
sub(r"""and passes a self-test suite; the recomputed rates match each official board exactly.""",
    r"""and passes a self-test suite; the recomputed rates match each official board to its published
precision, with one exception on the Test split, where our rate for one system differs by $1.22$~pp
because its submission file lists $241$ results over $213$ distinct instance ids.""",
    "abstract parity")

# --- the bound predicts from size and discordance -------------------------------
sub(r"""$d$, that predicts which pairs are decidable from a benchmark's size, validate it on $12{,}739$ pairs""",
    r"""$d$, that predicts which pairs are decidable from a benchmark's size and its near-pair discordance,
validate it on $12{,}739$ pairs""",
    "abstract bound")

# --- B9: the reshuffle column had no computation --------------------------------
sub(r"""full-size \textbf{Test} split ($2{,}294$ items) has a clear, stable \#1 that survives a $50$-way task
reshuffle every time; the crowded \textbf{Verified}, \textbf{Lite}, and \textbf{Multimodal} splits do
not.""",
    r"""full-size \textbf{Test} split ($2{,}294$ items) has a \#1 that leads every one of $2{,}000$ item
bootstraps; on the crowded \textbf{Verified} and \textbf{Lite} splits the leader holds its place in
only $38.3\%$ and $53.6\%$ of them (\texttt{top1\_stability.py}, seed $20260918$).""",
    "reshuffle text")

sub(r"""Split & Systems & Items & Adjacent unclear & \#1 survives reshuffle & Largest unclear gap\\""",
    r"""Split & Systems & Items & Adjacent unclear & \#1 holds under item bootstrap & Largest unclear gap\\""",
    "reshuffle header")

sub(r"""SWE-bench Test        & $24$  & $2{,}294$ & $9/23$  & every time & $1.61$ pp\\""",
    r"""SWE-bench Test        & $24$  & $2{,}294$ & $9/23$  & $100.0\%$ & $1.61$ pp\\""",
    "test row")

sub(r"""SWE-bench Verified\footnotemark[1] & $134$ & $500$     & $129/133$ & never    & $4.60$ pp\\""",
    r"""SWE-bench Verified & $134$ & $500$     & $129/133$ & $38.3\%$ & $4.60$ pp\\""",
    "verified row")

sub(r"""SWE-bench Lite        & $84$  & $300$     & $82/83$ & never      & $6.67$ pp\\""",
    r"""SWE-bench Lite        & $84$  & $300$     & $82/83$ & $53.6\%$   & $6.67$ pp\\""",
    "lite row")

# --- B8: the agreement figures were unlabelled ----------------------------------
sub(r"""with no free parameters and scoring against the exact test: agreement is $95.6\%$, $92.4\%$, $98.2\%$,
and $89.4\%$ across the four splits,""",
    r"""with no free parameters and scoring against the exact test: agreement is $95.6\%$ on Verified,
$92.4\%$ on Lite, $98.2\%$ on Test and $89.4\%$ on Multimodal,""",
    "agreement labels")

# --- N1: the corrected bibliography entry was itself wrong ----------------------
sub(r"""\bibitem[Mogstad et~al.(2020)]{mogstad2024} M.~Mogstad, J.~P. Romano, A.~M. Shaikh, and D.~Wilhelm. Inference for ranks with applications to mobility across neighborhoods. 2020.""",
    r"""\bibitem[Mogstad et~al.(2024)]{mogstad2024} M.~Mogstad, J.~P. Romano, A.~M. Shaikh, and D.~Wilhelm. Inference for Ranks with Applications to Mobility across Neighbourhoods and Academic Achievement across Countries. \emph{Review of Economic Studies}, 91(1):476--518, 2024.""",
    "mogstad bibitem")


NOTE3 = r"""

\medskip\noindent
\textbf{Version~3.} A re-audit of version~2 found that the correction had been incomplete in one
direction and wrong in another, so this version records both. Three items from the first pass reached
the deposit's description but never the paper: the abstract still said the recomputed rates match each
official board \emph{exactly}, where the results file records one mismatch of $1.22$~pp on the Test
split; the per-split agreement figures were given in an order different from the table above them; and
the abstract still derived the bound from size alone rather than from size and near-pair discordance.
The corrected Mogstad et al. entry, added in version~2, carried the wrong year, a truncated title and
no venue, which is the same defect the correction existed to remove. And Table~\ref{tab:splits} had a
column headed ``\#1 survives reshuffle'' whose values, ``every time'' and ``never'', had no computation
anywhere in the repository. Rather than reconstruct what those words meant, the check is now defined
explicitly and run: $2{,}000$ bootstrap resamples of the items per split, asking how often the
full-board leader also leads the resample. Test holds at $100\%$; Verified and Lite hold at $38.3\%$
and $53.6\%$, so ``never'' was wrong as well as unevidenced.
"""


def main():
    s = io.open(P, encoding="utf-8").read()
    applied, missed = 0, []
    for old, new, label in E:
        n = s.count(old)
        if n == 1:
            s = s.replace(old, new)
            applied += 1
        else:
            missed.append("%s(%d)" % (label, n))

    # N2: the footnote text belongs with its table, and its mark is gone from the row
    old_ft = ("\\footnotetext[1]{$134$ of the board's $180$ entries publish per-instance results; the\n"
              "census covers those. MTEB likewise: $181$ of $331$ models report all $41$ tasks.}")
    if s.count(old_ft) == 1:
        s = s.replace(old_ft, "")
        anchor = "SWE-bench Verified & $134$ & $500$     & $129/133$ & $38.3\\%$ & $4.60$ pp\\\\"
        note = ("\n\\noindent\\footnotesize $134$ of the board's $180$ entries publish per-instance results; the\n"
                "census covers those. MTEB likewise: $181$ of $331$ models report all $41$ tasks.\\normalsize\n")
        end = s.find("\\end{table}", s.find(anchor))
        if end > 0:
            end = s.find("\n", end) + 1
            s = s[:end] + note + s[end:]
        else:
            missed.append("table end")
    else:
        missed.append("footnotetext(%d)" % s.count(old_ft))

    if "\\textbf{Version~3.}" not in s:
        anchor = "\\begin{thebibliography}"
        if s.count(anchor) == 1:
            s = s.replace(anchor, NOTE3 + "\n" + anchor)
        else:
            missed.append("bib anchor")

    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("applied %d/%d %s" % (applied, len(E), ("MISSED: " + ", ".join(missed)) if missed else ""))
    print("footnotemark left:", s.count("footnotemark"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
