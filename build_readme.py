"""Generate README.md: the map of this repo, with the tool index read from the code.

Every tool's one-line description comes from the first line of its module
docstring, so the index cannot drift from the code without the code being
edited. Tools are grouped by what they are for; a tool that is not in any
group is listed under "unsorted" rather than dropped, so the file fails
loudly rather than quietly.

    python build_readme.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

GROUPS = [
    ("The standard and its reference implementation", [
        "leaderboard_standard.py", "export_card_data.py", "build_certified_page.py",
        "export_card_mteb.py", "build_certified_mteb.py", "build_laws_md.py", "build_readme.py",
    ]),
    ("Core measurement", [
        "rank_sets.py", "two_way_bootstrap.py", "leaderboard_entropy.py", "leaderboard_geometry.py",
        "invariant_core.py", "measurement_invariance.py", "ordinal_invariance.py",
        "benchmark_spectrum.py", "information_depletion.py", "recount_margin.py",
        "pattern_anomaly.py", "reweighting_polytope.py", "saturation_horizon.py",
        "deflated_benchmark.py", "progress_or_selection.py", "selection_sbi.py",
        "leaderboard_resolution.py", "resolution_law.py", "swebench_rank_noise.py",
    ]),
    ("The two laws", [
        "resolution_law_test.py", "entropy_law_test.py", "evidence_trajectory.py",
        "entropy_law_twin2.py", "entropy_decomposition.py", "residual_correlation.py",
        "law1_pairwise.py", "tenth_board.py", "universality.py",
    ]),
    ("Pair resolution (R10) and lineage", [
        "pair_sharpness.py", "kappa_reliability.py", "kappa_generality.py", "kappa_trend.py",
        "kappa_predicts_future.py", "lineage_detection.py", "independence_flag.py",
        "effective_entrants.py", "isotonic_families.py", "pairing_dividend.py",
        "prescription_pairwise.py", "incentive_asymmetry.py", "swebench_base_models.py",
    ]),
    ("SOTA claims through time", [
        "sota_audit.py", "sota_twin.py", "step_sizes.py", "fourth_board.py", "fifth_board.py",
        "sota_luck_law.py", "sota_families.py", "frontier_lineage.py", "leader_luck.py",
        "chase_model.py", "chase_correlated.py", "chase_refit.py", "sibling_chase.py",
        "lineage_tree.py", "broad_or_deep.py", "decisiveness_history.py", "crown_stability.py",
        "model_or_harness.py", "within_family_spread.py",
    ]),
    ("Robustness of the reading", [
        "alpha_sensitivity.py", "method_independence.py", "cluster_bootstrap.py",
        "sota_clustered.py", "half_split_replication.py", "crown_stability.py",
        "missing_entries.py", "crowding_penalty.py", "granularity.py",
    ]),
    ("Design questions a board can be asked", [
        "minimal_benchmark.py", "merge_boards.py", "cheap_entry.py", "composition_all.py",
        "time_to_decide.py", "item_side.py", "benchmark_health.py",
    ]),
    ("Prescription", ["refill_prescription.py", "refill_all.py"]),
    ("Matrix builders", [
        "swebench_matrix.py", "all_splits.py", "helm_matrix.py", "mteb_dates.py",
        "lmarena_resolution.py", "matharena/build_matrix.py", "casp/build_matrix.py",
        "proteingym/dates.py",
    ]),
]

MATRICES = [
    ("SWE-bench Verified", "134 x 500", "binary, per-instance resolve lists"),
    ("SWE-bench Lite", "84 x 300", "binary"),
    ("SWE-bench test", "24 x 2294", "binary, pre-selected entrants"),
    ("MTEB English v2", "181 x 41", "continuous task scores; 174 dated from HF createdAt"),
    ("HELM classic", "90 x 10", "win rates"),
    ("ProteinGym DMS", "96 x 217", "Spearman per assay; dated from reference URLs"),
    ("TabArena", "16 x 51 and 45 x 51", "min-max normalised within dataset"),
    ("CASP14", "101 x 42", "GDT_TS / 100"),
    ("LiveBench", "152 x 200", "judged scores, largest complete block"),
    ("MathArena 2025", "35 x 183", "per-problem correctness"),
    ("LMArena categories", "35 x 28", "category win rates; held out during the 2026-08-23 loop"),
]


def summary(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return "(unparsed)"
    doc = ast.get_docstring(tree) or ""
    first = doc.strip().splitlines()[0] if doc.strip() else "(no docstring)"
    return first.rstrip(".")


def main() -> int:
    here = Path(".")
    listed = {f for _, fs in GROUPS for f in fs}
    present = {str(p).replace("\\", "/") for p in here.glob("*.py")} | \
              {str(p).replace("\\", "/") for p in here.glob("*/*.py")}
    unsorted = sorted(p for p in present if p in {q for q in present} and p not in listed
                      and not p.startswith("_"))
    L = []
    p = L.append
    p("# Leaderboard measurement toolkit")
    p("")
    p("*What a systems x items matrix supports, and what a printed ranking claims.*")
    p("")
    p("Two documents state the results:")
    p("")
    # counted from the file, not asserted: the first draft of this README said
    # "six prohibited presentations" when there were five.
    std = Path("LEADERBOARD_STANDARD.md").read_text(encoding="utf-8")
    n_req = len(re.findall(r"^\| R\d+ \|", std, re.M))
    n_pro = len(re.findall(r"^- ", std.split("## 3. Prohibited")[1].split("## 4")[0], re.M)) if "## 3. Prohibited" in std else 0
    ver = re.search(r"draft ([\d.]+)", std)
    p(f"- **[LEADERBOARD_STANDARD.md](LEADERBOARD_STANDARD.md)** - what a leaderboard must")
    p(f"  publish beside its ranking (draft {ver.group(1) if ver else '?'}, {n_req} required fields and")
    p(f"  {n_pro} prohibited presentations), with `leaderboard_standard.py` as the reference")
    p("  implementation.")
    p("- **[LAWS.md](LAWS.md)** - two relations that predict how much ranking a benchmark")
    p("  supports and how much of a printed order is evidence, tested on ten boards from")
    p("  five fields, including one held out until after the relations were fixed.")
    p("")
    p("Everything is computed from the score matrix alone. No tool needs metadata, and")
    p("the ones that use names (lineage detection, base-model families) say so and are")
    p("validated against a permutation null.")
    p("")
    p("## Matrices")
    p("")
    p("| board | shape | notes |")
    p("|---|---|---|")
    for b, shape, note in MATRICES:
        p(f"| {b} | {shape} | {note} |")
    p("")
    p("## Tools")
    for title, files in GROUPS:
        p("")
        p(f"### {title}")
        p("")
        for f in files:
            fp = here / f
            if fp.exists():
                p(f"- `{f}` - {summary(fp)}")
            else:
                p(f"- `{f}` - MISSING FROM DISK")
    if unsorted:
        p("")
        p("### Unsorted (add to a group in build_readme.py)")
        p("")
        for f in unsorted:
            p(f"- `{f}` - {summary(here / f)}")
    p("")
    p("## How the results were produced")
    p("")
    p("Each tool prints its own self-checks first and refuses to print a table if one")
    p("fails. Thresholds and expectations are written into the module docstring and")
    p("committed to git BEFORE the run that tests them; when an expectation fails, the")
    p("failure is recorded in the same file rather than removed. Where a construction")
    p("was changed after a failed check, the change and its reason are in the docstring.")
    p("")
    p("Results live next to each tool as `*_results.txt` and are regenerated by running")
    p("the tool. `build_laws_md.py` and `build_readme.py` assemble the documents from")
    p("those files and from the code, so neither document can drift silently.")
    Path("README.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    missing = sum(1 for line in L if "MISSING FROM DISK" in line)
    print(f"wrote README.md: {len(L)} lines, {sum(len(f) for _, f in GROUPS)} tools listed, "
          f"{len(unsorted)} unsorted, {missing} missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
