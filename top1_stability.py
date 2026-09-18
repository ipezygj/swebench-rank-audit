"""Does the #1 system stay #1 when the items are resampled?

The paper's Table 2 carried a column headed "#1 survives reshuffle", with the
values "every time" and "never", and the text spoke of a 50-way task reshuffle.
No computation for either was ever stored in this repository. Rather than
reconstruct what the words might have meant, this script defines the check
explicitly and runs it.

Procedure, per split: resample the items with replacement B times; on each
resample recompute every system's resolve rate; record whether the system that
leads the full board also leads the resample. The reported number is the share
of resamples in which it does. Seeded, so it reproduces.
"""
import csv, io, json, sys
import numpy as np

SPLITS = [
    ("verified", "swebench_verified_matrix.csv"),
    ("lite", "swebench_lite_matrix.csv"),
    ("test", "swebench_test_matrix.csv"),
]
B = 2000
SEED = 20260918


def load(path):
    rows = list(csv.reader(io.open(path, encoding="utf-8")))
    systems = [r[0] for r in rows[1:]]
    M = np.array([[float(x) if x not in ("", "NA") else np.nan for x in r[1:]] for r in rows[1:]])
    return systems, M


def main():
    rng = np.random.default_rng(SEED)
    out = []
    for name, path in SPLITS:
        systems, M = load(path)
        n_sys, n_items = M.shape
        rates = np.nanmean(M, axis=1)
        top = int(np.argmax(rates))
        wins = 0
        ties = 0
        for _ in range(B):
            idx = rng.integers(0, n_items, n_items)
            r = np.nanmean(M[:, idx], axis=1)
            best = r.max()
            leaders = np.flatnonzero(r >= best - 1e-12)
            if top in leaders:
                if len(leaders) == 1:
                    wins += 1
                else:
                    ties += 1
        share = (wins + ties) / B
        out.append({
            "split": name, "systems": n_sys, "items": n_items,
            "top_system": systems[top], "top_rate_pct": round(100 * rates[top], 4),
            "bootstrap_B": B,
            "top1_retained_share": round(share, 4),
            "sole_leader_share": round(wins / B, 4),
        })
        print("%-9s n=%-5d systems=%-4d top=%-28s rate=%6.2f%%  retains #1 in %5.1f%% of resamples (sole in %5.1f%%)"
              % (name, n_items, n_sys, systems[top][:28], 100 * rates[top], 100 * share, 100 * wins / B))
    io.open("top1_stability_results.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps({"seed": SEED, "splits": out}, indent=2) + "\n")
    print("\nwritten: top1_stability_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
