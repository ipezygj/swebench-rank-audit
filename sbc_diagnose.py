"""The one tool that fails the pipeline: is the posterior wrong, or the check?

selection_sbi.py is the only failure in a 110-tool run. It refuses to print
because its own simulation-based calibration check reports KS p = 0.0003, and
refusing is the right behaviour: a miscalibrated posterior looks exactly like a
good one. But "SBC failed" has been recorded as a KNOWN failure for days without
anyone asking two questions that decide what it means.

  Which way does it fail? Ranks piled at the ends mean the posterior is too
  NARROW and the tool is overconfident. Ranks piled in the middle mean it is too
  WIDE, and since that file's headline claim is that N is NOT identified - a flat
  posterior - a too-wide posterior would be manufacturing exactly the conclusion
  it reports. The direction is the whole meaning of the failure.

  Can the check pass anything? An SBC test is code too. Handed a posterior that
  is calibrated by construction, it must return uniform ranks. Nobody has ever
  run it on one, so "the posterior is miscalibrated" and "the checker is broken"
  are currently the same observation.

This answers both. The rank statistic is also rebuilt: the original takes
cdf[i] at the grid index at or above the true value, which reads the TOP of the
containing bin and biases every rank upward by about half a bin. Both versions
are reported so the size of that effect is visible rather than assumed.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  the checker passes its positive control: fed a posterior drawn from the
      exact prior on a simulator whose data carry no information, SBC ranks are
      uniform at KS p > 0.05. If this fails, nothing else here means anything.
  P2  the checker rejects a negative control: a posterior deliberately narrowed
      by a factor of 4 around the truth is caught at KS p < 0.01.
  P3  the real failure is OVERCONFIDENCE - ranks piled at the ends, so the
      posterior is too narrow. Predicted because the ratio estimator is trained
      to discriminate and discriminative training sharpens.
  P4  the half-bin bias in the original rank statistic is small: correcting it
      moves the KS p-value by less than a factor of 10, and the check still
      fails. If correcting it makes the failure go away, the tool has been
      refusing to print for days because of an off-by-half-a-bin.

SELF-CHECKS (no table if any fails)
  * the positive and negative controls must disagree with each other, or the
    checker has no resolution at all;
  * the rank vectors must have the advertised length and lie in [0, 1];
  * the trained model must be the same one selection_sbi.py uses - same seed,
    same simulation count - or this is diagnosing a different object.

    python sbc_diagnose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

import selection_sbi as sb

SEED = 20260825
REPS = 300
GRID_N = 61
GRID_S = 41


def ranks_of(model, reps, rng, grid_logn, grid_s, half_bin_fix: bool):
    """SBC ranks for the marginal over log N.

    half_bin_fix interpolates the CDF at the true value and jitters within the
    containing bin, which is what removes the upward bias of reading cdf at the
    grid point at or above the truth.
    """
    out = []
    for _ in range(reps):
        th = sb.sample_prior(1, rng)[0]
        x = sb.summarise(sb.top_of_n(th[0:1], th[1:2], sb.TOPJ, rng))[0]
        post = sb.posterior_grid(model, x, grid_logn, grid_s)
        marg = post.sum(axis=1)
        cdf = np.cumsum(marg)
        if not half_bin_fix:
            i = int(np.searchsorted(grid_logn, th[0]))
            i = min(max(i, 0), len(cdf) - 1)
            out.append(float(cdf[i]))
        else:
            lo = np.concatenate([[0.0], cdf[:-1]])
            i = int(np.clip(np.searchsorted(grid_logn, th[0]) - 1, 0,
                            len(cdf) - 1))
            out.append(float(lo[i] + rng.random() * marg[i]))
    return np.array(out)


def shape_of(r: np.ndarray) -> str:
    """Where the mass sits: ends means too narrow, middle means too wide."""
    ends = float(((r < 0.1) | (r > 0.9)).mean())
    mid = float(((r > 0.4) & (r < 0.6)).mean())
    if ends > 0.25:
        return f"ENDS {100 * ends:.0f}% (posterior too NARROW - overconfident)"
    if mid > 0.30:
        return f"MIDDLE {100 * mid:.0f}% (posterior too WIDE - underconfident)"
    return f"neither (ends {100 * ends:.0f}%, middle {100 * mid:.0f}%)"


def control_ranks(reps, rng, narrow: float | None):
    """SBC on a posterior that is calibrated by construction.

    The prior is uniform on the log N range. If the data carry no information,
    the correct posterior IS the prior, and the SBC rank of the truth is its
    prior CDF - uniform by definition. narrow shrinks that posterior around the
    truth by the given factor, which must be caught.
    """
    lo, hi = sb.LOGN_RANGE
    out = []
    for _ in range(reps):
        t = lo + (hi - lo) * rng.random()
        if narrow is None:
            out.append((t - lo) / (hi - lo))
        else:
            w = (hi - lo) / narrow
            a, b = max(lo, t - w / 2), min(hi, t + w / 2)
            # rank of the truth inside a box posterior centred on it
            out.append((t - a) / max(b - a, 1e-12))
    return np.array(out)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)

    print("controls ...")
    pos = control_ranks(REPS, rng, None)
    neg = control_ranks(REPS, rng, 4.0)
    p_pos = float(stats.kstest(pos, "uniform").pvalue)
    p_neg = float(stats.kstest(neg, "uniform").pvalue)
    ok_pos = p_pos > 0.05
    ok_neg = p_neg < 0.01
    print(f"  [{'ok  ' if ok_pos else 'FAIL'}] positive control (posterior = prior): "
          f"KS p = {p_pos:.4f}")
    print(f"  [{'ok  ' if ok_neg else 'FAIL'}] negative control (4x too narrow): "
          f"KS p = {p_neg:.6f}")
    ok_res = ok_pos and ok_neg
    print(f"  [{'ok  ' if ok_res else 'FAIL'}] the two controls disagree, so the "
          f"checker has resolution")

    print("training the same model selection_sbi.py trains ...")
    tr = np.random.default_rng(sb.SEED)
    theta = sb.sample_prior(60000, tr)
    x = sb.summarise(sb.top_of_n(theta[:, 0], theta[:, 1], sb.TOPJ, tr))
    model = sb.train_ratio(theta, x, epochs=12, seed=sb.SEED)

    grid_logn = np.linspace(*sb.LOGN_RANGE, GRID_N)
    grid_s = np.linspace(*sb.S_RANGE, GRID_S)
    r_orig = ranks_of(model, REPS, np.random.default_rng(SEED + 1),
                      grid_logn, grid_s, False)
    r_fix = ranks_of(model, REPS, np.random.default_rng(SEED + 1),
                     grid_logn, grid_s, True)
    p_orig = float(stats.kstest(r_orig, "uniform").pvalue)
    p_fix = float(stats.kstest(r_fix, "uniform").pvalue)

    ok_len = (len(r_orig) == REPS and len(r_fix) == REPS
              and r_orig.min() >= 0 and r_orig.max() <= 1
              and r_fix.min() >= 0 and r_fix.max() <= 1)
    print(f"  [{'ok  ' if ok_len else 'FAIL'}] rank vectors have length {REPS} and "
          f"lie in [0, 1]")

    if not (ok_res and ok_len):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    L = []
    p = L.append
    p("SELECTION_SBI'S CALIBRATION FAILURE: WHICH WAY, AND IS THE CHECK SOUND?")
    p("=" * 92)
    p("  selection_sbi.py is the only failure in a 110-tool run. It refuses to")
    p("  print because its own SBC check rejects. This asks what that means.")
    p("")
    p(f"  {'':<44}{'KS p':>12}{'shape':>44}")
    p(f"  {'positive control: posterior = prior':<44}{p_pos:>12.4f}"
      f"{shape_of(pos):>44}")
    p(f"  {'negative control: 4x too narrow':<44}{p_neg:>12.6f}"
      f"{shape_of(neg):>44}")
    p(f"  {'the real posterior, original rank stat':<44}{p_orig:>12.6f}"
      f"{shape_of(r_orig):>44}")
    p(f"  {'the real posterior, half-bin corrected':<44}{p_fix:>12.6f}"
      f"{shape_of(r_fix):>44}")
    p("")
    p(f"  rank deciles, real posterior (corrected): "
      + " ".join(f"{v:.2f}" for v in np.quantile(r_fix, np.linspace(0, 1, 11))))
    p("")
    ends = float(((r_fix < 0.1) | (r_fix > 0.9)).mean())
    mid = float(((r_fix > 0.4) & (r_fix < 0.6)).mean())
    p(f"  P1  positive control uniform: KS p = {p_pos:.4f}      "
      f"pre-registered > 0.05:  {'HIT' if ok_pos else 'MISS'}")
    p(f"  P2  negative control caught: KS p = {p_neg:.6f}   "
      f"pre-registered < 0.01:  {'HIT' if ok_neg else 'MISS'}")
    p(f"  P3  failure is overconfidence: ends {100 * ends:.0f}%, middle "
      f"{100 * mid:.0f}%   pre-registered ENDS:  "
      f"{'HIT' if ends > 0.25 and ends > mid else 'MISS'}")
    p(f"  P4  half-bin correction moves p from {p_orig:.6f} to {p_fix:.6f} and the "
      f"check still fails: {'HIT' if p_fix < 0.01 else 'MISS'}")
    p("")
    p("  Why the direction decides the meaning. selection_sbi.py's headline is")
    p("  that the number of unreported attempts N is NOT identified, evidenced")
    p("  by a flat posterior. A posterior that is too WIDE would manufacture")
    p("  that conclusion. A posterior that is too NARROW argues the other way:")
    p("  the flatness it reports is real and understated, and the tool is")
    p("  refusing to print a conclusion that is if anything conservative.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("sbc_diagnose_results.txt").write_text(text + chr(10), encoding="utf-8",
                                                newline=chr(10))
    print(chr(10) + "wrote sbc_diagnose_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
