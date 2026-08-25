"""Law 1 names a function. Can nine boards tell that function from another?

Law 1 says the share of pairs a benchmark can order is Phibar(1/SNR), with
SNR = tau sqrt(2n) / (c sigma_p). It has no free parameters, which is to its
credit. But the claim has two parts and only one of them has been tested.

  THE COLLAPSE - that the established share is a function of ONE number.
  THE LINK     - that the function is the Gaussian upper tail.

Every test so far has scored the pair together. A reviewer put it that at nine
points, two of them 17 points off, the Gaussian tail cannot be distinguished
from any other monotone squash, and that the functional form should not be
called a finding until the alternatives are fitted and lose.

So they are fitted here. Three competitors, all parameter-free, all mapping
1/SNR in [0, inf) to a share in (0, 0.5] and agreeing with Phibar at both ends:

    Gaussian     Phibar(x)              the law
    logistic     1 / (1 + exp(x))
    Cauchy       0.5 - arctan(x) / pi
    algebraic    0.5 / (1 + x)

None has a parameter to tune, so this is a like-for-like contest, and it is run
twice: on simulated Gaussian boards where Phibar must win because it generated
them, and on the nine real boards where the question is open.

PRE-REGISTERED (2026-08-25, committed before the run)
  P1  on simulated boards Phibar wins and wins clearly: lowest mean absolute
      error of the four, by a factor of at least 2 over the runner-up. This is
      a sanity check on the derivation - a miss here means law 1's algebra is
      wrong, not that the link is undecidable.
  P2  on the nine REAL boards the four links are indistinguishable: the spread
      between the best and worst mean absolute error is smaller than law 1's
      own residual standard deviation across boards.
  P3  Phibar is NOT the best of the four on the real boards. If a link that
      did not generate the data fits it better, the Gaussian form is not what
      is doing the work.
  P4  the ranking is unstable: dropping the two TabArena boards, which law 1
      misses by 17 points, changes which link comes first.

  What P2 MISSING would mean: nine boards DO have the resolution to pick a
  link, the Gaussian tail is an empirical finding rather than a modelling
  convention, and law 1 is stronger than this file assumes.

SELF-CHECKS (no table if any fails)
  * every link must be parameter-free and must satisfy L(0) = 0.5 and
    L(x) -> 0, asserted numerically, or it is not in the same contest;
  * every link must be strictly decreasing on the swept range;
  * the real-board SNR and observed shares must be read from
    resolution_law_test_results.txt, not recomputed here, so this file cannot
    quietly use a different SNR than the law is stated with;
  * at least 9 real boards and at least 200 simulated ones.

    python link_competition.py
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

import rank_sets as rs

SEED = 20260825
J_SIM = 40
NS = (20, 40, 80, 160, 320, 640)
SPREADS = (0.004, 0.008, 0.014, 0.022, 0.032, 0.045, 0.065, 0.09, 0.13, 0.2)
REPS = 4

LINKS = {
    "Gaussian  Phibar(x)": lambda x: norm.sf(x),
    "logistic  1/(1+e^x)": lambda x: 1.0 / (1.0 + np.exp(np.clip(x, -50, 50))),
    "Cauchy    .5-atan/pi": lambda x: 0.5 - np.arctan(x) / math.pi,
    "algebraic .5/(1+x)": lambda x: 0.5 / (1.0 + x),
}


def read_real():
    """(name, SNR, observed share) from law 1's own results file."""
    out = []
    for line in Path("resolution_law_test_results.txt").read_text(
            encoding="utf-8").splitlines():
        m = re.match(r"\s{2}(\S.*?)\s{2,}(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%", line)
        if m:
            out.append((m.group(1).strip(), float(m.group(4)),
                        float(m.group(5)) / 100.0))
    return out


def sim_points(rng):
    """Simulated boards: (1/SNR, observed established share)."""
    pts = []
    for n in NS:
        for tau in SPREADS:
            for _ in range(REPS):
                p = np.clip(0.5 + rng.normal(0, tau, J_SIM), 0.02, 0.98)
                x = (rng.random((J_SIM, n)) < p[:, None]).astype(float)
                r = rs.rank_sets(x)
                J = J_SIM
                iu = np.triu_indices(J, k=1)
                sig = r["sigma"][iu]
                sig = sig[np.isfinite(sig) & (sig > 0)]
                if not len(sig):
                    continue
                sc = x.mean(axis=1)
                tau_o = float(sc.std(ddof=1))
                sp = float(np.median(sig))
                c = float(r["crit"])
                if tau_o <= 0 or sp <= 0:
                    continue
                snr = tau_o * math.sqrt(2 * n) / (c * sp)
                obs = float(r["beats"].sum() / (J * (J - 1)))
                pts.append((1.0 / snr, obs))
    return pts


def mae(link, pts):
    return float(np.mean([abs(link(x) - y) for x, y in pts]))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print("self-checks ...")
    xs = np.linspace(0, 12, 400)
    ok_ends = all(abs(f(0.0) - 0.5) < 1e-9 and f(50.0) < 1e-3
                  for f in LINKS.values())
    print(f"  [{'ok  ' if ok_ends else 'FAIL'}] every link is 0.5 at 0 and tends to 0")
    ok_mono = all(np.all(np.diff(f(xs)) <= 1e-12) for f in LINKS.values())
    print(f"  [{'ok  ' if ok_mono else 'FAIL'}] every link is decreasing on the swept range")

    real = read_real()
    ok_real = len(real) >= 9
    print(f"  [{'ok  ' if ok_real else 'FAIL'}] {len(real)} real boards read from "
          f"resolution_law_test_results.txt")

    sim = sim_points(np.random.default_rng(SEED))
    ok_sim = len(sim) >= 200
    print(f"  [{'ok  ' if ok_sim else 'FAIL'}] {len(sim)} simulated boards")

    if not (ok_ends and ok_mono and ok_real and ok_sim):
        print(chr(10) + "A CHECK FAILED - no table is printed.")
        return 1

    rpts = [(1.0 / s, o) for _, s, o in real]
    rpts_x = [(1.0 / s, o) for nm, s, o in real if "TabArena" not in nm]

    sm = {k: mae(f, sim) for k, f in LINKS.items()}
    rm = {k: mae(f, rpts) for k, f in LINKS.items()}
    xm = {k: mae(f, rpts_x) for k, f in LINKS.items()}

    L = []
    p = L.append
    p("CAN NINE BOARDS TELL THE GAUSSIAN TAIL FROM ANY OTHER SQUASH?")
    p("=" * 96)
    p("  Four parameter-free links, all 0.5 at 0 and decreasing to 0. Mean")
    p("  absolute error in PERCENTAGE POINTS of established share.")
    p("")
    p(f"  {'link':<24}{'simulated (' + str(len(sim)) + ')':>18}"
      f"{'real (9)':>12}{'real minus TabArena (7)':>26}")
    for k in LINKS:
        p(f"  {k:<24}{100 * sm[k]:>17.2f}{100 * rm[k]:>12.2f}"
          f"{100 * xm[k]:>25.2f}")
    p("")
    best_s = min(sm, key=sm.get)
    runner = sorted(sm.values())[1]
    best_r = min(rm, key=rm.get)
    best_x = min(xm, key=xm.get)
    spread = 100 * (max(rm.values()) - min(rm.values()))
    resid_sd = 100 * float(np.std([LINKS["Gaussian  Phibar(x)"](x) - y
                                   for x, y in rpts], ddof=1))
    p(f"  P1  simulated winner: {best_s}, by a factor of "
      f"{runner / sm[best_s]:.1f} over the runner-up")
    p(f"      pre-registered Gaussian by >= 2x:  "
      f"{'HIT' if best_s.startswith('Gaussian') and runner / sm[best_s] >= 2 else 'MISS'}")
    p(f"  P2  real-board spread between best and worst link: {spread:.2f} points,")
    p(f"      against law 1's own residual sd of {resid_sd:.2f} points")
    p(f"      pre-registered spread < residual sd:  "
      f"{'HIT' if spread < resid_sd else 'MISS'}")
    p(f"  P3  best link on the real boards: {best_r}")
    p(f"      pre-registered NOT Gaussian:  "
      f"{'HIT' if not best_r.startswith('Gaussian') else 'MISS'}")
    p(f"  P4  best without TabArena: {best_x}")
    p(f"      pre-registered ranking changes:  "
      f"{'HIT' if best_x != best_r else 'MISS'}")
    p("")
    p("  Law 1 makes two claims and only one of them has ever been tested. That")
    p("  the established share is a function of ONE number - the collapse - is")
    p("  the substantive claim, and it survives everything in this repository.")
    p("  That the function is the Gaussian upper tail is a modelling")
    p("  convention inherited from the derivation, and the table above is what")
    p("  nine boards can say about it.")
    text = chr(10).join(L)
    print(chr(10) + text)
    Path("link_competition_results.txt").write_text(text + chr(10),
                                                    encoding="utf-8", newline=chr(10))
    print(chr(10) + "wrote link_competition_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
