"""P18 power calculation (PREREG section 5). Writes results/power.json.

Test (pre-registered): per arm, the seed-level exact accuracy on e59 at T = 12n (one number per seed); A1 vs the best
alpha of DT-Recall progressive (A4*); Welch t, one-sided, alpha 0.05, "beats" = lower bound of the one-sided 95%
interval of mean(A1) - mean(A4*) > 0. The same test on e33 must also pass (intersection-union test, level 0.05).

Scenarios for the seed-level outcomes (accuracies clipped to [0, 1]):
  normal:  A1 ~ N(m1, sd), A4* ~ N(m4, sd), delta = m1 - m4.
  mixture: each seed trains (leaves the ln 2 plateau) with probability p; a trained seed scores N(mu, 0.1), an
           untrained seed 0. P13 DT-Recall (alpha 0.01, same recipe) left the plateau in 2 of 4 seeds by step 14k
           (about 1.4 L4-hours), a 3rd at 26k, the 4th not by 12k: p about 0.5 for DT-Recall at a 1.5-2 h budget.
The intersection with e33 is ignored here (power for e59 alone is an upper bound on power for both).
"""
import json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = np.random.default_rng(18)
NSIM = 20000


def welch_lb_pos(a, b, alpha=0.05):
    """rows of a (n1) and b (n2): True where the one-sided (1-alpha) lower bound of mean(a) - mean(b) is > 0"""
    n1, n2 = a.shape[1], b.shape[1]
    va, vb = a.var(1, ddof=1) / n1, b.var(1, ddof=1) / n2
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (n1 - 1) + vb ** 2 / (n2 - 1) + 1e-300)
    d = a.mean(1) - b.mean(1)
    lb = d - stats.t.ppf(1 - alpha, np.maximum(df, 1)) * se
    ok = np.where(se > 0, lb > 0, d > 0)
    return ok


def sim_normal(m1, m4, sd, n1, n4):
    a = np.clip(R.normal(m1, sd, (NSIM, n1)), 0, 1)
    b = np.clip(R.normal(m4, sd, (NSIM, n4)), 0, 1)
    return float(welch_lb_pos(a, b).mean())


def sim_mix(p1, mu1, p4, mu4, n1, n4, sd=0.1):
    a = np.where(R.random((NSIM, n1)) < p1, np.clip(R.normal(mu1, sd, (NSIM, n1)), 0, 1), 0.0)
    b = np.where(R.random((NSIM, n4)) < p4, np.clip(R.normal(mu4, sd, (NSIM, n4)), 0, 1), 0.0)
    return float(welch_lb_pos(a, b).mean())


def main():
    out = {"nsim": NSIM, "seed": 18, "normal": [], "mixture": [], "false_positive": []}
    designs = [(3, 3), (4, 3), (4, 4), (5, 5), (6, 6)]
    for sd in [0.15, 0.25, 0.35]:
        for delta in [0.3, 0.5, 0.7]:
            m4 = 0.25
            for n1, n4 in designs:
                out["normal"].append({"sd": sd, "delta": delta, "n1": n1, "n4": n4,
                                      "power": sim_normal(m4 + delta, m4, sd, n1, n4)})
    # mixture: A1 always trains; DT-Recall trains with p4; both score mu when trained
    for p4 in [0.25, 0.5, 0.75]:
        for mu1, mu4 in [(0.9, 0.9), (0.9, 0.5), (0.6, 0.9)]:
            for n1, n4 in designs:
                out["mixture"].append({"p1": 1.0, "mu1": mu1, "p4": p4, "mu4": mu4, "n1": n1, "n4": n4,
                                       "delta": round(mu1 - p4 * mu4, 3), "power": sim_mix(1.0, mu1, p4, mu4, n1, n4)})
    # false positives: equal arms (the test's size), normal and mixture
    for n1, n4 in designs:
        out["false_positive"].append({"model": "normal sd 0.25, equal means 0.5", "n1": n1, "n4": n4,
                                      "rate": sim_normal(0.5, 0.5, 0.25, n1, n4)})
        out["false_positive"].append({"model": "mixture p 0.5, mu 0.9 both arms", "n1": n1, "n4": n4,
                                      "rate": sim_mix(0.5, 0.9, 0.5, 0.9, n1, n4)})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "power.json"), "w"), indent=1)
    print("normal (m4 = 0.25): power of Welch one-sided 0.05")
    for sd in [0.15, 0.25, 0.35]:
        for delta in [0.3, 0.5, 0.7]:
            row = [r for r in out["normal"] if r["sd"] == sd and r["delta"] == delta]
            print(f"  sd {sd} delta {delta}: " + "  ".join(f"n={r['n1']}/{r['n4']}:{r['power']:.2f}" for r in row))
    print("mixture (A1 trains always; DT-Recall trains with p4)")
    for p4 in [0.25, 0.5, 0.75]:
        for mu1, mu4 in [(0.9, 0.9), (0.9, 0.5), (0.6, 0.9)]:
            row = [r for r in out["mixture"] if r["p4"] == p4 and r["mu1"] == mu1 and r["mu4"] == mu4]
            print(f"  p4 {p4} mu1 {mu1} mu4 {mu4} (delta {row[0]['delta']}): " +
                  "  ".join(f"n={r['n1']}/{r['n4']}:{r['power']:.2f}" for r in row))
    print("false positive rate (equal arms)")
    for r in out["false_positive"]:
        print(f"  {r['model']} n={r['n1']}/{r['n4']}: {r['rate']:.3f}")


if __name__ == "__main__":
    main()
