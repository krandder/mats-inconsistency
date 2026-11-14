"""Numerical checks for T1 lemma 2: exact LMSR settlement = log-score gain = integrated gradients
from the prior; expected value = information gain (mutual information from the base-rate prior).

Run: python3 check_L2.py   (numpy + scipy only). Writes out/L2.json. Seeds 1,2,3.
"""
import itertools, json, math, os
import numpy as np
from scipy.special import expit, log_expit, xlogy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SEEDS = [1, 2, 3]
SIG2MAX = 1 / (6 * math.sqrt(3))          # max |sigma''|


def sp(x):                                  # softplus = binary LMSR cost C(l) = log(1+e^l)
    return np.logaddexp(0.0, x)


def logscore(l, y):                         # log p_l(y)
    return y * log_expit(l) + (1 - y) * log_expit(-l)


def kl_bern(p1l, p0l):                      # KL(Bern(sig p1l) || Bern(sig p0l)), logits in
    p1 = expit(p1l)
    return p1 * (log_expit(p1l) - log_expit(p0l)) + (1 - p1) * (log_expit(-p1l) - log_expit(-p0l))


def pbar_exact(l0, l1):                     # average LMSR price along the path = cost per share
    d = l1 - l0
    small = np.abs(d) < 1e-8
    out = np.where(small, expit(l0), (sp(l1) - sp(l0)) / np.where(small, 1.0, d))
    return out


def pbar_rule(l0, l1, n, rule):
    d = l1 - l0
    if rule == "midpoint":
        a = (np.arange(n) + 0.5) / n
        return expit(l0[..., None] + a * d[..., None]).mean(-1)
    if rule == "left":
        a = np.arange(n) / n
        return expit(l0[..., None] + a * d[..., None]).mean(-1)
    if rule == "trapezoid":
        a = np.arange(n + 1) / n
        v = expit(l0[..., None] + a * d[..., None])
        return (v[..., 1:-1].sum(-1) + 0.5 * (v[..., 0] + v[..., -1])) / n
    raise ValueError(rule)


def main():
    res = {}
    # ---- (a) identities: payoff = log-score gain = tangent + KL(p1||p0)
    rng = np.random.default_rng(0)
    l0 = rng.uniform(-20, 20, 10 ** 6); l1 = rng.uniform(-20, 20, 10 ** 6); y = (rng.random(10 ** 6) < 0.5).astype(float)
    payoff = (l1 - l0) * y - sp(l1) + sp(l0)
    gain = logscore(l1, y) - logscore(l0, y)
    tangent = (l1 - l0) * (y - expit(l1))
    res["a"] = {"max_abs_payoff_minus_logscore_gain": float(np.abs(payoff - gain).max()),
                "max_abs_payoff_minus_tangent_minus_KL": float(np.abs(payoff - tangent - kl_bern(l1, l0)).max()),
                "min_payoff_minus_tangent": float((payoff - tangent).min()), "n": 10 ** 6}
    # ---- (b) path-integral (IG) shares: closed form vs 16-step rules; error bound
    res["b"] = {}
    for dmax in [5, 10, 20]:
        rng = np.random.default_rng(dmax)
        n = 200000
        l0 = rng.uniform(-3, 3, n)                         # base-rate prior logit
        d = rng.uniform(-dmax, dmax, n)
        l1 = l0 + d
        y = (rng.random(n) < expit(l1)).astype(float)
        E = logscore(l1, y) - logscore(l0, y)
        row = {}
        for rule in ["midpoint", "trapezoid", "left"]:
            for steps in [16, 64]:
                En = d * (y - pbar_rule(l0, l1, steps, rule))
                err = np.abs(En - E)
                big = np.abs(E) > 1e-2
                row[f"{rule}{steps}"] = {"max_abs_err_nats": float(err.max()), "p99_abs_err_nats": float(np.quantile(err, 0.99)),
                                         "p99_rel_err": float(np.quantile(err[big] / np.abs(E[big]), 0.99)),
                                         "mean_abs_err_over_mean_abs_E": float(err.mean() / np.abs(E).mean())}
        row["midpoint16_bound_at_dmax_nats"] = float(dmax ** 3 * SIG2MAX / (24 * 16 ** 2))
        row["closed_form_resid_max"] = float(np.abs(d * (y - pbar_exact(l0, l1)) - E).max())
        res["b"][f"|dl|<={dmax}"] = row
    # ---- (c) E[exact settlement from base-rate prior] = I(X;Y) for calibrated posterior; tangent mean 0
    res["c"] = {}
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(200)
    gh_w = gh_w / gh_w.sum()
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        a, b = rng.uniform(1, 4), rng.uniform(-1.5, 1.5)
        post = expit(a * gh_x + b)
        p0 = float((gh_w * post).sum())
        H = lambda p: -(xlogy(p, p) + xlogy(1 - p, 1 - p))
        I = float(H(p0) - (gh_w * H(post)).sum())
        n = 400000
        x = rng.normal(size=n); l1 = a * x + b; y = (rng.random(n) < expit(l1)).astype(float)
        l0 = np.full(n, math.log(p0 / (1 - p0)))
        E = logscore(l1, y) - logscore(l0, y)
        T = (l1 - l0) * (y - expit(l1))
        K = kl_bern(l1, l0)
        res["c"][seed] = {"a": a, "b": b, "I_quadrature_nats": I, "mean_exact": float(E.mean()), "se_exact": float(E.std(ddof=1) / math.sqrt(n)),
                          "z_exact_vs_I": float((E.mean() - I) / (E.std(ddof=1) / math.sqrt(n))),
                          "mean_tangent": float(T.mean()), "se_tangent": float(T.std(ddof=1) / math.sqrt(n)),
                          "mean_KL_p1_p0": float(K.mean())}
    # ---- (d) coalitions: IG share vs exact ablation value, Shapley vs IG, arrival-order spread
    res["d"] = {}
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        K = 8
        gaps, bounds, shap_ig, tan_ig, order_spread = [], [], [], [], []
        for _ in range(300):
            l0 = rng.uniform(-2, 2)
            c = rng.normal(0, rng.uniform(0.3, 3), K)
            l1 = l0 + c.sum()
            y = float(rng.random() < expit(l1))
            pb = pbar_exact(np.array(l0), np.array(l1))
            ig = c * (y - pb)
            v = {}
            for m in range(1 << K):
                S = [i for i in range(K) if m >> i & 1]
                v[m] = float(logscore(l0 + c[S].sum(), y) - logscore(l0, y))
            full = (1 << K) - 1
            # Shapley exact
            sh = np.zeros(K)
            for i in range(K):
                for m in range(1 << K):
                    if m >> i & 1:
                        continue
                    s = bin(m).count("1")
                    wgt = math.factorial(s) * math.factorial(K - s - 1) / math.factorial(K)
                    sh[i] += wgt * (v[m | 1 << i] - v[m])
            shap_ig.append(np.abs(sh - ig).sum() / (np.abs(ig).sum() + 1e-12))
            tan_ig.append(np.abs(c * (y - expit(l1)) - ig).sum() / (np.abs(ig).sum() + 1e-12))
            # a few arrival orders: spread of trader 0's sequential P&L
            pl = []
            for _o in range(20):
                perm = rng.permutation(K); pre = 0
                for j in perm:
                    if j == 0:
                        pl.append(v[pre | 1] - v[pre]); break
                    pre |= 1 << j
            order_spread.append(float(np.std(pl) / (abs(ig[0]) + 1e-3)))
            # coalition gap on random subsets
            for _s in range(10):
                m = int(rng.integers(1, full))
                S = [i for i in range(K) if m >> i & 1]
                cS = c[S].sum()
                abl = v[full] - v[full ^ m]
                share = ig[S].sum()
                gaps.append(abs(abl - share))
                lo = min(l0, l1, l1 - cS); hi = max(l0, l1, l1 - cS)
                bounds.append(abs(cS) * (expit(hi) - expit(lo)))
        gaps, bounds = np.array(gaps), np.array(bounds)
        res["d"][seed] = {"coalition_gap_median_nats": float(np.median(gaps)), "coalition_gap_p90_nats": float(np.quantile(gaps, 0.9)),
                          "frac_within_bound": float((gaps <= bounds + 1e-12).mean()),
                          "shapley_vs_IG_rel_L1_median": float(np.median(shap_ig)),
                          "tangent_vs_IG_rel_L1_median": float(np.median(tan_ig)),
                          "arrival_order_sd_over_IG_median": float(np.median(order_spread))}
    ms = lambda k: [float(np.mean([res["d"][s][k] for s in SEEDS])), float(np.std([res["d"][s][k] for s in SEEDS], ddof=1))]
    res["d_mean_sd"] = {k: ms(k) for k in res["d"][SEEDS[0]]}
    with open(os.path.join(OUT, "L2.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["a"]), "\n", json.dumps(res["b"]["|dl|<=10"]), "\n", json.dumps(res["c"]), "\n", json.dumps(res["d_mean_sd"]))


if __name__ == "__main__":
    main()
