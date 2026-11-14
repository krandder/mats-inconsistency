"""E1 floor at provenance `none`: the exact posterior of theta given every agent's s_i and own model (A_i, R_i),
with the shared-noise loadings B_i marginalised.

Own model R_i = B_i B_i^T + sigma_i^2 I fixes B_i only up to B_i -> B_i Q_i (Q_i in O(2)) and, strictly, does not
fix sigma_i either: any sigma_i^2 < lambda_min(R_i) splits R_i. Under the generative prior (B_i entries
N(0, factor_scale^2 / r), sigma_i log-uniform), given R_i:
  sigma_i | R_i  has density  prior(sigma) * Wishart_{r}(R_i - sigma^2 I; scale factor_scale^2/r I)
  B_i | R_i, sigma_i = (R_i - sigma_i^2 I)^{1/2} Q_i,  Q_i Haar on O(m)   (needs r = m; r = 0 means B_i empty).
For every sample of (sigma, Q) the posterior of theta is Gaussian (gaussian_global), so the floor posterior is a
mixture, weighted by the likelihood of all s under that sample (self-normalised importance sampling, proposal =
the conditional prior given the R_i, so the weight is the joint marginal likelihood N(s; 0, AA^T + BB^T + D);
the per-agent factors are constant given R_i and cancel).

Two floors:
  rot   sigma_i known (only the rotations Q_i unknown): the brief's definition;
  none  sigma_i also marginalised: what (A_i, R_i) actually reveals (the own model at commit 20ee6a0).
Regret = -log q(theta) + log N(theta; Bayes), as in metrics.per_episode for Gaussian envs.

    python3 scripts/e1_floor_none.py --n 2000 --K 4096 --out results/explore/e1-floor-20261021
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggbench import make_env  # noqa: E402
from aggbench.envs import GaussianEnv  # noqa: E402
from aggbench.metrics import bootstrap_ci, gaussian_nll, per_episode  # noqa: E402
from aggbench.baselines import sum_logodds  # noqa: E402


def haar_o(rng, K, m):
    q, r = np.linalg.qr(rng.standard_normal((K, m, m)))
    return q * np.sign(np.einsum("kii->ki", r))[:, None, :]


def sample_sigma(rng, R, K, cfg, ngrid=4000):
    """K draws of sigma | R from prior(sigma) * Wishart(R - sigma^2 I) (r = m = 2 here, general m with r = m).
    Substitution u = sqrt(lam_min - sigma^2) removes the (lam_min - sigma^2)^{-1/2}-type edge singularity."""
    m = R.shape[0]
    r = cfg.n_factors
    v = cfg.factor_scale ** 2 / r                  # Wishart scale v I with r dof
    lam = np.linalg.eigvalsh(R)
    lo, hi = cfg.noise_range
    smax = min(hi, np.sqrt(lam[0]))
    smin = lo
    if r != m:
        raise NotImplementedError("B_i = W^{1/2} Q_i needs n_factors == obs_dim")
    if smax <= smin:
        raise ValueError("R inconsistent with the noise prior")
    u_lo, u_hi = np.sqrt(max(lam[0] - smax ** 2, 0.0)), np.sqrt(lam[0] - smin ** 2)
    u = np.linspace(u_lo, u_hi, ngrid)
    s2 = lam[0] - u ** 2
    # density in sigma: sigma^-1 (log-uniform) |W|^a exp(-tr W / 2v), W = R - sigma^2 I, a = (r-m-1)/2;
    # eigenvalues of W are u^2 and lam_j - sigma^2 (j >= 1); |d sigma / d u| = u / sigma, so in u:
    # sigma^-2 u^(2a+1) prod_{j>=1} (lam_j - sigma^2)^a exp(-tr W / 2v)   (u-power is 0 for r = m = 2)
    ew = lam[None, :] - s2[:, None]
    a = (r - m - 1) / 2
    logp = -np.log(s2) + a * np.log(ew[:, 1:]).sum(1) - ew.sum(1) / (2 * v)
    if 2 * a + 1 != 0:
        logp = logp + (2 * a + 1) * np.log(np.maximum(u, 1e-300))
    p = np.exp(logp - logp.max())
    cdf = np.concatenate([[0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(u))])
    cdf /= cdf[-1]
    uu = np.interp(rng.random(K), cdf, u)
    return np.sqrt(lam[0] - uu ** 2)


def sqrtm_psd(W):
    w, V = np.linalg.eigh(W)
    return (V * np.sqrt(np.maximum(w, 0))[..., None, :]) @ np.swapaxes(V, -1, -2)


def floor_episode(A, R, sig_true, s, theta, cfg, K, rng, mode):
    """-log q(theta) under the floor mixture, the ESS, and the mixture's moment-matched mean/cov."""
    N, m, d = A.shape
    r = cfg.n_factors
    sig = np.empty((K, N))
    Bs = np.zeros((K, N, m, max(r, 0)))
    for i in range(N):
        if mode == "rot" or r == 0:
            sig[:, i] = sig_true[i] if r else np.sqrt(R[i, 0, 0])
        else:
            sig[:, i] = sample_sigma(rng, R[i], K, cfg)
        if r:
            W = R[i][None] - sig[:, i, None, None] ** 2 * np.eye(m)
            Bs[:, i] = sqrtm_psd(W) @ haar_o(rng, K, m)
    Ka = A.reshape(N * m, d)
    Bk = Bs.reshape(K, N * m, r)
    Rn = Bk @ np.swapaxes(Bk, 1, 2) + np.eye(N * m) * np.repeat(sig ** 2, m, axis=1)[:, None, :]
    Rinv = np.linalg.inv(Rn)
    KtR = np.einsum("ad,kab->kdb", Ka, Rinv)
    P = np.eye(d) + KtR @ Ka
    sv = s.ravel()
    mu = np.linalg.solve(P, (KtR @ sv)[..., None])[..., 0]
    # marginal log-likelihood of s: N(s; 0, Ka Ka^T + Rn)
    S = Ka @ Ka.T + Rn
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, np.broadcast_to(sv, (K, N * m))[..., None])[..., 0]
    logw = -0.5 * (z ** 2).sum(1) - np.log(np.einsum("kii->ki", L)).sum(1)
    logw -= logsumexp(logw)
    lN = -gaussian_nll(mu, P, np.broadcast_to(theta, (K, d)))
    nll = -logsumexp(logw + lN)
    w = np.exp(logw)
    ess = 1.0 / (w ** 2).sum()
    mmean = w @ mu
    cov = np.einsum("k,kij->ij", w, np.linalg.inv(P)) + np.einsum("k,ki,kj->ij", w, mu - mmean, mu - mmean)
    return nll, ess, mmean, np.linalg.inv(cov)


def run(env, n, K, mode, seed=0, start=0):
    cfg = env.config
    dv = env.sample("dev", n, start=start)
    o = dv.oracle
    R = np.einsum("enar,enbr->enab", o["Bf"], o["Bf"]) + o["sig"][..., None, None] ** 2 * np.eye(dv.obs.shape[-1])
    rng = np.random.default_rng(seed)
    nll, ess, mm, mp = np.empty(n), np.empty(n), [], []
    for e in range(n):
        nll[e], ess[e], a, b = floor_episode(o["A"][e], R[e], o["sig"][e], dv.obs[e], dv.truth[e], cfg, K, rng, mode)
        mm.append(a)
        mp.append(b)
    bnll = gaussian_nll(dv.bayes["mean"], dv.bayes["prec"], dv.truth)
    proj = {"mean": np.array(mm), "prec": np.array(mp)}
    return dv, nll - bnll, ess, per_episode(proj, dv)["regret"]


def summ(x):
    lo, hi = bootstrap_ci(x)
    return {"mean": float(np.mean(x)), "ci_low": lo, "ci_high": hi, "n": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--out", default="results/explore/e1-floor-20261021")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"n_dev": a.n, "K": a.K, "env": "E1 defaults", "regret_definition": "-log q(theta) - gaussian_nll(Bayes)"}
    t0 = time.time()

    # validations: r = 0 (no shared factors) and N = 1 (one agent: R_i is the whole likelihood) must give Bayes
    for tag, ov in (("r0", dict(n_factors=0)), ("N1", dict(n_agents=1))):
        for mode in ("rot", "none"):
            _, reg, ess, _ = run(GaussianEnv(**ov), 200, 512, mode)
            res[f"validate_{tag}_{mode}"] = {"max_abs_regret": float(np.abs(reg).max()), "min_ess": float(ess.min())}
            print(tag, mode, res[f"validate_{tag}_{mode}"], flush=True)

    # simulation-based calibration of sigma | R: rank of the true sigma among 99 draws must be uniform
    rng, ranks = np.random.default_rng(5), []
    cfg = make_env("E1").config
    for _ in range(3000):
        B = rng.standard_normal((2, 2)) / np.sqrt(2)
        st = np.exp(rng.uniform(*np.log(cfg.noise_range)))
        ranks.append(int((sample_sigma(rng, B @ B.T + st ** 2 * np.eye(2), 99, cfg) < st).sum()))
    h = np.histogram(ranks, bins=10, range=(0, 100))[0]
    from scipy.stats import chisquare
    res["validate_sigma_sampler_sbc"] = {"decile_counts": h.tolist(), "chi2_p": float(chisquare(h).pvalue)}
    print("sbc", res["validate_sigma_sampler_sbc"], flush=True)

    env = make_env("E1")
    for mode in ("rot", "none"):
        dv, reg, ess, preg = run(env, a.n, a.K, mode)
        res[f"floor_{mode}"] = summ(reg)
        res[f"floor_{mode}_gaussian_projection"] = summ(preg)
        res[f"floor_{mode}_ess"] = {"median": float(np.median(ess)), "p05": float(np.quantile(ess, 0.05)),
                                    "min": float(ess.min())}
        # importance-sampling error: rerun 500 episodes with 4x K and an independent seed
        _, reg4, _, _ = run(env, 500, 4 * a.K, mode, seed=1)
        res[f"floor_{mode}_IS_check_500ep"] = {"K": float(np.mean(reg[:500])), "4K_seed1": float(np.mean(reg4)),
                                               "paired_diff": summ(reg4 - reg[:500])}
        np.save(out / f"floor_{mode}_regret.npy", reg)
        print(mode, res[f"floor_{mode}"], res[f"floor_{mode}_ess"], res[f"floor_{mode}_IS_check_500ep"],
              f"{time.time() - t0:.0f}s", flush=True)
    pool = per_episode(sum_logodds(dv.view("none")), dv)["regret"]
    np.save(out / "product_of_local_posteriors_regret.npy", pool)
    res["product_of_local_posteriors_none"] = summ(pool)
    res["wall_seconds"] = time.time() - t0
    (out / "floor.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
