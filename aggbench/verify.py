"""Independent re-computation of the Bayes ceilings (PROTOCOL 9.6: check exact oracles by brute force
on >= 20 episodes and record the check in the output JSON).

E2/E3: enumerate Y in {0, 1}; enumerate the distinct pieces seen by any agent (from the agents'
       raw observations, each piece once); multiply their densities directly. No LLR shortcut.
E4:    Bayes: enumerate Y and multiply the densities of all K context pieces (oracle x). Reports ceiling:
       write every latent (lam_k given Y, e_k, b, each nu_it) as its own coordinate, map it to the reports
       step by step along each rollout's order, evaluate both class-conditional Gaussian densities in full
       (slogdet, quadratic form) and apply Bayes' rule; compare with baselines.oracle_clearing.
Zoo:   loop over the finite world states and symbols (brute_force_tabular).
E1:    condition the joint Gaussian of (theta, s) with covariance form (Sigma_ts Sigma_ss^-1),
       not the information-form solve used by the environment.
"""
import numpy as np

from .metrics import sigmoid


def brute_force_bernoulli(batch):
    """Max |P_env(Y=1) - P_brute(Y=1)| over episodes."""
    prior = batch.config["prior"]
    worst = 0.0
    for e in range(len(batch)):
        obs, mu = batch.obs[e], batch.oracle["mu"][e]
        seen = {}  # piece id -> value; the same piece seen by several agents is counted once
        for i in range(obs.shape[0]):
            for k in np.flatnonzero(~np.isnan(obs[i])):
                seen[int(k)] = obs[i, k]
        joint = []
        for y in (0, 1):
            p = prior if y else 1 - prior
            for k, xk in seen.items():
                mean = mu[k] if y else -mu[k]
                p *= np.exp(-0.5 * (xk - mean) ** 2) / np.sqrt(2 * np.pi)
            joint.append(p)
        post = joint[1] / (joint[0] + joint[1])
        worst = max(worst, abs(post - sigmoid(batch.bayes["logit"][e])))
    return worst


def brute_force_gaussian(batch):
    """Max abs difference of posterior mean and covariance over episodes."""
    worst = 0.0
    for e in range(len(batch)):
        A, Bf, sig = batch.oracle["A"][e], batch.oracle["Bf"][e], batch.oracle["sig"][e]
        N, m, d = A.shape
        Ak = A.reshape(N * m, d)
        Bk = Bf.reshape(N * m, -1)
        S_ss = Ak @ Ak.T + Bk @ Bk.T + np.diag(np.repeat(sig ** 2, m))
        gain = Ak.T @ np.linalg.inv(S_ss)
        mean = gain @ batch.obs[e].ravel()
        cov = np.eye(d) - gain @ Ak
        env_cov = np.linalg.inv(batch.bayes["prec"][e])
        worst = max(worst, np.abs(mean - batch.bayes["mean"][e]).max(), np.abs(cov - env_cov).max())
    return worst


def _log_gauss(v, mean, cov):
    """Log density of N(mean, cov) at v, on the support of cov when cov is singular (eigendecomposition and
    pseudo-determinant; v - mean must lie in the support, which is checked)."""
    vals, vecs = np.linalg.eigh(cov)
    keep = vals > 1e-10 * max(vals.max(), 1e-300)
    d = v - mean
    proj = vecs.T @ d
    assert np.abs(proj[~keep]).max(initial=0.0) < 1e-8 * (1 + np.abs(d).max()), "point outside the support"
    return -0.5 * ((proj[keep] ** 2 / vals[keep]).sum() + np.log(vals[keep]).sum() + keep.sum() * np.log(2 * np.pi))


def brute_force_rollouts(batch):
    """E4. Max |P_env(Y=1) - P_brute(Y=1)| over episodes, for the Bayes ceiling and the reports ceiling."""
    from .baselines import oracle_clearing
    c = batch.config
    prior, a, g = c["prior"], c["calibration"], c["recency"]
    l0 = np.log(prior / (1 - prior))
    ceiling = oracle_clearing(batch)["logit"]
    worst = 0.0
    for e in range(len(batch)):
        x, mu, order = batch.oracle["x"][e], batch.oracle["mu"][e], batch.oracle["order"][e]
        # Bayes ceiling from the raw pieces of the whole context
        joint = []
        for y in (0, 1):
            p = prior if y else 1 - prior
            for k in range(len(x)):
                p *= np.exp(-0.5 * (x[k] - (mu[k] if y else -mu[k])) ** 2) / np.sqrt(2 * np.pi)
            joint.append(p)
        worst = max(worst, abs(joint[1] / (joint[0] + joint[1]) - sigmoid(batch.bayes["logit"][e])))
        # reports ceiling: latent z = (lam_1..K, e_1..K, b, nu_11, nu_12, ...), report = l0 + T z
        K, (N, Lmax) = len(mu), order.shape
        sd = np.concatenate([2 * mu, np.full(K, c["read_noise"]), [c["shared_bias"]],
                             np.full(N * Lmax, c["step_noise"])])
        T = np.zeros((N, len(sd)))
        for i in range(N):
            T[i, 2 * K] = 1.0
            L = int((order[i] >= 0).sum())
            for t in range(L):
                k = order[i, t]
                weight = a * g ** (L - 1 - t)
                T[i, k] += weight
                T[i, K + k] += weight
                T[i, 2 * K + 1 + i * Lmax + t] = 1.0
        cov = (T * sd ** 2) @ T.T
        logp = []
        for y, py in ((0, 1 - prior), (1, prior)):
            zmean = np.concatenate([(1 if y else -1) * 2 * mu ** 2, np.zeros(len(sd) - K)])
            logp.append(np.log(py) + _log_gauss(batch.reports["logit"][e], l0 + T @ zmean, cov))
        post = np.exp(logp[1] - np.logaddexp(logp[0], logp[1]))
        worst = max(worst, abs(post - sigmoid(ceiling[e])))
    return worst


def brute_force_tabular(env, batch):
    """Structure zoo. Max |P_env(Y=1) - P_brute(Y=1)| over episodes for the local reports, the Bayes ceiling
    and the reports ceiling. Plain Python loops over world states w: P(w, o) = pi(w) prod_i lik[i, w, o_i];
    reports ceiling: the symbols an agent's report cannot tell apart are collected one by one (by report value)
    and their probabilities summed. Uses the tables (checked against the simulator in tests/aggbench/test_zoo.py)
    but none of the vectorized inference code."""
    worst = 0.0
    for e in range(len(batch)):
        prior_w, target, lik, _ = env.tables_for(batch, e)
        sym = batch.oracle["sym"][e]
        N, W, O = lik.shape

        def post(cols):
            joint = [0.0, 0.0]
            for w in range(W):
                p = float(prior_w[w])
                for i in range(N):
                    p *= float(cols[i][w])
                joint[int(target[w])] += p
            return joint[1] / (joint[0] + joint[1])

        reports = []
        for i in range(N):  # agent i alone, every symbol
            rep = []
            for o in range(O):
                j = [0.0, 0.0]
                for w in range(W):
                    j[int(target[w])] += float(prior_w[w]) * float(lik[i, w, o])
                rep.append(np.log(j[1]) - np.log(j[0]) if j[0] > 0 and j[1] > 0 else np.nan)
            reports.append(rep)
            worst = max(worst, abs(sigmoid(rep[sym[i]]) - sigmoid(batch.reports["logit"][e, i])))
        worst = max(worst, abs(post([lik[i, :, sym[i]] for i in range(N)]) - sigmoid(batch.bayes["logit"][e])))
        merged = []
        for i in range(N):
            r0 = reports[i][sym[i]]
            col = np.zeros(W)
            for o in range(O):
                if np.isfinite(reports[i][o]) and abs(reports[i][o] - r0) <= 1e-9 * (1 + abs(r0)):
                    col += lik[i, :, o]
            merged.append(col)
        worst = max(worst, abs(post(merged) - sigmoid(batch.oracle["report_logit"][e])))
    return worst


def brute_force_private_read(batch):
    """Z8. Max |P_env(Y=1) - P_brute(Y=1)| for Bayes (the joint Gaussian density of ALL readings given Y, full
    covariance: readings of one piece share x_k) and for the reports ceiling (the joint Gaussian density of the
    reports given Y, built latent by latent: x_k and every eps_ik as its own coordinate)."""
    c = batch.config
    prior, tau = c["prior"], c["read_noise"]
    l0 = np.log(prior / (1 - prior))
    worst = 0.0
    for e in range(len(batch)):
        cov, mu, obs = batch.oracle["cov"][e], batch.oracle["mu"][e], batch.obs[e]
        N, K = cov.shape
        ii, kk = np.nonzero(cov)
        # latent z = (x_1..x_K, eps_ik for each read): readings = M z, reports = l0 + Wr z
        M = np.zeros((len(ii), K + len(ii)))
        M[np.arange(len(ii)), kk] = 1.0
        M[np.arange(len(ii)), K + np.arange(len(ii))] = tau
        w = 2 * mu / (1 + tau ** 2)
        Wr = np.zeros((N, len(ii)))
        Wr[ii, np.arange(len(ii))] = w[kk]
        Wr = Wr @ M
        r = obs[ii, kk]
        cz = np.eye(K + len(ii))
        lp, lq = [], []
        for y, py in ((0, 1 - prior), (1, prior)):
            zmean = np.concatenate([(1 if y else -1) * mu, np.zeros(len(ii))])
            lp.append(np.log(py) + _log_gauss(r, M @ zmean, M @ cz @ M.T))
            lq.append(np.log(py) + _log_gauss(batch.reports["logit"][e], l0 + Wr @ zmean, Wr @ cz @ Wr.T))
        worst = max(worst, abs(np.exp(lp[1] - np.logaddexp(*lp)) - sigmoid(batch.bayes["logit"][e])),
                    abs(np.exp(lq[1] - np.logaddexp(*lq)) - sigmoid(batch.oracle["report_logit"][e])))
    return worst


def oracle_check(env, n=24, tol=1e-8):
    """Record for the results JSON: brute-force check of the Bayes ceiling on n smoke-split episodes
    (E4 and the structure zoo: also of the reports ceiling)."""
    batch = env.sample("smoke", n)
    if hasattr(env, "tables"):
        diff = brute_force_tabular(env, batch)
    elif hasattr(env, "report_ceiling"):
        diff = brute_force_private_read(batch)
    elif "W" in batch.oracle:
        diff = brute_force_rollouts(batch)
    else:
        diff = brute_force_bernoulli(batch) if batch.kind == "bernoulli" else brute_force_gaussian(batch)
    return {"env": env.name, "split": "smoke", "episodes_checked": n,
            "max_abs_diff": float(diff), "tolerance": tol, "verified": bool(diff < tol)}
