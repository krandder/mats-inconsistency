"""Closed-form aggregators from the literature, implemented from their published equations.

No official code exists for these (they are formulas); each function cites the equation it implements.
All take an aggbench View (reports + keys) and return an aggbench forecast. Keys are read through
`key_blocks`, which knows the aggbench key layouts; nothing here reads `batch.oracle` or `batch.bayes`.

pif_oracle        Partial Information Framework, Gaussian model (Satopaa, Jensen, Pemantle, Ungar,
                  arXiv:1505.06472, Sec. 2.2.1 eq. (1) and Sec. 2.2.2): revealed aggregator
                  X'' = diag(Sigma)' Sigma^{-1} (X - mu0 1) + mu0, with Cov(Y, X_j) = Var(X_j) (Prop. 2.2 iii).
                  Sigma (the information overlap) comes from the keys, or, at visibility "none", is fitted
                  from the train-split reports alone (no outcomes), compound-symmetric because agents are
                  exchangeable. E2/E3 use the real-valued link on log-odds with Sigma_ij = sum over pieces in
                  both agents' coverage of 2 mu_k^2 (the pieces' information): for honest agents and exact
                  keys this is algebraically the reports ceiling `oracle_clearing` (tested). E1 uses the
                  vector form E[theta | X] = Cov(theta, X) Cov(X)^+ X with Cov(theta, X_j) = Var(X_j).
winkler_gls       Winkler (1981, doi:10.1287/mnsc.27.4.479), Clemen & Winkler (1985, doi:10.1287/opre.33.2.427):
                  combined mean 1' S^{-1} m / 1' S^{-1} 1, variance 1 / 1' S^{-1} 1, S = covariance of the
                  experts' errors, estimated on train (E1, per coordinate, pooled). Formula from the literature
                  map; not re-checked against the 1981 paper text.
covariance_intersection  Julier & Uhlmann (1997, doi:10.1109/ACC.1997.609105); two-source form as in
                  Noack et al. (2017) eq. (10a,b): P = omega P_A + (1 - omega) P_B; here the N-source form
                  P = sum_i omega_i P_i, h = sum_i omega_i h_i, omega on the simplex minimizing det(P^{-1}). E1.
inverse_covariance_intersection  Noack, Sijs, Reinhardt, Hanebeck (2017, Automatica 79:35-41,
                  doi:10.1016/j.automatica.2017.01.019), Theorem 7, eqs. (17a,b), (18a,b); omega minimizes
                  det(C) (Remark 8). Two-source rule, applied sequentially over agents in the given order. E1.
extremized_mean_logit  Satopaa et al. (2014, doi:10.1016/j.ijforecast.2013.09.009) logit aggregator with
                  equal weights; this is aggbench.baselines.extremized (alpha fitted on train), re-exported.
surprisingly_popular   Prelec, Seung, McCoy (2017, doi:10.1038/nature21054): for a binary question pick the
                  answer whose actual vote share exceeds its mean predicted share. aggbench has no
                  meta-prediction field yet, so it does not appear in the table; tested on a toy input.
"""
import numpy as np

from ..baselines import extremized, fit_extremize  # noqa: F401  (re-export: Satopaa et al. 2014)

# ----------------------------------------------------------------------------- keys


def key_blocks(view):
    """Parse aggbench keys into named blocks; None for blocks a visibility does not provide.

    E2/E3 keys (noisy/full): [coverage bits (K), coverage x mu (K)]. E1 keys: [A_i (m x d), B_i (m x r), log sigma_i].
    Unaligned keys (agent_inputs(..., alignment="unaligned")) have the same layout with per-agent permuted slots."""
    keys = view.keys
    if keys.shape[-1] == 0:
        return None
    if view.kind == "bernoulli":
        K = view.public["n_pieces"]
        cov = keys[..., :K] > 0.5
        mu = np.where(cov, keys[..., K:2 * K], 0.0)
        return {"cov": cov, "mu": mu}
    d = view.public["d"]
    n, N, kd = keys.shape
    # kd = m*d + m*r + 1 with m = obs_dim; aggbench default m=2, r=2. Solve for m, r from public d and kd.
    m = view.public.get("obs_dim", 2)
    r = (kd - 1 - m * d) // m
    A = keys[..., :m * d].reshape(n, N, m, d)
    B = keys[..., m * d:m * d + m * r].reshape(n, N, m, r)
    sig = np.exp(keys[..., -1])
    return {"A": A, "B": B, "sig": sig}


def with_keys(view, keys):
    """A copy of `view` whose keys are replaced (e.g. by agent_inputs(..., alignment="unaligned") keys)."""
    from ..envs import View
    return View(view.kind, {k: v.copy() for k, v in view.reports.items()}, np.asarray(keys).copy(),
                dict(view.public))


# ----------------------------------------------------------------------------- PIF


def _revealed_scalar(X, S, rcond=1e-10):
    """diag(S)' S^+ X per episode; X (n, N), S (n, N, N)."""
    Sp = np.linalg.pinv(S, rcond=rcond, hermitian=True)
    dg = np.einsum("nii->ni", S)
    return np.einsum("ni,nij,nj->n", dg, Sp, X)


def fit_pif_compound(view):
    """Compound-symmetric information structure from reports alone (no outcomes; PIF Sec. 3 estimates Sigma
    from forecasts only). Bernoulli: (delta, rho) of the centred log-odds. Gaussian: (D, R) d x d blocks."""
    if view.kind == "bernoulli":
        X = view.reports["logit"] - view.public["prior_logit"]
        n, N = X.shape
        delta = float((X ** 2).mean())
        off = (X.sum(1) ** 2 - (X ** 2).sum(1)) / (N * (N - 1))
        return {"delta": delta, "rho": float(off.mean())}
    X = view.reports["mean"]
    n, N, d = X.shape
    D = np.einsum("nai,naj->ij", X, X) / (n * N)
    tot = X.sum(1)
    R = (np.einsum("ni,nj->ij", tot, tot) - np.einsum("nai,naj->ij", X, X)) / (n * N * (N - 1))
    return {"D": D, "R": R}


def pif_oracle(view, fitted=None):
    """PIF revealed aggregator. Sigma from keys if the view has keys, else from `fitted` (fit_pif_compound on train)."""
    kb = key_blocks(view)
    if view.kind == "bernoulli":
        l0 = view.public["prior_logit"]
        X = view.reports["logit"] - l0
        n, N = X.shape
        if kb is None:
            if fitted is None:
                raise ValueError("visibility 'none': pass fitted=fit_pif_compound(train_view)")
            S = np.full((N, N), fitted["rho"]) + np.eye(N) * (fitted["delta"] - fitted["rho"])
            S = np.broadcast_to(S, (n, N, N))
        else:
            info = 2 * kb["mu"] ** 2                       # (n, N, K): information of each covered piece
            cov = kb["cov"].astype(float)
            S = np.einsum("nak,nbk->nab", cov * info, cov)  # overlap information
        return {"logit": l0 + _revealed_scalar(X, S)}
    return _pif_gaussian(view, kb, fitted)


def _pif_gaussian(view, kb, fitted, rcond=1e-9):
    X = view.reports["mean"]                               # (n, N, d); prior mean 0, prior cov I
    n, N, d = X.shape
    Xf = X.reshape(n, N * d)
    if kb is None:
        if fitted is None:
            raise ValueError("visibility 'none': pass fitted=fit_pif_compound(train_view)")
        D, R = fitted["D"], fitted["R"]
        S = np.kron(np.eye(N), D - R) + np.kron(np.ones((N, N)), R)
        S = np.broadcast_to(S, (n, N * d, N * d))
        Cty = np.broadcast_to(np.tile(D, (1, N)), (n, d, N * d))
    else:
        A, B, sig = kb["A"], kb["B"], kb["sig"]
        m = A.shape[2]
        eye_m = np.eye(m)
        R_ii = np.einsum("naxr,nayr->naxy", B, B) + sig[..., None, None] ** 2 * eye_m
        Rinv = np.linalg.inv(R_ii)
        P = np.eye(d) + np.einsum("naxi,naxy,nayj->naij", A, Rinv, A)    # local precision from keys
        G = np.linalg.solve(P, np.einsum("naxi,naxy->naiy", A, Rinv))   # X_i = G_i s_i, (n, N, d, m)
        Css = np.einsum("naxi,nbyi->nabxy", A, A) + np.einsum("naxr,nbyr->nabxy", B, B)
        idx = np.arange(N)
        Css[:, idx, idx] += sig[:, :, None, None] ** 2 * eye_m
        S = np.einsum("naix,nabxy,nbjy->naibj", G, Css, G).reshape(n, N * d, N * d)
        Cty = np.einsum("nbyi,nbjy->nibj", A, G).reshape(n, d, N * d)   # Cov(theta, X_j) = A_j' G_j'
    Sp = np.linalg.pinv(S, rcond=rcond, hermitian=True)
    W = np.einsum("nik,nkl->nil", Cty, Sp)
    mean = np.einsum("nil,nl->ni", W, Xf)
    cov = np.eye(d) - np.einsum("nil,njl->nij", W, Cty)
    cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-6, None)                           # plug-in keys can make the bound indefinite
    prec = np.einsum("nij,nj,nkj->nik", V, 1 / w, V)
    return {"mean": mean, "prec": prec}


# ----------------------------------------------------------------------------- Winkler GLS (E1)


def fit_winkler(view, truth):
    """Error covariance S (N x N) of the experts' point forecasts on train, pooled over coordinates (E1)."""
    if view.kind != "gaussian":
        raise ValueError("winkler_gls: E1 only (E2/E3 agents are shuffled and exchangeable: GLS weights are equal)")
    err = view.reports["mean"] - truth[:, None, :]       # (n, N, d)
    n, N, d = err.shape
    return np.einsum("nai,nbi->ab", err, err) / (n * d)


def winkler_gls(view, S):
    m = view.reports["mean"]
    n, N, d = m.shape
    Sinv1 = np.linalg.solve(S, np.ones(N))
    w = Sinv1 / Sinv1.sum()
    var = 1.0 / Sinv1.sum()
    return {"mean": np.einsum("a,nai->ni", w, m), "prec": np.broadcast_to(np.eye(d) / var, (n, d, d)).copy()}


# ----------------------------------------------------------------------------- CI and ICI (E1)


def _info(view):
    P = view.reports["prec"]
    return P, np.einsum("naij,naj->nai", P, view.reports["mean"])


def covariance_intersection(view, iters=300, lr=0.5):
    """N-source CI: omega on the simplex maximizing log det(sum_i omega_i P_i) (= minimizing det of the fused
    covariance), by exponentiated-gradient ascent (the objective is concave in omega)."""
    if view.kind != "gaussian":
        raise ValueError("covariance_intersection: Gaussian reports (E1) only")
    P, h = _info(view)
    n, N, d, _ = P.shape
    logw = np.zeros((n, N))
    for _ in range(iters):
        w = np.exp(logw - logw.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        F = np.einsum("na,naij->nij", w, P)
        g = np.einsum("nij,naji->na", np.linalg.inv(F), P)   # d/d omega_a log det F = tr(F^{-1} P_a)
        logw = logw + lr * g / np.maximum(np.abs(g).max(1, keepdims=True), 1e-12)
    w = np.exp(logw - logw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    F = np.einsum("na,naij->nij", w, P)
    hf = np.einsum("na,nai->ni", w, h)
    return {"mean": np.linalg.solve(F, hf[..., None])[..., 0], "prec": F, "omega": w}


def _ici_pair(xa, Ca, xb, Cb, grid):
    """Noack et al. 2017, Theorem 7: fused (x, C) for each episode, omega minimizing det(C) over `grid`
    (then refined by golden section)."""
    Cai, Cbi = np.linalg.inv(Ca), np.linalg.inv(Cb)

    def fused_logdet_prec(om):
        G = om[:, None, None] * Ca + (1 - om)[:, None, None] * Cb
        return np.linalg.slogdet(Cai + Cbi - np.linalg.inv(G))[1]   # maximize det(C^{-1}) = minimize det(C)

    n = len(xa)
    vals = np.stack([fused_logdet_prec(np.full(n, g)) for g in grid], 1)
    j = vals.argmax(1)
    lo, hi = grid[np.maximum(j - 1, 0)], grid[np.minimum(j + 1, len(grid) - 1)]
    gr = (np.sqrt(5) - 1) / 2
    for _ in range(30):
        c, e = hi - gr * (hi - lo), lo + gr * (hi - lo)
        better = fused_logdet_prec(c) > fused_logdet_prec(e)
        hi, lo = np.where(better, e, hi), np.where(better, lo, c)
    om = (lo + hi) / 2
    G = om[:, None, None] * Ca + (1 - om)[:, None, None] * Cb
    Gi = np.linalg.inv(G)
    Cinv = Cai + Cbi - Gi                                          # (17b)
    C = np.linalg.inv(Cinv)
    K = C @ (Cai - om[:, None, None] * Gi)                         # (18a)
    L = C @ (Cbi - (1 - om)[:, None, None] * Gi)                   # (18b)
    x = np.einsum("nij,nj->ni", K, xa) + np.einsum("nij,nj->ni", L, xb)   # (17a)
    return x, C


def inverse_covariance_intersection(view):
    if view.kind != "gaussian":
        raise ValueError("inverse_covariance_intersection: Gaussian reports (E1) only")
    m, P = view.reports["mean"], view.reports["prec"]
    C = np.linalg.inv(P)
    grid = np.linspace(0.0, 1.0, 21)
    x, Cf = m[:, 0], C[:, 0]
    for i in range(1, m.shape[1]):
        x, Cf = _ici_pair(x, Cf, m[:, i], C[:, i], grid)
    Cf = 0.5 * (Cf + np.swapaxes(Cf, 1, 2))
    return {"mean": x, "prec": np.linalg.inv(Cf)}


# ----------------------------------------------------------------------------- surprisingly popular


def surprisingly_popular(votes, predicted_share_yes):
    """Prelec, Seung, McCoy (2017). votes (n, N) in {0, 1}; predicted_share_yes (n, N): each respondent's
    prediction of the fraction answering yes. Returns (n,) in {0, 1}: yes iff its actual share exceeds its mean
    predicted share. aggbench reports carry no meta-predictions, so this is not in the default table."""
    votes = np.asarray(votes, float)
    return (votes.mean(1) > np.asarray(predicted_share_yes, float).mean(1)).astype(int)
