"""Numerical sanity checks of the exact lemmas in the hardness note (He-ReLU, bias-free, W iid N(0, 2/n)).

usage: lemma_checks.py CHECK [args]
  J        : E[relu(g)^2 relu(g')^2] for corr c against the closed form J(c); Hermite tail fractions r_D of relu(g)^2
  norm     : single-input norm law  E G = 1, E G^2 = (1 + 5/n)^L
  backward : Var_W of E_x||h_m||^2/n (what a backward-only estimator cannot know) vs the proven bound, and
             E_W Var_x(||h_L||^2/n); prints N_* = E_W Var_x / Var_W(E[S|A]) (MC samples a backward-only estimator is worth)
  survival : one-step survival factor of a mean perturbation orthogonal to the state span vs 1 - E[theta]/pi
  hermite  : multiplicative accumulation lemma, per-layer Hermite truncation of the norm ratio, (a+b)^L - a^L
"""
import math, sys
import numpy as np

rng = np.random.default_rng(20261029)


def J_closed(c):
    th = np.arccos(np.clip(c, -1, 1))
    return (3 * np.sin(th) * np.cos(th) + (np.pi - th) * (1 + 2 * np.cos(th) ** 2)) / (2 * np.pi)


def check_J():
    for c in (-1.0, -0.5, 0.0, 1 / (4 * np.pi), 0.3, 0.7, 0.95, 1.0):
        M = 4_000_000
        g = rng.standard_normal(M)
        e = rng.standard_normal(M)
        gp = c * g + math.sqrt(max(0.0, 1 - c * c)) * e
        mc = np.mean(np.maximum(g, 0) ** 2 * np.maximum(gp, 0) ** 2)
        print(f"c={c:+.4f}  MC {mc:.4f}  closed {J_closed(c):.4f}")
    # Hermite coefficients of relu(g)^2 (probabilists'), k! c_k^2 = variance in degree k
    from numpy.polynomial.hermite_e import hermegauss, hermeval
    x, w = hermegauss(200)
    w = w / w.sum()
    f = np.maximum(x, 0) ** 2
    tot = np.sum(w * f ** 2) - np.sum(w * f) ** 2
    cum = 0.0
    for k in range(1, 12):
        coef = np.zeros(k + 1); coef[k] = 1
        ck = np.sum(w * f * hermeval(x, coef)) / math.factorial(k)
        cum += math.factorial(k) * ck ** 2
        print(f"degree {k}: k! c_k^2 = {math.factorial(k) * ck ** 2:.5f}   r_{k} = {1 - cum / tot:.5f}  (Var = {tot:.4f})")
    cs = 1 / (4 * np.pi)
    print("kappa_* = 4 (J(c_*) - 1/4) =", 4 * (J_closed(cs) - 0.25))


def relu_net(n, L, rs):
    return [rs.standard_normal((n, n)) * math.sqrt(2.0 / n) for _ in range(L)]


def check_norm(n=64, L=32, R=20000):
    G = np.ones(R)
    G2acc = []
    for r in range(R):
        x = rng.standard_normal(n)
        h = x.copy()
        g = 1.0
        for l in range(L):
            W = rng.standard_normal((n, n)) * math.sqrt(2.0 / n)
            hn = np.maximum(W @ h, 0)
            g *= (hn @ hn) / (h @ h) if h @ h > 0 else 0.0
            h = hn
        G[r] = g
    print(f"n={n} L={L}: E G = {G.mean():.4f} (+- {G.std() / math.sqrt(R):.4f}), E G^2 = {np.mean(G ** 2):.4f} "
          f"vs (1+5/n)^L = {(1 + 5 / n) ** L:.4f}")


def check_backward(n, m, L, R=300, M=1024):
    """m = depth of the first part A (L - k); estimate Var_W(E_x ||h_m||^2/n) and E_W Var_x(||h_L||^2/n)."""
    Sm = np.empty(R); Sm_mcvar = np.empty(R); VarL = np.empty(R)
    for r in range(R):
        Ws = relu_net(n, L, rng)
        X = rng.standard_normal((M, n))
        H = X
        for l in range(L):
            H = np.maximum(H @ Ws[l].T, 0)
            if l + 1 == m:
                s = (H * H).sum(1) / n
                Sm[r] = s.mean(); Sm_mcvar[r] = s.var(ddof=1) / M
        s = (H * H).sum(1) / n
        VarL[r] = s.var(ddof=1)
    var_A = Sm.var(ddof=1) - Sm_mcvar.mean()           # remove the inner MC noise
    cs = 1 / (4 * np.pi)
    dstar = 4 / n * (J_closed(cs) - 0.25)
    bound = (1 - 1 / n) * (1 + dstar) ** (m - 2) - 1
    EV = VarL.mean()
    print(f"n={n} L={L} m={m}: Var_W(E[S|A]) = {var_A:.4f} (proven asymptotic bound {bound:.4f});"
          f" E_W Var_x(f_L) = {EV:.4f};  N_* = {EV / var_A:.2f};  crude MC bound e^(5L/n)-1 = {math.exp(5 * L / n) - 1:.3f}")


def check_survival(n=256, L0=16, M=2048, R=50):
    """State law Q = law of h_{L0}(x), x ~ N(0, I); delta orthogonal to span of the M sampled states (M < n)."""
    rs = np.random.default_rng(7)
    Ws = relu_net(n, L0, rs)
    X = rs.standard_normal((min(M, n // 2), n))
    H = X
    for W in Ws:
        H = np.maximum(H @ W.T, 0)
    # delta orthogonal to the rows of H
    Q, _ = np.linalg.qr(H.T)
    d = rs.standard_normal(n); d -= Q @ (Q.T @ d); d /= np.linalg.norm(d)
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    C = np.clip(Hn @ Hn.T, -1, 1)
    iu = np.triu_indices(len(H), 1)
    th = np.arccos(C)
    # include h = h' pairs (theta = 0) with weight 1/M as in E over independent h, h'
    Mh = len(H)
    Eth = (th.sum()) / (Mh * Mh)
    pred = 1 - Eth / np.pi
    vals = []
    for r in range(R):
        W = rs.standard_normal((n, n)) * math.sqrt(2.0 / n)
        p = (H @ W.T > 0).mean(0)                     # fraction of the state law with neuron j on
        vals.append(np.sum((p * (W @ d)) ** 2))
    print(f"n={n} depth {L0}: E_W ||Delta||^2/||delta||^2 = {np.mean(vals):.4f} +- {np.std(vals) / math.sqrt(R):.4f}"
          f";  1 - E[theta]/pi = {pred:.4f}  (mean pairwise angle {th[iu].mean():.4f} rad)")


def check_hermite(n=32, L=64, D=2, R=40000):
    """Gain of one input with each layer's ratio replaced by its degree-<=D Hermite truncation (along the true path)."""
    from numpy.polynomial.hermite_e import hermegauss, hermeval
    x, w = hermegauss(200); w = w / w.sum()
    f = np.maximum(x, 0) ** 2
    cks = [np.sum(w * f * hermeval(x, np.eye(k + 1)[k])) / math.factorial(k) for k in range(D + 1)]
    tot = np.sum(w * f ** 2) - np.sum(w * f) ** 2
    lowvar = sum(math.factorial(k) * cks[k] ** 2 for k in range(1, D + 1))
    a = 1 + (4 / n) * lowvar      # E[rhohat^2]: (2/n)^2 [n^2/4 + n * lowvar]
    ab = 1 + (4 / n) * tot        # E[rho^2] = 1 + 5/n
    errs = np.empty(R); G2 = np.empty(R)
    for r in range(R):
        g = rng.standard_normal((L, n))           # normalized pre-activations along the true path (exact law)
        rho = (2 / n) * (np.maximum(g, 0) ** 2).sum(1)
        rhohat = (2 / n) * hermeval(g, cks).sum(1)
        errs[r] = (np.prod(rho) - np.prod(rhohat)) ** 2
        G2[r] = np.prod(rho) ** 2
    print(f"n={n} L={L} D={D}: E(G - Ghat)^2 = {errs.mean():.4f} +- {errs.std() / math.sqrt(R):.4f}; "
          f"lemma (a+b)^L - a^L = {ab ** L - a ** L:.4f};  E G^2 = {G2.mean():.3f} vs {ab ** L:.3f}")


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'J':
        check_J()
    elif what == 'norm':
        check_norm()
    elif what == 'backward':
        n = int(sys.argv[2]); L = int(sys.argv[3]); m = int(sys.argv[4]); R = int(sys.argv[5]) if len(sys.argv) > 5 else 300
        check_backward(n, m, L, R=R)
    elif what == 'survival':
        check_survival(n=int(sys.argv[2]), L0=int(sys.argv[3]))
    elif what == 'hermite':
        check_hermite(n=int(sys.argv[2]), L=int(sys.argv[3]), D=int(sys.argv[4]))


def check_forward_trunc(n, L, k, R=80, Ain=16, M=512):
    """Mirror of Theorem 6.5: estimators that ignore the LAST k layers. Estimate Var_B(E[S | B]) by nested sampling
    (outer: last-k layers B; inner: first L-k layers A), S = E_x ||h_L||^2 / n."""
    outer = np.empty(R); inner_var = np.empty(R)
    for r in range(R):
        B = relu_net(n, k, rng)
        vals = np.empty(Ain)
        for a in range(Ain):
            A = relu_net(n, L - k, rng)
            H = rng.standard_normal((M, n))
            for W in A + B:
                H = np.maximum(H @ W.T, 0)
            vals[a] = ((H * H).sum(1) / n).mean()
        outer[r] = vals.mean(); inner_var[r] = vals.var(ddof=1) / Ain
    var_B = outer.var(ddof=1) - inner_var.mean()
    print(f"n={n} L={L} last k={k}: Var_B(E[S|B]) = {var_B:.4f} (single-input log-variance scale 5k/n = {5 * k / n:.3f})")


if __name__ == '__main__' and sys.argv[1] == 'fwdtrunc':
    check_forward_trunc(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
