# Draft structure prior + three mechanisms; checks closed forms and measures reference gaps.
import numpy as np, time, sys
sys.path.insert(0, '/tmp/mslit/scripts')
from lop_check import lop, verify
rng = np.random.default_rng(1)


def draw_structure(rng, N=None):
    S = int(rng.integers(2, 9)); P = int(rng.integers(0, 3)); N = N or int(rng.integers(4, 33))
    sig_s = rng.choice([0.1, 0.5, 2.0], size=S)          # sender-noise variances (shared sources)
    sig_p = rng.choice([0.25, 1.0, 4.0], size=P)          # public-signal noise variances
    pop = rng.dirichlet(0.5 * np.ones(S))
    listen = []
    for i in range(N):
        m = min(int(rng.integers(1, 4)), S)
        L = list(rng.choice(S, size=m, replace=False, p=pop))
        priv = rng.random() < 0.5                         # trader-specific (truly private) source
        listen.append((L, priv, rng.choice([0.1, 1.0]), rng.choice([0.25, 1.0, 4.0])))
    return dict(S=S, P=P, N=N, sig_s=sig_s, sig_p=sig_p, listen=listen)


def build(st):
    """Latent z = (v, eta_1..S, zeta_1..P, receiver/private noises). Trader i sees x_i = A_i z (own sources + public)."""
    S, P = st['S'], st['P']
    var = [1.0] + list(st['sig_s']) + list(st['sig_p'])

    def newlat(v):
        var.append(v); return len(var) - 1
    pub = [{0: 1.0, 1 + S + k: 1.0} for k in range(P)]
    A = []
    for (L, priv, rv, pv) in st['listen']:
        ri = [{0: 1.0, 1 + s: 1.0, newlat(rv): 1.0} for s in L]
        if priv:
            ri.append({0: 1.0, newlat(pv): 1.0})
        A.append(ri)
    nz = len(var)

    def mat(rs):
        M = np.zeros((len(rs), nz))
        for a, r in enumerate(rs):
            for k, c in r.items():
                M[a, k] = c
        return M
    D = np.diag(var)
    Ai = [np.vstack([mat(ri), mat(pub)]) if P else mat(ri) for ri in A]
    Pm = mat(pub) if P else np.zeros((0, nz))
    return Ai, Pm, D, nz


def bc_equilibrium(Ai, D, r):
    """Finite-N beauty contest a_i = (1-r) E_i v + r E_i[mean_{j!=i} a_j]; linear strategies a_i = w_i' x_i.
    w_i = (1-r) c_i + r/(N-1) sum_j S_ii^-1 Cov(x_i, x_j) w_j  (one linear solve)."""
    N = len(Ai); e0 = np.zeros(D.shape[0]); e0[0] = 1
    iS = [np.linalg.inv(A @ D @ A.T) for A in Ai]
    c = [iS[i] @ Ai[i] @ D @ e0 for i in range(N)]
    sizes = [A.shape[0] for A in Ai]; off = np.cumsum([0] + sizes); K = off[-1]
    M = np.zeros((K, K)); b = np.zeros(K)
    for i in range(N):
        b[off[i]:off[i + 1]] = (1 - r) * c[i]
        for j in range(N):
            if j != i:
                M[off[i]:off[i + 1], off[j]:off[j + 1]] = r / (N - 1) * iS[i] @ Ai[i] @ D @ Ai[j].T
    w = np.linalg.solve(np.eye(K) - M, b)
    return [w[off[i]:off[i + 1]] for i in range(N)], c, (M, b, off)


def mse_lin(f, D):  # E[(v - f.z)^2]
    e = -f.copy(); e[0] += 1; return e @ D @ e


if __name__ == "__main__":
    # Morris-Shin special case: public y (precision alpha), private x_i (precision beta), diffuse prior, large N
    alpha, beta, r = 1.0, 3.0, 0.6
    N = 400; nz = 2 + N
    D = np.diag([1e6, 1 / alpha] + [1 / beta] * N)
    Ai = []
    for i in range(N):
        A = np.zeros((2, nz)); A[0, 0] = 1; A[0, 1] = 1; A[1, 0] = 1; A[1, 2 + i] = 1; Ai.append(A)
    w, c, _ = bc_equilibrium(Ai, D, r)
    print("Morris-Shin public weight: solver %.4f  formula %.4f  (Bayes %.4f)" % (
        w[0][0], alpha / (alpha + beta * (1 - r)), alpha / (alpha + beta)))
    # level-k (best-response iteration from the Bayes action) on the same case
    _, _, (M, b, off) = bc_equilibrium(Ai, D, r)
    wk = np.concatenate(c)
    out = []
    for k in range(6):
        out.append(wk[0]); wk = b + M @ wk
    print("level-k public weight k=0..5:", np.round(out, 4))

    res = dict(bc=[], lop_flip=[], lop_corr=[], seq=[], t_bc=[], t_lop=[], bc_pub=[])
    for ep in range(300):
        st = draw_structure(rng); Ai, Pm, D, nz = build(st); N = st['N']
        r = float(rng.choice([0.0, 0.3, 0.6, 0.85]))
        t0 = time.time(); w, c, _ = bc_equilibrium(Ai, D, r); res['t_bc'].append(time.time() - t0)
        Q = np.vstack([w[i] @ Ai[i] for i in range(N)])            # quotes as linear forms in z
        Sa = Q @ D @ Q.T; ca = Q @ D[:, 0]
        k = np.linalg.lstsq(Sa, ca, rcond=None)[0]
        m = Q.mean(0); s = (m @ D[:, 0]) / (m @ D @ m)
        res['bc'].append((r, mse_lin(k @ Q, D), mse_lin(m, D), mse_lin(s * m, D)))
        # LOP: traders see own sources + public; MM sees public signals; independent liquidity u
        su2 = float(rng.choice([0.25, 1.0, 4.0])); kM = Pm.shape[0]
        big = np.vstack([np.eye(nz)[0]] + Ai + ([Pm] if kM else []))
        Omz = big @ D @ big.T
        Om = np.zeros((Omz.shape[0] + 1,) * 2); Om[:-1, :-1] = Omz; Om[-1, -1] = su2
        ks = [A.shape[0] for A in Ai]
        t0 = time.time(); al, bD, bM = lop(Om, ks, kM); res['t_lop'].append(time.time() - t0)
        err = verify(Om, ks, kM, al, bD, bM); assert err[0] < 1e-8 and err[1] < 1e-6, err
        o = 0; flips = 0; tot = 0; cors = []
        for i, kk in enumerate(ks):
            a = al[o:o + kk]; cb = np.linalg.solve(Ai[i] @ D @ Ai[i].T, Ai[i] @ D[:, 0])
            fa = a @ Ai[i]; fb = cb @ Ai[i]
            cors.append((fa @ D @ fb) / np.sqrt((fa @ D @ fa) * (fb @ D @ fb)))
            npub = Pm.shape[0]; kp = kk - npub
            flips += np.sum(np.sign(a[:kp]) != np.sign(cb[:kp])); tot += kp
            res.setdefault('pubflip', []).extend(list(np.sign(a[kp:]) != np.sign(cb[kp:])))
            res.setdefault('ws', []).extend(list(a[:kp] / np.abs(a[:kp]).sum()) if kp else [])
            o += kk
        res['lop_flip'].append(flips / tot); res['lop_corr'].append(np.mean(cors))
        # Sequential myopic-Bayes market: price after N-1 traders = E[v | x_1..x_{N-1}]; trader N prices.
        X = np.vstack(Ai[:-1]); kx = np.linalg.lstsq(X @ D @ X.T, X @ D[:, 0], rcond=None)[0]
        fp = kx @ X; Vp = mse_lin(fp, D)
        Xn = np.vstack([fp, Ai[-1]]); kn = np.linalg.lstsq(Xn @ D @ Xn.T, Xn @ D[:, 0], rcond=None)[0]
        Vb = mse_lin(kn @ Xn, D)
        co = np.linalg.solve(Ai[-1] @ D @ Ai[-1].T, Ai[-1] @ D[:, 0]); fo = co @ Ai[-1]; Vo = mse_lin(fo, D)
        prec = 1 / Vp + 1 / Vo - 1.0            # product of the two posteriors divided by the N(0,1) prior
        fn = (fp / Vp + fo / Vo) / prec; vn = 1 / prec
        d2 = mse_lin(fn, D) - Vb                # E[(m_bayes - m_naive)^2]
        res['seq'].append(0.5 * (np.log(vn / Vb) + (Vb + d2) / vn - 1))
    bc = np.array(res['bc'])
    for rv in [0.0, 0.3, 0.6, 0.85]:
        s = bc[bc[:, 0] == rv]
        print("BC aggregator r=%.2f n=%d: MSE Bayes %.4f | mean quote %.4f | rescaled mean quote %.4f" % (
            rv, len(s), s[:, 1].mean(), s[:, 2].mean(), s[:, 3].mean()))
    lf = np.array(res['lop_flip'])
    print("LOP: share of signal components traded against own-Bayes sign %.3f; episodes with >=1 flip %.2f; mean corr %.3f" % (
        lf.mean(), (lf > 0).mean(), np.mean(res['lop_corr'])))
    print("LOP public components traded against own-Bayes sign: %.3f (n=%d)" % (np.mean(res['pubflip']), len(res['pubflip'])))
    sq = np.array(res['seq'])
    print("Sequential next trader KL(Bayes||independence) nats: median %.3f mean %.3f p90 %.3f frac>0.05 %.2f" % (
        np.median(sq), sq.mean(), np.quantile(sq, .9), (sq > 0.05).mean()))
    print("ms/episode numpy unbatched: BC %.2f  LOP %.2f" % (1e3 * np.mean(res['t_bc']), 1e3 * np.mean(res['t_lop'])))
