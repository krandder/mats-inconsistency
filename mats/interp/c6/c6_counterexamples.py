"""C6: inconsistency (energy) minimisation vs exact Bayes. Exact, tiny counterexamples.
Run: OMP_NUM_THREADS=1 python3 c6_counterexamples.py        (numpy only, a few seconds, one process)
     python3 c6_counterexamples.py scan                       (scan of the W1 observation size a)
(a) W1 point estimate vs volume   (b) W2 nullspace / gauge   (c1) W3-discrete tied global optima
(c2) revised after review: atom energy with correctly specified inputs recovers the truth
"""
import itertools, math, sys
import numpy as np
from math import erf, sqrt, log


def Phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def lse(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


# ---------------------------------------------------------------- (a) W1
# x in {0,1}^16; hypothesis R = pair of coords (120, uniform prior); cell c = (x_R0, x_R1);
# g_c ~ N(0,1); item v_x ~ N(g_c(x), tau^2); observation y ~ N(v_x, sigma^2).
# Query: a new item x_q = 0^16 (cell (0,0) under every R); report P(y_q > 0 | D).
D, TAU, SIG = 16, 0.3, 0.3
PAIRS = list(itertools.combinations(range(D), 2))
R1 = (0, 1)


def w1_items(a, aq=0.5):
    s = np.array([1, 0] * 7)                       # pattern on coords 2..15 (7 ones, 7 zeros)
    N = np.r_[0, 0, np.ones(14)]                   # R1: query cell.  R in {2..15}: cell (1,1)
    P = np.r_[1, 1, np.zeros(14)]                  # R1: cell (1,1).   R in {2..15}: query cell
    M1 = np.r_[1, 1, s]
    M2 = np.r_[1, 1, 1 - s]                        # share P's cell under R1 only
    Q0 = np.r_[0, 1, np.zeros(14)]                 # R1: own cell (0,1); query cell under R={0,j} and R in {2..15}
    Q1 = np.r_[1, 0, np.zeros(14)]                 # R1: own cell (1,0); query cell under R={1,j} and R in {2..15}
    X = np.array([N, P, M1, M2, Q0, Q1]).astype(int)
    y = np.array([-a, a, a, a, aq, aq], dtype=float)
    return X, y


def cells(X, R):
    return 2 * X[:, R[0]] + X[:, R[1]]


def w1_marginal(X, y, R, cq=0):
    """exact -log p(y|R) with g, v integrated out, and the exact predictive of y_q under R."""
    c = cells(X, R)
    n = len(y)
    S = (c[:, None] == c[None, :]) * 1.0 + np.eye(n) * (TAU**2 + SIG**2)   # one obs per item
    L = np.linalg.cholesky(S)
    al = np.linalg.solve(L, y)
    logml = -0.5 * al @ al - np.log(np.diag(L)).sum() - 0.5 * n * log(2 * math.pi)
    k = (c == cq) * 1.0
    Si_k = np.linalg.solve(S, k)
    mu = Si_k @ y
    var = 1 + TAU**2 + SIG**2 - k @ Si_k
    return logml, mu, var


def w1_joint_map(X, y, R, cq=0):
    """min over theta=(g_0..g_3, v_1..v_n) of -log p(y, theta | R): the pure point energy; returns (energy, g_q hat)."""
    c = cells(X, R)
    n = len(y)
    d = 4 + n
    Lam = np.zeros((d, d))
    b = np.zeros(d)
    Lam[:4, :4] += np.eye(4)
    for i in range(n):
        e = np.zeros(d)
        e[4 + i] = 1
        e[c[i]] = -1
        Lam += np.outer(e, e) / TAU**2
        Lam[4 + i, 4 + i] += 1 / SIG**2
        b[4 + i] += y[i] / SIG**2
    th = np.linalg.solve(Lam, b)
    g, v = th[:4], th[4:]
    nlp = (0.5 * (g @ g) + 0.5 * np.sum((v - g[c])**2) / TAU**2 + 0.5 * np.sum((y - v)**2) / SIG**2
           + 0.5 * (4 + 2 * n) * log(2 * math.pi) + n * log(TAU) + n * log(SIG))
    return nlp, g[cq]


def part_a(a=1.0, verbose=True):
    X, y = w1_items(a)
    lml, mus, vrs, ej, gq = [], [], [], [], []
    for R in PAIRS:
        l, m, v = w1_marginal(X, y, R)
        e, g = w1_joint_map(X, y, R)
        lml.append(l); mus.append(m); vrs.append(v); ej.append(e); gq.append(g)
    lml, mus, vrs, ej, gq = map(np.array, (lml, mus, vrs, ej, gq))
    lp = lml - log(len(PAIRS))
    post = np.exp(lp - lse(lp))
    pq = np.array([1 - Phi(-m / sqrt(v)) for m, v in zip(mus, vrs)])     # P(y_q>0 | R, D)
    i1 = PAIRS.index(R1)
    E_marg = -lp                     # energy over R, continuous latents integrated (includes their volume)
    E_joint = ej + log(len(PAIRS))   # pure point energy over (R, g, v)
    im, ij = int(np.argmin(E_marg)), int(np.argmin(E_joint))
    bayes_pq = float(post @ pq)
    pq_joint = 1 - Phi(-gq[ij] / sqrt(TAU**2 + SIG**2))
    opp = (pq > 0.5) & (np.arange(len(PAIRS)) != i1)
    srt = np.sort(E_marg)
    r = dict(a=a, argmin_marg=PAIRS[im], argmin_joint=PAIRS[ij], gap=srt[1] - srt[0],
             post_R1=post[i1], post_opp=post[opp].sum(), n_opp=int(opp.sum()),
             n_opp_2=int(np.sum(opp & (E_marg - E_marg[i1] < 2.0))),
             pq_R1=pq[i1], pq_bayes=bayes_pq, pq_joint=pq_joint)
    if not verbose:
        return r
    print("(a) W1 point estimate vs volume. tau=%.2f sigma=%.2f; items N,P,M1,M2,Q0,Q1 observed y=%s; query x_q=0^16"
          % (TAU, SIG, y))
    print("    argmin_R E_marg(R) = %s ; argmin E_joint(R,g,v) = %s ; E_marg gap to 2nd best = %.3f nats"
          % (r['argmin_marg'], r['argmin_joint'], r['gap']))
    dE = E_marg[opp] - E_marg[i1]
    print("    %d other hypotheses predict y_q>0 (P>0.5); their E_marg - E_marg(R1): min %.3f, median %.3f, "
          "%d within 2 nats" % (r['n_opp'], dE.min(), np.median(dE), r['n_opp_2']))
    nz = (np.abs(pq - 0.5) < 1e-9)
    print("    %d hypotheses have an empty query cell (P=0.5), posterior mass %.4f" % (nz.sum(), post[nz].sum()))
    print("    posterior P(R1|D) = %.4f ; P(opposite group|D) = %.4f ; mass ratio opposite/R1 = %.1f"
          % (post[i1], post[opp].sum(), post[opp].sum() / post[i1]))
    print("    P(y_q>0): energy-min R1 (exact within R1) = %.4f ; joint-MAP plug-in = %.4f ; Bayes = %.4f ; "
          "Bayes - energy-min = %+.4f" % (pq[i1], pq_joint, bayes_pq, bayes_pq - pq[i1]))
    print("    E[y_q]: R1 = %.3f ; Bayes = %.3f" % (mus[i1], post @ mus))
    grid = np.linspace(-8, 8, 32001)
    dx = grid[1] - grid[0]

    def dens(m, v):
        return np.exp(-(grid - m)**2 / (2 * v)) / np.sqrt(2 * math.pi * v)
    pb = sum(w * dens(m, v) for w, m, v in zip(post, mus, vrs))
    pe = dens(mus[i1], vrs[i1])
    kl = np.sum(pb * (np.log(pb + 1e-300) - np.log(pe + 1e-300))) * dx
    print("    KL(Bayes predictive || energy-min predictive) on y_q = %.3f nats" % kl)
    # free energy over hypotheses: F(w) = sum_R w_R E_marg(R) - H(w) is minimised exactly by the posterior
    Fpost = post @ E_marg + np.sum(post * np.log(post))
    print("    check: min_w [E_w E_marg - H(w)] = %.6f = -log p(D) = %.6f" % (Fpost, -lse(lp)))
    for T in (0.25, 0.5, 1.0, 2.0):
        wT = np.exp(-E_marg / T - lse(-E_marg / T))
        print("    temperature T=%.2f (w ~ exp(-E_marg/T)): P(R1) = %.3f, P(y_q>0) = %.3f" % (T, wT[i1], wT @ pq))
    return r


# ---------------------------------------------------------------- (b) W2 nullspace
# K=3 items v_k ~ N(0,1); S=2 sources with bias b_s ~ N(0, tb^2); SPOT quote y_ks = v_k + b_s + e, e ~ N(0, s^2);
# every source quotes every item m times, every quote = ybar. Query: v_1, P(v_1 > 0.5).
def part_b(tb=0.25, s=0.3, ybar=1.0):
    K, S = 3, 2

    def design(m, anchor=False):
        rows, ys = [], []
        for k in range(K):
            for j in range(S):
                for _ in range(m):
                    r = np.zeros(K + S); r[k] = 1; r[K + j] = 1
                    rows.append(r); ys.append(ybar)
        if anchor:   # m quotes of item 1 from a known-unbiased source: removes the nullspace
            for _ in range(m):
                r = np.zeros(K + S); r[0] = 1
                rows.append(r); ys.append(ybar)
        return np.array(rows), np.array(ys)

    A, yv = design(1)
    _, sv, Vt = np.linalg.svd(A)
    rank = int(np.sum(sv > 1e-10))
    null = Vt[-1] / Vt[-1][0]                       # scaled so v_1 moves by c
    P0 = np.diag([1.0] * K + [1 / tb**2] * S)
    th_mn = np.linalg.pinv(A) @ yv                  # min-norm residual optimum (= GD from 0)

    def resid(th):
        return np.sum((yv - A @ th)**2)

    def priorE(th):
        return 0.5 * th @ P0 @ th
    c_star = -(th_mn @ P0 @ null) / (null @ P0 @ null)
    th_pr = th_mn + c_star * null                   # prior-chosen point on the optimum line

    def bayes(m, anchor=False, tbb=tb):
        A_, y_ = design(m, anchor)
        P0_ = np.diag([1.0] * K + [1 / tbb**2] * S)
        C = np.linalg.inv(P0_ + A_.T @ A_ / s**2)
        mu = C @ A_.T @ y_ / s**2
        return mu[0], sqrt(C[0, 0])
    print("\n(b) W2 nullspace. K=3 items, S=2 biased sources, all 6 SPOT quotes = %.1f, tau_b=%.2f, sigma=%.2f"
          % (ybar, tb, s))
    print("    rank(A) = %d of %d unknowns -> nullity %d; null vector (v1,v2,v3,b1,b2) = %s"
          % (rank, K + S, K + S - rank, np.round(null, 3)))
    for c in (-3.0, 0.0, 3.0):
        th = th_mn + c * null
        print("    residual at min-norm %+.0f*null: %.1e (v_1 = %+.3f, v_1-v_2 = %+.3f, b = %s)"
              % (c, resid(th), th[0], th[0] - th[1], np.round(th[3:], 3)))
    dE = priorE(th_mn) - priorE(th_pr)
    print("    two residual optima: min-norm v_1 = %.3f, prior-chosen v_1 = %.3f; prior log density ratio = %.2f nats"
          % (th_mn[0], th_pr[0], dE))
    for m in (1, 10, 1000):
        mu, sd = bayes(m)
        mua, sda = bayes(m, anchor=True)
        print("    m=%4d quotes/pair: Bayes v_1 = %.3f +- %.3f, P(v_1>0.5) = %.3f | + m anchor quotes: %.3f +- %.3f"
              % (m, mu, sd, 1 - Phi((0.5 - mu) / sd), mua, sda))
    for tbb in (0.25, 1.0, 4.0):
        mu, sd = bayes(1000, tbb=tbb)
        print("    prior tau_b = %.2f, m=1000: Bayes v_1 = %.3f +- %.3f (residual optimum set identical)" % (tbb, mu, sd))


# ---------------------------------------------------------------- (c1) W3-discrete
# Root bit z (prior 1/2) with one root report of likelihood ratio L for z=1; k leaves c_j, each with a unary report
# "c_j = 0" (error d) and a relation report "c_j = z" (error e). Energy = -log p(z, c, reports): the properly weighted
# inconsistency (each violated report costs its log-odds) plus prior. Query: z.
def part_c1(k=4, d=0.2, e=0.25):
    def leaf(z, c):
        return ((1 - d) if c == 0 else d) * ((1 - e) if c == z else e)
    L_tie = 1.0
    for _ in range(k):
        L_tie *= max(leaf(0, 0), leaf(0, 1)) / max(leaf(1, 0), leaf(1, 1))
    for L, tag in ((L_tie, "tie"), (0.95 * L_tie, "lower")):
        st, E, cnt = [], [], []
        for z in (0, 1):
            for cs in itertools.product((0, 1), repeat=k):
                p = (L if z == 1 else 1.0) / (1 + L)
                for c in cs:
                    p *= leaf(z, c)
                st.append((z,) + cs)
                E.append(-log(p))
                cnt.append(int(z != 1) + sum(c != 0 for c in cs) + sum(c != z for c in cs))
        E, cnt = np.array(E), np.array(cnt)
        pz = np.array([t[0] for t in st])
        post = np.exp(-E - lse(-E))
        pb = post[pz == 1].sum()
        opt = [st[i] for i in np.where(E - E.min() < 1e-9)[0]]
        if tag == "tie":
            print("\n(c1) W3-discrete. k=%d leaves, unary error %.2f, relation error %.2f, root report LR = %.0f "
                  "(error %.5f)" % (k, d, e, L, 1 / (1 + L)))
            copt = [st[i] for i in np.where(cnt == cnt.min())[0]]
            print("     unweighted violation count: min %d, %d minimisers, query z takes values %s"
                  % (cnt.min(), len(copt), sorted({o[0] for o in copt})))
        else:
            print("     root LR lowered 5%% to %.2f:" % L)
        print("     weighted-energy global minimisers (z, c1..c%d): %s" % (k, opt))
        print("     Bayes P(z=1|D) = %.4f (unique); minimisers give z in %s; P(z=1) of minimiser set uniform = %.3f"
              % (pb, sorted({o[0] for o in opt}), np.mean([o[0] for o in opt])))


# ---------------------------------------------------------------- (c2) atom energy
def part_c2():
    """Revised after review. Atom energy F(mu) = sum_a mu(v_S) KL(mu(X_i|v_S) || q_a) - gamma H(mu) on 2 bits,
    with the atom set the real estimator uses (both full conditionals AND both unconditional marginals), minimised
    exactly (scipy L-BFGS over softmax logits, several starts). World: z1 ~ Bern(0.8), z2 = z1 xor Bern(eta)."""
    from scipy.optimize import minimize
    eta = 1e-3
    p = np.array([0.2 * (1 - eta), 0.2 * eta, 0.8 * eta, 0.8 * (1 - eta)])          # true joint over 00,01,10,11

    def conds(j):                                    # true conditionals from a joint j
        M = j.reshape(2, 2)
        return M / M.sum(1, keepdims=True), (M / M.sum(0, keepdims=True)).T   # q(z2|z1)[z1,z2], q(z1|z2)[z2,z1]

    def F(mu, q21, q12, m1, m2, gam):
        M = mu.reshape(2, 2)
        tot = 0.0
        if q21 is not None:
            tot += np.sum(M * (np.log(M) - np.log(M.sum(1, keepdims=True)) - np.log(q21)))
            tot += np.sum(M.T * (np.log(M.T) - np.log(M.sum(0)[:, None]) - np.log(q12)))
        if m1 is not None:
            for marg, q in ((M.sum(1), m1), (M.sum(0), m2)):
                tot += np.sum(marg * (np.log(marg) - np.log(q)))
        return tot + gam * np.sum(mu * np.log(mu))

    def fit(*args):
        np.seterr(all='ignore')
        best = None
        for seed in range(5):
            th0 = np.random.default_rng(seed).normal(size=4)
            f = lambda th: F(np.clip(np.exp(th - lse(th)), 1e-300, 1), *args)
            r = minimize(f, th0, method='L-BFGS-B', options=dict(maxiter=5000, ftol=1e-15, gtol=1e-12))
            if best is None or r.fun < best.fun:
                best = r
        mu = np.exp(best.x - lse(best.x))
        return mu, best.fun
    q21, q12 = conds(p)
    sym = np.array([[1 - eta, eta], [eta, 1 - eta]])
    m1 = p.reshape(2, 2).sum(1)
    m2 = p.reshape(2, 2).sum(0)                    # true marginals of z1 and z2
    print("\n(c2, revised) atom energy on 2 bits; world z1~Bern(0.8), z2 = z1 xor Bern(%g); truth P(z1=1)=0.8" % eta)
    for tag, args in (("true full conditionals only", (q21, q12, None, None)),
                      ("true conditionals + true marginals (real atom set)", (q21, q12, m1, m2)),
                      ("symmetric copy conditionals only (old c2 input)", (sym, sym, None, None)),
                      ("symmetric copy conditionals + true marginals", (sym, sym, m1, m2))):
        for gam in (0.0, 1e-4):
            mu, f = fit(*(args + (gam,)))
            print("     %-52s gamma=%g: P(z1=1) = %.4f, F = %.2e, KL(p||mu) = %.1e"
                  % (tag, gam, mu[2] + mu[3], f, np.sum(p * np.log(p / mu))))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        for a in np.arange(0.5, 2.01, 0.1):
            r = part_a(a, verbose=False)
            print("a=%.1f argmin=%s/%s gap=%.2f P(R1)=%.3f P(opp)=%.3f n_opp=%d(<2nats %d) pq_R1=%.3f pq_bayes=%.3f pq_joint=%.3f"
                  % (a, r['argmin_marg'], r['argmin_joint'], r['gap'], r['post_R1'], r['post_opp'], r['n_opp'],
                     r['n_opp_2'], r['pq_R1'], r['pq_bayes'], r['pq_joint']))
    else:
        part_a(float(sys.argv[1]) if len(sys.argv) > 1 else 1.2)
        part_b()
        part_c1()
        part_c2()
