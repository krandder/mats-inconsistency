"""LIR study, 2-trader toy (eurisko li-net, 2026-10-03).

Checks, each against a generic numerical PDG-inconsistency solver where possible
(inf over mu on the simplex by scipy), not only against closed forms:
 C1  market log loss = (<<M+x>> - <<M>>)/beta0                    (chain rule identity)
 C2  dL/dbeta_i = -(1/beta0)(l_i(x) - E_mu l_i);  E_mu[s_i] = 0   (zero-price contract)
 C3  trader focus T_i (own edge +, data inf, market copy pinned, own edge on copy -)
     has the same gradient in (beta_i, theta_i) as the global market loss
 C4  plain inconsistency <<M+x>> as the trader objective drives every stake to 0
     (and the "own edges + data" focus is that objective)
 C5  LIR schedule (price focus full control, trader foci small control) on samples
     reaches the direct optimum of the market loss; stakes: relevant>0, irrelevant->0
 C6  KKT / inexploitability: at the optimum E_q[s_i]=0 if beta_i>0, <=0 if beta_i=0
 C7  leave-one-out r_i = log mu - log mu_{-i}: grad wrt own params = -grad L (exact
     potential game); Kelly-stake P&L beta_i log(p_i/mu): cross-partials asymmetric
 C8  sum_a beta_a log(p_a(x)/mu(x)) = -<<M>> for every x (house takes the inconsistency)
 C9  normalized geometric pool averages evidence; Bayes form (negative prior edge) adds it
 C10 GAN Prop 3: discriminator / generator foci give -L_GAN - log2 and L_GAN + log2
"""
import numpy as np
from scipy.optimize import minimize
from itertools import product

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)
EPS = 1e-12

# ---------- generic solver: one variable X, unconditional edges with weights ----------
def inc_generic(edges, n):
    """<<M>> = inf_mu sum_a b_a KL(mu || p_a); edges = [(b_a, p_a)], b_a may be negative
    as long as sum b_a > 0. Solved numerically over softmax-parametrised mu."""
    def f(z):
        mu = np.exp(z - z.max()); mu /= mu.sum()
        return sum(b * np.sum(mu * (np.log(mu + EPS) - np.log(p))) for b, p in edges)
    best = None
    for s in range(5):
        r = minimize(f, rng.normal(size=n), method="BFGS", options={"gtol": 1e-10})
        if best is None or r.fun < best.fun: best = r
    z = best.x; mu = np.exp(z - z.max()); mu /= mu.sum()
    return best.fun, mu

def pool(edges):
    """closed form minimiser: mu ∝ prod p_a^{b_a/B}"""
    B = sum(b for b, _ in edges)
    lg = sum(b * np.log(p) for b, p in edges) / B
    Z = np.exp(lg).sum()
    return np.exp(lg) / Z, -B * np.log(Z)

print("C1/C8  identities on a random 3-valued X with 3 edges (one negative weight)")
n = 3
for trial in range(3):
    ps = [rng.dirichlet(np.ones(n)) for _ in range(3)]
    bs = [1.0, 0.7 + rng.random(), -0.4 * rng.random()]
    edges = list(zip(bs, ps))
    I_num, mu_num = inc_generic(edges, n)
    mu_cf, I_cf = pool(edges)
    x = rng.integers(n)
    I_x = sum(b * -np.log(p[x]) for b, p in edges)    # mu forced to delta_x by the data edge
    B = sum(bs)
    lhs = -np.log(mu_cf[x]); rhs = (I_x - I_num) / B
    pl = [sum(b * np.log(p[xx] / mu_cf[xx]) for b, p in edges) for xx in range(n)]
    print(f"  trial {trial}: <<M>> num {I_num:.6f} cf {I_cf:.6f} | |mu_num-mu_cf| {np.abs(mu_num-mu_cf).max():.1e}"
          f" | -log mu(x) {lhs:.6f} vs (<<M+x>>-<<M>>)/B {rhs:.6f} | sum_a b_a log(p_a/mu) per x {np.round(pl,6)} vs -<<M>> {-I_cf:.6f}")

# ---------- the 2-trader toy ----------
# X in {0,1}, prior 0.5 (house edge, beta0 = 1, fixed). Regime R in {0,1} uniform, seen by both.
# Trader i sees S_i: S_i = X w.p. ACC if R == i, else S_i ~ Bern(1/2) independent of X.
# Trader i's cpd (likelihood form): p_i(S_i = X | X) = a_i  (theta_i = logit a_i), regime-blind.
# Trader i's stake: beta_i(R), two numbers (its own confidence, a function of its information).
ACC = 0.8
states = []   # (prob, x, r, s1, s2)
for x, r, s1, s2 in product([0, 1], [0, 1], [0, 1], [0, 1]):
    px = 0.5; pr = 0.5
    def ps(s, i): return (ACC if s == x else 1 - ACC) if r == i else 0.5
    states.append((px * pr * ps(s1, 0) * ps(s2, 1), x, r, s1, s2))
q = np.array([s[0] for s in states]); assert abs(q.sum() - 1) < 1e-12
beta0 = 1.0; prior = np.array([0.5, 0.5])

def lik(theta, s):          # vector over x of log p_i(s | x)
    a = 1 / (1 + np.exp(-theta))
    return np.array([np.log(a) if s == x else np.log(1 - a) for x in (0, 1)])

def market(params, r, s, drop=None):
    """Bayes-form pool: mu ∝ prior * prod p_i(s_i|X)^{beta_i(r)/beta0}.
    As a PDG on X: prior edge weight beta0 - sum beta_i (may be negative), posterior edges
    p_i(X|s_i) ∝ prior*p_i(s_i|X) with weight beta_i; here computed in likelihood form."""
    th, be = params
    lg = np.log(prior).copy()
    for i in range(2):
        if i == drop: continue
        lg += be[i, r] / beta0 * lik(th[i], s[i])
    mu = np.exp(lg - lg.max()); return mu / mu.sum()

def L(params):
    return sum(qq * -np.log(market(params, r, (s1, s2))[x]) for qq, x, r, s1, s2 in states)

def flat(params): return np.concatenate([params[0], params[1].ravel()])
def unflat(v): return (v[:2].copy(), v[2:].reshape(2, 2).copy())
def num_grad(f, v, h=1e-6):
    g = np.zeros_like(v)
    for k in range(len(v)):
        e = np.zeros_like(v); e[k] = h; g[k] = (f(v + e) - f(v - e)) / (2 * h)
    return g

bayes = sum(qq * -np.log(  # exact posterior P(X=x|r,s1,s2)
    (lambda post: post[x] / post.sum())(np.array([sum(q2 for q2, x2, r2, a2, b2 in states
        if x2 == xx and r2 == r and a2 == s1 and b2 == s2) for xx in (0, 1)])))
    for qq, x, r, s1, s2 in states)
print(f"\ntoy: ACC={ACC}; Bayes-optimal market loss {bayes:.5f} nats; uninformed log2 {np.log(2):.5f}")

# C2: gradient of L wrt beta_i at a random point = -(1/beta0) E_q[l_i(x) - E_mu l_i]; E_mu s_i = 0
P = (rng.normal(size=2), np.abs(rng.normal(size=(2, 2))))
v = flat(P)
g_num = num_grad(lambda w: L(unflat(w)), v)
g_cf = np.zeros(6); Ems = []
for qq, x, r, s1, s2 in states:
    s = (s1, s2); mu = market(P, r, s)
    for i in range(2):
        l = lik(P[0][i], s[i])
        si = l[x] - mu @ l                       # zero-price contract (times 1/beta0)
        Ems.append(mu @ (l - mu @ l))
        g_cf[2 + 2 * i + r] += qq * -(1 / beta0) * si
        a = 1 / (1 + np.exp(-P[0][i])); dl = np.array([(1 - a) if s[i] == xx else -a for xx in (0, 1)])
        g_cf[i] += qq * -(P[1][i, r] / beta0) * (dl[x] - mu @ dl)
print("C2  grad L numeric ", g_num, "\n    closed form     ", g_cf, f"\n    max |E_mu s_i| = {np.abs(Ems).max():.1e}")

# C3: the trader focus T_i as an explicit PDG on (X, X'), solved generically, gradient vs global
def focus_T(i, params_i, m, x, s_i, r):
    """Variables X (data x, inf), X' (market copy m, inf). Edges: p_i(S_i|X) weight +beta_i,
    p_i(S_i|X') weight -beta_i, with S_i pinned to s_i. mu is fully pinned by the inf edges,
    so the inconsistency is the weighted sum evaluated at mu* = delta_x (x) m."""
    th, b = params_i
    l = lik(th, s_i)
    return b * (-l[x]) - b * (m @ -l)
gl = num_grad(lambda w: L(unflat(w)), v)
gT = np.zeros(6)
for qq, x, r, s1, s2 in states:
    s = (s1, s2); m = market(P, r, s)            # price focus: m <- pool (frozen in T_i)
    for i in range(2):
        f = lambda w: focus_T(i, (w[0], w[1]), m, x, s[i], r)
        gi = num_grad(f, np.array([P[0][i], P[1][i, r]]))
        gT[i] += qq * gi[0] / beta0; gT[2 + 2 * i + r] += qq * gi[1] / beta0
print("C3  global grad    ", gl, "\n    sum of T_i grads ", gT, f"\n    max diff {np.abs(gl-gT).max():.1e}")

# C4: plain inconsistency <<M+x>> = sum_i beta_i(-l_i(x)) + beta0(-log prior(x)) as objective
def plainInc(params):
    th, be = params
    return sum(qq * (beta0 * -np.log(prior[x]) + sum(be[i, r] * -lik(th[i], (s1, s2)[i])[x] for i in range(2)))
               for qq, x, r, s1, s2 in states)
w = flat((np.array([1.0, 1.0]), np.ones((2, 2))))
for it in range(3000):
    w -= 0.05 * num_grad(lambda u: plainInc(unflat(u)), w); w[2:] = np.maximum(w[2:], 0)
th_, be_ = unflat(w)
print(f"C4  plain-inconsistency descent: stakes {be_.ravel()} (all 0), market loss {L(unflat(w)):.5f} (= log2)")

# C5: direct optimum of L (projected GD, beta>=0) vs an LIR schedule on samples
def proj(w): w[2:] = np.maximum(w[2:], 0); return w
w = flat((np.array([0.5, 0.5]), np.full((2, 2), 0.5)))
for it in range(4000): w = proj(w - 0.3 * num_grad(lambda u: L(unflat(u)), w))
Pdir = unflat(w)
eff = lambda Pp: np.array([[Pp[1][i, r] * Pp[0][i] for r in (0, 1)] for i in (0, 1)])
print(f"C5  direct GD on L: loss {L(Pdir):.5f}; stakes beta_i(R) rows=trader cols=R:\n{Pdir[1]}"
      f"\n    effective log-odds weight beta*theta (target logit {np.log(ACC/(1-ACC)):.4f} on diagonal, 0 off):\n{eff(Pdir)}")

def focus_T_grad(i, th_i, b_i, m, x, s_i):
    """analytic gradient of the local focus T_i in (theta_i, beta_i); uses only m, x, s_i, own params"""
    l = lik(th_i, s_i); a = 1 / (1 + np.exp(-th_i))
    dl = np.array([(1 - a) if s_i == xx else -a for xx in (0, 1)])
    return np.array([-b_i * (dl[x] - m @ dl), -(l[x] - m @ l)])
th = np.array([0.5, 0.5]); be = np.full((2, 2), 0.5); ROUNDS, BATCH = 4000, 64
for t in range(ROUNDS):
    chi = 1.0 / (1 + t / 500)
    counts = rng.multinomial(BATCH, q)
    gth = np.zeros(2); gbe = np.zeros((2, 2))
    for k in np.nonzero(counts)[0]:
        _, x, r, s1, s2 = states[k]; s = (s1, s2)
        m = market((th, be), r, s)               # focus P: attend to all trader edges, full control of m
        for i in range(2):                       # focus T_i: own edge (+), data (inf), market copy (inf), own edge on copy (-)
            g = focus_T_grad(i, th[i], be[i, r], m, x, s[i])
            gth[i] += counts[k] * g[0] / BATCH; gbe[i, r] += counts[k] * g[1] / BATCH
    th -= chi * gth; be = np.maximum(0.0, be - chi * gbe)
chi_s = f"1/(1+t/500), {ROUNDS} rounds x {BATCH} samples"
Plir = (th, be)
print(f"    LIR schedule ({chi_s}): loss {L(Plir):.5f}; stakes:\n{be}\n    effective weight:\n{eff(Plir)}")

# the schedule the brief proposed: attention = own edge + data only, control = own cpd and stake
th2 = np.array([0.5, 0.5]); be2 = np.full((2, 2), 0.5)
idx = rng.choice(len(states), size=20000, p=q); chi = 0.05
for t, k in enumerate(idx):
    _, x, r, s1, s2 = states[k]; s = (s1, s2)
    for i in range(2):
        f = lambda u: u[1] * -lik(u[0], s[i])[x]
        g = num_grad(f, np.array([th2[i], be2[i, r]]))
        th2[i] -= chi * g[0]; be2[i, r] = max(0.0, be2[i, r] - chi * g[1])
print(f"    'own edge + data' focus: stakes {be2.ravel()} -> 0, loss {L((th2, be2)):.5f}")

# C6: KKT / inexploitability at the direct optimum
for i in range(2):
    for r in (0, 1):
        Es = sum(qq * (lik(Pdir[0][i], (s1, s2)[i])[x] - market(Pdir, rr, (s1, s2)) @ lik(Pdir[0][i], (s1, s2)[i]))
                 for qq, x, rr, s1, s2 in states if rr == r) / 0.5
        print(f"C6  trader {i} regime {r}: beta={Pdir[1][i, r]:.4f}  E_q[s_i | R={r}] = {Es:+.2e}")

# C7: leave-one-out is an exact potential game; Kelly-stake P&L is not
def LOO(i, params):
    return sum(qq * (np.log(market(params, r, (s1, s2))[x]) - np.log(market(params, r, (s1, s2), drop=i)[x]))
               for qq, x, r, s1, s2 in states)
v = flat(P); gL = num_grad(lambda u: L(unflat(u)), v)
for i in range(2):
    own = [i, 2 + 2 * i, 3 + 2 * i]
    gr = num_grad(lambda u: LOO(i, unflat(u)), v)
    print(f"C7  LOO trader {i}: grad of r_i on own params {gr[own]} vs -grad L {-gL[own]}")
# Kelly-stake game: normalized pool over 3 fixed beliefs on a 3-valued X, payoff pi_i = beta_i E_q log(p_i/mu)
qx = np.array([0.5, 0.3, 0.2]); bel = [np.array([.6, .3, .1]), np.array([.2, .5, .3]), np.array([.34, .33, .33])]
def pi(i, b):
    mu, _ = pool(list(zip(b, bel))); return b[i] * qx @ np.log(bel[i] / mu)
b0 = np.array([1.0, 0.7, 0.4]); h = 1e-4
def mixed(i, j, b):
    def d_i(bb):
        e = np.zeros(3); e[i] = h; return (pi(i, bb + e) - pi(i, bb - e)) / (2 * h)
    e = np.zeros(3); e[j] = h; return (d_i(b + e) - d_i(b - e)) / (2 * h)
print(f"    Kelly-stake P&L game: d2pi_0/db0db1 = {mixed(0,1,b0):+.5f}, d2pi_1/db1db0 = {mixed(1,0,b0):+.5f}"
      " (unequal -> no exact potential)")
print(f"    sum_i pi_i = {sum(pi(i,b0) for i in range(3)):.6f} = -<<M>> = {-pool(list(zip(b0,bel)))[1]:.6f}")

# C9: averaging vs adding evidence, two independent 0.9 signals for X=1, prior 1/2
p9 = np.array([0.1, 0.9])
mu_avg, _ = pool([(1, p9), (1, p9)])
mu_bayes, _ = pool([(1 - 2, prior), (1, p9), (1, p9)])     # prior edge weight beta0 - sum beta_i = -1
print(f"\nC9  normalized pool of two 0.9 posteriors: {mu_avg[1]:.4f}; with prior edge at weight -1 (Bayes): {mu_bayes[1]:.4f}"
      f" (exact {0.81/(0.81+0.01):.4f}); generic solver: {inc_generic([(-1.0, prior), (1, p9), (1, p9)], 2)[1][1]:.4f}")

# C10: GAN Prop 3 on a discrete image space
K = 5; pdata = rng.dirichlet(np.ones(K)); G = rng.dirichlet(np.ones(K)); D = rng.uniform(.05, .95, K)  # D = P(real|x)
LGAN = pdata @ np.log(D) + G @ np.log(1 - D)
mu = np.stack([0.5 * pdata, 0.5 * G], 1)  # mu*(x, c), c=0 real, 1 fake, pinned by inf edges
def incGAN(bD, be):
    muc = mu / mu.sum(1, keepdims=True)
    Dc = np.stack([D, 1 - D], 1); e = np.full_like(Dc, 0.5)
    return (mu * (bD * (np.log(muc) - np.log(Dc)) + be * (np.log(muc) - np.log(e)))).sum()
print(f"C10 GAN: L_GAN={LGAN:.5f}; disc focus (D+1,e-1): {incGAN(1,-1):.5f} vs -L-log2 {-LGAN-np.log(2):.5f};"
      f" gen focus (D-1,e+1): {incGAN(-1,1):.5f} vs L+log2 {LGAN+np.log(2):.5f}; sum of the two foci {incGAN(1,-1)+incGAN(-1,1):.1e}")
print(f"    corrected: -L/2-log2 = {-LGAN/2-np.log(2):.5f}, L/2+log2 = {LGAN/2+np.log(2):.5f} (paper's proof drops the 1/2 from the fair coin)")
for bD, be_ in [(1, 0), (0.7, 0.4), (2, -0.5)]:
    JS = 0.5*(pdata@np.log(pdata/(0.5*pdata+0.5*G))) + 0.5*(G@np.log(G/(0.5*pdata+0.5*G)))
    print(f"    general bD={bD}, be={be_}: numeric {incGAN(bD,be_):.5f}; -bD*L/2+(bD+be)JS-bD*log2 = {-bD*LGAN/2+(bD+be_)*JS-bD*np.log(2):.5f};"
          f" paper -bD*L+(bD+be)JS-bD*log2 = {-bD*LGAN+(bD+be_)*JS-bD*np.log(2):.5f}")
