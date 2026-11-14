"""Part 1: history-free traders for fixed-point properties (coherence, forced values, prior frequencies).
Atoms a,b,c; Gamma = {a->b, b->c}. Sentences S below. Worlds = assignments; models = PC(Gamma)."""
import itertools, numpy as np
from scipy.optimize import minimize, linprog
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)
imp = lambda x, y: (not x) or y
S = {  # name -> truth function of (a,b,c)
 'a': lambda a,b,c: a, 'b': lambda a,b,c: b, 'c': lambda a,b,c: c,
 'a->c': lambda a,b,c: imp(a,c), 'a&c': lambda a,b,c: a and c, '~b': lambda a,b,c: not b,
 'b|c': lambda a,b,c: b or c, 'a&~c': lambda a,b,c: a and not c, 'c->a': lambda a,b,c: imp(c,a)}
names = list(S)
allw = list(itertools.product([0,1], repeat=3))
gamma = lambda a,b,c: imp(a,b) and imp(b,c)
models = [w for w in allw if gamma(*w)]
M = np.array([[float(S[s](*w)) for s in names] for w in models])   # |models| x |S|
print('models (a,b,c):', models)

def project(P, A=M):
    """Euclidean projection of P onto conv(rows of A) (QP over the simplex)."""
    k = A.shape[0]
    f = lambda l: 0.5*np.sum((l@A - P)**2)
    g = lambda l: (l@A - P)@A.T
    r = minimize(f, np.ones(k)/k, jac=g, method='SLSQP', bounds=[(0,1)]*k,
                 constraints=[{'type':'eq','fun':lambda l: l.sum()-1,'jac':lambda l: np.ones(k)}],
                 options={'ftol':1e-15,'maxiter':500})
    return r.x@A

# (1a) coherence trader T(P) = pi(P) - P : sure profit >= dist^2 in every model
worst = np.inf
for _ in range(300):
    P = rng.random(len(names)); pi = project(P); d2 = np.sum((P-pi)**2)
    prof = (M - P)@(pi - P)               # value of trade in each model
    worst = min(worst, (prof - d2).min())
print('(1a) min over 300 random P and all models of [profit - dist^2] =', round(worst, 8), '(theorem: >= 0)')

# block dynamics P <- clip(P + eta*T(P))
P = rng.random(len(names)); eta = 0.5
for t in range(30):
    P = np.clip(P + eta*(project(P) - P), 0, 1)
print('(1a) after 30 blocks eta=0.5: dist to C =', np.linalg.norm(P-project(P)).round(10))
print('     P(a->c) [entailed] =', P[names.index('a->c')].round(6), ' P(a&~c) [refuted] =', P[names.index('a&~c')].round(6))

# (1b) single affine-constraint trader for Gamma |= a <= b (a->b): T = max(0,Pa-Pb)(e_b - e_a)
ia, ib = names.index('a'), names.index('b'); worst = np.inf
for _ in range(1000):
    P = rng.random(len(names)); x = np.zeros(len(names)); g = max(0, P[ia]-P[ib]); x[ib] += g; x[ia] -= g
    worst = min(worst, ((M-P)@x - g*g).min())
print('(1b) affine trader: min [profit - gap^2] =', round(worst, 10), '(theorem: >= 0)')

# (1c) Stiemke: stationary P is exploitable (weak Dutch book: >=0 in all models, >0 in one) iff P not in relint C
def weak_book(P):
    # maximise sum_w x.(w-P) s.t. x.(w-P) >= 0 for all w, |x_i|<=1
    D = M - P
    r = linprog(-D.sum(0), A_ub=-D, b_ub=np.zeros(len(models)), bounds=[(-1,1)]*len(names))
    return -r.fun
uniform = M.mean(0)
boundary = 0.5*M[0] + 0.5*M[1]
print('(1c) weak-book value at uniform-over-models P =', round(weak_book(uniform), 9), ' at boundary P =', round(weak_book(boundary), 6))

# (1d) prior traders. target m* = marginals of mu0(.|Gamma), mu0 = product prior with Pr(atom)=q
q = np.array([0.3, 0.6, 0.5])
mu0_all = np.array([np.prod([q[i] if w[i] else 1-q[i] for i in range(3)]) for w in allw])
Mall = np.array([[float(S[s](*w)) for s in names] for w in allw])
mask = np.array([bool(gamma(*w)) for w in allw])
mstar = (mu0_all[mask]/mu0_all[mask].sum())@M        # conditioned marginals (PDG: Gamma at confidence inf)
m0 = mu0_all@Mall                                     # unconditioned product-prior marginals
print('(1d) m* (conditioned)        =', mstar)
# fixed point of fix(V)=clip(V + T_coh(V) + T_prior(V)) with T_prior = m* - V
V = rng.random(len(names))
for t in range(200): V = np.clip(V + 0.3*((project(V)-V) + (mstar - V)), 0, 1)
print('     fixed point coh+prior(m*) =', V, ' |V-m*| =', np.linalg.norm(V-mstar).round(8))
# hard coherence + local prior m0: Euclidean projection of m0 onto C  vs  conditioning
pm0 = project(m0)
print('     Euclid proj of m0 onto C =', pm0, ' |pi(m0)-m*| =', np.linalg.norm(pm0-mstar).round(4), '(refutes: projection of prices == conditioning)')
# soft coherence weight w_c vs prior weight 1: fixed point distance to C
for wc in [1, 10, 100]:
    V = m0.copy()
    for t in range(4000): V = np.clip(V + 0.2/(1+wc)*(wc*(project(V)-V) + (m0 - V)), 0, 1)
    print(f'     weights coh={wc:>3}, prior(m0)=1: dist(V,C) = {np.linalg.norm(V-project(V)):.4f}')
# binary-KL Bregman projection of m0 onto C (sum_i KL on each sentence's Bernoulli)
def kl_proj(p0):
    k = M.shape[0]; eps=1e-12
    def f(l):
        v = np.clip(l@M, eps, 1-eps); return np.sum(v*np.log(v/p0) + (1-v)*np.log((1-v)/(1-p0)))
    r = minimize(f, np.ones(k)/k, method='SLSQP', bounds=[(0,1)]*k,
                 constraints=[{'type':'eq','fun':lambda l: l.sum()-1}], options={'ftol':1e-14,'maxiter':1000})
    return r.x@M
print('     binary-KL proj of m0 onto C =', kl_proj(m0), ' |.-m*| =', np.linalg.norm(kl_proj(m0)-mstar).round(4))
# joint-level I-projection of mu0 onto {mu : mu(Gamma)=1} = conditioning (Csiszar); check by direct minimisation
k = len(allw)
f = lambda m: np.sum(np.where(m>1e-15, m*np.log(np.maximum(m,1e-15)/mu0_all), 0))
r = minimize(f, np.where(mask, 1/mask.sum(), 0), method='SLSQP', bounds=[(0,1) if mk else (0,0) for mk in mask],
             constraints=[{'type':'eq','fun':lambda m: m.sum()-1}], options={'ftol':1e-15,'maxiter':1000})
print('     joint I-projection marginals =', r.x@Mall, ' |.-m*| =', np.linalg.norm(r.x@Mall - mstar).round(6),
      ' KL =', round(r.fun, 6), ' -log mu0(Gamma) =', round(-np.log(mu0_all[mask].sum()), 6))

# (1e) model-holding traders T_k(V) = w_k - V, w_k ~ mu0(.|Gamma): fixed point = empirical mean, coherent
pm = mu0_all[mask]/mu0_all[mask].sum()
for K in [10, 100, 10000]:
    idx = rng.choice(len(models), size=K, p=pm); V = M[idx].mean(0)
    print(f'(1e) K={K:>5} model-traders: |V-m*| = {np.linalg.norm(V-mstar):.4f}, dist(V,C) = {np.linalg.norm(V-project(V)):.2e}')
print('     dist(m0,C) =', np.linalg.norm(m0 - project(m0)).round(4), '-> predicted dist(V,C) = dist(m0,C)/(1+w_c):',
      [round(np.linalg.norm(m0 - project(m0))/(1+wc), 4) for wc in [1, 10, 100]])

# (1f) Bregman (binary-KL) coherence trader on undecided sentences: x = logit(pi_F(P)) - logit(P),
#      claim: x.(w-P) >= D(pi,P) + D(P,pi) for every model w.
und = [i for i in range(len(names)) if 0 < M[:, i].mean() < 1]; Mu = M[:, und]
Dkl = lambda u, v: np.sum(u*np.log(u/v) + (1-u)*np.log((1-u)/(1-v)))
logit = lambda u: np.log(u/(1-u))
worst = np.inf
for _ in range(100):
    P = 0.05 + 0.9*rng.random(len(und)); k = Mu.shape[0]
    f = lambda l: Dkl(np.clip(l@Mu, 1e-9, 1-1e-9), P)
    r = minimize(f, np.ones(k)/k, method='SLSQP', bounds=[(1e-6,1)]*k,
                 constraints=[{'type':'eq','fun':lambda l: l.sum()-1}], options={'ftol':1e-15,'maxiter':1000})
    pi = r.x@Mu; x = logit(pi) - logit(P)
    worst = min(worst, ((Mu - P)@x - (Dkl(pi,P) + Dkl(P,pi))).min())
print('(1f) Bregman/logit coherence trader: min over 100 P and models of [profit - symKL(pi,P)] =', round(worst, 6), '(theorem: >= 0)')
