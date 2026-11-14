"""Numerical checks for math/pdg-propositional.md.
Propositional PDG: atoms a_1..a_K, axioms (clauses, beta=inf), price table p_j for
sentences phi_j with confidence beta_j.  <<M>> = min_{mu in Delta(Mod Gamma)} sum_j beta_j d(mu(phi_j)||p_j).
"""
import itertools, numpy as np
from scipy.optimize import minimize, linprog
rng = np.random.default_rng(0)
EPS = 1e-12

def worlds(K): return np.array(list(itertools.product([0,1], repeat=K)), dtype=int)

def rand_clause(K, width):
    idx = rng.choice(K, size=width, replace=False); sg = rng.integers(0,2,size=width)
    return lambda W: np.any(W[:,idx]==sg, axis=1)

def rand_sentence(K):
    t = rng.integers(0,4)
    i, j = rng.choice(K, 2, replace=False)
    if t==0: return lambda W: W[:,i]==1
    if t==1: return lambda W: (W[:,i]==1)&(W[:,j]==1)
    if t==2: return lambda W: (W[:,i]==1)|(W[:,j]==0)
    return lambda W: (W[:,i]!=W[:,j])

def bkl(q,p):
    q=np.clip(q,EPS,1-EPS); p=np.clip(p,EPS,1-EPS)
    return q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p))

def inc_of_q(q,p,b): return float(np.sum(b*bkl(q,p)))

def solve_primal(V,p,b,mu0=None):
    """min over mu in simplex on models of sum b d(V^T mu || p). Mirror descent (exponentiated gradient), then polish with SLSQP."""
    n=V.shape[0]; mu=np.ones(n)/n if mu0 is None else mu0
    logit=lambda z: np.log(z/(1-z))
    for t in range(20000):
        q=np.clip(V.T@mu,1e-12,1-1e-12); g=V@(b*(logit(q)-logit(p)))
        mu=mu*np.exp(-0.5*g/ (1+np.max(b)*10)); mu/=mu.sum()
    f=lambda m: inc_of_q(V.T@m,p,b)
    r=minimize(f,mu,method='SLSQP',bounds=[(0,1)]*n,constraints=[{'type':'eq','fun':lambda m:m.sum()-1}],options={'ftol':1e-15,'maxiter':2000})
    mu=np.clip(r.x,0,None); mu/=mu.sum()
    return mu, f(mu)

def solve_dual(V,p,b):
    """max_{x,t} t - sum b log(1-p+p e^{x/b}) s.t. t <= V x  (each model). Returns value and x."""
    m=V.shape[1]
    obj=lambda z: -(z[-1]-np.sum(b*np.log(1-p+p*np.exp(z[:m]/b))))
    cons=[{'type':'ineq','fun':lambda z: V@z[:m]-z[-1]}]
    best=None
    for s in range(4):
        z0=np.r_[rng.normal(size=m),-1.0]
        r=minimize(obj,z0,method='SLSQP',constraints=cons,options={'ftol':1e-15,'maxiter':3000})
        if best is None or r.fun<best.fun: best=r
    return -best.fun, best.x[:m]

def coherent_LP(V,p):
    n=V.shape[0]
    r=linprog(np.zeros(n),A_eq=np.vstack([V.T,np.ones(n)]),b_eq=np.r_[p,1],bounds=[(0,None)]*n)
    return r.status==0

def maxent_given_q(V,q):
    n=V.shape[0]
    f=lambda m: np.sum(np.clip(m,1e-15,None)*np.log(np.clip(m,1e-15,None)))
    r=minimize(f,np.ones(n)/n,method='SLSQP',bounds=[(0,1)]*n,
               constraints=[{'type':'eq','fun':lambda m:m.sum()-1},{'type':'eq','fun':lambda m:V.T@m-q}],options={'ftol':1e-14,'maxiter':3000})
    return r.x

def instance(K,ncl,m):
    W=worlds(K)
    while True:
        cls=[rand_clause(K,rng.integers(2,4)) for _ in range(ncl)]
        ok=np.all(np.stack([c(W) for c in cls]),axis=0) if cls else np.ones(len(W),bool)
        if ok.sum()>=3: break
    Mod=W[ok]
    sents=[rand_sentence(K) for _ in range(m)]
    V=np.stack([s(Mod) for s in sents],axis=1).astype(float)   # n_models x m
    p=rng.uniform(0.05,0.95,size=m); b=rng.uniform(0.5,2.0,size=m)
    return Mod,V,p,b

print("=== (1)+(2)+(3) primal/dual/uniqueness/Pythagoras/bounds on random instances ===")
maxerr=dict(dual=0,xstar_rel_interior=0,uniq=0,pyth_viol=0,pyth_eq=0,pinsker_viol=0,chi2_viol=0)
for K in [4,5,6]:
    for trial in range(5):
        Mod,V,p,b=instance(K,ncl=rng.integers(1,4),m=rng.integers(4,8))
        mu,Inc=solve_primal(V,p,b); q=V.T@mu
        # uniqueness of q*: different random start
        mu2,Inc2=solve_primal(V,p,b,mu0=rng.dirichlet(np.ones(V.shape[0])))
        maxerr['uniq']=max(maxerr['uniq'],np.max(np.abs(V.T@mu2-q)))
        D,x=solve_dual(V,p,b)
        maxerr['dual']=max(maxerr['dual'],abs(D-Inc))
        logit=lambda z: np.log(z/(1-z))
        xs=b*(logit(np.clip(q,1e-9,1-1e-9))-logit(p))
        inter=(q>1e-5)&(q<1-1e-5)   # dual optimum is finite only on coordinates with q* in (0,1)
        if inter.any(): maxerr['xstar_rel_interior']=max(maxerr['xstar_rel_interior'],np.max(np.abs(xs-x)[inter]/(1+np.abs(xs[inter]))))
        if (~inter).any(): print(f"   boundary coords q*={np.round(q[~inter],6)} (sentence forced by axioms+other prices); dual x there = {np.round(x[~inter],1)} (diverges)")
        # Pythagorean dominance: LL(p,w)-LL(q*,w) >= Inc for every model, = on support
        LLp=np.array([inc_of_q(v,p,b) for v in V]); LLq=np.array([inc_of_q(v,q,b) for v in V])
        gain=LLp-LLq
        maxerr['pyth_viol']=max(maxerr['pyth_viol'],max(0,Inc-gain.min()))
        sup=mu>1e-6
        maxerr['pyth_eq']=max(maxerr['pyth_eq'],np.max(np.abs(gain[sup]-Inc)))
        # Pinsker: min_q sum b (q-p)^2 <= Inc/2 ; chi2: Inc <= sum b (q-p)^2/(p(1-p)) at Euclidean-weighted projection
        n=V.shape[0]
        r=minimize(lambda m_: np.sum(b*(V.T@m_-p)**2),np.ones(n)/n,method='SLSQP',bounds=[(0,1)]*n,
                   constraints=[{'type':'eq','fun':lambda m_:m_.sum()-1}],options={'ftol':1e-15})
        d2=r.fun; qe=V.T@r.x
        maxerr['pinsker_viol']=max(maxerr['pinsker_viol'],max(0,d2-Inc/2))
        maxerr['chi2_viol']=max(maxerr['chi2_viol'],max(0,Inc-np.sum(b*(qe-p)**2/(p*(1-p)))))
        print(f"K={K} |Mod|={n:2d} m={V.shape[1]} Inc={Inc:.6f} dual={D:.6f} coherentLP={coherent_LP(V,p)} wdist2={d2:.5f} minGain-Inc={gain.min()-Inc:+.2e}")
print("max errors:",{k:f"{v:.2e}" for k,v in maxerr.items()})

print("\n=== (3a) <<M>>=0 iff coherent (p in Q) ===")
cnt=[0,0]
for K in [4,5,6]:
    for trial in range(10):
        Mod,V,_,b=instance(K,ncl=2,m=6)
        if trial%2==0: p=V.T@rng.dirichlet(np.ones(len(V)))       # coherent table
        else: p=rng.uniform(0.05,0.95,size=V.shape[1])              # random table
        p=np.clip(p,1e-6,1-1e-6)
        mu,Inc=solve_primal(V,p,b)
        coh=coherent_LP(V,p)
        agree=(Inc<1e-7)==coh
        cnt[agree]+=1
        if not agree: print("DISAGREE",K,Inc,coh)
print("agreements:",cnt[1],"disagreements:",cnt[0])

print("\n=== (3b) linear-gain counterexamples ===")
for eps in [0.1,0.01,0.001]:
    # sentences A, notA ; no axioms; prices .5, .5+eps; beta=1
    V=np.array([[1,0],[0,1]],float); p=np.array([.5,.5+eps]); b=np.ones(2)
    mu,Inc=solve_primal(V,p,b)
    G=eps   # sell one share of each: guaranteed gain p_A+p_notA-1
    print(f"eps={eps}: one-share arbitrage gain G={G:.4g}, <<M>>={Inc:.4g} (G>Inc: {G>Inc}); bound sqrt(Inc*sum(y^2/b)/2)={np.sqrt(Inc*2/2):.4g}")
for delta in [1e-2,1e-4,1e-8]:
    # axiom: A valid. price p(A)=delta. one-share gain 1-delta; Inc = log(1/delta)
    V=np.array([[1.0]]); p=np.array([delta]); b=np.ones(1)
    Inc=inc_of_q(np.array([1.0]),p,b)
    print(f"delta={delta}: one-share gain={1-delta:.4f}, <<M>>={Inc:.3f}  (Inc unbounded while gain<=1)")

print("\n=== Frechet / relation trader: A, B, A&B with P(A&B) > P(A) ===")
W=worlds(2); V=np.stack([W[:,0],W[:,1],W[:,0]&W[:,1]],1).astype(float)
p=np.array([.4,.7,.5]); b=np.ones(3)
mu,Inc=solve_primal(V,p,b); q=V.T@mu
G=p[2]-p[0]  # sell A&B, buy A
print(f"p={p}, q*={np.round(q,5)}, <<M>>={Inc:.5f}, relation-trader gain={G:.3f}, Hoeffding lower bound 2G^2/sum(y^2/b)={2*G**2/2:.5f} <= Inc: {2*G**2/2<=Inc+1e-12}")

print("\n=== (2) maxent tie-break: gamma->0 limit of Inc - gamma*H equals maxent given q* ===")
H=lambda m: -np.sum(m[m>1e-15]*np.log(m[m>1e-15]))
for seed in [5,6,7]:
    rng=np.random.default_rng(seed)
    Mod,V,p,b=instance(5,ncl=1,m=3)
    mu,Inc=solve_primal(V,p,b); q=V.T@mu; n=V.shape[0]
    me=maxent_given_q(V,q)
    print(f"seed {seed}: |Mod|={n}, q*={np.round(q,4)}, H(maxent)={H(me):.6f}, H(primal solver's mu*)={H(mu):.6f}  (mu* not unique)")
    for gam in [1e-2,1e-3]:
        def F(th):
            m_=np.exp(th-th.max()); m_/=m_.sum()
            return inc_of_q(V.T@m_,p,b)+gam*np.sum(m_*np.log(m_+1e-300))
        r=minimize(F,np.zeros(n),method='L-BFGS-B',options={'maxiter':20000,'ftol':1e-16,'gtol':1e-12})
        mg=np.exp(r.x-r.x.max()); mg/=mg.sum()
        Ig=inc_of_q(V.T@mg,p,b)
        print(f"   gamma={gam}: ||mu_gamma - maxent||_1={np.abs(mg-me).sum():.2e}, H(mu_gamma)={H(mg):.6f}, Inc(mu_gamma)-<<M>>={Ig-Inc:.2e} <= gamma*log|Mod|={gam*np.log(n):.2e}: {Ig-Inc<=gam*np.log(n)+1e-9}")

print("\n=== (4a) local (pairwise) polytope relaxation gap: XOR triangle ===")
# atoms A,B,C; sentences A!=B, B!=C, A!=C priced 0.9 each, beta=1, no axioms.
W=worlds(3); X=[W[:,0]!=W[:,1],W[:,1]!=W[:,2],W[:,0]!=W[:,2]]
V=np.stack(X,1).astype(float); p=np.full(3,.9); b=np.ones(3)
mu,Inc=solve_primal(V,p,b)
print(f"exact <<M>> = {Inc:.5f}, q*={np.round(V.T@mu,4)} (Q: q1+q2+q3<=2). Pairwise-local relaxation: each XOR is a function of one pair, pair marginals with P(A!=B)=0.9 and singles 1/2 are locally consistent -> relaxed value 0.")
# closed form: q=2/3 each
print("closed form 3*d(2/3||0.9) =",3*bkl(np.array(2/3),np.array(.9)))

print("\n=== (4b) Frank-Wolfe with MaxSAT-style oracle (here brute force) + duality-gap certificate, K=6 ===")
Mod,V,p,b=instance(6,ncl=3,m=7)
mu_ref,Inc_ref=solve_primal(V,p,b)
logit=lambda z: np.log(z/(1-z))
q=V.mean(0)  # start: uniform over models (coherent)
for t in range(1,5001):
    g=b*(logit(np.clip(q,1e-12,1-1e-12))-logit(p))
    s=V[np.argmin(V@g)]           # linear minimisation oracle = weighted MaxSAT over models
    gap=g@(q-s)
    # exact line search on [0,1]
    lo,hi=0.0,1.0
    for _ in range(60):
        mid=(lo+hi)/2; d=q+mid*(s-q); dd=np.clip(d,1e-12,1-1e-12)
        if (b*(logit(dd)-logit(p)))@(s-q)>0: hi=mid
        else: lo=mid
    q=q+lo*(s-q)
    if t in (1,10,100,1000,5000):
        f=inc_of_q(q,p,b)
        print(f"iter {t:5d}: f={f:.7f}  FW gap={gap:.2e}  certified f-<<M>> <= gap; true f-<<M>>={f-Inc_ref:.2e}")
print("q_t is a convex combination of models at every iterate -> coherent on the query set at every step.")

print("\n=== (2b) q-space Pythagoras: for q in Q, f(q)-<<M>> >= sum b d(q||q*)  (training-loss certificate) ===")
rng=np.random.default_rng(11); worst=np.inf
for trial in range(6):
    Mod,V,p,b=instance(5,ncl=2,m=5)
    mu,Inc=solve_primal(V,p,b); qs=V.T@mu
    for _ in range(200):
        q=V.T@rng.dirichlet(np.ones(len(V))*0.3)
        worst=min(worst,(inc_of_q(q,p,b)-Inc)-inc_of_q(q,qs,b))
print("min over 1200 coherent q of [f(q)-<<M>> - D_b(q||q*)] =",f"{worst:.3e}","(>= 0 up to solver error)")

print("\n=== (2c) unpriced sentence psi: coherent interval vs maxent value ===")
rng=np.random.default_rng(5)
Mod,V,p,b=instance(5,ncl=1,m=3)
mu,Inc=solve_primal(V,p,b); q=V.T@mu; n=len(V)
psi=(Mod[:,0]==1)&(Mod[:,3]==1)
A=np.vstack([V.T,np.ones(n)]); rhs=np.r_[q,1]
lo=linprog(psi.astype(float),A_eq=A,b_eq=rhs,bounds=[(0,None)]*n).fun
hi=-linprog(-psi.astype(float),A_eq=A,b_eq=rhs,bounds=[(0,None)]*n).fun
print(f"psi = a1 & a4: coherent extensions give P(psi) in [{lo:.4f}, {hi:.4f}]; maxent mu* gives {mu@psi:.4f}; uniform-over-models frequency {psi.mean():.4f}")
