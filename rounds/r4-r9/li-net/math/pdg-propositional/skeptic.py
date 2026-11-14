"""Skeptic checks for math/pdg-propositional.md (each section tests one claim)."""
import itertools, numpy as np
from scipy.optimize import minimize, linprog
EPS=1e-12
def worlds(K): return np.array(list(itertools.product([0,1],repeat=K)),dtype=int)
def bkl(q,p):
    q=np.clip(q,EPS,1-EPS); p=np.clip(p,EPS,1-EPS)
    return q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p))
f=lambda q,p,b: float(np.sum(b*bkl(q,p)))
logit=lambda z: np.log(z/(1-z)); sig=lambda z: 1/(1+np.exp(-z))
def solve(V,p,b):
    n=len(V); best=None
    for s in range(5):
        th0=np.random.default_rng(s).normal(size=n)
        r=minimize(lambda th: f(V.T@(np.exp(th-th.max())/np.exp(th-th.max()).sum()),p,b),th0,method='L-BFGS-B',options={'ftol':1e-16,'gtol':1e-12,'maxiter':20000})
        if best is None or r.fun<best.fun: best=r
    m=np.exp(best.x-best.x.max()); m/=m.sum(); return m,best.fun

print("=== S1: R2(c) 'x*.v_w minimal exactly on supp mu*' -- only 'supp mu* is inside argmin' holds ===")
W=worlds(2); V=np.stack([W[:,0],W[:,1],W[:,0]&W[:,1]],1).astype(float)  # A, B, A&B
qstar=np.array([.5,.5,.5]); x=np.array([1.,0.,-1.]); b=np.ones(3)
p=sig(logit(qstar)-x/b)
mu,Inc=solve(V,p,b); q=V.T@mu
print("p =",np.round(p,4)," solver q* =",np.round(q,5)," (target .5,.5,.5)  <<M>> =",round(Inc,6),"= f(target)",round(f(qstar,p,b),6))
# fibre of q*=(.5,.5,.5): mu11=.5, mu10=mu01=0, mu00=.5 -> unique
print("worlds (A,B):",W.tolist()," mu* =",np.round(mu,4))
print("x*.v_w per world:",V@x,"-> argmin set has 3 worlds, supp mu* has 2 (world (0,1) is a minimiser with mu*=0)")
# also show dominance equality on non-support minimiser
gain=np.array([f(v,p,b)-f(v,q,b) for v in V]); print("LL gains - <<M>>:",np.round(gain-Inc,6))

print("\n=== S2: q*_j in {0,1} with interior prices only when Gamma decides phi_j (re-derive check.py's boundary case) ===")
rng=np.random.default_rng(0)
def rand_clause(K,width):
    idx=rng.choice(K,size=width,replace=False); sg=rng.integers(0,2,size=width)
    return lambda W: np.any(W[:,idx]==sg,axis=1)
def rand_sentence(K):
    t=rng.integers(0,4); i,j=rng.choice(K,2,replace=False)
    if t==0: return lambda W: W[:,i]==1
    if t==1: return lambda W: (W[:,i]==1)&(W[:,j]==1)
    if t==2: return lambda W: (W[:,i]==1)|(W[:,j]==0)
    return lambda W: (W[:,i]!=W[:,j])
def instance(K,ncl,m):
    W=worlds(K)
    while True:
        cls=[rand_clause(K,rng.integers(2,4)) for _ in range(ncl)]
        ok=np.all(np.stack([c(W) for c in cls]),axis=0) if cls else np.ones(len(W),bool)
        if ok.sum()>=3: break
    Mod=W[ok]; sents=[rand_sentence(K) for _ in range(m)]
    V=np.stack([s(Mod) for s in sents],axis=1).astype(float)
    p=rng.uniform(0.05,0.95,size=m); b=rng.uniform(0.5,2.0,size=m); return Mod,V,p,b
# replay check.py's rng stream exactly up to the 4th instance (K=4, trial 3)
found=0
for K in [4,5,6]:
    for trial in range(5):
        Mod,V,p,b=instance(K,ncl=rng.integers(1,4),m=rng.integers(4,8))
        # check.py also consumed rng in solve_primal(mu0=dirichlet) and solve_dual; replicate draws
        rng.dirichlet(np.ones(V.shape[0]))
        for s in range(4): rng.normal(size=V.shape[1])
        decided=[j for j in range(V.shape[1]) if V[:,j].min()==V[:,j].max()]
        if decided: print(f"K={K} trial={trial}: sentences decided by Gamma (constant over Mod): {decided}, values {[int(V[0,j]) for j in decided]}"); found+=1
print("instances with a decided sentence:",found)
# direct test: random instances, does any q*_j hit {0,1} without being decided?
rng=np.random.default_rng(42); bad=0; tot=0
for t in range(40):
    Mod,V,p,b=instance(4,ncl=3,m=5); mu,Inc=solve(V,p,b); q=V.T@mu
    for j in range(V.shape[1]):
        if V[:,j].min()!=V[:,j].max():
            tot+=1
            if min(q[j],1-q[j])<1e-4: bad+=1
print(f"undecided coordinates: {tot}, of which q*_j within 1e-4 of 0/1: {bad}")

print("\n=== S3: R1 finiteness wording: two hard prices each compatible, jointly not ===")
# atoms A,B; axiom not(A and B); sentences A, B priced 1, 1
W=worlds(2); Mod=W[~((W[:,0]==1)&(W[:,1]==1))]; V=Mod.astype(float)
print("Q vertices:",V.tolist(),"; p=(1,1): each coordinate alone attainable (q_A=1 at world (1,0)), jointly need world (1,1) excluded -> <<M>> = inf")
print("  f at best vertex (1,0):",f(np.array([1.,0.]),np.array([1.,1.]),np.ones(2)), "(EPS-clipped; true value inf)")

print("\n=== S4: Implication 1 -- f(q_out) is a valid certificate only if q_out in Q ===")
W=worlds(2); V=np.stack([W[:,0],W[:,1],W[:,0]&W[:,1]],1).astype(float); p=np.array([.4,.7,.5]); b=np.ones(3)
mu,Inc=solve(V,p,b)
print(f"<<M>>={Inc:.5f}; incoherent output q_out=p gives f=0 < <<M>> (loss rewards copying the input)")
# non-model proposal: world-vector (A=0,B=0,A&B=1) is not a model of the truth functions
nm=np.array([0.,0.,1.]); q=V.T@mu; best=min(f((1-e)*q+e*nm,p,b) for e in np.linspace(0,1,1001))
print(f"mixing q* with the non-model vector (0,0,1): min f = {best:.5f} < <<M>> = {Inc:.5f}  -> unchecked proposals let the loss go below <<M>>")

print("\n=== S5: R6 FW curvature constant: segment q_t -> vertex reaches q_j in {0,1}; C_f = sup 2/g^2 [f(q+g(s-q)) - f(q) - g grad.(s-q)] ===")
p=np.array([.4,.7,.5]); q=np.array([.45,.6,.4]); s=np.array([1.,1.,1.])
g_=b*(logit(q)-logit(p))
for gm in [0.5,0.9,0.99,0.999999]:
    y=q+gm*(s-q); print(f"  gamma={gm}: 2/g^2*(Bregman) = {2/gm**2*(f(y,p,b)-f(q,p,b)-gm*g_@(s-q)):.3f}  (finite, bounded by ~ sum b log terms)")
print("  local Hessian b/(q(1-q)) at gamma->1 diverges:",[round(float(1/((q+gm*(s-q))[0]*(1-(q+gm*(s-q))[0]))),1) for gm in [0.9,0.99,0.999]])
# per-iterate curvature: Bregman of f is sum b d(y||x) <= gamma^2 sum b (s-x)^2/(x(1-x)) (KL <= chi^2), so C(x) <= 2 sum b (s-x)^2/(x(1-x))
print("  per-iterate chi^2 curvature bound at x:",round(float(2*np.sum(b*(s-q)**2/(q*(1-q)))),3),">= sup over gamma of the curvature quotient above")
print("  global C_f (Jaggi) is infinite: x_j=delta->0, s_j=1, gamma=0.1:",[round(2/0.01*f(np.array([d+0.1*(1-d)]),np.array([.5]),np.ones(1)) - 2/0.01*(f(np.array([d]),np.array([.5]),np.ones(1)) + 0.1*(1-d)*(logit(d)-logit(.5))),2) for d in [1e-2,1e-4,1e-8]])
# FW run with per-iterate certificate h_t <= 2 C_max/(t+2), C_max over iterates
rng=np.random.default_rng(3); Mod,V,p,b=instance(6,ncl=3,m=7)
mu,Inc=solve(V,p,b); q=V.mean(0); undec=V.min(0)!=V.max(0); Cmax=0; viol=0
for t in range(1,3001):
    g=b*(logit(np.clip(q,1e-12,1-1e-12))-logit(p)); s=V[np.argmin(V@g)]
    Cmax=max(Cmax,2*np.sum((b*(s-q)**2/np.clip(q*(1-q),1e-300,None))[undec]))
    lo,hi=0.0,1.0
    for _ in range(60):
        mid=(lo+hi)/2; d=np.clip(q+mid*(s-q),1e-12,1-1e-12)
        if (b*(logit(d)-logit(p)))@(s-q)>0: hi=mid
        else: lo=mid
    q=q+lo*(s-q)
    if f(q,p,b)-Inc > 2*Cmax/(t+2)+1e-9: viol+=1
print(f"  FW K=6 3000 iters: C_max over iterates = {Cmax:.2f}, min_j q_t,j(1-q_t,j) stayed >= {np.min((q*(1-q))[undec]):.3f} at end; violations of h_t <= 2C_max/(t+2): {viol}; final h = {f(q,p,b)-Inc:.2e}")
