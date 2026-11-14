"""Dark room vs white-noise room for a Bayes-mixture market (finite analogue of a logical
inductor / Solomonoff predictor) and for agents that pick rooms by its traders' profit.

Run: cd this dir && OMP_NUM_THREADS=4 python3 toy.py > output.txt   (about 1 min)
All logs base 2 in printed numbers.
"""
import numpy as np

LN2 = np.log(2.0)
np.seterr(divide="ignore", invalid="ignore", over="ignore", under="ignore")

def gamma_len(n):  # Elias gamma code length of n >= 1
    return 2 * int(np.floor(np.log2(n))) + 1

# ---------------- traders (hypotheses) with Occam prior 2^-codelength ----------------
hyp_kind, hyp_par, hyp_len = [], [], []
# iid Bernoulli(theta): theta in {0,1} (constants) and dyadic j/2^d, d=1..4
for th in (0.0, 1.0):
    hyp_kind.append("iid"); hyp_par.append(th); hyp_len.append(2 + 1 + 1)
for d in range(1, 5):
    for j in range(1, 2 ** d, 2):
        hyp_kind.append("iid"); hyp_par.append(j / 2 ** d); hyp_len.append(2 + gamma_len(d + 1) + (d - 1))
# periodic patterns, period L = 2..10, every pattern
for L in range(2, 11):
    for code in range(2 ** L):
        pat = tuple((code >> b) & 1 for b in range(L))
        hyp_kind.append("per"); hyp_par.append(pat); hyp_len.append(2 + gamma_len(L) + L)
H = len(hyp_kind)
hyp_len = np.array(hyp_len, float)
logw = -hyp_len * LN2
logw -= np.logaddexp.reduce(logw)            # normalise to a proper prior
is_iid = np.array([k == "iid" for k in hyp_kind])
theta = np.array([p if k == "iid" else 0.0 for k, p in zip(hyp_kind, hyp_par)])
per_idx = np.where(~is_iid)[0]
Lmax = 10
# table: per_pred[t mod 2520][i] for periodic hyps (lcm(1..10)=2520)
LCM = 2520
pred_table = np.zeros((LCM, H))
pred_table[:, is_iid] = theta[is_iid]
for i in per_idx:
    pat = np.array(hyp_par[i]); pred_table[:, i] = pat[np.arange(LCM) % len(pat)]

def find_hyp(kind, par):
    for i, (k, p) in enumerate(zip(hyp_kind, hyp_par)):
        if k == kind and p == par:
            return i
    raise KeyError

# change-point prior q(k) = 2^-gamma_len(k) on k >= 1 (sums to 1)
KMAX = 200000
qk = np.array([2.0 ** -gamma_len(k) for k in range(1, KMAX + 1)])
logq = np.log(qk)                                    # logq[k-1] = log q(k)
tailS = np.concatenate([np.cumsum(qk[::-1])[::-1][1:], [0.0]])  # tailS[t-1] = sum_{k>t} q(k)
def logS(t):  # log sum_{k>t} q(k), t >= 0
    return 0.0 if t == 0 else np.log(tailS[t - 1])

# ---------------- streams ("rooms") ----------------
def stream(name, T, seed=0):
    if name == "dark":
        return np.zeros(T, int)
    if name == "noise":
        return np.random.default_rng(seed).integers(0, 2, T)
    if name == "structured":
        pat = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1]); return pat[np.arange(T) % 9]
    if name == "think":   # dark room + a self-posed computable problem (deductive stream)
        pat = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1]); return pat[np.arange(T) % 10]
    if name == "pseudorandom":  # deterministic LCG bit, period far beyond the trader class
        x, out = 12345, np.zeros(T, int)
        for t in range(T):
            x = (1103515245 * x + 12345) % 2 ** 31; out[t] = (x >> 30) & 1
        return out
    raise KeyError(name)

TRUE = {"dark": find_hyp("iid", 0.0), "noise": find_hyp("iid", 0.5),
        "structured": find_hyp("per", (1, 0, 1, 1, 0, 0, 1, 1, 1)),
        "think": find_hyp("per", (0, 1, 1, 0, 1, 0, 0, 1, 1, 1)),
        "pseudorandom": find_hyp("iid", 0.5)}  # best trader in class

def hb(p):
    p = np.clip(p, 0, 1)
    with np.errstate(all="ignore"):
        r = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.nan_to_num(r)

class Market:
    """Bayes mixture over base traders, each with 'never switch' (1/2) and
    'switch to constant c at time k' (1/2 * q(k) * 1/2) variants, change_points=True.
    Kelly traders: wealth of a component = its posterior weight."""
    def __init__(self, change_points=True):
        self.cp = change_points
        self.t = 0
        self.logL = np.zeros(H)                   # log P_i(x_1:t)
        self.logA = np.full((2, H), -np.inf)      # switched-to-c mass (incl. q(k)), c=0,1
        self.logM = 0.0                            # log M(x_1:t)
    def components(self):
        t = self.t
        p = pred_table[t % LCM]
        if not self.cp:
            return [(logw + self.logL, p)]
        base = logw + np.log(0.5) + self.logL + np.logaddexp(0.0, logS(t + 1))
        now = logw + np.log(0.5) + self.logL + logq[t] + np.log(0.5)   # k = t+1
        comps = [(base, p)]
        for c in (0, 1):
            comps.append((np.logaddexp(now, logw + np.log(0.5) + np.log(0.5) + self.logA[c]),
                          np.full(H, float(c))))
        return comps
    def predict(self):
        comps = self.components()
        lw = np.concatenate([c[0] for c in comps]); pr = np.concatenate([c[1] for c in comps])
        lZ = np.logaddexp.reduce(lw); w = np.exp(lw - lZ)
        m1 = float(np.sum(w * pr))
        eig = float(hb(m1) - np.sum(w * hb(pr)))   # I(hypothesis; next bit) in bits
        return m1, eig, w, pr
    def update(self, x):
        m1, eig, w, pr = self.predict()
        t = self.t
        p = pred_table[t % LCM]
        lik = np.where(x == 1, p, 1 - p)
        if self.cp:
            for c in (0, 1):
                if x == c:
                    self.logA[c] = np.logaddexp(self.logA[c], self.logL + logq[t])
                else:
                    self.logA[c] = np.full(H, -np.inf)
        self.logL = self.logL + np.log(lik)
        mx = m1 if x == 1 else 1 - m1
        self.logM += np.log(mx)
        self.t += 1
        return m1, eig, w, pr

def true_trader_logwealth(mk, i):
    """log2 of Kelly wealth multiple of the 'never-switching' trader i (start = prior share)."""
    return (mk.logL[i] - mk.logM) / LN2

def occam_cap(i, cp=True):
    return (-(logw[i] + (np.log(0.5) if cp else 0.0))) / LN2

print("Traders:", H, " Occam code lengths (bits, after normalisation):")
for r, i in TRUE.items():
    print(f"  {r:13s} best trader {hyp_kind[i]} {hyp_par[i]}  -log2 prior = {occam_cap(i):.2f}")

# =============== Part 1: passive watching, beliefs and profits ===============
T = 4000
checkpoints = [1, 10, 30, 100, 300, 1000, 4000]
rooms = ["dark", "noise", "pseudorandom", "structured", "think"]
print("\n=== Part 1: watch each room for T steps (market with change-point traders) ===")
summary = {}
for r in rooms:
    xs = stream(r, T); mk = Market(True)
    eigs, losses, expmoney = [], [], []
    rows = []
    for t in range(T):
        m1, eig, w, pr = mk.update(xs[t])
        eigs.append(eig)
        mx = m1 if xs[t] == 1 else 1 - m1
        losses.append(-np.log2(mx))
        # expected money profit, at market prices, of a trader buying 1 share of each
        # component's favourite side: E_M[1{x}-price] = 0 identically; record max |.|
        expmoney.append(abs((m1 * (1 - m1)) - ((1 - m1) * m1)))
        if t + 1 in checkpoints:
            rows.append((t + 1, m1, eig, sum(losses), sum(eigs),
                         true_trader_logwealth(mk, TRUE[r])))
    summary[r] = dict(eig=np.array(eigs), loss=np.array(losses), mk=mk)
    print(f"\n-- room: {r}   (Occam cap for best trader = {occam_cap(TRUE[r]):.2f} bits;"
          f" max |E_M money profit| = {max(expmoney):.1e})")
    print("   t   P(x=1 before obs)  EIG_t(bits)  cum logloss  cum EIG   log2 wealth of best trader")
    for (t, m1, eig, cl, ce, lw) in rows:
        print(f"{t:5d}   {m1:8.4f}          {eig:9.2e}   {cl:9.2f}  {ce:8.2f}   {lw:8.2f}")
    if r in ("noise", "pseudorandom"):
        print(f"   cum logloss - T*1 bit (excess over the stream's entropy) = {sum(losses) - T:.2f}")

# =============== Part 1b: noise with a non-Occam (continuous KT) market ===============
print("\n=== Part 1b: noise room, KT (Beta(1/2,1/2)) market vs simple trader 'theta=1/2' ===")
from math import lgamma
xs = stream("noise", 10 ** 6, seed=1)
c1 = np.cumsum(xs)
for n in [10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6]:
    k = int(c1[n - 1])
    logKT = lgamma(k + .5) + lgamma(n - k + .5) - 2 * lgamma(.5) - lgamma(n + 1) + lgamma(1.0)
    # log Beta(k+1/2,n-k+1/2)/Beta(1/2,1/2): lgamma(k+.5)+lgamma(n-k+.5)-lgamma(n+1) - (2lgamma(.5)-lgamma(1))
    lw = (-n * np.log(2) - logKT) / LN2
    print(f"  n={n:8d}  log2 wealth of theta=1/2 trader vs KT = {lw:7.2f}   (0.5*log2 n = {0.5*np.log2(n):.2f})")

# =============== Part 2: objectives that rank rooms ===============
print("\n=== Part 2: what each objective prefers (costs per step: dark 0, think 0, others 1) ===")
cost = {"dark": 0.0, "think": 0.0, "noise": 1.0, "pseudorandom": 1.0, "structured": 1.0}
for Th in [100, 1000, 4000]:
    print(f"\n horizon T={Th}")
    print("  room          criterion   E_M[money]  realised log-profit  cum ex-ante EIG   cost*T")
    for r in rooms:
        ig = summary[r]["eig"][:Th].sum()
        # realised log-profit of the market's traders = cum logloss saved by the best trader
        lp = summary[r]["loss"][:Th].sum() - (Th * (1.0 if r in ("noise", "pseudorandom") else 0.0))
        print(f"  {r:13s}  (no rank)   {0.0:8.2f}    {lp:12.2f}       {ig:10.2f}     {cost[r]*Th:7.0f}")
# lambda thresholds: lambda*IG - cost*T, structured vs dark
print("\n exchange rate lambda (cost units per bit) needed for structured to beat dark,"
      " objective = lambda*IG - cost*T")
for Th in [20, 100, 1000, 4000, 10 ** 5]:
    Th_eff = min(Th, T)
    ig_s = summary["structured"]["eig"][:Th_eff].sum(); ig_d = summary["dark"]["eig"][:Th_eff].sum()
    lam = (cost["structured"] - cost["dark"]) * Th / (ig_s - ig_d)
    print(f"  T={Th:6d}: IG_struct={ig_s:6.2f} IG_dark={ig_d:5.2f}  lambda* = {lam:9.1f}"
          + ("   (IG saturated, lambda* grows linearly in T)" if Th > T else ""))

# =============== Part 3: sequential greedy agent ===============
def run_agent(lam, cost, rooms, T=3000):
    mks = {r: Market(True) for r in rooms}
    xs = {r: stream(r, T) for r in rooms}
    visits = {r: [] for r in rooms}
    last_switch = 0; seq = []
    for t in range(T):
        vals = {}
        for r in rooms:
            m1, eig, _, _ = mks[r].predict()
            vals[r] = lam * eig - cost[r]
        best = max(rooms, key=lambda r: (vals[r], -rooms.index(r)))
        x = xs[best][mks[best].t]; mks[best].update(x)
        visits[best].append(t); seq.append(best)
    last = {r: (v[-1] if v else None) for r, v in visits.items()}
    return {r: len(v) for r, v in visits.items()}, last, seq

print("\n=== Part 3: greedy agent, each step picks argmax_r lambda*EIG_r - cost_r (T=3000) ===")
for scen, rs, cst in [
    ("A: world = dark/noise/pseudorandom/structured, costs 0/1/1/1",
     ["dark", "noise", "pseudorandom", "structured"], cost),
    ("B: add 'think' (dark room + self-posed problem), cost 0",
     ["dark", "noise", "pseudorandom", "structured", "think"], cost),
    ("C: all costs equal (0)",
     ["dark", "noise", "pseudorandom", "structured"], {r: 0.0 for r in cost}),
]:
    print("\n" + scen)
    lams = [0.0, 1.0, 10.0, 100.0, 1e4] if "C" not in scen[:2] else [1.0]
    for lam in lams:
        n, last, seq = run_agent(lam, cst, rs)
        tail = seq[-500:]
        tailmix = {r: tail.count(r) for r in rs}
        print(f"  lambda={lam:8.0f}  visits {n}  last visit {last}  last-500 mix {tailmix}")
