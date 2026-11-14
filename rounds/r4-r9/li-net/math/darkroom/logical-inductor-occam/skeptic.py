"""Skeptic checks for logical-inductor-occam (2026-10-03).

Question: is there a hidden fitted parameter that decides the 'white noise: solved' verdict?
Candidates varied here:
  D      = precision (max dyadic depth) of the iid traders' frequency grid. toy.py fixes D = 4.
  theta  = bias of the noise coin. toy.py uses exactly 1/2, which is a 4-bit trader in its class.
The model (Occam prior 2^-codelength, Kelly/Bayes mixture, change-point variants) is the toy's.
Run: cd this dir && OMP_NUM_THREADS=8 python3 skeptic.py > skeptic_output.txt
"""
import numpy as np
LN2 = np.log(2.0)
np.seterr(all="ignore")

def gamma_len(n):
    return 2 * int(np.floor(np.log2(n))) + 1

def iid_class(D):
    th, ln = [0.0, 1.0], [4, 4]
    for d in range(1, D + 1):
        for j in range(1, 2 ** d, 2):
            th.append(j / 2 ** d); ln.append(2 + gamma_len(d + 1) + (d - 1))
    return np.array(th), np.array(ln, float)

def hb(p):
    p = np.clip(p, 0, 1)
    r = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.nan_to_num(r)

# ---------- S1: passive noise room, iid traders only, exact from counts ----------
# (periodic traders die within ~30 steps on noise; they only shift normalisation by O(1) bits.
#  The periodic block gets its toy prior mass, so the iid block's share is the toy's.)
per_mass = sum(2 ** L * 2.0 ** -(2 + gamma_len(L) + L) for L in range(2, 11))
def noise_room(D, theta, T, seed=1):
    th, ln = iid_class(D)
    logw = -ln * LN2
    lZ = np.logaddexp(np.logaddexp.reduce(logw), np.log(per_mass))
    logw = logw - lZ
    lt, l1t = np.log(th), np.log1p(-th)
    x = (np.random.default_rng(seed).random(T) < theta).astype(int)
    k = np.concatenate([[0], np.cumsum(x)])
    # EIG at every step up to 4000, then on a log grid (trapezoid in n; EIG ~ c/n is smooth)
    ns = np.unique(np.concatenate([np.arange(0, min(T, 4000)),
                                   np.geomspace(4000, T - 1, 3000).astype(int)])) if T > 4000 \
        else np.arange(T)
    eig = np.empty(len(ns))
    for a, n in enumerate(ns):
        lp = logw + np.where(k[n] > 0, k[n] * lt, 0.0) + np.where(n - k[n] > 0, (n - k[n]) * l1t, 0.0)
        w = np.exp(lp - np.logaddexp.reduce(lp)); m = np.sum(w * th)
        eig[a] = hb(m) - np.sum(w * hb(th))
    cum = np.concatenate([[0], np.cumsum((eig[1:] + eig[:-1]) / 2 * np.diff(ns))]) + eig[0]
    def regret(n):  # log2 P_theta(x_1:n) / M(x_1:n): what the 'true' coin trader takes from the market
        kk = k[n]
        lp = logw + np.where(kk > 0, kk * lt, 0.0) + np.where(n - kk > 0, (n - kk) * l1t, 0.0)
        lM = np.logaddexp.reduce(lp)
        lP = kk * np.log(theta) + (n - kk) * np.log1p(-theta)
        return (lP - lM) / LN2
    return ns, eig, cum, regret

print("=== S1: noise room payout vs trader-grid precision D and coin bias theta ===")
print("cum EIG = O3 (plannable payout, bits); regret = O4 (log2 wealth of the true-coin trader)")
print("structured room, toy class: O3 = 8.62, O4 = 17.28 bits (toy output.txt, T=4000)\n")
T = 10 ** 6
for theta in (0.5, 1 / np.pi):
    for D in (4, 8, 12, 16):
        ns, eig, cum, regret = noise_room(D, theta, T)
        row = []
        for n in (100, 1000, 4000, 10 ** 4, 10 ** 5, 10 ** 6 - 1):
            i = np.searchsorted(ns, n); row.append(f"{cum[min(i, len(cum)-1)]:6.2f}/{regret(n):6.2f}")
        print(f" theta={theta:.4f} D={D:2d}  O3/O4 at n=100,1e3,4e3,1e4,1e5,1e6: " + "  ".join(row)
              + f"   EIG at 1e6 = {eig[-1]:.1e}")
    print()

# ---------- S2: toy.py's market (change-point traders included), class precision D varied ----------
def build(D):
    th_iid, ln_iid = iid_class(D)
    kinds, pars, lens = [], [], []
    for t_, l_ in zip(th_iid, ln_iid):
        kinds.append("iid"); pars.append(float(t_)); lens.append(l_)
    for L in range(2, 11):
        for code in range(2 ** L):
            kinds.append("per"); pars.append(tuple((code >> b) & 1 for b in range(L)))
            lens.append(2 + gamma_len(L) + L)
    Hn = len(kinds); lens = np.array(lens, float)
    lw = -lens * LN2; lw -= np.logaddexp.reduce(lw)
    tab = np.zeros((2520, Hn))
    for i, (k_, p_) in enumerate(zip(kinds, pars)):
        tab[:, i] = p_ if k_ == "iid" else np.array(p_)[np.arange(2520) % len(p_)]
    return lw, tab

KMAX = 200000
qk = np.array([2.0 ** -gamma_len(k) for k in range(1, KMAX + 1)])
logq = np.log(qk)
tailS = np.concatenate([np.cumsum(qk[::-1])[::-1][1:], [0.0]])
def logS(t):
    return 0.0 if t == 0 else np.log(tailS[t - 1])

class Market:  # identical update rule to toy.py's Market(change_points=True)
    def __init__(self, lw, tab):
        self.lw, self.tab = lw, tab; Hn = len(lw)
        self.t = 0; self.logL = np.zeros(Hn); self.logA = np.full((2, Hn), -np.inf)
    def predict(self):
        t, lw = self.t, self.lw; p = self.tab[t % 2520]
        base = lw + np.log(.5) + self.logL + np.logaddexp(0.0, logS(t + 1))
        now = lw + np.log(.5) + self.logL + logq[t] + np.log(.5)
        ws = [base] + [np.logaddexp(now, lw + 2 * np.log(.5) + self.logA[c]) for c in (0, 1)]
        prs = [p, np.zeros_like(p), np.ones_like(p)]
        L = np.concatenate(ws); P = np.concatenate(prs)
        w = np.exp(L - np.logaddexp.reduce(L)); m1 = float(np.sum(w * P))
        return m1, float(hb(m1) - np.sum(w * hb(P)))
    def update(self, x):
        t = self.t; p = self.tab[t % 2520]
        for c in (0, 1):
            self.logA[c] = np.logaddexp(self.logA[c], self.logL + logq[t]) if x == c \
                else np.full(len(self.lw), -np.inf)
        self.logL = self.logL + np.log(np.where(x == 1, p, 1 - p)); self.t += 1

def stream(name, T, theta=0.5, seed=0):
    if name == "dark": return np.zeros(T, int)
    if name == "noise": return (np.random.default_rng(seed).random(T) < theta).astype(int)
    if name == "structured": return np.array([1, 0, 1, 1, 0, 0, 1, 1, 1])[np.arange(T) % 9]
    raise KeyError

def run_agent(lw, tab, lam, cost, theta, T):
    rooms = ["dark", "noise", "structured"]
    mks = {r: Market(lw, tab) for r in rooms}; xs = {r: stream(r, T, theta) for r in rooms}
    seq = []
    for t in range(T):
        vals = {}
        for r in rooms:
            vals[r] = lam * mks[r].predict()[1] - cost[r]
        b = max(rooms, key=lambda r: (vals[r], -rooms.index(r)))
        mks[b].update(xs[b][mks[b].t]); seq.append(b)
    tot = {r: seq.count(r) for r in rooms}; tail = seq[-1000:]
    return tot, {r: tail.count(r) for r in rooms}

print("=== S2a: passive O3 (cum ex-ante EIG, bits) per room, toy market incl. change-points, T=4000 ===")
for D in (4, 12):
    lw, tab = build(D)
    out = []
    for r, th in (("dark", .5), ("noise", .5), ("noise", 1 / np.pi), ("structured", .5)):
        mk = Market(lw, tab); xs = stream(r, 4000, th); s = 0.0
        for x in xs:
            s += mk.predict()[1]; mk.update(x)
        out.append(f"{r}{'(1/pi)' if th != .5 else ''}={s:.2f}")
    print(f" D={D:2d}: " + "  ".join(out))

print("\n=== S2b: greedy agent argmax lambda*EIG - cost, rooms dark/noise/structured, T=6000 ===")
print(" (toy.py: D=4, theta=1/2.)   last-1000 = rooms chosen in the final 1000 steps")
for D in (4, 12):
    lw, tab = build(D)
    for theta in (0.5, 1 / np.pi):
        for label, lam, cost in (("C equal costs", 1.0, {"dark": 0, "noise": 0, "structured": 0}),
                                 ("A costs 0/1/1, lambda=1e4", 1e4, {"dark": 0, "noise": 1, "structured": 1})):
            tot, tail = run_agent(lw, tab, lam, cost, theta, 6000)
            print(f" D={D:2d} theta={theta:.3f} {label:26s} visits {tot}  last-1000 {tail}")

# ---------- S3: the change-point prior's entropy (Orseau-Lattimore-Hutter 2013, Prop. 11-13) ----------
# Undiscounted KL-KSA value (= total O3) is finite iff the prior has finite entropy. toy.py's
# change-point prior q(k) = 2^-gamma_len(k) has finite entropy; a universal-style 2^-K(k)
# (proxy: q(k) ~ 1/(k log2(k)^2), same order as 2^-K(k) for most k up to log factors) does not.
print("\n=== S3: entropy sum_k<=N q(k) log2 1/q(k) of the change-point prior (summed per dyadic level) ===")
J = np.arange(1, 10 ** 6 + 1, dtype=float)               # level j: k in [2^j, 2^(j+1))
eg = 2.0 ** (-J - 1) * (2 * J + 1)                       # Elias gamma: 2^j values of mass 2^-(2j+1)
# heavy prior: q(k) = c / (k (log2 k)^2): level mass ~ c ln2 / j^2 ... use per-level mass m_j = 1/(j(j+1)),
# spread uniformly over 2^j values -> -log2 q = j + log2(j(j+1)); normalised (sum m_j = 1)
mj = 1 / (J * (J + 1)); hv = mj * (J + np.log2(J * (J + 1)))
for lev in (10, 30, 100, 10 ** 3, 10 ** 6):
    print(f"  k up to 2^{lev:<8d}  Elias-gamma (toy): {eg[:lev].sum():6.3f} bits   1/(k log^2 k)-type: {hv[:lev].sum():6.3f} bits")
