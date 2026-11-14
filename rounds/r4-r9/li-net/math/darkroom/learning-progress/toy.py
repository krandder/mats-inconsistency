"""Learning progress / compression progress / predictive information in a three-room toy.

Rooms (binary sensory streams):
  D  dark      : constant 0
  N  noise     : i.i.d. fair coin
  S  structure : noisy XOR chain, x_t = x_{t-1} XOR x_{t-2} with prob 1-eps (eps=0.05).
                 Its order-0 and order-1 statistics are exactly uniform: below order 2 it looks like noise.
  S* structure, already learned: same chain, learner pre-trained on N0 samples.

Learners: KT (Beta(1/2,1/2)) Markov models of fixed order k, and a Bayesian mixture over orders 0..D
(uniform prior over orders; the Occam penalty comes from the KT marginal likelihoods).

Objectives (all in bits):
  A  raw prediction error (loss)                          -- Pathak-style curiosity, baseline
  B  error-drop learning progress (Oudeyer-Kaplan), windows tau, offset theta
  C1 Schmidhuber compression progress: recompress the history with old vs new plug-in model
  C2 agent-side expected information gain I(model; x_t | history) (Bayesian surprise / KL-KSA reward)
  C3 realized information gained about the true source = excess code length vs the truth
     (pointwise version of Bialek-Nemenman-Tishby predictive information of the learner)
  E  excess entropy (predictive information of the ergodic source itself), exact and plug-in estimates
"""
import numpy as np
from math import log2
from scipy.special import digamma

LN2 = np.log(2.0)
EPS = 0.05
T = 8192
SEEDS = 16
D = 8
N0 = 20000
rng0 = np.random.default_rng(20261003)


def gen(room, n, rng):
    if room == "D":
        return np.zeros(n, dtype=np.int8)
    if room == "N":
        return rng.integers(0, 2, n).astype(np.int8)
    x = np.zeros(n, dtype=np.int8)
    x[:2] = rng.integers(0, 2, 2)
    flips = rng.random(n) < EPS
    for t in range(2, n):
        x[t] = (x[t - 1] ^ x[t - 2]) ^ flips[t]
    return x


def true_logp(room, x):
    """log2 mu(x_t | x_<t) under the true source."""
    n = len(x)
    if room == "D":
        return np.zeros(n)
    if room == "N":
        return np.full(n, -1.0)
    lp = np.full(n, -1.0)
    pred = x[1:-1] ^ x[:-2]
    lp[2:] = np.where(x[2:] == pred, log2(1 - EPS), log2(EPS))
    return lp


def h2(p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def exp_beta_entropy(a, b):
    """E_{theta~Beta(a,b)} H(Bern theta), in bits."""
    s = a + b
    return (digamma(s + 1) - (a / s) * digamma(a + 1) - (b / s) * digamma(b + 1)) / LN2


class Learner:
    def __init__(self, D):
        self.D = D
        self.c = np.zeros((D + 1, 2 ** D, 2))  # counts[k, context, symbol]
        self.logml = np.zeros(D + 1)           # log2 marginal likelihood of each order
        self.hist = 0                          # last D bits as int (most recent = bit 0)
        self.ks = np.arange(D + 1)
        self.masks = (1 << self.ks) - 1

    def step(self, x):
        ctx = self.hist & self.masks
        n0 = self.c[self.ks, ctx, 0]
        n1 = self.c[self.ks, ctx, 1]
        q = (n1 + 0.5) / (n0 + n1 + 1.0)                 # P(x=1) per order
        px = np.where(x == 1, q, 1 - q)
        post = np.exp2(self.logml - self.logml.max()); post /= post.sum()
        qmix = float(post @ q)
        pmix = qmix if x == 1 else 1 - qmix
        # agent-side expected information gain (before seeing x)
        eh = exp_beta_entropy(n1 + 0.5, n0 + 0.5)
        eig_k = h2(q) - eh
        eig_mix = float(h2(np.array(qmix)) - post @ eh)
        # update
        n0n = n0 + (x == 0); n1n = n1 + (x == 1)
        qn = (n1n + 0.5) / (n0n + n1n + 1.0)
        # Schmidhuber: recompress the history in this context with the old vs new plug-in parameter
        C_old = -(n0n * np.log2(1 - q) + n1n * np.log2(q))
        C_new = -(n0n * np.log2(1 - qn) + n1n * np.log2(qn))
        cp_k = C_old - C_new
        self.c[self.ks, ctx, x] += 1
        self.logml += np.log2(px)
        self.hist = ((self.hist << 1) | int(x)) & ((1 << self.D) - 1)
        return -np.log2(px), -log2(pmix), eig_k, eig_mix, cp_k


def run(room, rng, pretrain=0):
    L = Learner(D)
    if pretrain:
        xp = gen(room, pretrain + T, rng)
        for v in xp[:pretrain]:
            L.step(int(v))
        x = xp[pretrain:]
        lp = true_logp(room, xp)[pretrain:]
    else:
        x = gen(room, T, rng)
        lp = true_logp(room, x)
    out = {k: [] for k in ("loss_k", "loss_mix", "eig_k", "eig_mix", "cp_k")}
    for v in x:
        r = L.step(int(v))
        for key, val in zip(out, r):
            out[key].append(val)
    out = {k: np.array(v) for k, v in out.items()}
    out["true"] = -lp
    return out


ROOMS = [("D", 0), ("N", 0), ("S", 0), ("S*", N0)]
res = {}
for name, pre in ROOMS:
    runs = [run(name.rstrip("*"), np.random.default_rng(1000 * s + 7), pre) for s in range(SEEDS)]
    res[name] = {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0]}
    res[name]["runs"] = runs

checkpoints = [16, 64, 256, 1024, 4096, 8192]
P = print
P(f"T={T} seeds={SEEDS} max order D={D} eps={EPS} pretrain(S*)={N0}\n")

P("== A. Raw prediction error: mean bits/step of the order-mixture over the last 1024 steps ==")
for n, _ in ROOMS:
    P(f"  {n:3s} {res[n]['loss_mix'][-1024:].mean():.4f}")

P("\n== C3. Information gained about the true source, cumulative bits up to t ==")
P("   (= excess code length vs truth; asymptotically (k_eff/2) log2 t, k_eff = parameters the data pin down)")
hdr = "  learner      room " + " ".join(f"{c:>7d}" for c in checkpoints)
P(hdr)
for lab, getter in [("mix0..8", lambda r: r["loss_mix"])] + [
        (f"order{k}", (lambda k: lambda r: r["loss_k"][:, k])(k)) for k in (0, 1, 2, 4, 8)]:
    for n, _ in ROOMS:
        cum = np.cumsum(getter(res[n]) - res[n]["true"])
        P(f"  {lab:10s} {n:4s} " + " ".join(f"{cum[c-1]:7.2f}" for c in checkpoints))

P("\n== C2. Agent-side expected information gain, cumulative bits up to t (computable by the agent) ==")
P(hdr)
for lab, getter in [("mix0..8", lambda r: r["eig_mix"])] + [
        (f"order{k}", (lambda k: lambda r: r["eig_k"][:, k])(k)) for k in (0, 1, 2, 4, 8)]:
    for n, _ in ROOMS:
        cum = np.cumsum(getter(res[n]))
        P(f"  {lab:10s} {n:4s} " + " ".join(f"{cum[c-1]:7.2f}" for c in checkpoints))
P("  per-step rate x t at t=T (approaches k_eff/(2 ln2)=0.72*k_eff for a Bayesian learner):")
for lab, getter in [("mix0..8", lambda r: r["eig_mix"]), ("order8", lambda r: r["eig_k"][:, 8])]:
    P("   " + lab + "  " + "  ".join(f"{n}:{getter(res[n])[-512:].mean() * T:.2f}" for n, _ in ROOMS))

P("\n== C1. Schmidhuber compression progress (plug-in, fixed order), cumulative bits up to t ==")
P(hdr)
for k in (0, 1, 2, 4, 8):
    for n, _ in ROOMS:
        cum = np.cumsum(res[n]["cp_k"][:, k])
        P(f"  order{k:<5d} {n:4s} " + " ".join(f"{cum[c-1]:7.2f}" for c in checkpoints))

P("\n== B. Oudeyer-Kaplan error-drop LP(t) = mean loss in [t-theta-tau, t-theta) - mean loss in [t-tau, t) ==")
P("   per run, then averaged; order-mixture learner; bits/step (x1000)")
for tau, theta in [(8, 8), (32, 32), (128, 128), (32, 512)]:
    row = []
    for t in (64, 256, 1024, 4096, 8192):
        if t - theta - tau < 0:
            row.append("   n/a" * 4); continue
        vals = []
        for n, _ in ROOMS:
            lps = [r["loss_mix"][t - theta - tau:t - theta].mean() - r["loss_mix"][t - tau:t].mean()
                   for r in res[n]["runs"]]
            vals.append(1000 * np.mean(lps))
        row.append(" ".join(f"{v:6.1f}" for v in vals))
    P(f"  tau={tau:4d} theta={theta:4d} | t=64,256,1024,4096,8192 ; each cell D N S S*")
    for t, cell in zip((64, 256, 1024, 4096, 8192), row):
        P(f"     t={t:5d}: {cell}")
P("  cumulative error drop from step 1 to the end (first-16 mean minus last-1024 mean):")
for n, _ in ROOMS:
    v = np.mean([r["loss_mix"][:16].mean() - r["loss_mix"][-1024:].mean() for r in res[n]["runs"]])
    P(f"     {n:3s} {v:+.3f} bits/step")

# ---- E: excess entropy
P("\n== E. Excess entropy E = I(past; future) of the ergodic source (Bialek et al. 2001; Crutchfield-Feldman) ==")
hS = float(h2(np.array(EPS)))
# noisy XOR chain: pair (x_{t-1},x_t) is a Markov chain; transition matrix is doubly stochastic -> uniform pairs
# E = H(X1,X2) - 2 h for an order-2 chain
E_S = 2.0 - 2 * hS
P(f"  exact: D 0.000   N 0.000   S {E_S:.3f}   S* {E_S:.3f} (already learned: unchanged)   entropy rate S {hS:.3f}")
P("  plug-in estimate E_L = L*H(L-1) - (L-1)*H(L) from one run of length n (mean over seeds):")
def block_H(x, L):
    if L == 0: return 0.0
    w = np.lib.stride_tricks.sliding_window_view(x, L) @ (1 << np.arange(L))
    p = np.bincount(w, minlength=2 ** L) / len(w); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
for n_len in (256, 4096, 65536):
    for L in (3, 6, 10):
        vals = []
        for room in ("D", "N", "S"):
            ests = []
            for s in range(8):
                x = gen(room, n_len, np.random.default_rng(55 + s))
                ests.append(L * block_H(x, L - 1) - (L - 1) * block_H(x, L))
            vals.append(np.mean(ests))
        P(f"   n={n_len:6d} L={L:2d}:  D {vals[0]:6.3f}   N {vals[1]:6.3f}   S {vals[2]:6.3f}")

# ---- Costs: one-shot choice of a room for T steps, J = info - lam * cost * T
P("\n== Costs. Choose a room for a horizon t; J = (cumulative info gain, C2 mixture) - lam * cost * t ==")
P("   costs: D 0, N 1, S 1 (units arbitrary). Critical lam* above which D beats S:")
cumS = np.cumsum(res["S"]["eig_mix"]); cumD = np.cumsum(res["D"]["eig_mix"]); cumN = np.cumsum(res["N"]["eig_mix"])
cumSs = np.cumsum(res["S*"]["eig_mix"])
for c in checkpoints:
    P(f"   t={c:5d}  infoS-infoD={cumS[c-1]-cumD[c-1]:6.2f}  lam*={(cumS[c-1]-cumD[c-1])/c:.5f}"
      f"   (S* vs D: {(cumSs[c-1]-cumD[c-1])/c:+.6f})")
P("   continuation value after t: E-gain over the next t steps, per step (all -> 0):")
for c in (1024, 4096):
    P("    " + "  ".join(f"{n}:{(np.cumsum(res[n]['eig_mix'])[2*c-1 if 2*c<=T else T-1]-np.cumsum(res[n]['eig_mix'])[c-1])/c:.5f}"
                          for n, _ in ROOMS) + f"   (window {c}..{min(2*c,T)})")
