"""Goal as a prior component: a PDG with a preference edge at ordinary confidence (Kelvin, 2026-10-03).

Variables: Z = f* (one of the 16 boolean functions of 2 bits), A = the agent's act (its guess of f*(q),
q = (1,1)), W = 1[A = f*(q)] (deterministic given Z, A), X = the stream seen in the chosen room.
Edges: belief p(Z) (beta_b = 1), likelihood p(X | Z) (beta_b = 1), observation x (hard),
       preference  pref(W=1) = 0.9  (beta_p = 1 by default: same confidence as any other claim).

Inc(x, A) = inf_mu sum_edges beta * KL.  Decompose mu(Z) = mu(W) mu(Z | W): the optimum puts
mu(Z|W) = p(Z|W, x), so
   Inc(x, A) = beta_b * (-log p(x))                                  [surprise of the data]
             + inf_{mu_W} beta_b KL(mu_W || p_W^A(.|x)) + beta_p KL(mu_W || pref)
             = beta_b * (-log p(x)) - (beta_b + beta_p) * log sum_w p_W^A(w|x)^a pref(w)^(1-a),
   a = beta_b / (beta_b + beta_p).
Gap(x) = min_A of the second line = Inc(with preference) - Inc(without preference): the part of the
inconsistency the preference edge is responsible for ("how well the preferred worlds reconcile with
everything else").  All quantities in nats.

Objectives compared, per room and n:
  CUR   plain curiosity: I(all latents of the room; X)  (expected KL on whatever the room's latent is)
  VOI   plain decision VOI: E[max_A P(W=1|x)] - max_A P(W=1)
  INC   raw PDG inconsistency reduction: Inc(prior) - E Inc(x)   (contains the surprise term)
  GAP   preference-attributable reduction: Gap(prior) - E Gap(x)  (Kelvin's objective)
Rooms: D dark (c ~ U{0,1}); N0 known fair noise; N1 noise with unknown bias; S_irr learnable stream of
an independent function g; N_side noisy labels (flip 0.3) of f*(0,0); S learnable stream of f*;
N_goal noisy labels (flip 0.3) of f*(1,1).
"""
import numpy as np
from math import lgamma, log, comb

rng = np.random.default_rng(3)
EPS, FL = 0.05, 0.3
Q = (1, 1)
F = np.array([[[(i >> (2 * a + b)) & 1 for b in (0, 1)] for a in (0, 1)] for i in range(16)])
PREF = np.array([0.1, 0.9])  # pref(W=0), pref(W=1)


def prior_Z(kappa, f0=6):
    p = np.ones(16); p[f0] = np.exp(kappa); return p / p.sum()


def gap(postZ, bb=1.0, bp=1.0):
    a = bb / (bb + bp)
    s1 = float(np.clip(np.sum(postZ * F[:, Q[0], Q[1]]), 0.0, 1.0))  # P(f*(q)=1)
    best = np.inf
    for s in (s1, 1 - s1):  # A = 1 or A = 0: success probability
        if PREF[0] == 0.0:  # hard preference: mu_W must be delta_1; Gap = -beta_b log s, beta_pref drops out
            best = min(best, -bb * np.log(max(s, 1e-300))); continue
        pw = np.array([1 - s, s])
        best = min(best, -(bb + bp) * np.log(np.sum(pw ** a * PREF ** (1 - a))))
    return best


def succ(postZ):
    s1 = np.sum(postZ * F[:, Q[0], Q[1]]); return max(s1, 1 - s1)


def gen_S(f, n):
    x = np.zeros(max(n, 2), int); x[:2] = rng.integers(0, 2, 2); fl = rng.random(n) < EPS
    for t in range(2, n):
        x[t] = F[f, x[t - 2], x[t - 1]] ^ fl[t]
    return x[:n]


def loglik_S(x):
    ll = np.zeros(16)
    for t in range(2, len(x)):
        ll += np.where(F[:, x[t - 2], x[t - 1]] == x[t], np.log(1 - EPS), np.log(EPS))
    return ll


def room_stats(room, n, pz, bb, bp, mc=3000):
    """returns CUR (nats), VOI, INC, GAP, mean surprise; exact enumeration where the room allows it."""
    g0, s0 = gap(pz, bb, bp), succ(pz)
    if room == "D":  # posterior on Z = prior; surprise log 2 once; own latent c: log 2
        return dict(CUR=log(2) if n else 0, VOI=0.0, INC=-bb * (log(2) if n else 0), GAP=0.0, SURP=log(2) if n else 0)
    if room == "N0":
        return dict(CUR=0.0, VOI=0.0, INC=-bb * n * log(2), GAP=0.0, SURP=n * log(2))
    if room == "N1":  # exact over k: Beta(1,1)-Bernoulli sequence; p(seq with k ones) = k!(n-k)!/(n+1)!
        H, I = 0.0, 0.0
        for k in range(n + 1):
            lp = lgamma(k + 1) + lgamma(n - k + 1) - lgamma(n + 2)
            pk = comb(n, k) * np.exp(lp)  # = 1/(n+1)
            H += pk * (-lp)
        # I(theta; X) = H(X) - H(X|theta); H(X|theta) = n * E[h(theta)] = n/2 nats for theta ~ U(0,1)
        I = H - n * 0.5
        return dict(CUR=I, VOI=0.0, INC=-bb * H, GAP=0.0, SURP=H)
    if room in ("N_side", "N_goal"):
        inp = (0, 0) if room == "N_side" else Q
        bit = F[:, inp[0], inp[1]]
        cur = voi = gp = surp = 0.0
        p1 = np.sum(pz * bit)
        for k in range(n + 1):  # k = number of labels equal to 1
            lk1 = k * log(1 - FL) + (n - k) * log(FL)  # likelihood of the sequence given bit = 1
            lk0 = k * log(FL) + (n - k) * log(1 - FL)
            lik = np.where(bit == 1, np.exp(lk1), np.exp(lk0))
            px_seq = np.sum(pz * lik)
            pk = comb(n, k) * px_seq
            post = pz * lik / px_seq
            cur += pk * np.sum(post * np.log(np.maximum(post, 1e-300) / pz))
            voi += pk * (succ(post) - s0)
            gp += pk * (g0 - gap(post, bb, bp))
            surp += pk * (-log(px_seq))
        return dict(CUR=cur, VOI=voi, INC=gp - bb * surp, GAP=gp, SURP=surp)
    # S and S_irr: Monte Carlo
    cur = voi = gp = surp = 0.0
    for _ in range(mc):
        f = rng.choice(16, p=pz) if room == "S" else rng.integers(16)
        x = gen_S(f, n); ll = loglik_S(x)
        prior_here = pz if room == "S" else np.full(16, 1 / 16)  # S_irr: latent g, uniform
        lpx = np.log(np.sum(prior_here * np.exp(ll - ll.max()))) + ll.max() + 2 * log(0.5) * (n >= 2) + log(0.5) * (n == 1)
        post = prior_here * np.exp(ll - ll.max()); post /= post.sum()
        cur += np.sum(post * np.log(np.maximum(post, 1e-300) / prior_here))
        surp += -lpx
        if room == "S":
            voi += succ(post) - s0; gp += g0 - gap(post, bb, bp)
    cur, voi, gp, surp = cur / mc, voi / mc, gp / mc, surp / mc
    return dict(CUR=cur, VOI=voi, INC=gp - bb * surp, GAP=gp, SURP=surp)


ROOMS = ["D", "N0", "N1", "S_irr", "N_side", "S", "N_goal"]
out = []
P = out.append


def table(n, kappa, bb=1.0, bp=1.0, keys=("CUR", "VOI", "INC", "GAP")):
    pz = prior_Z(kappa)
    P(f"-- n={n}, prior kappa={kappa}, beta_belief={bb}, beta_pref={bp}; prior P(success)={succ(pz):.3f}, Gap(prior)={gap(pz,bb,bp):.4f}")
    P("room     " + " ".join(f"{k:>9}" for k in keys) + "   (CUR, INC, GAP in nats; VOI in success prob)")
    res = {}
    for r in ROOMS:
        st = room_stats(r, n, pz, bb, bp); res[r] = st
        P(f"{r:8} " + " ".join(f"{st[k]:9.4f}" for k in keys))
    for k in keys:
        P(f"   argmax {k}: {max(res, key=lambda r: res[r][k])}")
    return res


P("== F. PDG with a preference edge at ordinary confidence (Kelvin's sharper objective)")
for n in [1, 4, 16, 64]:
    table(n, 0.0)
P("\n-- weight check: GAP at n=16 under different preference confidences (belief confidence 1)")
for bp in [0.25, 1.0, 4.0, 100.0]:
    pz = prior_Z(0.0)
    row = {r: room_stats(r, 16, pz, 1.0, bp)["GAP"] for r in ROOMS}
    P(f"   beta_pref={bp:<6}: " + " ".join(f"{r}:{v:.4f}" for r, v in row.items()) + f"   ratio S/N_goal {row['S']/row['N_goal']:.3f}")
P("\n-- informative prior (kappa=2: f0 = XOR favoured, so the agent already leans to A = 0 at q)")
r2 = table(1, 2.0)
r2b = table(4, 2.0)
P("\n-- sign check on weak, goal-relevant evidence (N_goal with flip 0.45 instead of 0.3), kappa=2:")
FL = 0.45
for n in [1, 2, 4, 8, 32]:
    st = room_stats("N_goal", n, prior_Z(2.0), 1.0, 1.0)
    P(f"   n={n:<3} VOI {st['VOI']:+.5f}  GAP {st['GAP']:+.5f}  CUR {st['CUR']:.5f}")
FL = 0.3
P("\n-- wishful thinking: optimal mu_W(success) vs belief P(success), prior kappa=0, A chosen:")
pz = prior_Z(0.0)
for bp in [1.0, 4.0]:
    s = succ(pz); a = 1 / (1 + bp); m = (s ** a * 0.9 ** (1 - a)); m = m / (m + (1 - s) ** a * 0.1 ** (1 - a))
    P(f"   beta_pref={bp}: belief {s:.3f} -> reconciled mu_W(success) {m:.3f}")
P("\n-- costs as one more ordinary edge: pref(rest=1) = 0.6 on R = 1[room is D], beta 1.")
P(f"   choosing a non-dark room costs -log 0.4 + log 0.6 = {log(0.6/0.4):.4f} nats extra inconsistency, once;")
for n in [4, 16, 64]:
    g = room_stats("S", n, prior_Z(0.0), 1.0, 1.0)["GAP"]
    P(f"   S at n={n}: GAP {g:.4f} -> {'leave the dark room' if g > log(1.5) else 'stay in the dark room'}")
P("\n-- the preference's own probability: GAP by room, kappa=0, beta 1/1, pref(W=1) = p")
for p in [0.9, 0.99, 1.0]:
    PREF = np.array([1 - p, p])
    for n in [4, 16, 64]:
        pz = prior_Z(0.0)
        row = {r: room_stats(r, n, pz, 1.0, 1.0)["GAP"] for r in ["D", "N0", "N1", "S_irr", "N_side", "S", "N_goal"]}
        P(f"   p={p:<5} n={n:<3}: " + " ".join(f"{r}:{v:+.4f}" for r, v in row.items()))
P("   hard preference p=1: same table for beta_pref = 0.25 and 100 (beta_pref should drop out):")
for bp in [0.25, 100.0]:
    pz = prior_Z(0.0)
    row = {r: room_stats(r, 16, pz, 1.0, bp)["GAP"] for r in ["D", "N0", "S", "N_goal"]}
    P(f"   beta_pref={bp:<6} n=16: " + " ".join(f"{r}:{v:+.4f}" for r, v in row.items()))
P("   hard preference, sign check (N_goal flip 0.45, kappa=2): GAP = E log P(success|x) - log P(success)")
FL = 0.45
for n in [1, 2, 4, 8, 32]:
    st = room_stats("N_goal", n, prior_Z(2.0), 1.0, 1.0)
    P(f"   n={n:<3} VOI {st['VOI']:+.5f}  GAP {st['GAP']:+.5f}")
FL = 0.3
print("\n".join(out))
