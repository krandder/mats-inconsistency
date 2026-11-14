"""Value of information (Howard 1966) in the three-room toy, plus the goal-as-prior variant.

Rooms (binary streams; the agent's prior over each room's latent is correct, so EVSI is exact Bayes):
  D     dark      : x_t = c, c ~ U{0,1}
  N0    noise     : x_t ~ Bern(1/2), law fully known (white noise, no latent)
  N1    noise     : x_t ~ Bern(theta), theta ~ Beta(1,1) (noise with an unknown bias)
  S     learnable : x_t = f(x_{t-2}, x_{t-1}) xor flip(eps), f uniform over the 16 boolean functions of 2 bits
  S_irr learnable : same as S but driven by an independent function g (irrelevant to the world goal)
  N_rel relevant noise: each step shows a uniform input (a,b) and y = f*(a,b) xor flip(0.4)

Goals (decisions taken after n observation steps in one room):
  G1 own-room forecast : bet 0-1 (or log score) on x_{n+1} of the same room
  G2 world goal        : predict f*(a,b) for a uniform random input; S and N_rel are driven by f*
  G0 no decision       : a single action; VOI is identically 0
Parts: A decision VOI, B remove the decision, C scalar weights, D self-market price moves,
       E goal replaced by a prior component (Kelvin addendum 2026-10-03).
"""
import numpy as np

rng = np.random.default_rng(20261003)
EPS = 0.05
FLIP_REL = 0.4
NS = [1, 2, 3, 4, 6, 8, 16, 64, 256]
MC = 4000
F = np.array([[[(idx >> (2 * a + b)) & 1 for b in (0, 1)] for a in (0, 1)] for idx in range(16)])  # F[f,a,b]
LOG95, LOG05 = np.log(1 - EPS), np.log(EPS)


def gen_S(f, n):
    x = np.zeros(max(n, 2), dtype=int)
    x[:2] = rng.integers(0, 2, 2)
    fl = rng.random(n) < EPS
    for t in range(2, n):
        x[t] = F[f, x[t - 2], x[t - 1]] ^ fl[t]
    return x[:n]


def post_S(x):
    """posterior over the 16 functions after stream x (length >= 0)."""
    cnt = np.zeros((2, 2, 2))
    for t in range(2, len(x)):
        cnt[x[t - 2], x[t - 1], x[t]] += 1
    ll = np.zeros(16)
    for a in (0, 1):
        for b in (0, 1):
            fy = F[:, a, b]
            ll += np.where(fy == 1, cnt[a, b, 1] * LOG95 + cnt[a, b, 0] * LOG05,
                           cnt[a, b, 0] * LOG95 + cnt[a, b, 1] * LOG05)
    p = np.exp(ll - ll.max())
    return p / p.sum()


def post_rel(fstar, n):
    a = rng.integers(0, 2, n); b = rng.integers(0, 2, n)
    y = F[fstar, a, b] ^ (rng.random(n) < FLIP_REL)
    ll = np.zeros(16)
    for t in range(n):
        ll += np.where(F[:, a[t], b[t]] == y[t], np.log(1 - FLIP_REL), np.log(FLIP_REL))
    p = np.exp(ll - ll.max())
    return p / p.sum()


def world_belief(room, n, fstar):
    """posterior over f* after n steps in room; independent rooms leave the prior untouched."""
    if room == "S":
        return post_S(gen_S(fstar, n))
    if room == "N_rel":
        return post_rel(fstar, n)
    if room == "S_irr":
        _ = post_S(gen_S(rng.integers(16), n))  # the agent learns g; g says nothing about f*
    return np.full(16, 1 / 16)


def pf1(post):  # P(f*(a,b)=1) for each input, shape (2,2)
    return np.einsum("f,fab->ab", post, F)


def g2_payoff(post, fstar):
    """realised 0-1 payoff of predicting f*(a,b) at a uniform input, averaged over inputs."""
    p = pf1(post)
    guess = (p > 0.5).astype(int)
    tie = np.isclose(p, 0.5)
    return np.mean(np.where(tie, 0.5, (guess == F[fstar]).astype(float)))


def g1(room, n):
    """own-room forecast of x_{n+1}: expected 0-1 payoff and log2 score, MC under the room's prior."""
    acc, lg = [], []
    for _ in range(MC):
        if room == "D":
            p1, nxt = (0.5, rng.integers(0, 2)) if n == 0 else (1.0, 1)
        elif room == "N0":
            p1, nxt = 0.5, rng.integers(0, 2)
        elif room == "N1":
            th = rng.random(); k = rng.binomial(n, th)
            p1, nxt = (k + 1) / (n + 2), int(rng.random() < th)
        elif room == "S":
            f = rng.integers(16); x = gen_S(f, n + 1)
            if n < 2:
                p1 = 0.5
            else:
                post = post_S(x[:n])
                q = F[:, x[n - 2], x[n - 1]]
                p1 = np.sum(post * np.where(q == 1, 1 - EPS, EPS))
            nxt = x[n]
        pn = p1 if nxt == 1 else 1 - p1
        acc.append(0.5 if p1 == 0.5 else float((p1 > 0.5) == (nxt == 1)))
        lg.append(np.log2(max(pn, 1e-300)))
    return np.mean(acc), np.mean(lg)


def g2(room, n, mc=MC):
    v = []
    for _ in range(mc):
        fs = rng.integers(16)
        v.append(g2_payoff(world_belief(room, n, fs), fs))
    return np.mean(v)


out = []
P = lambda *s: out.append(" ".join(str(z) for z in s))

P("== A. Decision VOI (EVSI = E[payoff | n obs] - payoff with no obs), MC =", MC)
P("G1 own-room forecast of the next symbol, 0-1 payoff (no-info payoff 0.5):")
P("n      " + " ".join(f"{r:>8}" for r in ["D", "N0", "N1", "S"]))
G1V, G1L = {}, {}
for n in [1, 2, 4, 8, 16, 64]:
    row = []
    for r in ["D", "N0", "N1", "S"]:
        a, l = g1(r, n)
        G1V[(r, n)], G1L[(r, n)] = a - 0.5, l + 1
        row.append(a - 0.5)
    P(f"{n:<6} " + " ".join(f"{z:8.3f}" for z in row))
P("same, log2-score VOI (bits gained on the next symbol):")
for n in [1, 4, 16, 64]:
    P(f"{n:<6} " + " ".join(f"{G1L[(r, n)]:8.3f}" for r in ["D", "N0", "N1", "S"]))
P("value of choosing the room (not VOI) at n=16, 0-1:", {r: round(G1V[(r, 16)] + 0.5, 3) for r in ["D", "N0", "N1", "S"]})

P("\nG2 world goal: predict f*(a,b) at a random input (no-info payoff 0.5). VOI:")
rooms2 = ["D", "N0", "N1", "S_irr", "S", "N_rel"]
P("n      " + " ".join(f"{r:>8}" for r in rooms2))
G2V = {}
for n in [2, 4, 8, 16, 64, 256]:
    row = []
    for r in rooms2:
        mc = MC if r in ("S", "N_rel") else 200
        if r == "N_rel" and n == 256:
            mc = 800
        G2V[(r, n)] = g2(r, n, mc) - 0.5
        row.append(G2V[(r, n)])
    P(f"{n:<6} " + " ".join(f"{z:8.3f}" for z in row))
P("EVPI (perfect knowledge of f*) = 0.5. Per-step entropy of streams: D 0, N0 1, N1 ~0.72 (avg), S 0.29+, N_rel 2.97 bits")

P("\nMarginal VOI of the n-th step in S under G2 (VOI(n) - VOI(n_prev)):")
ns = [2, 4, 8, 16, 64, 256]
P(" ".join(f"{ns[i]}:{(G2V[('S', ns[i])] - (G2V[('S', ns[i-1])] if i else 0)) / (ns[i] - (ns[i-1] if i else 0)):.4f}/step" for i in range(len(ns))))

P("\n== B. Remove the decision")
P("G0: one available action -> E[max_a U | obs] - max_a E U = 0 for every room and every n (identity).")
P("Goal-free substitute 1: score the room's own next symbol by log loss (Bernardo 1979: the log-score")
P("EVSI on a latent equals mutual information). Total EIG on each room's own latent, bits:")


def eig_latent(room, n, mc=MC):
    v = []
    for _ in range(mc):
        if room == "D":
            v.append(1.0 if n >= 1 else 0.0); continue
        if room == "N0":
            v.append(0.0); continue
        if room == "N1":
            th = rng.random(); k = rng.binomial(n, th)
            # I(theta; x) = E[log p(x|theta) - log p(x)] (Beta-binomial, sequence-level)
            from math import lgamma, log
            lpx = lgamma(k + 1) + lgamma(n - k + 1) - lgamma(n + 2)
            lpxt = k * log(max(th, 1e-12)) + (n - k) * log(max(1 - th, 1e-12))
            v.append((lpxt - lpx) / log(2)); continue
        f = rng.integers(16); post = post_S(gen_S(f, n))
        v.append(4 + np.sum(post * np.log2(np.maximum(post, 1e-300))))
    return np.mean(v)


for n in [1, 4, 16, 64, 256, 4096]:
    P(f"n={n:<5} " + " ".join(f"{r}:{eig_latent(r, n, 1500 if r != 'S' or n <= 256 else 300):7.3f}" for r in ["D", "N0", "N1", "S"]))
P("-> D capped at 1 bit, S capped at 4 bits (log2 16), N1 grows ~ (1/2) log2 n with no cap: the noisy TV returns.")

P("\n== C. Scalar weights")
r16 = {r: G2V[(r, 16)] for r in rooms2}
P("Scale invariance: G2 VOI x 7 ranks the rooms the same:", sorted(r16, key=lambda r: -r16[r]) == sorted(r16, key=lambda r: -7 * r16[r]))
for c in [0.001, 0.01, 0.05]:
    best = min((c * n / G2V[("S", n)], n) for n in [2, 4, 8, 16, 64, 256] if G2V[("S", n)] > 0)
    P(f"cost c={c}/step in S, dark costs 0: leave the dark room iff stakes m > {best[0]:.3f} (best n={best[1]})")
v1d, v1s = G1V[("D", 16)], G1V[("S", 16)]
v2s = G2V[("S", 16)]
w = v2s / ((v1d - v1s) + v2s)
P(f"Two goals, w*G1 + (1-w)*G2 at n=16: VOI D = w*{v1d:.3f}, S = w*{v1s:.3f} + (1-w)*{v2s:.3f}; dark wins iff w > {w:.3f}")

P("\n== D. Self-market: information is worth what it moves a price by")
P("Query q, price P0 = prior P(q), P_n = posterior. Decision 'buy or sell one share of q at P0' has VOI = E|P_n - P0|.")
chk = []
for _ in range(2000):
    fs = rng.integers(16); post = post_S(gen_S(fs, 6)); Pn = pf1(post)[0, 1]; truth = F[fs, 0, 1]
    pay = truth - 0.5 if Pn > 0.5 else (0.5 - truth if Pn < 0.5 else 0.0)
    chk.append((pay, abs(Pn - 0.5), Pn - 0.5))
chk = np.array(chk)
P(f"check on q = 'f*(0,1)=1', room S, n=6: realised trading payoff {chk[:,0].mean():.4f} vs E|dP| {chk[:,1].mean():.4f}; signed E[dP] {chk[:,2].mean():+.4f} (martingale)")


def price_moves(room, n, mc):
    l1, l2, kl, zo = [], [], [], []
    for _ in range(mc):
        fs = rng.integers(16); post = world_belief(room, n, fs); p = pf1(post).ravel()
        l1.append(np.mean(np.abs(p - 0.5))); l2.append(np.mean((p - 0.5) ** 2))
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        kl.append(np.mean(pc * np.log2(2 * pc) + (1 - pc) * np.log2(2 * (1 - pc))))
        zo.append(g2_payoff(post, fs) - 0.5)
    return np.mean(l1), np.mean(l2), np.mean(kl), np.mean(zo)


P("Q_world = {f*(a,b)=1 : 4 inputs}; mean over queries of E|dP|, E dP^2, E KL(P_n||P0), and 0-1 VOI:")
for n in [4, 16, 64]:
    for r in ["D", "N0", "S_irr", "S", "N_rel"]:
        mc = 1500 if r in ("S", "N_rel") else 100
        a = price_moves(r, n, mc)
        P(f"  n={n:<3} {r:6} L1 {a[0]:.3f}  L2 {a[1]:.3f}  KL {a[2]:.3f}  0-1 {a[3]:.3f}")
P("Q_own_next = {next symbol of the room}: E|dP| after 16 obs -> D 0.5, N0 0, N1 ~E|theta-1/2|-ish, S ~0.45 (see G1 table: same ordering, dark first)")
# Q_own_bits: price of 'x_t = 1' just before x_t is seen vs just after
for r in ["D", "N0", "S"]:
    mv = []
    for _ in range(300):
        if r == "D":
            mv.append(np.r_[0.5, np.zeros(255)]); continue
        if r == "N0":
            mv.append(np.full(256, 0.5)); continue
        f = rng.integers(16); x = gen_S(f, 256); m = np.zeros(256)
        for t in range(256):
            if t < 2:
                p1 = 0.5
            else:
                post = post_S(x[:t]); q = F[:, x[t - 2], x[t - 1]]; p1 = np.sum(post * np.where(q == 1, 1 - EPS, EPS))
            m[t] = abs(x[t] - p1)
        mv.append(m)
    mv = np.array(mv).sum(1).mean()
    P(f"Q_own_bits (every 'x_t=1' as it resolves), total |dP| over 256 steps, room {r}: {mv:.1f}")
P("-> pricing every sentence about your own sensor stream is prediction-error curiosity: noise moves 0.5/step forever.")

P("\n== E. Goal replaced by a prior component (Kelvin addendum)")
P("E1. Preferred variable Z = f* (a world latent), ordinary prior with confidence kappa: p(f) ∝ exp(kappa*[f==f0])")
P("    value of a room = expected KL(posterior on Z || prior on Z) = I(Z; obs), bits. f* drawn from that prior.")


def eig_Z(room, n, kappa, mc):
    pr = np.ones(16); f0 = 6; pr[f0] = np.exp(kappa); pr /= pr.sum()
    v = []
    for _ in range(mc):
        fs = rng.choice(16, p=pr)
        if room == "S":
            x = gen_S(fs, n); lik = post_S(x)  # post_S uses a uniform prior: that is the normalised likelihood
            po = lik * pr; po /= po.sum()
        elif room == "N_rel":
            lik = post_rel(fs, n); po = lik * pr; po /= po.sum()
        else:
            po = pr.copy()
        v.append(np.sum(po * np.log2(np.maximum(po, 1e-300) / pr)))
    return np.mean(v)


for kappa in [0.0, 1.0, 3.0, 6.0]:
    P(f"  kappa={kappa:3.0f} n=16: " + " ".join(f"{r}:{eig_Z(r, 16, kappa, 1200 if r in ('S','N_rel') else 5):6.3f}" for r in ["D", "N0", "N1", "S_irr", "S", "N_rel"]))
P("E2. Same, but Z drifts (f* redrawn with hazard h=0.02/step, an ordinary belief about the world).")
P("    steady-state EIG per step on the current f*, room S vs D/N (D, N0, N1, S_irr: 0 by independence):")
h = 0.02; T = 3000; reps = 30; tot = []
for _ in range(reps):
    fs = rng.integers(16); x = np.zeros(T, int); x[:2] = rng.integers(0, 2, 2)
    b = np.full(16, 1 / 16); gains = []
    for t in range(2, T):
        if rng.random() < h:
            fs = rng.integers(16)
        x[t] = F[fs, x[t - 2], x[t - 1]] ^ (rng.random() < EPS)
        prior_t = (1 - h) * b + h / 16
        lik = np.where(F[:, x[t - 2], x[t - 1]] == x[t], 1 - EPS, EPS)
        b = prior_t * lik; b /= b.sum()
        gains.append(np.sum(b * np.log2(np.maximum(b, 1e-300) / prior_t)))
    tot.append(np.mean(gains[500:]))
P(f"    S: {np.mean(tot):.3f} bits/step (realised KL per step, t>500), D/N0/N1/S_irr: 0.000")
P("E3. Z = all latents of all rooms (c, theta, f, g): EIG = Part B table. N1 pays (1/2)log2 n without cap -> noise wins for n > ~256.")

P("E4. Friston reading: the prior component is a preference over outcomes, held as an ordinary belief.")
P("    Outcome per step: change (x_t != x_{t-1}) or same. Belief rho ~ Beta(mu*s, (1-mu)*s), mu = 0.6.")
P("    Room choice by pragmatic value E_room[log p(o | belief)]; the belief updates on what the agent sees.")
x = gen_S(6, 200000); rS = np.mean(x[1:] != x[:-1])
rates = {"D": 0.0, "N0": 0.5, "S(XOR)": rS}
P(f"    change rates: {', '.join(f'{k} {v:.3f}' for k, v in rates.items())}")


def pick(m):
    sc = {k: r * np.log(m) + (1 - r) * np.log(1 - m) for k, r in rates.items()}
    return max(sc, key=sc.get)


P(f"    with mu=0.6 the agent starts in: {pick(0.6)}")
for s in [1, 2, 10, 100, 1000]:
    k = 0
    while pick(0.6 * s / (s + k)) != "D":
        k += 1
    # after the flip, check absorbing: stay 1000 steps in D
    m = 0.6 * s / (s + k); stay = all(pick(0.6 * s / (s + k + j)) == "D" for j in range(1000))
    P(f"    strength s={s:<5}: forced dark steps before the agent prefers the dark room k*={k:<5} (k*/s={k/s:.2f}); absorbing afterwards: {stay}")
P("    a fixed (never-updated) preference is infinite strength: k* = infinity.")
print("\n".join(out))
