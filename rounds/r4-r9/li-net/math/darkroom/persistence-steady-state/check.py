"""Persistence / nonequilibrium-steady-state framing of the dark room and white-noise room.

Three readings of "the agent is a NESS with a Markov blanket":
  R1 self-evidencing: the system's flow descends its OWN stationary surprisal -log p*.
  R2 entropy bound:   the system keeps the entropy of its sensory states low (min average surprisal).
  R3 persistence:     the system keeps existing; existence costs free energy, which it must harvest.
Toy: three rooms, each a binary sensory stream.
  D dark      constant 0000...
  N noise     iid fair bits
  S structured Markov chain, flips with prob 0.95 (learnable, entropy rate H_b(0.05))
  A alternating 0101... (deterministic structured stream, entropy rate 0)
Units: bits; energy in k_B T ln 2.
Run: python3 check.py  (seconds)
"""
import numpy as np
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)
log2 = np.log2

def Hb(p):
    p = np.clip(p, 1e-300, 1); q = np.clip(1 - p, 1e-300, 1)
    return float(-(p * log2(p) + q * log2(q)))

print("=" * 78)
print("PART A. R1 is an identity: every stationary system descends its own surprisal")
print("=" * 78)

# A1. Linear diffusion dx = -B x dt + sqrt(2D) dW.  Stationary p* = N(0, S), S solves B S + S B^T = 2D.
# Claim (Friston 2019, Parr et al 2020): flow f = (Q - D) grad(-log p*) with Q antisymmetric.
# Here grad(-log p*) = S^{-1} x, so f = -B x requires Q = D - B S; check Q antisymmetric for random B.
from scipy.linalg import solve_continuous_lyapunov
worst = 0.0
for trial in range(1000):
    n = 4
    M = rng.normal(size=(n, n))
    B = M + (abs(np.linalg.eigvals(M).real).max() + 0.5) * np.eye(n)  # stable
    L = rng.normal(size=(n, n)); D = L @ L.T / n + 0.1 * np.eye(n)
    S = solve_continuous_lyapunov(B, 2 * D)
    Q = D - B @ S
    worst = max(worst, np.abs(Q + Q.T).max())
print(f"A1 linear NESS, 1000 random stable systems: max |Q+Q^T| = {worst:.2e}"
      " -> every one is a 'gradient flow on its own surprisal' (identity, no constraint)")

# A2. Discrete chain over (room, sensory bit). Policies differ in where the agent dwells.
rooms = ["D", "N", "S"]
def stream(room, s_prev):
    """P(s'=1 | room, previous bit)."""
    if room == "D": return 0.0
    if room == "N": return 0.5
    if room == "S": return 0.05 if s_prev == 1 else 0.95   # flips w.p. 0.95
def kernel(pi):
    """pi[r, r'] room-move matrix -> transition matrix over states (r, s), index 2r+s."""
    P = np.zeros((6, 6))
    for r in range(3):
        for s in range(2):
            for r2 in range(3):
                p1 = stream(rooms[r2], s if r2 == r else -1) if rooms[r2] != "S" or r2 == r else 0.5
                P[2*r+s, 2*r2+1] += pi[r, r2] * p1
                P[2*r+s, 2*r2+0] += pi[r, r2] * (1 - p1)
    return P
def stationary(P):
    w, v = np.linalg.eig(P.T); i = np.argmin(abs(w - 1)); p = np.real(v[:, i]); return p / p.sum()
def KL(p, q):
    m = p > 0; return float((p[m] * np.log(p[m] / q[m])).sum())

policies = {}
for k, name in enumerate(["dark-dweller", "noise-dweller", "structure-dweller"]):
    pi = np.full((3, 3), 0.01); pi[:, k] = 0.98
    pi /= pi.sum(1, keepdims=True); policies[name] = pi
policies["roamer"] = np.full((3, 3), 1/3)
print("\nA2 chain over (room, bit). Room surprisal -log2 p*(room) under each policy's own NESS,")
print("   and whether KL(p_t || p*) is monotone non-increasing from 200 random starts:")
print(f"   {'policy':18s} {'D':>7s} {'N':>7s} {'S':>7s}  argmin  KL-monotone  H(sensory|own model)")
for name, pi in policies.items():
    P = kernel(pi); ps = stationary(P)
    room_p = ps.reshape(3, 2).sum(1)
    mono = True
    for _ in range(200):
        supp = ps > 1e-15                      # start on the support of p* (D never emits a 1)
        p = np.zeros(6); p[supp] = rng.dirichlet(np.ones(supp.sum())); prev = KL(p, ps)
        for t in range(60):
            p = p @ P; cur = KL(p, ps)
            if cur > prev + 1e-12: mono = False
            prev = cur
    # entropy rate of the sensory+room process under its own stationary law (best model = the chain)
    hr = -sum(ps[i] * P[i, j] * log2(P[i, j]) for i in range(6) for j in range(6) if P[i, j] > 0)
    sur = -log2(room_p)
    am = "tie" if np.ptp(sur) < 1e-9 else rooms[int(np.argmin(sur))]
    print(f"   {name:18s} {sur[0]:7.3f} {sur[1]:7.3f} {sur[2]:7.3f}  {am:>6s}  {str(mono):>11s}  {hr:.3f} bits/step")
print("   -> every persisting policy is a 'free-energy minimiser' w.r.t. its own p*; the dark-dweller's"
      "\n      lowest-surprisal room is D. 'The dark room has high free energy' holds iff the system is not a dark-dweller.")

print("\n" + "=" * 78)
print("PART B. Stream statistics: block-conditional entropies h_k = H(X_t | X_{t-k..t-1})")
print("=" * 78)
streams = {
    "D dark (constant)":   (np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]])),
    "N noise (iid)":       (np.array([0.5, 0.5]), np.array([[0.5, 0.5], [0.5, 0.5]])),
    "S structured (q=.95)": (np.array([0.5, 0.5]), np.array([[0.05, 0.95], [0.95, 0.05]])),
    "A alternating (q=1)": (np.array([0.5, 0.5]), np.array([[0.0, 1.0], [1.0, 0.0]])),
}
def hk(p0, T, k):
    if k == 0:
        return Hb(p0[1])
    # first-order Markov: h_k = h_1 for k>=1
    return float(sum(p0[i] * Hb(T[i, 1]) for i in range(2)))
H = {}
for name, (p0, T) in streams.items():
    H[name] = [hk(p0, T, k) for k in range(3)]
    print(f"   {name:22s} h_0={H[name][0]:.3f}  h_1={H[name][1]:.3f}  h_2={H[name][2]:.3f}  (h_mu = {H[name][2]:.3f})")

print("\nR2 (keep sensory entropy low; expected surprisal under the best model = h_mu):")
order = sorted(H, key=lambda n: H[n][2])
print("   ranking, most preferred first:", " > ".join(f"{n.split()[0]}({H[n][2]:.3f})" for n in order))
print("   -> prefers D (tied with A only if A's model is free); N always last.")

print("\nWhite-noise theorem check (Gibbs): mean work 1 + E[log2 q(X)] for X uniform, over 10^5 random q:")
qs = rng.uniform(1e-6, 1 - 1e-6, size=100000)
w = 1 + 0.5 * (log2(qs) + log2(1 - qs))
print(f"   max over q = {w.max():.2e} (<= 0, = 0 only at q = 1/2). Any model, any memory: noise yields no free energy.")

print("\n" + "=" * 78)
print("PART C. R3 persistence. Fuel = work harvested from the stream by an information ratchet")
print("  (Boyd-Mandal-Crutchfield: <W> <= kT ln2 (h'_mu - h_mu) <= kT ln2 (1 - h_mu) for binary output).")
print("  Ideal protocol tuned to the agent's predictive q extracts W(x) = 1 + log2 q(x | past) per symbol.")
print("  Energy: E <- min(Emax, E + W - c - eps*k); k = model order (memory bits); death at E < 0.")
print("=" * 78)

def work_samples(name, k, n, T_steps):
    p0, T = streams[name]
    # simulate the stream
    x = np.zeros((n, T_steps), dtype=int)
    x[:, 0] = rng.random(n) < p0[1]
    for t in range(1, T_steps):
        x[:, t] = rng.random(n) < T[x[:, t-1], 1]
    if k == 0:
        q1 = p0[1]
        q = np.where(x == 1, q1, 1 - q1)
    else:
        q = np.empty_like(x, dtype=float)
        q[:, 0] = np.where(x[:, 0] == 1, p0[1], p0[0])
        qn = T[x[:, :-1], 1]
        q[:, 1:] = np.where(x[:, 1:] == 1, qn, 1 - qn)
    with np.errstate(divide="ignore"):
        return 1 + log2(q)

def survival(name, k, c, eps, n=4000, T_steps=1000, Emax=20.0, E0=10.0, dark_absent=False):
    if dark_absent and name.startswith("D"):
        W = np.zeros((n, T_steps))
    else:
        W = work_samples(name, k, n, T_steps)
    E = np.full(n, E0); alive = np.ones(n, bool)
    for t in range(T_steps):
        E = np.minimum(Emax, E + W[:, t] - c - eps * k)
        alive &= E >= 0
    return alive.mean()

names = list(streams)
def best_room(c, eps, dark_absent=False, verbose=False, only=None):
    rows = []
    for name in (only or names):
        cands = []
        for k in (0, 1):
            m = (1 - H[name][k]) - c - eps * k
            if dark_absent and name.startswith("D"): m = -c - eps * k
            cands.append((survival(name, k, c, eps, dark_absent=dark_absent), m, -k))
        s_best, mean_net, mk = max(cands)       # best survival, then best margin, then smaller model
        rows.append((name, -mk, s_best, mean_net))
    if verbose:
        for name, k, s, m in rows:
            print(f"     {name:22s} best order k={k}  survival(1000 steps)={s:.3f}  mean net gain={m:+.3f}/step")
    top = max(r[2] for r in rows)
    if top == 0:
        return ["none survive"], ["none survive"]
    winners = [r[0].split()[0] for r in rows if r[2] >= top - 1e-9]
    # tie-break among survivors by mean net gain (persistence margin)
    if len(winners) > 1:
        sub = [r for r in rows if r[0].split()[0] in winners]
        m = max(r[3] for r in sub)
        winners_margin = [r[0].split()[0] for r in sub if r[3] >= m - 1e-9]
        return winners, winners_margin
    return winners, winners

print("\nC1. Dark room = constant stream (the 'dark/constant' room of the brief). c=0.5, eps=0.05:")
w, wm = best_room(0.5, 0.05, verbose=True)
print(f"   survivors-max: {w}; by margin: {wm}")

print("\nC2. Sweep metabolic cost c and memory cost eps (dark = constant stream):")
print(f"   {'c':>5s} {'eps':>6s}  max-survival rooms        by-margin")
for c in (-0.2, 0.2, 0.5, 0.7, 0.9, 1.1):
    for eps in (0.0, 0.01, 0.2):
        w, wm = best_room(c, eps)
        print(f"   {c:5.2f} {eps:6.2f}  {str(w):26s} {wm}")

print("\nC3. Dark room = NO stream (darkness = no input, so no fuel). Sweep c, eps=0.05:")
print(f"   {'c':>5s}  max-survival rooms        by-margin")
for c in (-0.2, 0.0, 0.2, 0.5, 0.7, 0.9):
    w, wm = best_room(c, 0.05, dark_absent=True)
    print(f"   {c:5.2f}  {str(w):26s} {wm}")

print("\nC4. The brief's three rooms only (D = no stream, N, S), eps=0.05. S's break-even c = 1 - h_S - eps = %.3f:" % (1 - H["S structured (q=.95)"][1] - 0.05))
print(f"   {'c':>5s}  max-survival rooms        by-margin")
three = [n for n in names if not n.startswith("A")]
for c in (-0.2, 0.0, 0.3, 0.6, 0.7, 0.9):
    w, wm = best_room(c, 0.05, dark_absent=True, only=three)
    print(f"   {c:5.2f}  {str(w):26s} {wm}")
print("   detail at c=0.0 and c=0.7:")
for c in (0.0, 0.7):
    print(f"   c={c}"); best_room(c, 0.05, dark_absent=True, only=three, verbose=True)
print("   -> S wins exactly when it is the only viable fuel; with food elsewhere (c <= 0) persistence is indifferent.")

print("""
Summary
  R1 (self-evidencing): identity; ranks nothing. Every persisting policy, incl. the dark-dweller, passes.
  R2 (min sensory entropy): picks D among viable rooms; N last for every model (Gibbs).
  R3 (persistence, fuel from the stream): free energy of a stream is 1 - h_mu, maximal at h_mu = 0.
     Constant dark stream: best fuel at zero model cost -> D wins whenever eps > 0, ties A at eps = 0,
       beats S always. N never wins (zero fuel, any model). When c < 0 (food from elsewhere) all survive: tie.
     Dark as no input: D avoided iff c > 0 -- a fact about the room's physics, not about darkness.
""")
