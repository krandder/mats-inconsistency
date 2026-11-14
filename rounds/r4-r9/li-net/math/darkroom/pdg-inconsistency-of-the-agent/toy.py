"""Dark room / white-noise room toy for the candidate 'PDG inconsistency of the agent'.
Key: pdg-inconsistency-of-the-agent.  Run: python3 toy.py > output.txt
Parts:
  A. Constitution edges: does the confidence of 'alive=1' matter? (PDG inconsistency, Richardson 2022 eq.1)
  B. Three rooms (dark / white noise / structured), one Bayesian learner per room.
     Per-step: sample inconsistency s_t = -log q(o_t|past) (what the agent's PDG measures),
     law-level inconsistency k_t = KL(law(.|ctx) || q(.|past)) (reducible part, god's eye),
     irreducible part H_t = H(law(.|ctx)), agent-side estimate of reducible part g_t = I(theta; o_t | past).
  C. Decisions: O1 = minimise sample inconsistency + constitution (free energy of the agent PDG),
     O2 = LIR reading: spend control where it reduces inconsistency (max expected reducible inconsistency).
"""
import numpy as np
from scipy.special import digamma, gammaln
np.set_printoptions(precision=4, suppress=True)
K = 4
LOGK = np.log(K)

# ---------------------------------------------------------------- Part A
def kl_b(m, p):
    """binary KL(Bern(m)||Bern(p)), with 0 log 0 = 0 and inf where needed"""
    out = 0.0
    for a, b in ((m, p), (1 - m, 1 - p)):
        if a > 0:
            out += np.inf if b == 0 else a * np.log(a / b)
    return out

def inc_alive(beta_phys, h, beta_c, eps):
    """inf over mu(Alive) of beta_phys*KL(mu||Bern(1-h)) + beta_c*KL(mu||Bern(1-eps)); grid + endpoints"""
    grid = np.concatenate([[0.0, 1.0], np.linspace(1e-9, 1 - 1e-12, 200001)])
    best = np.inf
    for m in grid:
        v = (beta_phys * kl_b(m, 1 - h) if beta_phys != np.inf else (0 if m == 1 - h else np.inf))
        if v == np.inf: continue
        v += beta_c * kl_b(m, 1 - eps) if beta_c > 0 else 0
        best = min(best, v)
    return best

def partA():
    print("=== Part A: constitution edge 'Alive=1' plus the agent's physics edge p(Alive)=1-h ===")
    h = 0.01
    print(f"hazard h={h}; -log(1-h) = {-np.log(1-h):.6f}")
    print("A1 degenerate constitution (eps=0): inconsistency vs beta_c (beta_phys=1)")
    for bc in [0.01, 0.1, 1, 10, 1e3]:
        print(f"   beta_c={bc:>7}: <<M>> = {inc_alive(1, h, bc, 0.0):.6f}")
    print("A2 soft constitution (eps=0.001, 'alive w.p. 0.999'): beta_c now decides the value")
    for bc in [0.01, 0.1, 1, 10, 1e3]:
        print(f"   beta_c={bc:>7}: <<M>> = {inc_alive(1, h, bc, 1e-3):.6f}")
    print("A3 degenerate constitution, physics confidence beta_phys sweep: <<M>> = beta_phys * (-log(1-h))")
    for bp in [0.1, 1, 10]:
        print(f"   beta_phys={bp:>5}: <<M>> = {inc_alive(bp, h, 1.0, 0.0):.6f}")
    print("A4 physics as law (beta_phys=inf) + degenerate constitution: <<M>> =", inc_alive(np.inf, h, 1.0, 0.0),
          "(infinite for every h>0: no room can be compared)")
    # observation edge, gamma = 0 vs 1: mu is forced to delta_o, so IDef = -H(mu)+alpha*H(O)=0
    q = np.array([0.5, 0.3, 0.15, 0.05])
    print("A5 perception PDG {q(O), O=o}: inconsistency = -log q(o) for gamma=0 and gamma=1 (mu forced to delta_o, H=0):",
          [f"{-np.log(x):.4f}" for x in q])
    print()

# ---------------------------------------------------------------- learners
class Learner:
    """Bayesian mixture over Markov orders 0..2 (KT Dirichlet(1/2) per context),
    optionally plus 'simple-law atoms': uniform iid and the K constant streams (Occam)."""
    def __init__(self, atom_mass=0.0, orders=(0, 1, 2), a0=0.5):
        self.orders = orders; self.a0 = a0
        self.counts = [np.zeros((K ** o, K)) for o in orders]
        natoms = 1 + K if atom_mass > 0 else 0
        self.atom_preds = []
        if natoms:
            self.atom_preds = [np.full(K, 1.0 / K)] + [np.eye(K)[c] for c in range(K)]
        w = [(1 - atom_mass) / len(orders)] * len(orders)
        if natoms:  # uniform atom gets half the atom mass, constants share the rest
            w += [atom_mass / 2] + [atom_mass / 2 / K] * K
        with np.errstate(divide='ignore'):
            self.logw = np.log(np.array(w))
        self.hist = []

    def ctx(self, o):
        if o == 0: return 0
        if len(self.hist) < o: return None
        c = 0
        for s in self.hist[-o:]: c = c * K + s
        return c

    def components(self):
        """predictive and expected entropy E_theta[H(p_theta)] for each component"""
        preds, eh = [], []
        for o, C in zip(self.orders, self.counts):
            c = self.ctx(o)
            a = (C[c] if c is not None else np.zeros(K)) + self.a0
            A = a.sum()
            preds.append(a / A)
            eh.append(digamma(A + 1) - np.sum(a / A * digamma(a + 1)))
        for p in self.atom_preds:
            preds.append(p)
            nz = p[p > 0]; eh.append(-np.sum(nz * np.log(nz)))
        return np.array(preds), np.array(eh)

    def predict(self):
        P, EH = self.components()
        w = np.exp(self.logw - np.max(self.logw)); w /= w.sum()
        q = w @ P
        Hq = -np.sum(q[q > 0] * np.log(q[q > 0]))
        g = Hq - w @ EH  # mutual information I(model,theta ; next obs) = expected reducible inconsistency
        return q, max(g, 0.0), P

    def update(self, x, P=None):
        if P is None: P, _ = self.components()
        with np.errstate(divide='ignore'):
            self.logw = self.logw + np.log(P[:, x])
        for o, C in zip(self.orders, self.counts):
            c = self.ctx(o)
            if c is not None: C[c, x] += 1
        self.hist.append(x)

# ---------------------------------------------------------------- rooms
class Room:
    def __init__(self, kind, rng):
        self.kind = kind; self.rng = rng; self.prev = 0
        if kind == 'struct':  # order-1 Markov: next = prev+1 mod K w.p. 0.9, else one of the other 3
            M = np.full((K, K), 0.1 / (K - 1))
            for i in range(K): M[i, (i + 1) % K] = 0.9
            self.M = M
    def law(self):
        if self.kind == 'dark': return np.eye(K)[0]
        if self.kind == 'noise': return np.full(K, 1.0 / K)
        return self.M[self.prev]
    def step(self):
        p = self.law(); x = self.rng.choice(K, p=p); self.prev = x; return x

def H(p):
    nz = p[p > 0]; return -np.sum(nz * np.log(nz))
def KL(p, q):
    m = p > 0; return np.sum(p[m] * np.log(p[m] / q[m]))

def run_alone(kind, atom_mass, T, S):
    s = np.zeros(T); k = np.zeros(T); g = np.zeros(T); h = np.zeros(T)
    for seed in range(S):
        rng = np.random.default_rng(1000 + seed)
        R, L = Room(kind, rng), Learner(atom_mass)
        for t in range(T):
            q, gi, P = L.predict(); lw = R.law()
            x = R.step()
            s[t] += -np.log(q[x]); k[t] += KL(lw, q); g[t] += gi; h[t] += H(lw)
            L.update(x, P)
    return s / S, k / S, g / S, h / S

ROOMS = ['dark', 'noise', 'struct']

def partB(T=2000, S=40):
    print(f"=== Part B: each room alone, T={T} steps, {S} seeds, K={K} symbols ===")
    print("columns: total sample inconsistency S=sum E[-log q(o_t)]; law-level reducible inc. R=sum E[KL(law||q_t)];")
    print("         irreducible I=sum E[H(law)]; agent-estimated reducible G=sum E[I(theta;o_t)]; check S-(R+I)")
    curves = {}
    for am, name in [(0.0, 'flat (orders 0-2, KT)'), (0.5, 'occam (flat + uniform/constant atoms, mass 0.5)'),
                     (1e-6, 'occam (atom mass 1e-6)')]:
        print(f"-- learner: {name}")
        for r in ROOMS:
            s, k, g, h = run_alone(r, am, T, S)
            curves[(am, r)] = (s, k, g, h)
            tail = slice(T // 2, T)
            print(f"   {r:6s}: S={s.sum():9.2f}  R={k.sum():8.2f}  I={h.sum():9.2f}  G={g.sum():8.2f}  "
                  f"S-(R+I)={s.sum()-k.sum()-h.sum():7.2f}   per-step in 2nd half: s={s[tail].mean():.4f} "
                  f"r={k[tail].mean():.2e} g={g[tail].mean():.2e}")
        # log-growth of reducible inconsistency: R(T)-R(T/2) ~ c*log 2 means unbounded c*log T growth
        print("   growth of R between T/2 and T, in units of log 2 (0 = bounded, ~d/2 = parametric regret):",
              {r: round(float(curves[(am, r)][1][T // 2:].sum() / np.log(2)), 2) for r in ROOMS})
    print()
    return curves

def partC(curves, T=2000):
    print("=== Part C1: O1 = free energy of the agent PDG = sample inconsistency + constitution, static room choice ===")
    scen = {'equal hazard': {'dark': 1e-3, 'noise': 1e-3, 'struct': 1e-3},
            'dark starves': {'dark': 1e-2, 'noise': 1e-3, 'struct': 1e-3},
            'struct costly': {'dark': 1e-3, 'noise': 1e-3, 'struct': 2e-3}}
    for am in [0.0, 0.5]:
        print(f"-- learner atom_mass={am}")
        for sname, hz in scen.items():
            for bp in [0.1, 1.0, 10.0]:
                J = {r: curves[(am, r)][0].sum() + bp * T * -np.log(1 - hz[r]) for r in ROOMS}
                best = min(J, key=J.get)
                print(f"   {sname:13s} beta_phys={bp:5}: " + "  ".join(f"{r}={J[r]:8.1f}" for r in ROOMS) + f"  -> {best}")
        # flip threshold for 'dark starves'
        hz = scen['dark starves']
        Sd, Ss = curves[(am, 'dark')][0].sum(), curves[(am, 'struct')][0].sum()
        bstar = (Ss - Sd) / (T * (-np.log(1 - hz['dark']) + np.log(1 - hz['struct'])))
        print(f"   'dark starves': struct beats dark only if beta_phys > {bstar:.3f} (decided by the physics/perception exchange rate)")
    print()

def sequential(policy, atom_mass, hz, T, S, W=1.0, burn=10):
    """Agent chooses a room every step; rooms are unlabelled (same prior for each); first `burn`
    steps in each room are forced (so a myopic policy is not trapped by ignorance of the other rooms).
    policy 'O1':  maximise -H(q_r) - W*(-log(1-h_r))   [O1: predicted next-step sample inconsistency
                  plus constitution; W = beta_phys, the physics/perception exchange rate]
    policy 'LIR': maximise W*g_r - (-log(1-h_r))       [g_r = agent's predicted reducible inconsistency
                  of room r's perception edge = what control over theta can remove; W = number of future
                  ticks the improved model is used (1 = myopic, 1/h = expected remaining life)]."""
    share = np.zeros(3); endR = np.zeros(3); total_s = 0.0
    for seed in range(S):
        rng = np.random.default_rng(5000 + seed)
        perm = rng.permutation(3)  # random labelling, so ties are broken arbitrarily
        rooms = [Room(ROOMS[i], np.random.default_rng(seed * 7 + i)) for i in perm]
        Ls = [Learner(atom_mass) for _ in range(3)]
        cache = [L.predict() for L in Ls]
        cost = np.array([-np.log(1 - hz[ROOMS[i]]) for i in perm])
        for t in range(T):
            if t < 3 * burn:
                j = t // burn
            else:
                if policy == 'O1':
                    score = np.array([-H(c[0]) for c in cache]) - W * cost
                else:
                    score = W * np.array([c[1] for c in cache]) - cost
                score = score + 1e-12 * rng.standard_normal(3)
                j = int(np.argmax(score))
            q, g, P = cache[j]
            x = rooms[j].step(); total_s += -np.log(q[x])
            Ls[j].update(x, P); cache[j] = Ls[j].predict()
            share[perm[j]] += 1
        for jj in range(3):
            endR[perm[jj]] += KL(rooms[jj].law(), cache[jj][0])
    return share / (S * T), endR / S, total_s / S

def partD(T=1500, S=20):
    print(f"=== Part C2: sequential choice, rooms unlabelled, 10 forced visits each, T={T}, {S} seeds ===")
    print("    share = fraction of time in [dark, noise, struct]; final KL = law-level inconsistency left per room")
    hz_eq = {r: 1e-3 for r in ROOMS}
    hz_ds = {'dark': 1e-2, 'noise': 1e-3, 'struct': 1e-3}
    hz_dc = {'dark': 1e-3, 'noise': 1e-3, 'struct': 2e-3}
    for am in [0.0, 0.5, 1e-6]:
        for pol, Ws in [('O1', [1.0, 100.0]), ('LIR', [1.0, 1000.0])]:
            for hn, hz in [('equal', hz_eq), ('dark starves', hz_ds), ('struct costly', hz_dc)]:
                for W in ([1.0] if hn == 'equal' else Ws):
                    sh, endR, ts = sequential(pol, am, hz, T, S, W)
                    print(f"   atoms={am:<6} {pol:3s} hazards={hn:13s} W={W:<6}: share={sh}  "
                          f"final KL={endR}  sample inc={ts:8.1f}")
    print()

def partE(T=20000, S=4):
    print(f"=== Part D: long run, LIR equal hazards, T={T}, {S} seeds: time share [dark, noise, struct] in each quarter ===")
    for am in [0.0, 1e-6, 0.5]:
        rows = []
        for seed in range(S):
            rng = np.random.default_rng(9000 + seed); perm = rng.permutation(3)
            rooms = [Room(ROOMS[i], np.random.default_rng(seed * 11 + i)) for i in perm]
            Ls = [Learner(am) for _ in range(3)]; cache = [L.predict() for L in Ls]
            visits = np.zeros((4, 3))
            for t in range(T):
                j = t // 10 if t < 30 else int(np.argmax(np.array([c[1] for c in cache]) + 1e-15 * rng.standard_normal(3)))
                q, g, P = cache[j]; x = rooms[j].step(); Ls[j].update(x, P); cache[j] = Ls[j].predict()
                visits[min(3, 4 * t // T), perm[j]] += 1
            rows.append(visits / (T / 4))
        print(f"   atoms={am}: " + "  ".join(str(np.round(np.mean(rows, 0)[k], 4)) for k in range(4)))
    print()

if __name__ == '__main__':
    partA()
    c = partB()
    partC(c)
    partD()
    partE()
