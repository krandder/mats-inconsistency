"""Skeptic checks for the EFE note: which hidden choices decide the rooms?

S1  Model class of the parameter prior (Dirichlet vs a discrete class
    {uniform, deterministic}): does it flip the noise-vs-structure ranking?
S2  Markov order of the agent's model (order 1 vs order 0): does the
    "structured" room stay structured?
S3  Planning horizon H (greedy vs H-step committed policies, Monte Carlo):
    does it flip any long-run verdict?
S4  F_amb long-run G is the cross-entropy of the outcome marginal against C,
    so it is blind to structure for every C (closed form check).
Run: python3 skeptic.py   (about 1-2 min)
"""
import numpy as np
from scipy.special import digamma

K = 4; LNK = np.log(K); ROOMS = ["dark", "noise", "struct"]


def ent(p):
    p = p[p > 1e-300]; return float(-(p * np.log(p)).sum())


def nov_dir(alpha):
    a0 = alpha.sum(); pred = alpha / a0
    return ent(pred) - float((pred * (digamma(a0 + 1) - digamma(alpha + 1))).sum())


class DiscreteRow:
    """theta in {uniform (prior w)} U {one-hot e_j (prior (1-w)/K each)}."""
    def __init__(self, w): self.w = w; self.n = 0; self.sym = None; self.mixed = False
    def post_u(self):
        if self.mixed: return 1.0
        if self.n == 0: return self.w
        lu = np.log(self.w) - self.n * LNK; ld = np.log((1 - self.w) / K)
        return float(1 / (1 + np.exp(ld - lu)))
    def pred(self):
        pu = self.post_u()
        if self.n == 0 or self.mixed: return np.full(K, 1 / K) if (self.mixed or self.n == 0) else None
        p = np.full(K, pu / K); p[self.sym] += 1 - pu; return p
    def novelty(self):
        if self.mixed: return 0.0
        return ent(self.pred()) - self.post_u() * LNK
    def update(self, o):
        if self.n > 0 and o != self.sym: self.mixed = True
        self.sym = o if self.n == 0 else self.sym; self.n += 1


class Room:
    def __init__(self, kind, rng):
        self.kind, self.rng, self.o = kind, rng, 0
        perm = rng.permutation(K); self.sigma = np.empty(K, int)
        for i in range(K): self.sigma[perm[i]] = perm[(i + 1) % K]
    def step(self):
        if self.kind == "dark": self.o = 0
        elif self.kind == "noise": self.o = int(self.rng.integers(K))
        else: self.o = int(self.sigma[self.o])
        return self.o


def run(prior, family, lam, cost, order=1, T=20000, seed=0, a0=1.0, w=0.5):
    rng = np.random.default_rng(seed)
    rooms = [Room(k, rng) for k in ROOMS]
    nrow = K if order == 1 else 1
    if prior == "dir":
        tabs = [np.full((nrow, K), a0) for _ in ROOMS]
    else:
        tabs = [[DiscreteRow(w) for _ in range(nrow)] for _ in ROOMS]
    last = [0, 0, 0]; vis = np.zeros((T, 3), bool)
    for t in range(T):
        G = np.empty(3)
        for r in range(3):
            row = last[r] if order == 1 else 0
            if prior == "dir":
                al = tabs[r][row]; pred = al / al.sum(); nv = nov_dir(al)
            else:
                d = tabs[r][row]; pred = d.pred(); nv = d.novelty()
            sal = ent(pred) if family == "F_state" else 0.0
            G[r] = lam * cost[r] - sal - nv
        best = np.flatnonzero(np.isclose(G, G.min(), atol=1e-12))
        r = int(rng.choice(best)); o = rooms[r].step()
        row = last[r] if order == 1 else 0
        if prior == "dir": tabs[r][row, o] += 1
        else: tabs[r][row].update(o)
        last[r] = o; vis[t, r] = True
    return vis[:2000].mean(0), vis[int(0.9 * T):].mean(0)


def fmt(v): return "(" + " ".join(f"{x:.2f}" for x in v) + ")"


EQ = (0.05, 0.05, 0.05); DARKBAD = (0.2, 0.05, 0.05); DARKCHEAP = (0.0, 0.05, 0.05)

print("shares are (dark, noise, struct); 'first2k' = steps 0-1999, 'tail' = last 10% of T=20000\n")
print("== S1. Prior model class (F_amb, order 1). Equal costs and 'C dislikes dark'. ==")
for prior, kw in [("dir", dict(a0=1.0)), ("dir", dict(a0=0.1)), ("disc", dict(w=0.5)), ("disc", dict(w=0.01)), ("disc", dict(w=0.99))]:
    for cname, cost in [("equal", EQ), ("dark bad", DARKBAD)]:
        lam = 1.0
        f, tl = run(prior, "F_amb", lam, cost, **kw)
        print(f"  {prior:4s} {str(kw):12s} costs={cname:8s} lam={lam}: first2k={fmt(f)} tail={fmt(tl)}")
# total novelty available per room under the discrete class
for w in [0.5, 0.01, 0.99]:
    d = DiscreteRow(w); tot = 0
    for o in [2] * 50: tot += d.novelty(); d.update(o)
    rng = np.random.default_rng(3); d2 = DiscreteRow(w); tot2 = 0
    for o in rng.integers(K, size=50): tot2 += d2.novelty(); d2.update(int(o))
    print(f"  discrete class w={w}: total novelty in one row: deterministic stream {tot:.3f} nats, noise stream {tot2:.3f} nats")
print()

print("== S2. Markov order of the model (Dirichlet a0=1, equal costs). ==")
for fam in ["F_state", "F_amb"]:
    for order in [1, 0]:
        f, tl = run("dir", fam, 1.0, EQ, order=order)
        print(f"  {fam:7s} order={order}: first2k={fmt(f)} tail={fmt(tl)}")
print("  -> order 0: the 4-cycle has a uniform marginal, so the 'structured' room is a second noise room.\n")


def mc_horizon(H, lam, cost, family="F_amb", T=4000, a0=1.0, nroll=12, seed=0):
    """Committed H-step policies: pick a room, stay H steps. Value = MC estimate of the
    summed one-step G along rollouts from the agent's own predictive (counts updated in
    imagination, as in sophisticated/deep active inference with parameter learning)."""
    rng = np.random.default_rng(seed)
    rooms = [Room(k, rng) for k in ROOMS]; tabs = [np.full((K, K), a0) for _ in ROOMS]
    last = [0, 0, 0]; vis = []
    t = 0
    while t < T:
        G = np.zeros(3)
        for r in range(3):
            acc = 0.0
            for _ in range(nroll):
                tb = tabs[r].copy(); s = last[r]
                for h in range(H):
                    al = tb[s]; pred = al / al.sum()
                    acc += lam * cost[r] - (ent(pred) if family == "F_state" else 0.0) - nov_dir(al)
                    o = rng.choice(K, p=pred); tb[s, o] += 1; s = o
            G[r] = acc / nroll
        best = np.flatnonzero(np.isclose(G, G.min(), atol=1e-9)); r = int(rng.choice(best))
        for h in range(H):
            o = rooms[r].step(); tabs[r][last[r], o] += 1; last[r] = o; vis.append(r); t += 1
    v = np.array(vis[:T]); tail = v[int(0.75 * T):]
    return np.array([(tail == i).mean() for i in range(3)])


print("== S3. Planning horizon (committed H-step policies, F_amb, Dirichlet a0=1, T=4000, tail = last 25%) ==")
for cname, cost, lam in [("dark cheapest", DARKCHEAP, 1.0), ("dark bad", DARKBAD, 1.0), ("equal", EQ, 1.0)]:
    row = []
    for H in [1, 4, 16]:
        row.append(f"H={H}: {fmt(mc_horizon(H, lam, cost))}")
    print(f"  {cname:13s} lam={lam}: " + "  ".join(row))
print()

print("== S4. F_amb, parameters learned: G = KL(q(o)||C) + E H[P(o|s)] = cross-entropy of the outcome")
print("   marginal m against C, for ANY stream with marginal m. Check on a non-uniform marginal. ==")
rng = np.random.default_rng(11)
# structured stream with marginal m: deterministic cycle 0->1->0->2->0->3 (m = (1/2,1/6,1/6,1/6)),
# needs order-2 context; use the exact per-step predictive under the true process.
cyc = [0, 1, 0, 2, 0, 3]; m = np.bincount(cyc, minlength=K) / len(cyc)
worst = 0
for _ in range(1000):
    C = rng.dirichlet(np.ones(K) * rng.choice([0.3, 1, 3])); lnC = np.log(C)
    g_noise = float((m * (np.log(m) - lnC)).sum()) + ent(m)          # i.i.d. m: risk + ambiguity
    g_struct = float(np.mean([-lnC[o] for o in cyc]))                 # deterministic: risk only
    worst = max(worst, abs(g_noise - g_struct))
print(f"  max |G_struct - G_noise| over 1000 random C = {worst:.2e}  (both equal -E_m ln C)")
