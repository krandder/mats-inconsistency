"""Empowerment toy: dark room, white-noise room, passive structure, controllable rooms.

n-step empowerment E_n(s0) = max_{p(a_1..a_n)} I(A_1..A_n ; S_{n}) (open-loop, Salge et al. 2013 sec 4.4.3),
computed with Blahut-Arimoto on the exact channel p(sensor at step n | action sequence).
All logs base 2 (bits). numpy only. Run: python3 toy.py  (seconds).
"""
import itertools
import numpy as np

np.set_printoptions(precision=3, suppress=True)
K = 8                    # sensor alphabet / ring size
ACTS = (-1, 0, +1)       # left, stay, right
HORIZONS = range(1, 7)


def blahut_arimoto(P, iters=3000, tol=1e-12):
    """Capacity (bits) of channel P[a, s] = p(s|a). Returns (C, p_a, q_s)."""
    P = np.asarray(P, float)
    m = P.shape[0]
    p = np.full(m, 1.0 / m)
    for _ in range(iters):
        q = p @ P
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(P > 0, P * np.log2(P / q), 0.0).sum(1)   # KL(P_a || q)
        pn = p * np.exp2(d)
        pn /= pn.sum()
        if np.abs(pn - p).max() < tol:
            p = pn
            break
        p = pn
    q = p @ P
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(P > 0, P * np.log2(P / q), 0.0).sum(1)
    C = float(p @ d)
    return max(C, 0.0), p, q


def kl_rows(P, q):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(P > 0, P * np.log2(P / q), 0.0).sum(1)


# ---------------------------------------------------------------- rooms
# A room = (init_state, step(state, action) -> list[(prob, state)], sense(state) -> np.array over K')
def ring_step(s, a):
    return [(1.0, (s + a) % K)]


def make_rooms(rng):
    rooms = {}
    const = np.eye(K)[0]
    unif = np.full(K, 1.0 / K)
    pattern = rng.permutation(K)          # a fixed, learnable periodic stream (period K)

    rooms["dark (sensor constant)"] = (0, ring_step, lambda s: const)
    rooms["dark + proprioception"] = (0, ring_step, lambda s: np.eye(K)[s])
    rooms["white noise (iid, action-independent)"] = (0, ring_step, lambda s: unif)
    # passive structured stream: state = (pos, phase); sensor = pattern[phase]; phase advances each step
    rooms["passive periodic stream"] = (
        (0, 0),
        lambda st, a: [(1.0, ((st[0] + a) % K, (st[1] + 1) % K))],
        lambda st: np.eye(K)[pattern[st[1]]],
    )
    rooms["lit ring (controllable)"] = (0, ring_step, lambda s: np.eye(K)[s])
    for eps in (0.25, 0.75):
        rooms[f"lit ring + noise eps={eps}"] = (
            0, ring_step, (lambda e: (lambda s: (1 - e) * np.eye(K)[s] + e * unif))(eps))
    # "noisy TV with a remote": sensor = fixed random hash (64 symbols) of the whole action history
    H = 64
    table = {}

    def tv_sense(hist):
        if hist not in table:
            table[hist] = rng.integers(H)
        return np.eye(H)[table[hist]]
    rooms["hash TV (sensor = random hash of action history)"] = (
        (), lambda h, a: [(1.0, h + (a,))], tv_sense)
    # dark corridor of length 3, then a lit ring: positions 0..2 dark, >=3 lit (ring of K after)
    L = 3

    def cor_step(s, a):
        if s < L:
            return [(1.0, max(0, s + a))]
        return [(1.0, L + ((s - L + a) % K))]

    def cor_sense(s):
        return const if s < L else np.eye(K)[s - L]
    rooms["dark corridor (exit 3 steps, first new sensation 4) -> lit ring"] = (0, cor_step, cor_sense)
    return rooms


def channel(room, n):
    s0, step, sense = room
    rows = []
    for seq in itertools.product(ACTS, repeat=n):
        dist = {s0: 1.0}
        for a in seq:
            nd = {}
            for s, p in dist.items():
                for pp, s2 in step(s, a):
                    nd[s2] = nd.get(s2, 0.0) + p * pp
            dist = nd
        rows.append(sum(p * sense(s) for s, p in dist.items()))
    return np.array(rows)


def part_a(rooms):
    print("A. n-step empowerment E_n (bits), exact channel, |A|=3, open loop")
    names = list(rooms)
    print(f"{'room':55s}" + "".join(f"  n={n}" for n in HORIZONS))
    table = {}
    for nm in names:
        vals = [blahut_arimoto(channel(rooms[nm], n))[0] for n in HORIZONS]
        table[nm] = vals
        print(f"{nm:55s}" + "".join(f"{v:6.2f}" for v in vals))
    print("  upper bound n*log2(3):", " ".join(f"{n*np.log2(3):.2f}" for n in HORIZONS))
    return table


def part_b(table):
    print("\nB. external room costs: J = E_n - lam * cost; which room wins? (n=3)")
    cost = {"dark (sensor constant)": 0.0, "white noise (iid, action-independent)": 0.5,
            "passive periodic stream": 0.5, "lit ring (controllable)": 1.0}
    for lam in (0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 5.0):
        J = {k: table[k][2] - lam * c for k, c in cost.items()}
        best = max(J.values())
        winners = [k for k, v in J.items() if abs(v - best) < 1e-9]
        print(f"  lam={lam:4.1f}: winners = {winners}")
    lam_star = table["lit ring (controllable)"][2] / 1.0
    print(f"  switch point lam* = E_3(lit)/(c_lit - c_dark) = {lam_star:.3f}: above it the dark room wins.")
    print("  passive stream never wins: it ties dark at E=0 and costs more.")


def part_c():
    print("\nC. endogenous cost: lit ring drains 1 energy/step, energy e0=3, at 0 the agent is dead")
    print("   (absorbing, sensor constant). Dark room: no drain, sensor constant.")
    e0 = 3
    const = np.eye(K + 1)[K]

    def step(st, a):
        pos, e = st
        if e == 0:
            return [(1.0, st)]
        return [(1.0, ((pos + a) % K, e - 1))]

    def sense(st):
        return const if st[1] == 0 else np.eye(K + 1)[st[0]]
    lit = ((0, e0), step, sense)
    dark = (0, ring_step, lambda s: np.eye(K + 1)[K])
    for n in HORIZONS:
        El = blahut_arimoto(channel(lit, n))[0]
        Ed = blahut_arimoto(channel(dark, n))[0]
        verdict = "lit" if El > Ed + 1e-9 else "tie (indifferent between death and the dark room)"
        print(f"  n={n}: E(lit, e0={e0})={El:.2f}  E(dark)={Ed:.2f}  -> {verdict}")


def part_d(rng):
    print("\nD. empowerment estimated from data: plug-in channel from N samples per action, n=1")
    print("   true values: white noise 0.00, lit ring log2(3)=1.585")
    for Kn in (8, 64, 1024):
        for N in (10, 100, 1000):
            samples = rng.integers(Kn, size=(3, N))
            counts = np.stack([np.bincount(r, minlength=Kn) for r in samples]).astype(float)
            est = {}
            for lab, a0 in (("plug-in", 0.0), ("Dirichlet a0=1", 1.0), ("Dirichlet a0=1/K", 1.0 / Kn)):
                Pemp = (counts + a0) / (counts + a0).sum(1, keepdims=True)
                est[lab] = blahut_arimoto(Pemp)[0]
            print(f"  noise K={Kn:5d} N={N:5d}: " + "  ".join(f"{k}={v:.3f}" for k, v in est.items()))
            # same estimators on a controllable room with the same alphabet: action a -> symbol a, true E=log2(3)
            cc = np.zeros((3, Kn)); cc[np.arange(3), np.arange(3)] = N
            ec = {lab: blahut_arimoto((cc + a0) / (cc + a0).sum(1, keepdims=True))[0]
                  for lab, a0 in (("plug-in", 0.0), ("Dirichlet a0=1", 1.0), ("Dirichlet a0=1/K", 1.0 / Kn))}
            print(f"  ctrl  K={Kn:5d} N={N:5d}: " + "  ".join(f"{k}={v:.3f}" for k, v in ec.items()))


def part_e(rooms):
    print("\nE. min-max (redundancy = capacity) check: E = min_q max_a KL(p(S|a) || q)")
    print("   = inconsistency control over A can force on the PDG {tau(S|A) beta=inf, q(S) beta=1}")
    for nm in ["dark (sensor constant)", "white noise (iid, action-independent)",
               "passive periodic stream", "lit ring (controllable)", "lit ring + noise eps=0.25",
               "dark corridor (exit 3 steps, first new sensation 4) -> lit ring"]:
        P = channel(rooms[nm], 3)
        C, p, q = blahut_arimoto(P)
        mx = kl_rows(P, q).max()
        qu = np.full(P.shape[1], 1.0 / P.shape[1])
        print(f"  {nm:50s} C={C:.4f}  max_a KL(.||q*)={mx:.4f}  max_a KL(.||uniform)={kl_rows(P, qu).max():.4f}")


def part_f(rng):
    print("\nF. belief control instead of world control (LIR control over parameters, passive rooms)")
    print("   learner: order-1 Markov predictor with KT (add-1/2) counts, prior = uniform over K.")
    print("   reducible = T*log2(K) - cumulative log-loss (bits); final = log-loss rate in last 200 steps.")
    T = 4000
    pattern = rng.permutation(K)
    streams = {"dark": np.zeros(T, int), "white noise": rng.integers(K, size=T),
               "periodic": pattern[np.arange(T) % K]}
    for nm, x in streams.items():
        cnt = np.full((K, K), 0.5)
        prev = 0
        losses = []
        for t in range(T):
            pr = cnt[prev] / cnt[prev].sum()
            losses.append(-np.log2(pr[x[t]]))
            cnt[prev, x[t]] += 1
            prev = x[t]
        losses = np.array(losses)
        red = T * np.log2(K) - losses.sum()
        print(f"  {nm:12s}: reducible total={red:9.1f} bits  final rate={losses[-200:].mean():.3f}"
              f"  steps until rate<0.1 bit: {int(np.argmax(np.convolve(losses, np.ones(50)/50, 'valid') < 0.1)) if (np.convolve(losses, np.ones(50)/50, 'valid') < 0.1).any() else 'never'}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rooms = make_rooms(rng)
    table = part_a(rooms)
    part_b(table)
    part_c()
    part_d(np.random.default_rng(1))
    part_e(rooms)
    part_f(np.random.default_rng(2))
