"""Skeptic checks for pdg-inconsistency-of-the-agent. Run: python3 skeptic.py > skeptic_output.txt
Reuses toy.py. LIR policy = argmax of agent-estimated reducible inconsistency g_r (equal hazards, so W is irrelevant).
S1  generic noise: i.i.d. noise whose law is NOT the uniform atom (a fixed biased law, and a law drawn per seed).
S2  model-class choice: maximum Markov order of the learner (0, 0-1, 0-2, 0-3).
S3  prior concentration a0 of the Dirichlet (KT 0.5, Laplace 1, 0.05).
S4  O1 (surprisal) with a low-entropy noise room and with a safer noise room.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import toy
from toy import Learner, Room, K, H
np.set_printoptions(precision=4, suppress=True)

class LawRoom:
    """i.i.d. room with a given law (generic white-ish noise)."""
    def __init__(self, p, rng): self.p = np.asarray(p, float); self.rng = rng
    def law(self): return self.p
    def step(self): return self.rng.choice(K, p=self.p)

def lir_run(make_rooms, make_learner, T, S, burn=10, seed0=7000):
    """rooms: list of 3 factories (rng, seed)->room, in the order [dark, noise, struct]. Returns shares per quarter."""
    out = np.zeros((4, 3))
    for seed in range(S):
        rng = np.random.default_rng(seed0 + seed); perm = rng.permutation(3)
        rooms = [make_rooms[i](np.random.default_rng(seed * 13 + i), seed) for i in perm]
        Ls = [make_learner() for _ in range(3)]; cache = [L.predict() for L in Ls]
        for t in range(T):
            j = t // burn if t < 3 * burn else int(np.argmax(np.array([c[1] for c in cache]) + 1e-15 * rng.standard_normal(3)))
            q, g, P = cache[j]; x = rooms[j].step(); Ls[j].update(x, P); cache[j] = Ls[j].predict()
            out[min(3, 4 * t // T), perm[j]] += 1
    return out / (S * T / 4)

dark = lambda rng, s: Room('dark', rng)
struct = lambda rng, s: Room('struct', rng)
uniform = lambda rng, s: Room('noise', rng)
biased_fixed = lambda rng, s: LawRoom([0.4, 0.3, 0.2, 0.1], rng)
biased_random = lambda rng, s: LawRoom(np.random.default_rng(424242 + s).dirichlet(np.ones(K)), rng)

def show(name, sh):
    print(f"   {name:58s} " + "  ".join(str(np.round(sh[k], 3)) for k in range(4)))

if __name__ == '__main__':
    T, S = int(sys.argv[1]) if len(sys.argv) > 1 else 12000, int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(f"LIR, equal hazards, T={T}, {S} seeds. Time share [dark, noise, struct] per quarter of the run.\n")
    print("=== S1: is the noise room dismissed when its law is not an Occam atom? ===")
    for am in [0.0, 0.5]:
        mk = lambda am=am: Learner(am)
        for nname, nf in [('uniform noise (= atom)', uniform), ('biased noise [.4,.3,.2,.1]', biased_fixed),
                          ('noise law ~ Dirichlet(1) per seed', biased_random)]:
            show(f"atoms={am}  {nname}", lir_run([dark, nf, struct], mk, T, S))
    print("\n=== S2: model class = max Markov order (atoms 0.5, uniform noise) ===")
    for orders in [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]:
        show(f"orders={orders}", lir_run([dark, uniform, struct], lambda o=orders: Learner(0.5, orders=o), T, S))
    print("   same, flat prior (atoms 0):")
    for orders in [(0,), (0, 1, 2), (0, 1, 2, 3)]:
        show(f"orders={orders} atoms=0", lir_run([dark, uniform, struct], lambda o=orders: Learner(0.0, orders=o), T, S))
    print("\n=== S3: Dirichlet concentration a0, flat prior (atoms 0), uniform noise ===")
    for a0 in [0.05, 0.5, 1.0, 4.0, 16.0]:
        show(f"a0={a0}", lir_run([dark, uniform, struct], lambda a=a0: Learner(0.0, a0=a), T, S))
    print("\n=== S4: O1 (minimise surprisal + beta_phys * hazard), long-run per-step cost, analytic entropy rates ===")
    p_low = np.array([0.91, 0.03, 0.03, 0.03])
    Hs = {'dark': 0.0, 'uniform noise': np.log(K), 'low-entropy noise [.91,.03,.03,.03]': H(p_low),
          'struct': 0.9 * -np.log(0.9) + 0.1 * -np.log(0.1 / 3)}
    for k, v in Hs.items(): print(f"   entropy rate {k:38s} {v:.4f} nats/step")
    print("   -> without the dark room, O1 picks the low-entropy noise room over struct (an environment fact, no parameter).")
    hn, hs = 1e-3, 1e-2
    bstar = (np.log(K) - Hs['struct']) / (-np.log(1 - hs) + np.log(1 - hn))
    print(f"   struct hazard {hs}, uniform-noise hazard {hn}: O1 picks white noise over struct iff beta_phys > {bstar:.1f}")
