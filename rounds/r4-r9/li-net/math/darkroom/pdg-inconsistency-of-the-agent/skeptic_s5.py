"""S5: the 'noisy TV' flip. Alphabet K=16. Struct room uses 2 symbols (alternate 0<->1 w.p. 0.9, entropy 0.325 nats);
noise room is i.i.d. over all 16 symbols. LIR (argmax agent-estimated reducible inconsistency), equal hazards.
Run: python3 skeptic_s5.py T S > skeptic_s5_output.txt"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import toy
toy.K = 16
from skeptic import lir_run, show, LawRoom
from toy import Learner, Room
KK = 16
class Alt:
    def __init__(self, rng): self.rng = rng; self.prev = 0
    def law(self):
        p = np.zeros(KK); p[1 - self.prev] = 0.9; p[self.prev] = 0.1; return p
    def step(self):
        x = self.rng.choice(KK, p=self.law()); self.prev = x; return x
class Law16(LawRoom):
    def step(self): return self.rng.choice(KK, p=self.p)
dark = lambda rng, s: Room('dark', rng)
alt = lambda rng, s: Alt(rng)
uni = lambda rng, s: Law16(np.full(KK, 1 / KK), rng)
rnd = lambda rng, s: Law16(np.random.default_rng(777 + s).dirichlet(np.ones(KK)), rng)
if __name__ == '__main__':
    T, S = int(sys.argv[1]), int(sys.argv[2])
    print(f"K=16, LIR, equal hazards, T={T}, {S} seeds, orders 0-1. Share [dark, noise, struct] per quarter.")
    for a0 in [0.5, 0.05]:
        for am in [0.0, 0.5]:
            for nn, nf in [('uniform noise (= atom)', uni), ('noise, law ~ Dirichlet(1)', rnd)]:
                show(f"a0={a0} atoms={am} {nn}", lir_run([dark, nf, alt], lambda a=am, c=a0: Learner(a, orders=(0, 1), a0=c), T, S))
