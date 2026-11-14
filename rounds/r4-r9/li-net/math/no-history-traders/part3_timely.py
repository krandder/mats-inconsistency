"""Part 3: block-depth timeliness metrics on a toy family.
Instance: Gamma = facts b_1..b_n (bits, each 1 w.p. q, i.i.d.), query phi = OR(b_1..b_n).
Reference derivation reads one fact per block: phi is derived at block d = index of first 1 (or n, refuted, if none).
Reasoners (history-free: P_t is a function of the block-t state = which facts were read):
  bayes : P_t = E[phi | first t facts]  = 1 if a 1 seen else 1-(1-q)^(n-t)
  ded   : P_t = 1 if derived true, 0 if derived false, else 0.5      (deduction only, no pattern)
  prior : P_t = 1 if derived true, 0 if derived false, else 1-(1-q)^n  (ignores evidence of absence)
Metrics per block t on instances still undecided at t (d > t): Brier, ECE, NENU drift E[P_{t+k}-P_t | bin(P_t)]."""
import numpy as np
rng = np.random.default_rng(1)
n, q, N = 40, 0.03, 200000
bits = rng.random((N, n)) < q
anyone = bits.any(1); first = np.where(anyone, bits.argmax(1) + 1, n)   # derivation depth d
truth = anyone.astype(float)
T = n + 1
def traj(kind):
    P = np.empty((N, T))
    for t in range(T):
        derived_true = anyone & (first <= t); derived_false = (~anyone) & (t >= n)
        und = 1 - (1-q)**(n - t) if kind == 'bayes' else (0.5 if kind == 'ded' else 1-(1-q)**n)
        P[:, t] = np.where(derived_true, 1.0, np.where(derived_false, 0.0, und))
    return P
def ece(p, y, bins=10):
    b = np.minimum((p*bins).astype(int), bins-1); e = 0.0
    for k in range(bins):
        m = b == k
        if m.any(): e += m.mean()*abs(p[m].mean() - y[m].mean())
    return e
print(f'n={n}, q={q}, Pr(phi)={1-(1-q)**n:.4f}, mean derivation depth={first.mean():.1f} blocks')
print(f'{"reasoner":>7} {"t":>3} {"Brier(und)":>10} {"ECE(und)":>9} {"NENU drift k=10":>16} {"Brier(all)":>10}')
for kind in ['bayes', 'ded', 'prior']:
    P = traj(kind)
    for t in [0, 5, 10, 20, 30]:
        und = first > t
        k = min(10, n - t)
        drift = np.abs((P[und, t+k] - P[und, t]).mean())
        print(f'{kind:>7} {t:>3} {np.mean((P[und,t]-truth[und])**2):10.4f} {ece(P[und,t], truth[und]):9.4f} {drift:16.4f} {np.mean((P[:,t]-truth)**2):10.4f}')
# preemptive learning via optional stopping: for bayes, Pr(hit >= p before end) vs P_0
P = traj('bayes'); p = 0.9
hit = (P >= p).any(1)
print(f'bayes: P_0 = {P[0,0]:.4f}; Pr(P_t ever >= {p}) = {hit.mean():.4f}; bound P_0 >= p*Pr(hit) -> {P[0,0]:.4f} >= {p*hit.mean():.4f}')
# self-trust (ensemble form): E[phi | P_t' > p] >= p
for tp in [10, 30]:
    m = P[:, tp] > 0.5
    print(f'bayes self-trust: t\'={tp}: E[phi | P_t\' > 0.5] = {truth[m].mean():.4f} (>= 0.5), frac={m.mean():.3f}')
