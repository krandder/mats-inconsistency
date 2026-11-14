"""Expected free energy (EFE) on the three-room toy: dark, white noise, structured.

Rooms (K symbols per observation):
  D  dark:       o_t = 0 forever
  N  noise:      o_t ~ Uniform(K), i.i.d.
  S  structured: o_{t+1} = sigma(o_t), sigma a fixed K-cycle unknown to the agent

Agent model of every room: first-order Markov over symbols, each row of the
transition table Dirichlet(a0)-distributed and learned from counts. Two
*observationally equivalent* ways to write that same predictive model as an
active-inference generative model (they assign identical probability to every
observation sequence, so model evidence can never tell them apart):

  F_state: hidden state s_{t+1} = o_{t+1} (A = identity); randomness of the next
           symbol lives in the transition B (state uncertainty).
  F_amb:   hidden state s_{t+1} = o_t (known, deterministic B); randomness of the
           next symbol lives in the likelihood A (ambiguity).

EFE in the outcome form (Da Costa et al. 2020, eqs. 13-16):
  G = -E_q(o)[ln p~(o)]  -  salience  -  novelty
  salience = I(s;o) under q (expected info gain about hidden states)
  novelty  = I(theta;o) (expected info gain about the Dirichlet parameters)
With flat C over symbols the first term is the constant ln K. Cost enters as a
second outcome modality with ln p~(cost) = -lam * c_room (the only place a cost
can enter in active inference: prior preferences C). Policy precision gamma
multiplies G uniformly and does not change argmin, so we select greedily.
"""
import numpy as np
from scipy.special import digamma

K = 4
LN_K = np.log(K)
ROOMS = ["dark", "noise", "struct"]


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def novelty(alpha):
    """I(theta; o) for theta ~ Dir(alpha), o ~ Cat(theta): H(pred) - E H(Cat theta)."""
    a0 = alpha.sum()
    pred = alpha / a0
    eh = float((pred * (digamma(a0 + 1) - digamma(alpha + 1))).sum())
    return ent(pred) - eh


def efe_terms(alpha_row, family, known_params=False):
    pred = alpha_row / alpha_row.sum()
    sal = ent(pred) if family == "F_state" else 0.0   # state revealed by o vs state already known
    nov = 0.0 if known_params else novelty(alpha_row)
    return sal, nov


class Room:
    def __init__(self, kind, rng):
        self.kind, self.rng, self.o = kind, rng, 0
        self.sigma = None
        if kind == "struct":  # a single K-cycle
            perm = rng.permutation(K)
            self.sigma = np.empty(K, int)
            for i in range(K):
                self.sigma[perm[i]] = perm[(i + 1) % K]

    def step(self):
        if self.kind == "dark":
            self.o = 0
        elif self.kind == "noise":
            self.o = int(self.rng.integers(K))
        else:
            self.o = int(self.sigma[self.o])
        return self.o


def simulate(family, a0, lam, cost, T=20000, seed=0, known=False):
    rng = np.random.default_rng(seed)
    rooms = [Room(k, rng) for k in ROOMS]
    counts = [np.full((K, K), a0, float) for _ in ROOMS]
    if known:  # parameters known exactly: put huge counts on the true rows
        counts[0][:, :] = 1e-9; counts[0][:, 0] = 1e9
        counts[1][:, :] = 1e9
        counts[2][:, :] = 1e-9
        for i in range(K):
            counts[2][i, rooms[2].sigma[i]] = 1e9
    last = [0, 0, 0]
    visits = np.zeros((T, 3), bool)
    ties = 0
    for t in range(T):
        G = np.empty(3)
        for r in range(3):
            sal, nov = efe_terms(counts[r][last[r]], family, known)
            G[r] = lam * cost[r] - sal - nov  # + ln K constant dropped
        best = np.flatnonzero(np.isclose(G, G.min(), atol=1e-12))
        ties += len(best) > 1
        r = int(rng.choice(best))  # exact ties broken uniformly at random
        o = rooms[r].step()
        if not known:
            counts[r][last[r], o] += 1
        last[r] = o
        visits[t, r] = True
    frac = visits.mean(0)
    tail = visits[int(0.9 * T):].mean(0)
    # last time the agent was outside the dark room
    outside = np.flatnonzero(~visits[:, 0])
    settle = int(outside[-1]) + 1 if len(outside) else 0
    return frac, tail, settle, ties


def main():
    np.set_printoptions(precision=3, suppress=True)
    print(f"K = {K}, ln K = {LN_K:.4f} nats\n")

    print("== 1. Known parameters (after learning is complete), per-step epistemic terms ==")
    print("room     F_state: sal  nov   F_amb: sal  nov   predictive H")
    preds = {"dark": np.eye(K)[0], "noise": np.full(K, 1 / K), "struct": np.eye(K)[1]}
    for k, p in preds.items():
        print(f"{k:7s}  {ent(p):9.4f} {0:5.1f}  {0:10.4f} {0:5.1f}   {ent(p):.4f}")
    print("-> F_state: G_noise = G_dark - ln K forever; F_amb: G equal for all rooms, cost decides.\n")

    print("== 2. Observational equivalence (same Dirichlet-Markov predictive) ==")
    rng = np.random.default_rng(1)
    seq = rng.integers(K, size=2000)
    c = np.full((K, K), 1.0); logev = 0.0; sal_state = 0.0; sal_amb = 0.0; nov_sum = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        row = c[a]; p = row / row.sum()
        logev += np.log(p[b])
        s1, n1 = efe_terms(row, "F_state"); s2, n2 = efe_terms(row, "F_amb")
        sal_state += s1; sal_amb += s2; nov_sum += n1
        c[a, b] += 1
    print(f"noise seq, 2000 obs: log evidence (both families, identical by construction) = {logev:.2f}")
    print(f"  cumulative salience F_state = {sal_state:.1f} nats, F_amb = {sal_amb:.1f} nats; novelty both = {nov_sum:.2f}\n")

    print("== 3. Novelty (parameter info gain) per visit to a row with n counts, times n ==")
    print("   a0     noise row (uniform)   deterministic row (vertex)   theory vertex (K-1)a0(psi(a0+1)-ln a0)")
    for a0 in [0.1, 0.5, 1.0, 2.0, 10.0]:
        n = 100000
        nr = novelty(np.full(K, a0 + n / K)) * n
        vr = novelty(np.r_[a0 + n, np.full(K - 1, a0)]) * n
        th = (K - 1) * a0 * (digamma(a0 + 1) - np.log(a0))
        print(f"  {a0:5.1f}   {nr:10.4f}   (=(K-1)/2={(K-1)/2:.2f})   {vr:10.4f}          {th:.4f}")
    print("-> per row visit, noise gives >= the novelty of a learned deterministic stream for every a0;")
    print("   all novelty decays like 1/n, so any constant cost difference eventually dominates.\n")

    print("== 4. Greedy EFE agent choosing a room each step, T = 20000 ==")
    print("cost per step (dark, noise, struct); fractions = share of steps in (dark, noise, struct)")
    print("F_state threshold: noise beats dark iff lam*(c_noise-c_dark) < ln K, i.e. lam < ln4/0.05 = %.1f" % (LN_K / 0.05))
    configs = [
        ("dark cheapest", (0.0, 0.05, 0.05)),
        ("C dislikes dark (no food there)", (0.2, 0.05, 0.05)),
    ]
    for fam in ["F_state", "F_amb"]:
        for known in [True, False]:
            for a0 in ([1.0] if known else [0.1, 1.0]):
                for name, cost in configs:
                    for lam in [0.0, 0.1, 1.0, 10.0, 20.0, 30.0, 100.0]:
                        frac, tail, settle, ties = simulate(fam, a0, lam, cost, known=known)
                        tag = "known " if known else f"a0={a0:<4}"
                        print(f"{fam:8s} {tag} {name:32s} lam={lam:6.1f}  all={frac}  last10%={tail}"
                              f"  settled-dark-after={settle if tail[0] == 1 else '-'}  ties={ties}")
        print()

    print("== 5. Can a preference C over the symbols separate noise from structure? ==")
    print("Learned rooms, long-run average of G = KL(q(o)||C) + ambiguity over one structured cycle.")
    rng = np.random.default_rng(7)
    sigma = np.roll(np.arange(K), -1)
    worst = -np.inf; best = np.inf
    for trial in range(2000):
        C = rng.dirichlet(np.full(K, rng.choice([0.2, 1.0, 5.0])))
        lnC = np.log(C)
        # noise room: q(o) uniform at every step
        kl_noise = float((np.full(K, 1 / K) * (np.log(1 / K) - lnC)).sum())
        # struct room: q(o) = delta at sigma(prev); average over the cycle
        kl_struct = float(np.mean([-lnC[sigma[i]] for i in range(K)]))
        for fam, amb_noise in [("F_state", 0.0), ("F_amb", LN_K)]:
            d = kl_struct - (kl_noise + amb_noise)   # G_struct - G_noise
            if trial < 3:
                print(f"  C={C.round(3)} {fam}: G_struct - G_noise = {d:+.4f}")
            if fam == "F_state":
                worst = max(worst, d); best = min(best, d)
            else:
                amb_d = d
    print(f"  F_state over 2000 random C: G_struct - G_noise in [{best:.4f}, {worst:.4f}] (ln K = {LN_K:.4f})")
    print(f"  F_amb: G_struct - G_noise = {amb_d:+.2e} for every C")
    print("-> a single-outcome preference never favours the learnable stream over noise.")


if __name__ == "__main__":
    main()
