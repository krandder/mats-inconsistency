"""Synthesis check (DARKROOM.md): does goal-relative information value survive when the agent does NOT know
which room bears on its goal?  value-of-information.md assumed the agent's model already says the dark room and
the noise room are independent of the goal.  Here that independence has to be learned.

World: episodes e = 1..E.  A fresh goal bit Z_e ~ Bern(1/2) every episode ("which door is open").  The agent
takes L = 3 looks, each in a room of its choice (every look costs the same), then guesses Z_e by its exact
posterior; Z_e is then revealed (outcome feedback).
Rooms:
  D    dark            K=2,  x = 0 always
  N    white noise     K=2,  x ~ uniform, independent of Z
  TV   noisy TV        K=64, x ~ a fixed generic law drawn from Dirichlet(1) per seed, independent of Z
  S    the useful room K=2,  x = Z xor Bern(0.25)   (0.131 nats about Z per look at even odds)
Agent's model, the same for every room (it is told only the alphabet size K_r):
  x | Z=z ~ Cat(theta_{r,z}).  Two priors on the channel:
  slab   theta_{r,1}, theta_{r,0} ~ Dir(a0) independently: every room is a priori relevant (relevance has prior 1)
  spike  with prob rho the room is irrelevant (theta_{r,1} = theta_{r,0} ~ Dir(a0)), else slab.
         The spike is the hypothesis "this stream does not depend on the goal", whatever its law.
Policies (per look):
  goal-greedy    argmax_r I(Z_e ; x_r | everything seen)          myopic goal VOI (log score)
  goal-thompson  sample a channel from the posterior once per episode, argmax_r I(Z_e ; x_r) under it
  curiosity      argmax_r I(Z_e, H_r, theta_r ; x_r)               information about every latent (EFE epistemic)
  surprise-min   argmin_r H(x_r)                                    free energy / persistence reading
  random
Exact conjugate Bayes throughout.  Nats.  No weight anywhere: every policy ranks one quantity at equal cost.
Run: python3 relevance.py [E] [seeds]   (default 2000 episodes, 16 seeds, ~3 min on 16 cores)
"""
import sys
from multiprocessing import Pool
import numpy as np
from scipy.special import gammaln, digamma, logsumexp

NAMES = ["D", "N", "TV", "S"]
TVK = 64  # size of the noisy TV alphabet (tvsize.py varies it)
KS = [2, 2, TVK, 2]
L = 3
FLIP = 0.25


def H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def logDM(alpha, w):
    """log probability of a particular sequence with counts w under Dirichlet(alpha)-categorical"""
    return float(gammaln(alpha.sum()) - gammaln(alpha.sum() + w.sum()) + (gammaln(alpha + w) - gammaln(alpha)).sum())


def EH_dir(alpha):
    A = alpha.sum()
    return float(digamma(A + 1) - (alpha * digamma(alpha + 1)).sum() / A)


def run(args):
    policy, E, seed, a0, prior, rho, forced = args[:7]
    if len(args) > 7:
        KS[2] = args[7]
    rng = np.random.default_rng(seed)
    tv_law = rng.dirichlet(np.ones(KS[2]))
    C = [np.zeros((2, k)) for k in KS]  # past counts by revealed z
    looks = np.zeros((E, 4), int)
    succ = np.zeros(E)
    lr = np.log(rho) if prior == "spike" else -np.inf
    lnr = np.log1p(-rho) if prior == "spike" else 0.0
    for e in range(E):
        z_true = int(rng.random() < 0.5)
        # history posterior over H per room: log weights [irr, rel]
        lwH = []
        for r in range(4):
            a = np.full(KS[r], a0)
            li = lr + logDM(a, C[r][0] + C[r][1])
            lre = lnr + logDM(a, C[r][0]) + logDM(a, C[r][1])
            m = logsumexp([li, lre]); lwH.append(np.array([li - m, lre - m]))
        if policy == "goal-thompson":
            th = []
            for r in range(4):
                a = np.full(KS[r], a0)
                if rng.random() < np.exp(lwH[r][0]):
                    t = rng.dirichlet(a + C[r][0] + C[r][1]); th.append(np.stack([t, t]))
                else:
                    th.append(np.stack([rng.dirichlet(a + C[r][0]), rng.dirichlet(a + C[r][1])]))
        W = [np.zeros(k) for k in KS]  # within-episode counts
        for step in range(L + 1):
            # per room, per z: log M_r(z) and next-symbol predictive and E[H(theta)]
            logM = np.zeros((4, 2)); pred = []; eh = np.zeros((4, 2))
            for r in range(4):
                a = np.full(KS[r], a0); pr = []
                for z in (0, 1):
                    al_i = a + C[r][0] + C[r][1] + W[r]
                    al_r = a + C[r][z] + W[r]
                    t_i = lwH[r][0] + logDM(a + C[r][0] + C[r][1], W[r])
                    t_r = lwH[r][1] + logDM(a + C[r][z], W[r])
                    lm = logsumexp([t_i, t_r]); logM[r, z] = lm
                    pi, prr = np.exp(t_i - lm), np.exp(t_r - lm)
                    pr.append(pi * al_i / al_i.sum() + prr * al_r / al_r.sum())
                    if policy == "curiosity":
                        eh[r, z] = pi * EH_dir(al_i) + prr * EH_dir(al_r)
                pred.append(pr)
            lz = logM.sum(0); pz = np.exp(lz - logsumexp(lz))
            if step == L:
                guess = int(pz[1] > pz[0]) if pz[1] != pz[0] else int(rng.random() < 0.5)
                succ[e] = guess == z_true
                break
            score = np.zeros(4)
            if policy == "goal-thompson":
                lzs = np.zeros(2)
                for r in range(4):
                    for z in (0, 1):
                        lzs[z] += (W[r] * np.log(np.maximum(th[r][z], 1e-300))).sum()
                pzs = np.exp(lzs - logsumexp(lzs))
            for r in range(4):
                mix = pz[0] * pred[r][0] + pz[1] * pred[r][1]
                if policy == "goal-greedy":
                    score[r] = H(mix) - pz[0] * H(pred[r][0]) - pz[1] * H(pred[r][1])
                elif policy == "goal-thompson":
                    q = th[r]; score[r] = H(pzs[0] * q[0] + pzs[1] * q[1]) - pzs[0] * H(q[0]) - pzs[1] * H(q[1])
                elif policy == "curiosity":
                    score[r] = H(mix) - pz[0] * eh[r, 0] - pz[1] * eh[r, 1]
                elif policy == "surprise-min":
                    score[r] = -H(mix)
            if e < forced:
                r = int(rng.integers(3))  # forced start in D, N or TV
            else:
                best = np.flatnonzero(score >= score.max() - 1e-12); r = int(rng.choice(best))
            if r == 0: x = 0
            elif r == 1: x = int(rng.integers(2))
            elif r == 2: x = int(rng.choice(KS[2], p=tv_law))
            else: x = z_true ^ int(rng.random() < FLIP)
            W[r][x] += 1; looks[e, r] += 1
        for r in range(4):
            C[r][z_true] += W[r]
    return looks, succ


def report(pool, label, policy, E, seeds, a0=1.0, prior="slab", rho=0.5, forced=0):
    res = pool.map(run, [(policy, E, 1000 + s, a0, prior, rho, forced) for s in range(seeds)])
    cps = [E // 16, E // 4, E]
    q = np.array([lk[3 * E // 4:].sum(0) / lk[3 * E // 4:].sum() for lk, _ in res]).mean(0)
    tv = np.array([[lk[:c, 2].sum() for c in cps] for lk, _ in res]).mean(0)
    dn = np.array([[lk[:c, :2].sum() for c in cps] for lk, _ in res]).mean(0)
    sc = np.mean([su[3 * E // 4:].mean() for _, su in res])
    stuck = sum((lk[3 * E // 4:, 3].sum() / lk[3 * E // 4:].sum()) < 0.5 for lk, _ in res)
    print(f"{label:34s} {policy:13s} last-qtr " + " ".join(f"{n}:{v:.3f}" for n, v in zip(NAMES, q))
          + f" | succ {sc:.3f} | not-in-S {stuck}/{seeds}"
          + " | cum looks D+N " + "/".join(f"{v:.0f}" for v in dn) + "  TV " + "/".join(f"{v:.0f}" for v in tv)
          + f"  (at E={'/'.join(str(c) for c in cps)})", flush=True)


def main():
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    print(f"E={E} episodes x {L} looks, {seeds} seeds. Rooms {NAMES}, K={KS}. Success: S only 0.84, nothing 0.50.")
    print("last-qtr = share of looks in the last quarter; cum looks = cumulative looks, averaged over seeds\n")
    with Pool(16) as pool:
        print("== 1. slab prior Dir(1): every room a priori relevant")
        for pol in ["goal-greedy", "goal-thompson", "curiosity", "surprise-min", "random"]:
            report(pool, "slab a0=1", pol, E, seeds)
        print("\n== 2. forced dark start: first 5 episodes look only at D, N, TV")
        for pol in ["goal-greedy", "goal-thompson", "curiosity"]:
            report(pool, "slab a0=1 forced", pol, E, seeds, forced=5)
        print("\n== 3. spike prior: P(room irrelevant to the goal) = rho")
        for rho in [0.5, 0.1, 0.9]:
            for pol in ["goal-greedy", "goal-thompson", "curiosity"]:
                report(pool, f"spike rho={rho} a0=1", pol, E, seeds, prior="spike", rho=rho)
        print("\n== 4. Dirichlet concentration a0 (the knob that flipped other candidates)")
        for a0 in [0.1, 0.5, 4.0]:
            for prior in ["slab", "spike"]:
                for pol in ["goal-thompson", "curiosity"]:
                    report(pool, f"{prior} a0={a0}", pol, E, seeds, a0=a0, prior=prior)
        print("\n== 5. spike prior, forced dark start")
        for pol in ["goal-greedy", "goal-thompson"]:
            report(pool, "spike rho=0.5 forced", pol, E, seeds, prior="spike", forced=5)


if __name__ == "__main__":
    main()
