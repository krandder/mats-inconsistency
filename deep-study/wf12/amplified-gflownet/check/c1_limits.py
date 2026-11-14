"""Check of claims 2-6: limits of the three amplifications, distillation rate, alien coordinate.
Own world (cw.py), fresh seeds.  Output: out/c1_limits.out"""
import numpy as np
from cw import *

L = []
rng = np.random.default_rng(20261110)
for ws in (101, 102, 103):
    W = make_world(ws)
    R, T = W["R"], W["T"]
    hid = lambda x: x.get(0) == 1 and x.get(1) == 1
    pols = {"A": noisy_policy(R, 0.5, 7 + ws), "B": noisy_policy(R, 1.5, 9 + ws, lean=2.0),
            "S": noisy_policy(R, 1.5, 9 + ws, lean=2.0, hide=hid)}
    hidden = T[(DXV[:, 0] == 1) & (DXV[:, 1] == 1)].sum()
    L.append(f"\n== world {ws}: Z = {W['Z']:.4f}; T-mass hidden by S = {hidden:.4f}")
    for name, (Q, root) in pols.items():
        P = law(Q, root)
        sup = P > 0
        pb = rng.dirichlet([1, 1], size=D).T                   # a random normalised backward law over the alien bit
        w = np.where(sup, R[None, :] * pb / np.where(sup, P, 1), 0)
        # exact: E[w 1_d] = R(d) on the support
        Ew = (P * w).sum(0)
        dsup = sup.all(0)
        L.append(f"policy {name}: TV(policy, T) = {tv(P.sum(0), T):.4f}; max |E[w 1_d] - R(d)| on full-support d = "
                 f"{np.abs(Ew - R)[dsup].max():.2e}; demonstrations shown {int((P.sum(0) > 0).sum())}/48")
        # Monte Carlo expected SNIS law
        row = []
        for K in (4, 64, 1024, 8192):
            reps = 3000 if K <= 1024 else 400
            idx = rng.choice(2 * D, size=(reps, K), p=P.ravel())
            Wt = w.ravel()[idx]
            Wn = Wt / Wt.sum(1, keepdims=True)
            m = np.zeros(D)
            np.add.at(m, (idx % D).ravel(), Wn.ravel())
            row.append(tv(m / reps, T))
        L.append(f"   weighting (random backward law), TV(expected law, T) at K = 4, 64, 1024, 8192: "
                 + ", ".join(f"{v:.4f}" for v in row))
        # soft tree, exact fixed point with every terminal edge read; and with none read
        for opt in (1, 2):
            Qt = {k: v.copy() for k, v in Q.items()}
            rt = np.zeros(2)
            for a in (0, 1):
                for n in sorted(NODES, key=len, reverse=True):
                    for ci, c in enumerate(kids(n)):
                        ch = n + (c,)
                        if len(ch) == N:
                            if opt == 1:
                                Qt[(a, n)][ci] = np.log(R[DIDX[ch]]) + np.log(0.5)
                        else:
                            Qt[(a, ch)]  # exists
                    # backups of deeper nodes are done first because of the sort
                for n in sorted(NODES, key=len, reverse=True):
                    if len(n) < N - 1:
                        for ci, c in enumerate(kids(n)):
                            Qt[(a, n)][ci] = np.logaddexp.reduce(Qt[(a, n + (c,))])
                rt[a] = np.logaddexp.reduce(Qt[(a, ())])
            lim = law(Qt, rt).sum(0)
            L.append(f"   soft tree fully backed up, terminal edges {'read the score' if opt == 1 else 'keep the network numbers'}:"
                     f" TV to T = {tv(lim, T):.4f}" + (f"; log Z from root {np.logaddexp.reduce(rt):.4f} vs {np.log(W['Z']):.4f}" if opt == 1 else ""))
        # back-and-forth kernel (own construction): components (a, s1, s2, s3); drop last j, rebuild from the policy
        Pf = P.ravel()
        Rf = np.tile(R, 2)
        full = [(t // D,) + PATH[t % D] for t in range(2 * D)]
        for rule in ("mh", "printed", "greedy"):
            Kt = np.zeros((2 * D, 2 * D))
            for j in (1, 2, 3, 4):
                keep = 4 - j
                for t in range(2 * D):
                    if Pf[t] == 0:
                        Kt[t, t] += 0.25
                        continue
                    grp = np.array([u for u in range(2 * D) if full[u][:keep] == full[t][:keep]])
                    q = Pf[grp] / Pf[grp].sum()
                    for u, qu in zip(grp, q):
                        if u == t or qu == 0:
                            continue
                        if rule == "mh":      # target(u) q(t|u) / (target(t) q(u|t))
                            acc = min(1.0, (Rf[u] * Pf[t]) / (Rf[t] * Pf[u]))
                        elif rule == "printed":   # R(u)/R(t) * q(u|t)/q(t|u), eq. after l.108 of 4_method.tex
                            acc = min(1.0, (Rf[u] * Pf[u]) / (Rf[t] * Pf[t]))
                        else:
                            acc = float(Rf[u] > Rf[t])
                        Kt[t, u] += 0.25 * qu * acc
                    Kt[t, t] += 0.25 - (Kt[t, grp].sum() - Kt[t, t]) * 0  # placeholder, fixed below
            Kt[np.arange(2 * D), np.arange(2 * D)] = 0
            Kt[np.arange(2 * D), np.arange(2 * D)] = 1 - Kt.sum(1)
            lim = (Pf @ np.linalg.matrix_power(Kt, 2 ** 22)).reshape(2, D).sum(0)
            inv = np.abs((np.tile(T, 2) / 2) @ Kt - np.tile(T, 2) / 2).max()
            L.append(f"   refinement {rule:8s}: limit TV to T = {tv(lim, T):.4f}; T x 1/2 invariant: {'yes' if inv < 1e-12 else f'no ({inv:.3f})'}")
            if rule == "mh" and name != "S":
                # counterexample hunt: does one MH move ever raise the TV over demonstrations?
                worst_j, worst_m = -1, -1
                for trial in range(4000):
                    c = rng.dirichlet([0.3, 0.3], size=D).T
                    base = T if trial % 2 == 0 else rng.dirichlet(np.full(D, 0.5 + 3 * rng.random()))
                    P0 = (base[None, :] * c).ravel()
                    P1 = P0 @ Kt
                    tj0, tj1 = tv(P0, np.tile(T, 2) / 2), tv(P1, np.tile(T, 2) / 2)
                    tm0, tm1 = tv(P0.reshape(2, D).sum(0), T), tv(P1.reshape(2, D).sum(0), T)
                    worst_j, worst_m = max(worst_j, tj1 - tj0), max(worst_m, tm1 - tm0)
                L.append(f"      one MH move from 4,000 random start laws: largest rise of TV on (alien, demonstration) = {worst_j:.4f};"
                         f" largest rise of TV over demonstrations = {worst_m:.4f}")
open("out/c1_limits.out", "w").write("\n".join(L) + "\n")
print("\n".join(L))
