"""Closed-form T1 pay game (population level). 3 traders, K=2 codes as functions of z*=(b1,b2).
Per-output tabular reader refit to each profile (Bayes price per visited bucket; unvisited buckets
quote the x-only price). Coalition value v(C) = E log mixture price with absent codes marginalised
by their own marginal, minus E log x-only price. AS-IG path: q_j(t)=(1-t)*marginal_j + t*onehot_j."""
import itertools, json, sys
import numpy as np
from math import comb, factorial
EPS = 0.05; NG = 8
Z = [(b1, b2) for b1 in (0, 1) for b2 in (0, 1)]
def fvec(b1, b2, toy):
    a, d = (b1, b2) if toy == 'natural' else (b1, b1 ^ b2)
    return (a, d, a & d, a | d)
S = {'const': (0, 0, 0, 0), 'b1': tuple(b1 for b1, b2 in Z), 'b2': tuple(b2 for b1, b2 in Z),
     'xor': tuple(b1 ^ b2 for b1, b2 in Z)}
for k, (b1, b2) in enumerate(Z):
    S['one%d%d' % (b1, b2)] = tuple(int(j == k) for j in range(4))
names = list(S)
M = np.array(list(itertools.product(range(NG + 1), repeat=4)))
B = np.array(list(itertools.product((0, 1), repeat=3)))  # 8 buckets x 3
def setup(toy):
    F = np.array([fvec(*z, toy) for z in Z])
    q = np.where(F == 1, 1 - EPS, EPS)
    C = np.array([comb(NG, int(x)) for x in range(NG + 1)])
    Pm = np.stack([np.prod([C[M[:, g]] * q[z, g] ** M[:, g] * (1 - q[z, g]) ** (NG - M[:, g]) for g in range(4)], axis=0) for z in range(4)])
    p0 = q.mean(0)
    lx = (M * np.log(p0) + (NG - M) * np.log(1 - p0)).sum(1)
    return q, Pm, lx
def game(profile, toy, cache={}):
    if toy not in cache: cache[toy] = setup(toy)
    q, Pm, lx = cache[toy]
    codes = np.array([S[s] for s in profile]).T  # 4 x 3
    p0 = q.mean(0)
    pb = np.stack([q[[z for z in range(4) if tuple(codes[z]) == tuple(b)]].mean(0) if any(tuple(codes[z]) == tuple(b) for z in range(4)) else p0 for b in B])
    P = np.exp((M[None] * np.log(pb)[:, None, :] + (NG - M[None]) * np.log(1 - pb)[:, None, :]).sum(2))  # 8 x 6561
    marg = np.stack([[1 - codes[:, j].mean(), codes[:, j].mean()] for j in range(3)])  # 3x2
    onehot = np.stack([np.eye(2)[codes[:, j]] for j in range(3)])  # 3 x 4 x 2
    prior = np.repeat(marg[:, None, :], 4, 1)
    def W(Q):  # 4 x 8
        return np.prod(np.stack([Q[j][:, B[:, j]] for j in range(3)]), 0)
    def v_of_q(Q):
        mix = W(Q) @ P
        return 0.25 * (Pm * (np.log(mix) - lx)).sum()
    def dv(Q, i, d):  # directional derivative of v wrt Q_i along d (4x2)
        mix = W(Q) @ P
        others = np.prod(np.stack([Q[j][:, B[:, j]] for j in range(3) if j != i]), 0)  # 4x8
        g = (others * d[:, B[:, i]]) @ P  # 4 x 6561
        return 0.25 * (Pm * g / mix).sum()
    N = (0, 1, 2)
    V = {}
    for r in range(4):
        for Cc in itertools.combinations(N, r):
            V[Cc] = v_of_q(np.stack([onehot[j] if j in Cc else prior[j] for j in N]))
    vs = lambda Cc: V[tuple(sorted(Cc))]
    loo = [vs(N) - vs([k for k in N if k != i]) for i in N]
    shap, banz = [], []
    for i in N:
        o = [k for k in N if k != i]; sh = bz = 0.
        for r in range(3):
            for Cc in itertools.combinations(o, r):
                d = vs(list(Cc) + [i]) - vs(Cc)
                sh += factorial(r) * factorial(2 - r) / 6 * d; bz += d / 4
        shap.append(sh); banz.append(bz)
    casc = [vs([0]), vs([0, 1]) - vs([0]), vs(N) - vs([0, 1])]
    def asig(n):
        out = np.zeros(3)
        for t in (np.arange(n) + 0.5) / n:
            Q = (1 - t) * prior + t * onehot
            for i in N:
                out[i] += dv(Q, i, onehot[i] - prior[i]) / n
        return out
    a20 = asig(20); a400 = asig(400)
    return dict(vN=vs(N), v0=vs(()), loo=loo, shap=shap, banz=banz, casc=casc, asig=a20.tolist(),
                asig_err20=float(a20.sum() - (vs(N) - vs(())) ), asig_err400=float(a400.sum() - (vs(N) - vs(()))))
if __name__ == '__main__':
    toy = sys.argv[1] if len(sys.argv) > 1 else 'natural'
    out = {prof: game(prof, toy) for prof in itertools.product(names, repeat=3)}
    rules = ['loo', 'shap', 'banz', 'asig', 'casc']
    vmax = max(g['vN'] for g in out.values())
    summ = {'toy': toy, 'vmax': vmax}
    for r in rules:
        ne = []
        for prof, g in out.items():
            ok = strict = True
            for i in range(3):
                for alt in names:
                    if alt == prof[i]: continue
                    p2 = list(prof); p2[i] = alt
                    a = out[tuple(p2)][r][i]
                    if a > g[r][i] + 1e-7: ok = False
                    if a > g[r][i] - 1e-7: strict = False
            if ok: ne.append([list(prof), strict, g['vN']])
        summ[r] = ne
    json.dump({'summary': summ, 'profiles': {'|'.join(k): v for k, v in out.items()}}, open('/home/agent/workspaces/calibrated-ai/trader-economy/p3-latent-markets/oracle/t1_%s.json' % toy, 'w'))
    print('toy', toy, 'vmax %.4f nats/example' % vmax)
    for r in rules:
        canon = {}
        for prof, strict, vn in summ[r]:
            key = tuple(sorted(prof)) if r != 'casc' else tuple(prof)
            canon.setdefault(key, (strict, vn))
        print(r, 'NE (up to trader order):')
        for k, (strict, vn) in canon.items():
            print('   ', k, 'strict' if strict else 'weak', 'vN=%.4f' % vn, 'SUFFICIENT' if abs(vn - vmax) < 1e-6 else '')
    for prof in [('b1', 'b2', 'const'), ('b1', 'xor', 'const'), ('b2', 'xor', 'const'), ('b1', 'b2', 'xor'), ('b1', 'b2', 'b1')]:
        g = out[prof]
        print(prof, {k: (np.round(g[k], 4).tolist() if isinstance(g[k], list) else round(g[k], 5)) for k in g})
