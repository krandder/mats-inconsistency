"""Per-output (factorised) score version of the T1 pay game. v(C)=sum_i E[log p_i(codes_C) - log p0_i].
mode 'mix': absent codes marginalised by own marginal through the full 8-bucket reader (unvisited -> x-only price)
mode 'refit': each coalition priced by its own Bayes reader."""
import itertools, json, sys
import numpy as np
from math import factorial
sys.path.insert(0, '/home/agent/workspaces/calibrated-ai/trader-economy/p3-latent-markets/oracle')
from t1_game import S, names, B, Z, fvec, EPS, NG
def game(profile, toy, mode):
    F = np.array([fvec(*z, toy) for z in Z]); q = np.where(F == 1, 1 - EPS, EPS)  # 4z x 4g
    p0 = q.mean(0)
    codes = np.array([S[s] for s in profile]).T
    def bayes(cols):
        keys = [tuple(codes[z, cols]) for z in range(4)]
        return np.stack([q[[w for w in range(4) if keys[w] == keys[z]]].mean(0) for z in range(4)])  # price per z
    def score(pz):  # pz: 4z x 4g prices of bit=1
        return NG * 0.25 * (q * np.log(pz) + (1 - q) * np.log(1 - pz) - q * np.log(p0) - (1 - q) * np.log(1 - p0)).sum()
    pb = np.stack([q[[z for z in range(4) if tuple(codes[z]) == tuple(b)]].mean(0) if any(tuple(codes[z]) == tuple(b) for z in range(4)) else p0 for b in B])  # 8x4
    marg = np.stack([[1 - codes[:, j].mean(), codes[:, j].mean()] for j in range(3)])
    onehot = np.stack([np.eye(2)[codes[:, j]] for j in range(3)]); prior = np.repeat(marg[:, None, :], 4, 1)
    W = lambda Q: np.prod(np.stack([Q[j][:, B[:, j]] for j in range(3)]), 0)  # 4x8
    vq = lambda Q: score(W(Q) @ pb)
    N = (0, 1, 2); V = {}
    for r in range(4):
        for Cc in itertools.combinations(N, r):
            V[Cc] = score(bayes(list(Cc))) if mode == 'refit' else vq(np.stack([onehot[j] if j in Cc else prior[j] for j in N]))
    vs = lambda Cc: V[tuple(sorted(Cc))]
    loo = [vs(N) - vs([k for k in N if k != i]) for i in N]; shap = []; banz = []
    for i in N:
        o = [k for k in N if k != i]; sh = bz = 0.
        for r in range(3):
            for Cc in itertools.combinations(o, r):
                d = vs(list(Cc) + [i]) - vs(Cc); sh += factorial(r) * factorial(2 - r) / 6 * d; bz += d / 4
        shap.append(sh); banz.append(bz)
    casc = [vs([0]), vs([0, 1]) - vs([0]), vs(N) - vs([0, 1])]
    def asig(n, h=1e-6):
        out = np.zeros(3)
        for t in (np.arange(n) + 0.5) / n:
            Q = (1 - t) * prior + t * onehot
            for i in N:
                d = onehot[i] - prior[i]; Qp = Q.copy(); Qm = Q.copy(); Qp[i] += h * d; Qm[i] -= h * d
                out[i] += (vq(Qp) - vq(Qm)) / (2 * h) / n
        return out
    a = asig(20)
    return dict(vN=vs(N), v0=vs(()), single=[vs([i]) for i in N], loo=loo, shap=shap, banz=banz, casc=casc, asig=a.tolist(),
                asig_err=float(a.sum() - (vq(onehot) - vq(prior))))
def run(toy, mode):
    out = {p: game(p, toy, mode) for p in itertools.product(names, repeat=3)}
    rules = ['loo', 'shap', 'banz', 'casc'] + (['asig'] if mode == 'mix' else [])
    vmax = max(g['vN'] for g in out.values())
    print('== toy', toy, 'mode', mode, 'vmax %.4f nats/example' % vmax)
    res = {}
    for r in rules:
        canon = {}
        for prof, g in out.items():
            ok = strict = True
            for i in range(3):
                for alt in names:
                    if alt == prof[i]: continue
                    p2 = list(prof); p2[i] = alt; a = out[tuple(p2)][r][i]
                    if a > g[r][i] + 1e-9: ok = False
                    if a > g[r][i] - 1e-9: strict = False
            if ok:
                k = tuple(sorted(prof)) if r != 'casc' else prof
                canon.setdefault(k, (strict, g['vN']))
        res[r] = {'|'.join(k): v for k, v in canon.items()}
        print(' ', r, '; '.join('%s %s%s' % (k, 's' if s else 'w', '*' if abs(vn - vmax) < 1e-9 else '') for k, (s, vn) in canon.items()))
    for prof in [('b1', 'b2', 'const'), ('b1', 'xor', 'const'), ('b2', 'xor', 'const'), ('b1', 'b2', 'xor')]:
        g = out[prof]; print('   ', prof, {k: np.round(g[k], 4).tolist() for k in ['single', 'loo', 'shap', 'asig', 'casc'] if k in g}, 'asig_err %.1e' % g['asig_err'])
    return res
if __name__ == '__main__':
    allres = {}
    for toy in ('natural', 'mirrored'):
        for mode in ('mix', 'refit'):
            allres[toy + '/' + mode] = run(toy, mode)
    json.dump(allres, open('/home/agent/workspaces/calibrated-ai/trader-economy/p3-latent-markets/oracle/t1_perout.json', 'w'), default=str)
