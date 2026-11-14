"""Identity test: true copy vs new-identity copy vs independent source with the same query reading (PREREG addendum).

usage: PYTHONPATH=... python split_copy.py RUNS_DIR OUT_JSON [n]
"""
import glob, json, os, sys
import numpy as np, torch
import mech2, split, world as W
from pfn import PFN, DEV


def variants(base, rng):
    d, semb, qemb, gemb, s0 = base
    n = len(s0); ar = np.arange(n)
    o, lab = d['o'], d['lab']
    newg = rng.standard_normal((n, 1, W.DS)) / 4
    news = rng.standard_normal((n, 1, W.DS)) / 4
    gemb7 = np.concatenate([gemb, newg], 1)                      # label 6 = a new singleton group
    fresh = d['y'][:, None, :] + rng.standard_normal((n, 1, W.Q)) * np.sqrt(W.SG2) \
        + rng.standard_normal((n, 1, W.Q)) * np.sqrt(W.SE2)
    fresh[:, 0, W.QL] = o[ar, s0, W.QL]
    out = {}
    src6 = np.tile(np.arange(W.S), (n, 1))
    # (i) true copy: extra row pointing at s0
    out['copy'] = dict(semb=semb, gemb=gemb, o=o, lab=lab, src=np.concatenate([src6, s0[:, None]], 1))
    # (ii) same values, new identity (row 6), same group label as s0
    o7 = np.concatenate([o, o[ar, s0][:, None]], 1)
    semb7 = np.concatenate([semb, news], 1)
    lab_ii = np.concatenate([lab, lab[ar, s0][:, None]], 1)
    out['newid'] = dict(semb=semb7, gemb=gemb, o=o7, lab=lab_ii, src=np.concatenate([src6, np.full((n, 1), 6)], 1))
    # (iv) independent new source, new singleton group, query reading = o_s0
    o7b = np.concatenate([o, fresh], 1)
    lab_iv = np.concatenate([lab, np.full((n, 1), 6)], 1)
    out['indep'] = dict(semb=semb7, gemb=gemb7, o=o7b, lab=lab_iv, src=np.concatenate([src6, np.full((n, 1), 6)], 1))
    return out


def tokens(v, base, keys):
    d, semb, qemb, gemb, s0 = base
    valid = np.ones_like(v['src'], dtype=bool)
    return W.assemble(v['semb'], qemb, v['gemb'], v['src'], valid, v['o'], d['y'], v['lab'], keys)


def main(runs, outp, n=600):
    base = mech2.base_episodes(n, np.random.default_rng(mech2.SWEEP_SEED))
    d, s0 = base[0], base[4]
    V = variants(base, np.random.default_rng(60000))
    odup = d['o'][np.arange(n), s0, W.QL]
    # identity-blind references: same query readings in every variant
    coef = split.fit_r1()
    q6 = d['o'][:, :, W.QL]; q7 = np.concatenate([q6, odup[:, None]], 1)
    r1 = lambda Q: np.array([coef[Q.shape[1]][:-1] @ split.basis(q).sum(0) + coef[Q.shape[1]][-1] for q in Q])
    ref = {'R1': mech2.slope(r1(q7) - r1(q6), odup - r1(q6)),
           'R2': mech2.slope(split.r2_mean(q7) - split.r2_mean(q6), odup - split.r2_mean(q6))}
    res = {'ref': ref, 'models': []}
    for f in sorted(glob.glob(os.path.join(runs, 'L*_k*_s*/meta.json'))):
        meta = json.load(open(f)); keys = meta['keys']
        torch.manual_seed(meta['seed'])
        model = PFN(meta['fin'], meta['depth']).to(DEV)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(f), 'model.pt'), map_location=DEV))
        model.eval()
        X1, v1, _ = mech2.tokens_with_copies(base, 1, keys)
        with torch.no_grad():
            mu1 = mech2.run(model, torch.tensor(X1), torch.tensor(v1))[0].numpy()
            p = {}
            for name, v in V.items():
                X, vm = tokens(v, base, keys)
                mu = mech2.run(model, torch.tensor(X), torch.tensor(vm))[0].numpy()
                p[name] = mech2.slope(mu - mu1, odup - mu1)
        res['models'].append({'depth': meta['depth'], 'keys': keys, 'seed': meta['seed'], 'pull': p})
        print(meta['depth'], keys, meta['seed'], {k: round(x, 3) for k, x in p.items()}, flush=True)
    print('ref', ref)
    json.dump(res, open(outp, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600)
