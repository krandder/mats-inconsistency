"""Does query attention lower the copies' scores, beyond softmax arithmetic?

For one source copied to m rows, if the query's score for each copy equals the original's (no deduplication in the
scores), the total mass on the copies' query-question tokens is m*q1/(1 + (m-1)*P1), with q1 the m=1 mass on the
original query-question token and P1 the m=1 mass on the original's whole row (9 tokens), per episode and head.
Perfect deduplication by the readout scores keeps the total at q1. Per layer and head we report
dedup fraction = (arith - observed) / (arith - q1), mean over episodes: 0 = no score change, 1 = full dedup.
At depth 1 the identity is exact by construction (a control). For deeper layers the m=1 mass comes from the m=1
run, whose other tokens also differ, so it is a reference, not an identity.

usage: PYTHONPATH=... python mech3.py RUNS_DIR OUT_JSON [n]
"""
import glob, json, os, sys
import numpy as np, torch
import mech2, world as W
from pfn import PFN, DEV


def fractions(model, keys, base, ms=(2, 4, 8)):
    s0 = base[4]
    att, rowm = {}, {}
    for m in (1,) + tuple(ms):
        X, vm, src = mech2.tokens_with_copies(base, m, keys)
        with torch.no_grad():
            _, _, _, A = mech2.run(model, torch.tensor(X, device=DEV), torch.tensor(vm, device=DEV))
        rows = src == s0[:, None]; R = src.shape[1]
        tokrow = np.repeat(rows, W.Q, 1)
        qtok = np.zeros(R * W.Q, bool); qtok[np.arange(R) * W.Q + W.QL] = True
        att[m] = [(a[:, :, -1, :-1].cpu().numpy() * (tokrow & qtok)[:, None]).sum(-1) for a in A]   # per layer (n, h)
        rowm[m] = [(a[:, :, -1, :-1].cpu().numpy() * tokrow[:, None]).sum(-1) for a in A]
    out = {}
    for m in ms:
        per_layer = []
        for li in range(len(att[1])):
            q1, P1, obs = att[1][li], rowm[1][li], att[m][li]
            arith = m * q1 / (1 + (m - 1) * P1)
            per_layer.append({'q1': q1.mean(0).tolist(), 'arith': arith.mean(0).tolist(), 'obs': obs.mean(0).tolist(),
                              'frac': ((arith.mean(0) - obs.mean(0)) / (arith.mean(0) - q1.mean(0))).tolist()})
        out[m] = per_layer
    return out


def main(runs, outp, n=600):
    base = mech2.base_episodes(n, np.random.default_rng(mech2.SWEEP_SEED))
    res = []
    for f in sorted(glob.glob(os.path.join(runs, 'L*_k*_s*/meta.json'))):
        meta = json.load(open(f))
        for trained in (True, False):
            torch.manual_seed(meta['seed'])
            model = PFN(meta['fin'], meta['depth']).to(DEV)
            if trained:
                model.load_state_dict(torch.load(os.path.join(os.path.dirname(f), 'model.pt'), map_location=DEV))
            model.eval()
            r = fractions(model, meta['keys'], base)
            res.append({'depth': meta['depth'], 'keys': meta['keys'], 'seed': meta['seed'], 'trained': trained, 'frac': r})
            print(meta['depth'], meta['keys'], meta['seed'], trained,
                  [[round(v, 2) for v in l['frac']] for l in r[2]], flush=True)
    json.dump(res, open(outp, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600)
