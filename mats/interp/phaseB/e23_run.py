"""E23 (PREREG-B amendment 30): a layer-1 attention signature of the A/B groups, tested on fresh nets (E21's GPU nets).
For each acquired E21 net (runs/e21_w*_s*, pfn2 models with standard layer-2 heads, loadable as pfn.PFN): rho = Spearman correlation,
over the first 4,000 test episodes, of the query-question token's within-source layer-1 attention share (per head, normalised over
its source's four labelled tokens) with each token's |error| (exactly as e22_post.py); then compared with the net's middle-range rom
from results/e21/ (E21's measures).
usage: python e23_run.py -> results/e23.json
"""
import glob, json, math, os
import numpy as np
import torch
from scipy.stats import spearmanr
import pfn as P
import nnoise as N
import e21_summ as S

torch.set_num_threads(1)
QQ = [4, 9, 14]


def rho_of(run, X, v, absr):
    meta = json.load(open(os.path.join(run, 'meta.json'))); m = P.build(meta)
    m.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu')); m.eval()
    with torch.no_grad():
        blk = m.blocks[0]; allow = N.NoisyPFN.allow_mask(v); x = m.inp(X); B, T, d = x.shape
        q, k, _ = blk.qkv(blk.ln1(x)).split(d, -1); sp = lambda t: t.view(B, T, blk.h, d // blk.h).transpose(1, 2)
        a = (sp(q) @ sp(k).transpose(-1, -2) / math.sqrt(d // blk.h)).masked_fill(~allow[:, None], float('-inf')).softmax(-1).numpy()
    share = np.stack([a[:, :, qi, 5 * s_:5 * s_ + 4] for s_, qi in enumerate(QQ)], 2); share = share / share.sum(-1, keepdims=True)
    return float(spearmanr(np.broadcast_to(absr[:, None], share.shape).ravel(), share.ravel())[0])


def main():
    tz = np.load('/home/agent/workspaces/mats/work/tiny-pfn/runs/testset.npz'); n = 4000
    X = torch.from_numpy(tz['X'][:n]); v = torch.from_numpy(tz['valid'][:n]); absr = np.abs(tz['o'][:n, :, :4] - tz['y'][:n, None, :4])
    z = np.load('results/e12_pairs.npz'); out = {}
    for d in sorted(glob.glob('runs/e21_w*_s*')):
        tag = os.path.basename(d)[4:]; f = f'results/e21/{tag}.npz'
        if not os.path.exists(f):
            continue
        st, _ = S.stats(np.load(f), z)
        if st['gap'] >= 0.10:
            out[tag] = dict(acquired=False); continue
        out[tag] = dict(acquired=True, rho=rho_of(d, X, v, absr), middle=st['middle']['rom'], large=st['large']['rom'])
        print(tag, out[tag], flush=True)
    acq = {k: o for k, o in out.items() if o.get('acquired')}
    pos = [o for o in acq.values() if o['rho'] > 0]
    agree = sum((o['rho'] > 0) == (o['middle'] > 0) for o in acq.values())
    out['_P130'] = dict(n=len(acq), n_rho_pos=len(pos), n_rho_pos_B=int(sum(o['middle'] > 0 for o in pos)), agree=int(agree),
                        holds=bool(len(acq) > 0 and all(o['middle'] > 0 for o in pos) and agree >= 0.8 * len(acq)))
    json.dump(out, open('results/e23.json', 'w'), indent=1)
    print(out['_P130'])


if __name__ == '__main__':
    main()
