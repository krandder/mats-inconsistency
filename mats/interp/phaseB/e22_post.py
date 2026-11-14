"""E22 post hoc (written after E22's registered results): how the query-question token's layer-1 attention within its own source's
labelled tokens depends on each token's error, in the real clean nets.
For each E22 net, on the first 4,000 test episodes: the within-group share a_q = attention(QQ_s -> labelled token q of s) / sum over
the four, per head; Spearman correlation of a_q with |r_q| pooled over episodes, sources, tokens and heads; and the mean share by
|r| quintile. Uniform share = 0.25.
usage: python e22_post.py -> results/e22_post.json
"""
import json, math
import numpy as np
import torch
from scipy.stats import spearmanr
import pfn as P
import nnoise as N

torch.set_num_threads(1)
QQ = [4, 9, 14]


def main():
    tz = np.load('/home/agent/workspaces/mats/work/tiny-pfn/runs/testset.npz'); n = 4000
    X = torch.from_numpy(tz['X'][:n]); v = torch.from_numpy(tz['valid'][:n])
    absr = np.abs(tz['o'][:n, :, :4] - tz['y'][:n, None, :4])            # (n, 3, 4)
    q_edges = np.quantile(absr, [0.2, 0.4, 0.6, 0.8])
    nets = json.load(open('results/e22_nets.json')); out = {}
    for r in nets:
        meta = json.load(open(r['run'] + '/meta.json')); m = P.build(meta); m.load_state_dict(torch.load(r['run'] + '/model.pt', map_location='cpu')); m.eval()
        with torch.no_grad():
            blk = m.blocks[0]; allow = N.NoisyPFN.allow_mask(v); x = m.inp(X); B, T, d = x.shape
            q, k, _ = blk.qkv(blk.ln1(x)).split(d, -1); sp = lambda t: t.view(B, T, blk.h, d // blk.h).transpose(1, 2)
            s = (sp(q) @ sp(k).transpose(-1, -2) / math.sqrt(d // blk.h)).masked_fill(~allow[:, None], float('-inf')); a = s.softmax(-1).numpy()
        share = np.stack([a[:, :, qi, 5 * s_:5 * s_ + 4] for s_, qi in enumerate(QQ)], 2)      # (n, h, 3, 4)
        share = share / share.sum(-1, keepdims=True)
        rr = np.broadcast_to(absr[:, None], share.shape)
        rho = float(spearmanr(rr.ravel(), share.ravel())[0])
        bins = np.digitize(rr.ravel(), q_edges)
        byq = [float(share.ravel()[bins == b].mean()) for b in range(5)]
        out[r['tag']] = dict(cfg=r['cfg'], spearman=rho, share_by_quintile=byq)
        print(r['tag'], 'rho %.3f' % rho, 'share by |r| quintile', np.round(byq, 3), flush=True)
    rhos = [o['spearman'] for o in out.values()]
    out['_summary'] = dict(n=len(rhos), n_negative=int(sum(x < 0 for x in rhos)), median=float(np.median(rhos)),
                           quintile_mean=[float(np.mean([o['share_by_quintile'][b] for o in out.values() if 'cfg' in o])) for b in range(5)])
    json.dump(out, open('results/e22_post.json', 'w'), indent=1)
    print(out['_summary'])


if __name__ == '__main__':
    main()
