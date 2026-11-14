"""E20.3 (PREREG-B amendment 27 addendum 2): frozen-model attention intervention. For each trained ln model (attention pooling),
evaluate the matched pairs with the attention weights replaced by uniform weights (1/4 each), keeping the embedding, LayerNorm,
noise and readout unchanged, and recompute the departure exactly as e20_summ.py does.
usage: python e20_freeze.py -> results/e20_freeze.json
"""
import glob, json, os
import numpy as np
import torch
import minimal as M
import e20_summ as S


def main():
    y, sid, qid, R, rg, pb = S.pairs()
    Pb = (pb / M.VLEV[None]).sum(1)
    out = {}
    for f in sorted(glob.glob('results/e20/ln_s*_sig*.json')):
        meta = json.load(open(f)); net = M.Net('ln'); net.load_state_dict(torch.load(f[:-5] + '.pt')); net.eval(); net.sigma = meta['sigma']
        res = {}
        for mode in ('attention', 'uniform'):
            torch.manual_seed(0); P = {}
            with torch.no_grad():
                for v in ('big', 'spread'):
                    X = M.tokens(y, R[v], sid, qid); ps = 0; reps = 16 if meta['sigma'] > 0 else 1
                    for _ in range(reps):
                        h = net.tok(X)
                        if mode == 'uniform':
                            logit = net.read(h.mean(1))
                        else:
                            a = torch.softmax(net.k(h) @ net.q / np.sqrt(h.shape[-1]), 1); logit = net.read((a[:, :, None] * h).sum(1))
                        ps = ps + torch.softmax(logit, -1).numpy() / reps
                    P[v] = (ps / M.VLEV[None]).sum(1)
            res[mode] = {r: float((P['big'] - P['spread'])[rg == r].sum() / Pb[rg == r].sum()) for r in ('large', 'middle')}
        out[os.path.basename(f)[:-5]] = dict(sigma=meta['sigma'], **res)
        print(os.path.basename(f)[:-5], res, flush=True)
    s0 = [v for v in out.values() if v['sigma'] == 0.0]
    out['P124'] = dict(holds=bool(all(v['uniform']['middle'] > 0 and v['uniform']['large'] > v['attention']['large'] for v in s0)),
                       n=len(s0))
    json.dump(out, open('results/e20_freeze.json', 'w'), indent=1)
    print('P124', out['P124'])


if __name__ == '__main__':
    main()
