"""N5 (scale-free-theory, 2026-10-29): prototype of measurement M1 (interaction profile) on trained tiny-PFN nets.

For a cut C of the tokens and its complement, with a, a' two draws of C's errors and b, b' two draws of the rest's,
    Delta = L(a, b) - L(a', b) - L(a, b') + L(a', b'),   L(x) = log p(y1 | x) - log p(y2 | x)  (y1, y2 fixed per episode).
Share = Var(Delta) / (Var(main effect of C) + Var(main effect of rest)).
Exact Bayes: Delta = 0 for source-aligned cuts (log-odds add over sources given y*); non-zero within a source.
Cuts: 'source' = all 5 tokens of source 0 (its v_0 and errors redrawn); 'within' = source 0's labelled questions 0, 1
(errors redrawn with the same v_0; the complement's redraw keeps v_0 and redraws everything else).
IDs (random per episode) are held fixed across the four variants. Read-only use of work/tiny-pfn (world.py, pfn.py).
"""
import json, math, sys, os
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import torch
import world as W
import pfn as P

RUNS = '/home/agent/workspaces/mats/work/tiny-pfn/runs'
DY = np.array([-0.6, -0.3, 0.3, 0.6])          # log-odds of y* + DY versus y*


def bayes_logp(d, yq):
    """log p(y | x) for exact Bayes at points yq (n, m)."""
    out, nll, mom, w = W.references(d)
    means, vars_, wts = out['bayes']
    comp = -0.5 * np.log(2 * np.pi * vars_)[None, None] - 0.5 * (yq[:, :, None] - means[:, None, :]) ** 2 / vars_[None, None]
    mx = comp.max(2, keepdims=True)
    return mx[:, :, 0] + np.log((wts[:, None, :] * np.exp(comp - mx)).sum(2))


def net_logp(model, X, m, yq):
    with torch.no_grad():
        head = model(torch.from_numpy(X), torch.from_numpy(m))
    head = [t.numpy().astype(np.float64) for t in head]
    if len(head) == 2:
        mu, logs = head
        return -logs[:, None] - 0.5 * math.log(2 * math.pi) - 0.5 * ((yq - mu[:, None]) / np.exp(logs)[:, None]) ** 2
    mu, logs, logits = head
    lw = logits - logits.max(-1, keepdims=True); lw = lw - np.log(np.exp(lw).sum(-1, keepdims=True))
    lp = lw[:, None, :] - logs[:, None, :] - 0.5 * math.log(2 * math.pi) - 0.5 * ((yq[:, :, None] - mu[:, None, :]) / np.exp(logs)[:, None, :]) ** 2
    mx = lp.max(-1, keepdims=True)
    return mx[:, :, 0] + np.log(np.exp(lp - mx).sum(-1))


def variants(n, rng, cut):
    """Four episodes per base: (a,b), (a',b), (a,b'), (a',b'). Returns list of dicts and fixed IDs."""
    S, Q = W.S, W.Q
    vidx = rng.integers(0, W.NV, (n, S)); v = W.VLEV[vidx]
    y = rng.standard_normal((n, Q))
    e = rng.standard_normal((n, S, Q)) * np.sqrt(v)[:, :, None]
    # alternative draws
    vidx2 = rng.integers(0, W.NV, (n, S)); v2 = W.VLEV[vidx2]
    e2_newv = rng.standard_normal((n, S, Q)) * np.sqrt(v2)[:, :, None]
    e2_samev = rng.standard_normal((n, S, Q)) * np.sqrt(v)[:, :, None]
    def make(vv, vi, ee):
        return dict(vidx=vi, v=vv, y=y, o=y[:, None, :] + ee)
    if cut == 'source':
        # C = source 0 (redraw v_0 and errors); rest = sources 1, 2 (redraw v and errors)
        vA = v.copy(); viA = vidx.copy(); eA = e.copy()
        vA[:, 0] = v2[:, 0]; viA[:, 0] = vidx2[:, 0]; eA[:, 0] = e2_newv[:, 0]
        vB = v.copy(); viB = vidx.copy(); eB = e.copy()
        vB[:, 1:] = v2[:, 1:]; viB[:, 1:] = vidx2[:, 1:]; eB[:, 1:] = e2_newv[:, 1:]
        vAB = v2.copy(); viAB = vidx2.copy(); eAB = e2_newv.copy()
        ds = [make(v, vidx, e), make(vA, viA, eA), make(vB, viB, eB), make(vAB, viAB, eAB)]
    else:
        # C = source 0's labelled q0, q1 (same v_0); rest = source 0's q2, q3, query (same v_0) + sources 1, 2 (new v)
        cmask = np.zeros((S, Q), bool); cmask[0, 0] = cmask[0, 1] = True
        alt = e2_samev.copy(); alt[:, 1:] = e2_newv[:, 1:]
        valt = v.copy(); valt[:, 1:] = v2[:, 1:]; vialt = vidx.copy(); vialt[:, 1:] = vidx2[:, 1:]
        eA = np.where(cmask[None], alt, e)
        eB = np.where(cmask[None], e, alt); vB = valt; viB = vialt
        eAB = alt
        ds = [make(v, vidx, e), make(v, vidx, eA), make(vB, viB, eB), make(vB, viB, eAB)]
    return ds


def shares(Lv):
    L00, L10, L01, L11 = Lv
    D = L00 - L10 - L01 + L11
    A = L10 - L00; B = L01 - L00
    return float(np.var(D) / (np.var(A) + np.var(B))), float(np.sqrt(np.mean(D ** 2))), float(np.sqrt(np.mean(A ** 2))), float(np.sqrt(np.mean(B ** 2)))


def main():
    n = 4000
    cfgs = sys.argv[1:] if len(sys.argv) > 1 else ['w32_h2_L2', 'w32_h2_L2_mix4', 'w32_h2_L2_mix4_ws', 'w16_h1_L2_mix4', 'w32_h1_L2',
                                                   'w32_h2_L2_attn', 'w32_h2_L1', 'w16_h1_L1']
    res = {}
    for cut in ['source', 'within']:
        rng = np.random.default_rng(11 if cut == 'source' else 12)
        ds = variants(n, rng, cut)
        ystar = ds[0]['y'][:, W.QL]
        yq = ystar[:, None] + np.concatenate([[0.0], DY])[None]
        # Bayes
        Lb = []
        for d in ds:
            lp = bayes_logp(d, yq)
            Lb.append(lp[:, 1:] - lp[:, :1])
        res.setdefault('bayes', {})[cut] = shares([x.ravel() for x in Lb])
        LB_ref = [x.copy() for x in Lb]
        print(cut, 'bayes', res['bayes'][cut], flush=True)
        # Bayes moment-matched Gaussian: the right reference for Gaussian-head nets
        Lg = []
        for d in ds:
            out, nll, mom, w = W.references(d)
            mm, vv = mom['bayes']
            lp = -0.5 * (yq - mm[:, None]) ** 2 / vv[:, None]
            Lg.append(lp[:, 1:] - lp[:, :1])
        res.setdefault('bayes_gauss', {})[cut] = shares([x.ravel() for x in Lg])
        LG_ref = [x.copy() for x in Lg]
        print(cut, 'bayes_gauss', res['bayes_gauss'][cut], flush=True)
        for cfg in cfgs:
            for s in range(5):
                rd = os.path.join(RUNS, f'{cfg}_s{s}')
                if not os.path.exists(os.path.join(rd, 'model.pt')):
                    continue
                meta = json.load(open(os.path.join(rd, 'meta.json')))
                model = P.build(meta); model.load_state_dict(torch.load(os.path.join(rd, 'model.pt'))); model.eval()
                rngid = np.random.default_rng(99)
                semb = rngid.standard_normal((n, W.S, meta['ds'])) / 4
                qemb = rngid.standard_normal((n, W.Q, meta['ds'])) / 4
                Ln = []
                for d in ds:
                    X, m = W.assemble(semb, qemb, d['o'], d['y'])
                    lp = net_logp(model, X, m, yq)
                    Ln.append(lp[:, 1:] - lp[:, :1])
                key = f'{cfg}_s{s}'
                res.setdefault(key, {})[cut] = shares([x.ravel() for x in Ln])
                # deviation from Bayes (and from Bayes-Gaussian): how non-additive is the ERROR itself
                res[key][cut + '_dev'] = shares([(x - y).ravel() for x, y in zip(Ln, LB_ref)])
                res[key][cut + '_devg'] = shares([(x - y).ravel() for x, y in zip(Ln, LG_ref)])
                ev = json.load(open(os.path.join(rd, 'eval.json'))) if os.path.exists(os.path.join(rd, 'eval.json')) else {}
                res[key]['kl_bayes'] = ev.get('best', {}).get('kl', None)
                print(cut, key, [round(x, 4) for x in res[key][cut]], flush=True)
    json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n5_interaction_v2.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
