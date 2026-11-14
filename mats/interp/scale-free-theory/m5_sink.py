"""M5 (scale-free-theory addendum, PREREG-SF.md): is the query token's attention to itself the prior sink?

Identity (6.1): with Gaussian messages and a sink of logit log tau_0, the sink weight is tau_0 x posterior variance.
For each net, in the combining layer (block 2 of 2-layer nets; block 1 of 1-layer nets), per head, a_self(x) = the
query row's attention on the query token. Report, per head, the correlation and slope of log a_self against
log Var_Bayes(x) (and against log of the net's own predictive variance) on test episodes.
Clean nets: work/scale-free-theory/nets/clean; noisy phase-B nets: nets/noisy (attention averaged over K noise draws).
"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/scale-free-theory')
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp-B')
from sf_net import load_clean, forward, episodes, allow_mask, NETS
import world as W
import nnoise as NN

CLEAN = [f'w32_h2_L2_s{s}' for s in range(5)] + [f'w32_h2_L2_mix4_s{s}' for s in range(5)] + [f'w32_h2_L1_s{s}' for s in range(5)]
NOISY = ['L2_s0.05', 'L2seed1_s0.05', 'L2mx0_s0.05', 'L2mx1_s0.05', 'L2mx2_s0.05', 'L2nb6_s0.05',
         'L2seed2_s0.05', 'L2ws0_s0.05', 'L2ws3_s0.05', 'L2mx4_s0.05', 'L2nb5_s0.05', 'L2nb7_s0.05']
TEST_SEED, KD = 502, 8


def head_var(head):
    if len(head) == 2:
        return (2 * head[1]).exp().numpy().astype(np.float64)
    mu, ls, lg = head
    w = torch.softmax(lg, -1)
    m = (w * mu).sum(-1)
    return ((w * (ls.exp() ** 2 + mu ** 2)).sum(-1) - m ** 2).numpy().astype(np.float64)


def stats(x, y):
    r = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(y, x, 1)[0])
    return r, slope


def noisy_attn_self(model, X, valid, layer, gen):
    """Forward of a NoisyPFN replicating nnoise.forward, returning the query row's self-attention per head at `layer`
    (1-based) and the head outputs."""
    allow = NN.NoisyPFN.allow_mask(valid)
    sg = model.sigma
    x = model.noise(model.f_emb(X), sg, gen)
    aself = None
    for l in range(1, model.depth + 1):
        b = model.blocks[l - 1]
        if l == layer:
            B, T, d = x.shape
            q, k, v = b.qkv(b.ln1(x)).split(b.h * b.dh, -1)
            sp = lambda t: t.view(B, T, b.h, b.dh).transpose(1, 2)
            q, k = sp(q), sp(k)
            s = (q @ k.transpose(-1, -2) / math.sqrt(b.dh)).masked_fill(~allow[:, None], float('-inf'))
            aself = s.softmax(-1)[:, :, T - 1, T - 1]
        xa = model.noise(model.f_a(l, x, allow), sg, gen)
        m = model.noise(model.f_m(l, xa), sg, gen)
        x = model.noise(model.f_r(l, xa, m), sg, gen)
    return aself, model.head(x)


def main():
    t0 = time.time()
    res = {'clean': {}, 'noisy': {}}
    d, XT, mT = episodes(4000, TEST_SEED, 16)
    out, nll, mom, w = W.references(d)
    lvb = np.log(mom['bayes'][1])
    for name in CLEAN:
        model, meta = load_clean(name)
        with torch.no_grad():
            head, attns = forward(model, XT, mT)
        layer = len(model.blocks) - 1
        A = attns[layer].numpy()
        lvn = np.log(head_var(head))
        rec = {}
        for h in range(A.shape[1]):
            la = np.log(np.maximum(A[:, h, 15, 15], 1e-30))
            rb, sb = stats(la, lvb); rn, sn = stats(la, lvn)
            rec[f'head{h}'] = {'r_bayes_var': rb, 'slope_bayes_var': sb, 'r_net_var': rn, 'slope_net_var': sn,
                               'mean_aself': float(A[:, h, 15, 15].mean())}
        res['clean'][name] = rec
        print(name, json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in rec.items()}), flush=True)
    g = torch.Generator().manual_seed(7)
    for name in NOISY:
        model, meta = NN.load_run(os.path.join('/home/agent/workspaces/mats/work/scale-free-theory/nets/noisy', name))
        acc = None; mus, lss, lgs = [], [], []
        with torch.no_grad():
            for kd in range(KD):
                a, hd = noisy_attn_self(model, XT, mT, model.depth, g)
                acc = a if acc is None else acc + a
                mus.append(hd[0]); lss.append(hd[1]); lgs.append(torch.log_softmax(hd[2], -1) - math.log(KD))
        A = (acc / KD).numpy()
        mu = torch.cat(mus, -1); ls = torch.cat(lss, -1); lg = torch.cat(lgs, -1)
        lvn = np.log(head_var((mu, ls, lg)))
        rec = {}
        for h in range(A.shape[1]):
            la = np.log(np.maximum(A[:, h], 1e-30))
            rb, sb = stats(la, lvb); rn, sn = stats(la, lvn)
            rec[f'head{h}'] = {'r_bayes_var': rb, 'slope_bayes_var': sb, 'r_net_var': rn, 'slope_net_var': sn,
                               'mean_aself': float(A[:, h].mean())}
        res['noisy'][name] = rec
        print(name, json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in rec.items()}), round(time.time() - t0), 's', flush=True)
    json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/m5_sink.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
