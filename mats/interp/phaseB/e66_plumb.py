"""E66 plumbing (PREREG-B amendment 74; disclosed, not results), on E63's explored nets (E43 newinit s0, A; s1, B) and an untrained
own net (E60 own s0's step-0 weights):
 1. Surg with no zeroed head reproduces the net's own forward exactly;
 2. with a head zeroed, the summary rows' logits of labelled keys change by exactly the removed odd term, and nothing else changes in
    block 1 (other rows, other heads, non-labelled keys);
 3. the odd term removed equals E63's decomposition: sqrt(d / dh) q.W_K(gamma u~) r_j / n_j (up to LayerNorm's eps).
usage: python e66_plumb.py
"""
import json, math, sys
import numpy as np
import torch

import e66_run as E

S, Q, QL, T = E.S, E.Q, E.QL, E.T


def logits(net, X, v):
    """Block-1 logits as Surg computes them (before masking)."""
    m = net.m; B = X.shape[0]; x = m.inp(X); blk = m.blocks[0]; ln = blk.ln1
    qq, kk, _ = blk.qkv(ln(x)).split(blk.h * blk.dh, -1)
    sp = lambda t: t.view(B, T, blk.h, blk.dh).transpose(1, 2)
    qq, kk = sp(qq), sp(kk)
    sc = qq @ kk.transpose(-1, -2) / math.sqrt(blk.dh)
    if net.zero:
        Wt = m.inp.weight; u = Wt[:, 32] - Wt[:, 32].mean(); Wk = blk.qkv.weight[blk.h * blk.dh:2 * blk.h * blk.dh]
        ku = (Wk @ (ln.weight * u)).view(blk.h, blk.dh)
        xc = x - x.mean(-1, keepdim=True); sig = torch.sqrt((xc ** 2).mean(-1) + ln.eps)
        r = X[:, :, 32] - X[:, :, 33]; odd = (r / sig)[:, E.LAB]; sc = sc.clone()
        for hh in net.zero:
            for s in range(S):
                coef = (qq[:, hh, E.SUMM[s], :] * ku[hh][None]).sum(-1) / math.sqrt(blk.dh)
                sc[:, hh, E.SUMM[s], E.LAB] -= coef[:, None] * odd
    return sc, qq, ku if net.zero else None


def main():
    z7 = np.load('results/e27_data.npz'); sl = slice(0, 200)
    for run in ('runs/e43/e43_newinit_s0', 'runs/e43/e43_newinit_s1'):
        model, arm, meta = E.E30.load(run)
        X, v = E.F.make_X(z7['o'][sl, 0], z7['y'][sl], z7['semb'][sl], z7['qemb'][sl], arm)
        Xt, vt = torch.from_numpy(X).float(), torch.from_numpy(v)
        with torch.no_grad():
            y0 = model(Xt, vt); y1 = E.Surg(model, [], False)(Xt, vt)
            print(run, 'base reproduces the forward: max diff %.1e' % max(float((a - b).abs().max()) for a, b in zip(y0, y1)))
            s0, _, _ = logits(E.Surg(model, [], False), Xt, vt); s1, qq, ku = logits(E.Surg(model, [0], False), Xt, vt)
            dif = (s1 - s0).abs()
            mask = torch.zeros_like(dif, dtype=torch.bool)
            for s in range(S):
                mask[:, 0, E.SUMM[s], E.LAB] = True
            print('   zero head 0: max change outside (head 0, summary rows, labelled keys) %.1e; inside, mean |change| %.3f' % (
                float(dif[~mask].max()), float(dif[mask].mean())))
            # compare with E63's sqrt(d/dh) q.W_K(gamma u~) r / n
            m = model; blk = m.blocks[0]; x = m.inp(Xt); xc = x - x.mean(-1, keepdim=True); n = xc.norm(dim=-1)
            r = Xt[:, :, 32] - Xt[:, :, 33]; d = x.shape[-1]
            e63 = math.sqrt(d / blk.dh) * (qq[:, 0, E.SUMM[0], :] * ku[0][None]).sum(-1)[:, None] * (r / n)[:, E.LAB]
            print('   removed term vs E63 decomposition (source 0, head 0): max relative difference %.1e' % float(
                ((s0 - s1)[:, 0, E.SUMM[0], E.LAB] - e63).abs().max() / e63.abs().max()))
    model, arm, meta = E.E30.load('runs/e60/e60_own_s0')
    model.load_state_dict(torch.load('runs/e60/e60_own_s0/ckpt/step_000000.pt', map_location='cpu'))
    X, v = E.F.make_X(z7['o'][sl, 0], z7['y'][sl], z7['semb'][sl], z7['qemb'][sl], arm)
    with torch.no_grad():
        y0 = model(torch.from_numpy(X).float(), torch.from_numpy(v)); y1 = E.Surg(model, [], True)(torch.from_numpy(X).float(), torch.from_numpy(v))
    print('own step 0: base reproduces the forward: max diff %.1e' % max(float((a - b).abs().max()) for a, b in zip(y0, y1)))


if __name__ == '__main__':
    main()
