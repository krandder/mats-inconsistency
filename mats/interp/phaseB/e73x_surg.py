"""E73 exploration (eighth agent; DEV NETS ONLY, disclosed in a later amendment; not a registered test): does the source-to-source
variation of the pooling heads' routing coefficients make part of the departure and of the KL? Evaluation-time surgery in block 1:
for every summary row and pooling head, the attention over the source's four own labelled keys is replaced by
    P_row * softmax_j((L_h + K_h r_j) g_j),   g_j = A_j / n_j (real, differentiable),
with L_h, K_h the head's global mean norm-scaled own logit and signed coefficient (E63) and P_row the row's real total mass on those
keys; every other attention entry, the values, block 2 and the head are unchanged (block 2's inputs change). Variants:
  surg     as above;  surg_src  the same formula with the row's own source-mean coefficients (a control: should nearly equal base);
  detq     the base forward with block 1's summary-row queries detached (isolates the path o* -> query -> pooling weights).
Measures: rom on E12's pairs (autograd weights on o*_s) and the test-NLL gap on tiny-pfn's 16,000 test episodes, base and surgery.
usage: python e73x_surg.py OUTJSON TAG=RUNDIR [...]
"""
import json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e63_run as E63  # noqa: E402
import e72_run as R  # noqa: E402
import pfn as P  # noqa: E402

torch.set_num_threads(1)
S, Q, QL, T = 3, 5, 4, 16
SUMM = [4, 9, 14]; LABS = [[s * Q + q for q in range(QL)] for s in range(S)]; QI = 15; OC, YC = 32, 33
TP = '/home/agent/workspaces/mats/work/tiny-pfn'
MODES = tuple(os.environ.get('E73X_MODES', 'base,surg,surg_src').split(','))


def allow_mask(B):
    a = torch.ones(B, T, T, dtype=torch.bool); a[:, :, QI] = False; a[:, QI, QI] = True
    return a


def forward(model, Xt, pool, coef, mode):
    """mode: base | surg | surg_src. Returns the mixture head (mu, logs, logits)."""
    B = Xt.shape[0]; blk = model.blocks[0]; al = allow_mask(B)
    x = model.inp(Xt); h = blk.ln1(x)
    q, k, v = blk.qkv(h).split(blk.h * blk.dh, -1)
    sp = lambda t: t.view(B, T, blk.h, blk.dh).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    if mode == 'detq':                                                     # summary rows' block-1 queries as constants
        q = q.clone(); q[:, :, SUMM] = q[:, :, SUMM].detach()
    a = ((q @ k.transpose(-1, -2)) / math.sqrt(blk.dh)).masked_fill(~al[:, None], float('-inf')).softmax(-1)
    if mode not in ('base', 'detq'):
        X0 = Xt.clone(); X0[..., OC] = Xt[..., YC]
        a0 = model.inp(X0); a0 = a0 - a0.mean(-1, keepdim=True); xc = x - x.mean(-1, keepdim=True)
        g = a0.norm(dim=-1) / xc.norm(dim=-1); r = Xt[..., OC] - Xt[..., YC]                    # (B, T)
        a = a.clone()
        for s in range(S):
            gs, rs = g[:, LABS[s]], r[:, LABS[s]]                                                    # (B, 4)
            for h_ in pool:
                if mode == 'surg':
                    Lh, Kh = coef[h_]
                else:
                    Lh, Kh = coef['src'][h_][:, s:s + 1], coef['srcK'][h_][:, s:s + 1]
                z = (Lh + Kh * rs) * gs
                Prow = a[:, h_, SUMM[s], LABS[s]].sum(-1, keepdim=True)
                a[:, h_, SUMM[s], LABS[s]] = Prow * torch.softmax(z, -1)
    x = x + blk.o((a @ v).transpose(1, 2).reshape(B, T, blk.h * blk.dh))
    x = x + blk.mlp(blk.ln2(x))
    for b2 in model.blocks[1:]:
        q2, k2, v2 = b2.qkv(b2.ln1(x)).split(b2.h * b2.dh, -1)
        sp2 = lambda t: t.view(B, T, b2.h, b2.dh).transpose(1, 2)
        q2, k2, v2 = sp2(q2), sp2(k2), sp2(v2)
        a2 = ((q2 @ k2.transpose(-1, -2)) / math.sqrt(b2.dh)).masked_fill(~al[:, None], float('-inf')).softmax(-1)
        x = x + b2.o((a2 @ v2).transpose(1, 2).reshape(B, T, b2.h * b2.dh))
        x = x + b2.mlp(b2.ln2(x))
    y = model.out(model.ln(x[:, QI])); K = model.mixk
    return y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K]


def src_coef(model, X, pool):
    _, qh, blk = E63.block1_parts(model, X); L, ratio, odd = E63.h3_logits(model, X, qh, blk)
    own = [[QL * s + q for q in range(QL)] for s in range(S)]
    Ls = np.stack([L[:, :, s, own[s]].mean(-1) for s in range(S)], 1); Ks = np.stack([odd[:, :, s, own[s]].mean(-1) for s in range(S)], 1)
    return {h: torch.from_numpy(Ls[:, :, h]).float() for h in pool}, {h: torch.from_numpy(Ks[:, :, h]).float() for h in pool}


def weights(model, X, src, pool, coef, mode, bs=1000):
    out = []
    for i in range(0, len(X), bs):
        Xt = torch.from_numpy(X[i:i + bs]).float().requires_grad_(True)
        c = coef
        if mode == 'surg_src':
            Ls, Ks = src_coef(model, X[i:i + bs], pool); c = dict(src=Ls, srcK=Ks)
        mu, ls, lg = forward(model, Xt, pool, c, mode)
        m = (torch.softmax(lg, -1) * mu).sum(1)
        gr, = torch.autograd.grad(m.sum(), Xt)
        out.append(gr[torch.arange(len(Xt)), torch.tensor([SUMM[s] for s in src[i:i + bs]]), OC].numpy().astype(np.float64))
    return np.concatenate(out)


def gap(model, pool, coef, mode, tz, bs=2000):
    nll = []
    with torch.no_grad():
        for i in range(0, len(tz['X']), bs):
            X = tz['X'][i:i + bs]; c = coef
            if mode == 'surg_src':
                Ls, Ks = src_coef(model, X, pool); c = dict(src=Ls, srcK=Ks)
            hd = forward(model, torch.from_numpy(X).float(), pool, c, mode)
            nll.append(P.head_nll_np(tuple(t.numpy() for t in hd), tz['ystar'][i:i + bs]))
    return float((np.concatenate(nll) - tz['nll_bayes']).mean())


def main():
    outp = sys.argv[1]; res = json.load(open(outp)) if os.path.exists(outp) else {}
    z = np.load('results/e12_pairs.npz'); src, rg = z['src'], z['range']; Wb = z['WB0'][np.arange(len(src)), src]
    tz = np.load(os.path.join(TP, 'runs/testset.npz'))
    for spec in sys.argv[2:]:
        tag, run = spec.split('=', 1)
        if tag in res:
            continue
        t0 = time.time(); model, arm, meta = E63.load(run, None); assert arm == 'ctrl' and not meta.get('variant')
        hd = R.heads_of(model, arm, meta); pool = hd['pool']; coef = {h: (hd['L'][h], hd['K'][h]) for h in pool}
        o = dict(pool=pool)
        for mode in MODES:
            ws = [weights(model, z['X%d' % k], src, pool, coef, mode) for k in (0, 1)]
            o[mode] = {rr: float((ws[1] - ws[0])[rg == rr].sum() / Wb[rg == rr].sum()) for rr in ('large', 'middle')}
            o[mode]['gap'] = gap(model, pool, coef, mode, tz)
        o['seconds'] = time.time() - t0; res[tag] = o
        json.dump(res, open(outp, 'w'), indent=1)
        print(tag, json.dumps(o), flush=True)


if __name__ == '__main__':
    main()
