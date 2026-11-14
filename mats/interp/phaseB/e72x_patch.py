"""E72 exploration, part 3 (eighth agent; DEV NETS ONLY, disclosed in the E72 amendment; not a registered test): through which routes
does the big-miss departure reach the output? Route patching on E12's matched pairs (d0 = spread, d1 = big miss, source s).
In the big-miss episode d1, one route at a time sees the spread episode's source-s labelled tokens instead:
  P_i    block 1, the summary row of s reads the keys and values of s's four labelled tokens from d0 (E25's site);
  P_ii   block 1, the query row reads them from d0;
  P_iii  block 2 (only the query row reaches the head) reads s's labelled tokens' block-1 states from d0 (E34's route).
The effective weight is d mean / d o*_s (autograd on the input's o column of s's summary token, as e15_run.eff_w); substituted keys,
values and states are detached, so each patch has a detach-only control (the same substitution with d1's own tensors). Route
contribution c_k = sum (w_ctrl_k - w_P_k) / sum W_Bayes over a range: the part of the departure that disappears when route k sees
the spread version. Also the unpatched rom (should equal the registered evaluation's).
usage: python e72x_patch.py OUTJSON TAG=RUNDIR:EVALFILE [...]
"""
import json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e63_run as E63  # noqa: E402

torch.set_num_threads(1)
S, Q, QL, T = 3, 5, 4, 16
SUMM = [4, 9, 14]; LABS = [[s * Q + q for q in range(QL)] for s in range(S)]; QI = 15; OC = 32


def allow_mask(B):
    a = torch.ones(B, T, T, dtype=torch.bool); a[:, :, QI] = False; a[:, QI, QI] = True
    return a


def qkv(blk, x):
    B = x.shape[0]
    q, k, v = blk.qkv(blk.ln1(x)).split(blk.h * blk.dh, -1)
    sp = lambda t: t.view(B, T, blk.h, blk.dh).transpose(1, 2)
    return sp(q), sp(k), sp(v)


def finish(blk, x, q, k, v, allow):
    B = x.shape[0]
    a = ((q @ k.transpose(-1, -2)) / math.sqrt(blk.dh)).masked_fill(~allow[:, None], float('-inf')).softmax(-1)
    out = x + blk.o((a @ v).transpose(1, 2).reshape(B, T, blk.h * blk.dh))
    return out + blk.mlp(blk.ln2(out)) if not blk.attn_only else out


def mean_from_x1(model, x1):
    x2 = finish(model.blocks[1], x1, *qkv(model.blocks[1], x1), allow_mask(x1.shape[0]))
    y = model.out(model.ln(x2[:, QI])); K = model.mixk
    return (torch.softmax(y[:, :K], -1) * y[:, K:2 * K]).sum(1)


def weights(model, X1, X0, src, modes=('full', 'ctrl_i', 'P_i', 'ctrl_ii', 'P_ii', 'ctrl_iii', 'P_iii')):
    """Effective weights on o*_src in d1 for: full, and per route k in (i, ii, iii) the detach-only control and the patch."""
    B = len(X1); bi = torch.arange(B); labs = torch.tensor([LABS[s] for s in src]); summ = torch.tensor([SUMM[s] for s in src])
    out = {}
    blk = model.blocks[0]; al = allow_mask(B)
    with torch.no_grad():
        x0_0 = model.inp(torch.from_numpy(X0).float()); q0, k0, v0 = qkv(blk, x0_0)
        x1_0 = finish(blk, x0_0, q0, k0, v0, al)
    for mode in modes:
        Xt = torch.from_numpy(X1).float().requires_grad_(True)
        x0 = model.inp(Xt); q, k, v = qkv(blk, x0); x1 = finish(blk, x0, q, k, v, al)
        if mode != 'full':
            route = mode.split('_')[1]
            if route in ('i', 'ii'):
                srck, srcv = (k0, v0) if mode.startswith('P') else (k.detach(), v.detach())
                km, vm = k.clone(), v.clone()
                for j in range(QL):
                    km[bi, :, labs[:, j]] = srck[bi, :, labs[:, j]]; vm[bi, :, labs[:, j]] = srcv[bi, :, labs[:, j]]
                xm = finish(blk, x0, q, km, vm, al)
                row = summ if route == 'i' else torch.full((B,), QI)
                x1 = x1.clone(); x1[bi, row] = xm[bi, row]
            else:
                srcx = x1_0 if mode.startswith('P') else x1.detach()
                x1 = x1.clone()
                for j in range(QL):
                    x1[bi, labs[:, j]] = srcx[bi, labs[:, j]]
        m = mean_from_x1(model, x1)
        g, = torch.autograd.grad(m.sum(), Xt)
        out[mode] = g[bi, summ, OC].numpy().astype(np.float64)
    return out


def main():
    outp = sys.argv[1]; res = json.load(open(outp)) if os.path.exists(outp) else {}
    z = np.load('results/e12_pairs.npz'); src, rg = z['src'], z['range']; Wb = z['WB0'][np.arange(len(src)), src]
    for spec in sys.argv[2:]:
        tag, rest = spec.split('=', 1); run, evf = rest.split(':', 1)
        if tag in res:
            continue
        t0 = time.time(); model, arm, meta = E63.load(run, None); evz = np.load(evf, allow_pickle=True)
        PX = []
        for k in (0, 1):
            ps, pq = E63.F.ids_from_X(z['X%d' % k])
            PX.append(E63.F.make_X(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), ps, pq, arm)[0])
        acc = {}; w0 = []
        for i in range(0, len(src), 1000):
            sl = slice(i, i + 1000)
            for kk, vv in weights(model, PX[1][sl], PX[0][sl], src[sl]).items():
                acc.setdefault(kk, []).append(vv)
            w0.append(weights(model, PX[0][sl], PX[0][sl], src[sl], ('full',))['full'])
        acc = {k: np.concatenate(v) for k, v in acc.items()}; w0 = np.concatenate(w0)
        o = dict(net_rom_large=float(evz['rom_large']), net_rom_middle=float(evz['rom_middle']))
        for rr in ('large', 'middle'):
            m = rg == rr; den = Wb[m].sum()
            o['rom_' + rr] = float((acc['full'] - w0)[m].sum() / den)
            for route in ('i', 'ii', 'iii'):
                o['c_%s_%s' % (route, rr)] = float((acc['ctrl_' + route] - acc['P_' + route])[m].sum() / den)
                o['detach_%s_%s' % (route, rr)] = float((acc['full'] - acc['ctrl_' + route])[m].sum() / den)
        o['seconds'] = time.time() - t0; res[tag] = o
        json.dump(res, open(outp, 'w'), indent=1)
        print(tag, json.dumps({k: round(v, 4) for k, v in o.items()}), flush=True)


if __name__ == '__main__':
    main()
