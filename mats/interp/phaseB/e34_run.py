"""E34 per-net measures (PREREG-B amendment 42): how much of the sign-odd dependence travels by one path, block 2's query read of
the labelled tokens? Path patching with real counterfactual states (the coordinator's design).
For each base, its negation orbit (8 patterns; E33) gives real block-1 states for every sign pattern. In block 2 only the query
token's row reaches the head. Two hybrids per orbit member m:
  path    the query's block-2 attention reads the labelled tokens' keys and values from member m's block-1 states, and everything
          else (its own state and query, the other tokens' keys and values) from the base's: only this path sees the new signs;
  rev     the reverse: everything from member m, except the labelled tokens' keys and values, which come from the base.
Measures, on E28's first 3,000 bases:
  negation shifts for the three single-source negations s: dm_full = m(member s) - m(base), dm_path = m(path hybrid) - m(base),
  dm_rev = m(member s) - m(rev hybrid); shares share_path = sum dm_path dm_full / sum dm_full^2, share_rev likewise;
  KL-exact (uncorrected) negation cost over E28's 9-sample negation family: full, path-only (each sample replaced by its path hybrid)
  and all-but-path (each sample replaced by its rev hybrid); mean KL of each family; minimum grid mass;
  marker: the query's block-2 attention mass on the labelled tokens and on the summary tokens (base episodes).
usage: python e34_run.py SET I0 I1 -> results/e34/<SET>/<tag>.npz   (sets and nets as in results/e26_nets.json)
"""
import json, math, os, sys, time
import numpy as np
import torch

import e26_run as RU
import e28_data as E8
import e33_run as E33
import world as W

torch.set_num_threads(1)
NB = 3000
LAB = [5 * s + q for s in range(3) for q in range(4)]
SUMM = [4, 9, 14]
SINGLE = [E33.SIGNS.index(sg) for sg in [(-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)]]


def qkv(blk, x):
    B, T, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
    sp = lambda t: t.view(B, T, blk.h, d // blk.h).transpose(1, 2)
    return sp(q), sp(k), sp(v)


def query_out(model, xq, q, K, V, allow_q):
    """Block-2 output for the query token and the head. xq (B, d) its block-1 state; q (B, h, dh); K, V (B, h, T, dh)."""
    blk = model.blocks[1]; B, h, T, dh = K.shape
    s = (q[:, :, None, :] @ K.transpose(-1, -2))[:, :, 0] / math.sqrt(dh)            # (B, h, T)
    a = s.masked_fill(~allow_q[:, None, :], float('-inf')).softmax(-1)
    o = (a[:, :, :, None] * V).sum(2).reshape(B, h * dh)
    x = xq + blk.o(o)
    if not blk.attn_only:
        x = x + blk.mlp(blk.ln2(x))
    y = model.out(model.ln(x)); Kh = model.mixk
    if Kh == 0:
        return y[:, 0:1], y[:, 1:2].clamp(-6, 3), torch.zeros_like(y[:, 0:1])
    return y[:, Kh:2 * Kh], y[:, 2 * Kh:].clamp(-6, 3), torch.log_softmax(y[:, :Kh], -1)


def mean_of(h):
    mu, ls, lw = h
    return (np.exp(lw) * mu).sum(1)


def main():
    which, i0, i1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = [r for r in json.load(open('results/e26_nets.json')) if r['set'] == which][i0:i1]
    z = np.load('results/e28_data.npz')
    y = z['y'][:NB]; semb = z['semb'][:NB]; qemb = z['qemb'][:NB]; res0 = z['res'][:NB, 0]; o0 = z['o'][:NB, 0]
    fam = E8.fam_idx('negate_ind')
    sgn = np.sign((z['res'][:NB][:, fam] * res0[:, None]).sum(-1)); sgn[sgn == 0] = 1.0
    member = np.zeros((NB, len(fam)), np.int64)
    for i, sg in enumerate(E33.SIGNS):
        member[np.all(sgn == np.array(sg)[None, None], axis=2)] = i
    g = z['bmean'][:NB, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][:NB], z['bvars'], z['bwts'][:NB], g)
    os.makedirs(os.path.join('results/e34', which), exist_ok=True)
    rows_idx = np.arange(NB)
    for row in rows:
        p = os.path.join('results/e34', which, row['tag'] + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, kind = RU.load(row); assert kind in ('clean', 'random')
        with torch.no_grad():
            X1, QKV = [], []
            for sg in E33.SIGNS:
                o = o0.copy(); o[:, :, :W.QL] = y[:, None, :W.QL] + res0 * np.array(sg)[None, :, None]
                X, valid = W.assemble(semb, qemb, o, y)
                Xt, vt = torch.from_numpy(X), torch.from_numpy(valid)
                allow = E33.allow_mask(vt)
                x1 = E33.block(model.blocks[0], model.inp(Xt), allow)
                X1.append(x1); QKV.append(qkv(model.blocks[1], x1))
            allow_q = allow[:, -1]
            qb, kb, vb = QKV[0]
            a0 = (qb[:, :, -1:, :] @ kb.transpose(-1, -2))[:, :, 0] / math.sqrt(kb.shape[-1])
            a0 = a0.masked_fill(~allow_q[:, None, :], float('-inf')).softmax(-1)
            out = dict(m_lab=float(a0[:, :, LAB].sum(-1).mean()), m_summ=float(a0[:, :, SUMM].sum(-1).mean()))
            H = {'full': [], 'path': [], 'rev': []}
            for i in range(len(E33.SIGNS)):
                qm, km, vm = QKV[i]
                H['full'].append(query_out(model, X1[i][:, -1], qm[:, :, -1], km, vm, allow_q))
                K = kb.clone(); V = vb.clone(); K[:, :, LAB] = km[:, :, LAB]; V[:, :, LAB] = vm[:, :, LAB]
                H['path'].append(query_out(model, X1[0][:, -1], qb[:, :, -1], K, V, allow_q))
                K = km.clone(); V = vm.clone(); K[:, :, LAB] = kb[:, :, LAB]; V[:, :, LAB] = vb[:, :, LAB]
                H['rev'].append(query_out(model, X1[i][:, -1], qm[:, :, -1], K, V, allow_q))
            Hn = {k: [[t.numpy().astype(np.float64) for t in h] for h in v] for k, v in H.items()}
        M = {k: np.stack([mean_of(h) for h in v], 1) for k, v in Hn.items()}             # (n, 8) predictive means
        dfull = np.stack([M['full'][:, i] - M['full'][:, 0] for i in SINGLE], 1)
        dpath = np.stack([M['path'][:, i] - M['full'][:, 0] for i in SINGLE], 1)
        drev = np.stack([M['full'][:, i] - M['rev'][:, i] for i in SINGLE], 1)
        out.update(dm_full2=float((dfull ** 2).sum()), dm_path_full=float((dpath * dfull).sum()), dm_rev_full=float((drev * dfull).sum()),
                   dm_path2=float((dpath ** 2).sum()), dm_rev2=float((drev ** 2).sum()))
        mass = 1.0
        for k in ('full', 'path', 'rev'):
            hds = []
            for j in range(len(fam)):
                mi = member[:, j]
                hds.append(tuple(np.stack([Hn[k][i][c] for i in range(len(E33.SIGNS))], 0)[mi, rows_idx] for c in range(3)))
            kl, pc, m = RU.set_stats(hds, lpB, g)
            out['pc_' + k], out['kl_' + k] = float(pc.mean()), float(kl.mean()); mass = min(mass, float(m.min()))
        out['mass'] = mass
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(which, row['tag'], {k: round(v, 5) for k, v in out.items()}, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
