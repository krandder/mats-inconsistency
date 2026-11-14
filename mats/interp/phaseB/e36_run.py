"""E36 per-net measures (PREREG-B amendment 44): which routes carry the single-source negation shift? Path patching with real states.
For each base and each single-source negation s (E33's orbit members), the predictive mean of hybrids that take one computation
from the negated run and everything else from the base run:
  summ_own   block 1's attention output at source s's summary token (5s+4) from the negated run;
  summ_all   block 1's attention outputs at all three summary tokens from the negated run;
  query1     block 1's attention output at the query token from the negated run;
  lab2       block 2's query read of the labelled tokens (their keys and values) from the negated run (E34's path).
(Summary and query tokens have the same embeddings in both runs, so only their block-1 attention outputs differ.)
Shares: share_r = sum dm_r dm_full / sum dm_full^2 over bases and the three negations, dm_r = m(hybrid r) - m(base),
dm_full = m(negated) - m(base). On the first 3,000 bases of E28's data.
usage: python e36_run.py SET I0 I1 -> results/e36/<SET>/<tag>.npz
"""
import json, math, os, sys, time
import numpy as np
import torch

import e26_run as RU
import e33_run as E33
import e34_run as E34
import world as W

torch.set_num_threads(1)
NB = 3000
SUMM = [4, 9, 14]


def attn_out(blk, x, allow):
    B, T, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
    sp = lambda t: t.view(B, T, blk.h, d // blk.h).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    a = (q @ k.transpose(-1, -2) / math.sqrt(d // blk.h)).masked_fill(~allow[:, None], float('-inf')).softmax(-1)
    return blk.o((a @ v).transpose(1, 2).reshape(B, T, d))


def finish_block1(blk, x_in, att):
    x = x_in + att
    return x + blk.mlp(blk.ln2(x)) if not blk.attn_only else x


def out_mean(model, x1, allow):
    q, k, v = E34.qkv(model.blocks[1], x1)
    h = E34.query_out(model, x1[:, -1], q[:, :, -1], k, v, allow[:, -1])
    return E34.mean_of([t.numpy().astype(np.float64) for t in h])


def main():
    which, i0, i1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = [r for r in json.load(open('results/e26_nets.json')) if r['set'] == which][i0:i1]
    z = np.load('results/e28_data.npz')
    y = z['y'][:NB]; semb = z['semb'][:NB]; qemb = z['qemb'][:NB]; res0 = z['res'][:NB, 0]; o0 = z['o'][:NB, 0]
    os.makedirs(os.path.join('results/e36', which), exist_ok=True)
    pats = [E33.SIGNS[0]] + [E33.SIGNS[i] for i in E34.SINGLE]
    for row in rows:
        p = os.path.join('results/e36', which, row['tag'] + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, kind = RU.load(row); assert kind in ('clean', 'random')
        b1 = model.blocks[0]
        with torch.no_grad():
            XIN, ATT = [], []
            for sg in pats:
                o = o0.copy(); o[:, :, :W.QL] = y[:, None, :W.QL] + res0 * np.array(sg)[None, :, None]
                X, valid = W.assemble(semb, qemb, o, y); Xt, vt = torch.from_numpy(X), torch.from_numpy(valid)
                allow = E33.allow_mask(vt)
                xin = model.inp(Xt); XIN.append(xin); ATT.append(attn_out(b1, xin, allow))
            X1 = [finish_block1(b1, XIN[i], ATT[i]) for i in range(4)]
            m0 = out_mean(model, X1[0], allow)
            acc = {k: [0.0, 0.0] for k in ('summ_own', 'summ_all', 'query1', 'lab2')}
            full2 = 0.0
            for j, s in enumerate(range(3)):
                i = j + 1
                dfull = out_mean(model, X1[i], allow) - m0; full2 += float((dfull ** 2).sum())
                hyb = {}
                for name, toks in (('summ_own', [SUMM[s]]), ('summ_all', SUMM), ('query1', [15])):
                    att = ATT[0].clone(); att[:, toks] = ATT[i][:, toks]
                    hyb[name] = out_mean(model, finish_block1(b1, XIN[0], att), allow)
                qb, kb, vb = E34.qkv(model.blocks[1], X1[0]); qm, km, vm = E34.qkv(model.blocks[1], X1[i])
                K = kb.clone(); V = vb.clone(); K[:, :, E34.LAB] = km[:, :, E34.LAB]; V[:, :, E34.LAB] = vm[:, :, E34.LAB]
                h = E34.query_out(model, X1[0][:, -1], qb[:, :, -1], K, V, allow[:, -1])
                hyb['lab2'] = E34.mean_of([t.numpy().astype(np.float64) for t in h])
                for name, mh in hyb.items():
                    d = mh - m0; acc[name][0] += float((d * dfull).sum()); acc[name][1] += float((d ** 2).sum())
        out = dict(full2=full2)
        for name, (pr, sq) in acc.items():
            out['proj_' + name], out['sq_' + name] = pr, sq
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(which, row['tag'], {k: round(out['proj_' + k] / full2, 3) for k in acc}, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
