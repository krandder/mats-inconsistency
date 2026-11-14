"""E33 per-net measures (PREREG-B amendment 41): which tokens carry the sign-odd dependence out of block 1?
For each base, its negation orbit is the 8 episodes obtained by negating any subset of the three sources' labelled-residual
vectors; all have the same Bayes posterior, and E28's negation family (9 samples) lies inside it. Block 1's output x1 is computed
for all 8 orbit members; x1bar is their mean, which is even under negating any source. An arm replaces x1 at a set of token
positions by x1bar, in every orbit member, and runs block 2 and the head unchanged:
  none   no replacement;
  summ   the three sources' summary tokens (query-question tokens 4, 9, 14);
  lab    the twelve labelled tokens;
  query  the query token (15);
  all    every token (block 2 then sees the same input for all orbit members, so the negation cost is exactly 0: a check).
Per arm, on E28's negation family (first 3,000 bases): the KL-exact (uncorrected) negation cost, its mean KL, and the base KL.
usage: python e33_run.py SET I0 I1 -> results/e33/<SET>/<tag>.npz   (sets and nets as in results/e26_nets.json)
"""
import itertools, json, math, os, sys, time
import numpy as np
import torch

import e26_run as RU
import e28_data as E8
import world as W

torch.set_num_threads(1)
NB = 3000
ARMS = {'none': [], 'summ': [4, 9, 14], 'lab': [5 * s + q for s in range(3) for q in range(4)], 'query': [15], 'all': list(range(16))}
SIGNS = list(itertools.product([1.0, -1.0], repeat=3))          # orbit members: per-source global signs


def allow_mask(valid):
    B, T = valid.shape; qi = T - 1
    ctx = valid.clone(); ctx[:, qi] = False
    allow = ctx[:, None, :].expand(B, T, T).clone(); allow[:, qi, qi] = True
    eye = torch.eye(T, dtype=torch.bool)[None]
    return allow | (eye & ~valid[:, :, None])


def block(blk, x, allow):
    B, T, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
    sp = lambda t: t.view(B, T, blk.h, d // blk.h).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    s = (q @ k.transpose(-1, -2) / math.sqrt(d // blk.h)).masked_fill(~allow[:, None], float('-inf'))
    x = x + blk.o((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, d))
    return x + blk.mlp(blk.ln2(x)) if not blk.attn_only else x


def head_of(model, x):
    y = model.out(model.ln(x[:, -1])); K = model.mixk
    if K == 0:
        return y[:, 0:1], y[:, 1:2].clamp(-6, 3), torch.zeros_like(y[:, 0:1])
    lg = torch.log_softmax(y[:, :K], -1)
    return y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), lg


def main():
    which, i0, i1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = [r for r in json.load(open('results/e26_nets.json')) if r['set'] == which][i0:i1]
    z = np.load('results/e28_data.npz')
    y = z['y'][:NB]; semb = z['semb'][:NB]; qemb = z['qemb'][:NB]; res0 = z['res'][:NB, 0]; o0 = z['o'][:NB, 0]
    # map each negation-family sample to its orbit member index (its per-source signs relative to the base)
    fam = E8.fam_idx('negate_ind')
    sgn = np.sign((z['res'][:NB][:, fam] * res0[:, None]).sum(-1))         # (n, 9, 3) = +1 / -1 per source
    sgn[sgn == 0] = 1.0
    member = np.zeros((NB, len(fam)), np.int64)
    for i, sg in enumerate(SIGNS):
        member[np.all(sgn == np.array(sg)[None, None], axis=2)] = i
    g = z['bmean'][:NB, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][:NB], z['bvars'], z['bwts'][:NB], g)
    os.makedirs(os.path.join('results/e33', which), exist_ok=True)
    for row in rows:
        p = os.path.join('results/e33', which, row['tag'] + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, kind = RU.load(row); assert kind in ('clean', 'random')
        out = {}
        with torch.no_grad():
            X1 = []
            for sg in SIGNS:
                o = o0.copy(); o[:, :, :W.QL] = y[:, None, :W.QL] + res0 * np.array(sg)[None, :, None]
                X, valid = W.assemble(semb, qemb, o, y)
                Xt, vt = torch.from_numpy(X), torch.from_numpy(valid)
                X1.append(block(model.blocks[0], model.inp(Xt), allow_mask(vt)))
            X1 = torch.stack(X1, 0)                                         # (8, n, T, d)
            x1bar = X1.mean(0)
            allow = allow_mask(vt)
            for arm, pos in ARMS.items():
                H = []
                for i in range(len(SIGNS)):
                    x = X1[i].clone()
                    if pos:
                        x[:, pos] = x1bar[:, pos]
                    x = block(model.blocks[1], x, allow)
                    H.append([t.numpy().astype(np.float64) for t in head_of(model, x)])
                rows_idx = np.arange(NB)
                hds = []
                for j in range(len(fam)):                                   # the 9 family samples, as orbit members
                    mi = member[:, j]
                    hds.append(tuple(np.stack([H[i][c] for i in range(len(SIGNS))], 0)[mi, rows_idx] for c in range(3)))
                kl, pc, m = RU.set_stats(hds, lpB, g)
                out['pc_' + arm], out['kl_' + arm] = float(pc.mean()), float(kl.mean())
                kl0, _, _ = RU.set_stats([tuple(H[0][c] for c in range(3))], lpB, g)
                out['klbase_' + arm] = float(kl0.mean())
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(which, row['tag'], {k: round(v, 5) for k, v in out.items()}, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
