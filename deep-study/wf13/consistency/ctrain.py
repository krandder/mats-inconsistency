"""ctrain.py DATADIR OUTDIR JOBLIST SHARD NSHARDS [STEPS] : train amplified predictors and write their test forecasts.

A job line is "ARM N D R": ARM = data | FORM@WEIGHT | fixed:NAME ; N = number of real check vectors (0 = none);
D = which disjoint subset of real vectors; R = initialisation seed.
Loss = log loss on REAL outcomes of the N vectors' training part (3/4; the last 1/4, at most 1,024 vectors, is the
validation part that picks the checkpoint) + WEIGHT x a consistency term that reads NO real outcome:
  i       order against order on pairs: the joint of (X, Y) given C built as A(X|C) A(Y|C,X) against A(Y|C) A(X|C,Y);
          mean over triples of sum_{x,y} w(x,y) (log J1 - log J2)^2, w = stop-gradient of (J1 + J2) / 2
  iio     variance over two sampled orders of the summed log step predictions for one terminal check vector, / 30
  iis     variance over sampled trajectories (terminal vector, order) of [summed log step predictions - log R-hat(x)],
          log R-hat = the score's value of the terminal vector (cgen.score_ref), / 30
  iii     over compute (wf10's form): the K forecast predicts the stop-gradient 4K forecast (KL)
  iiisym  the same without stop-gradient (symmetric KL): the constructed test of "reliable dragged to unreliable"
  synth   reference, not a consistency term: log loss on outcomes sampled from M itself (the score's own samples)
All consistency contexts are M's own samples (sets c1, c2). No exact inference is imported here.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
import cmodel as CM  # noqa: E402
from cmodel import LEVELS  # noqa: E402

EVAL_EVERY = 100
SMALL = int(os.environ.get('SMALL', 0))             # CPU fallback: half batches, smaller validation pass
B_DATA, B_PAIR, B_ORD, B_UNL = (128, 48, 6, 128) if SMALL else (256, 96, 12, 256)
N_VAL, N_VTG = (512, 4) if SMALL else (1024, 8)


def subset(lab, n, d):
    """Context indices (train, val) of real-vector subset d of size n."""
    vec, copy = lab.meta['vec'], lab.meta['copy']
    nv = int(vec.max()) + 1
    cnt = torch.bincount(vec)                                  # contexts per vector
    lo, hi = (d * n, (d + 1) * n) if 2 * n <= nv else (0, nv)
    c = int(cnt[lo:hi].min())
    nval = min(n // 4, 1024)
    in_set = (vec >= lo) & (vec < hi) & (copy < c)
    tr = torch.where(in_set & (vec < hi - nval))[0]
    va = torch.where(in_set & (vec >= hi - nval))[0]
    return tr, va, c


def rand_target(mask, g):
    r = torch.rand(mask.shape, generator=g, device=mask.device).masked_fill(mask, -1.0)
    return r.argmax(1)


def rsel(B, Kmax, K, g, dev):
    return torch.rand(B, Kmax, generator=g, device=dev).argsort(1)[:, :K]


def lp2(z):
    """logit (B,) -> log P(value 0), log P(value 1) stacked (B,2)."""
    return torch.stack([F.logsigmoid(-z), F.logsigmoid(z)], -1)


def pair_joints(z):
    """z (n,6): logits of A(X|C), A(Y|C), A(Y|C,X=0), A(Y|C,X=1), A(X|C,Y=0), A(X|C,Y=1).
    Returns log J1, log J2 (n,2,2) indexed [x, y]: J1 = A(X) A(Y|X), J2 = A(Y) A(X|Y)."""
    lx, ly = lp2(z[:, 0]), lp2(z[:, 1])
    lyx = torch.stack([lp2(z[:, 2]), lp2(z[:, 3])], 1)          # [x, y]
    lxy = torch.stack([lp2(z[:, 4]), lp2(z[:, 5])], 2)          # [x, y]: A(X = x | Y = y)
    return lx[:, :, None] + lyx, ly[:, None, :] + lxy


def cons_i(model, C1, g, Bc, dev):
    n = C1.N // 5
    K = LEVELS[int(torch.randint(0, len([k for k in LEVELS if k <= C1.K]), (1,), generator=g, device=dev))]
    t = torch.randint(0, n, (Bc,), generator=g, device=dev)
    X, Y = C1.meta['X'][t], C1.meta['Y'][t]
    idx = torch.cat([t, t, n + t, 2 * n + t, 3 * n + t, 4 * n + t])
    tgt = torch.cat([X, Y, Y, Y, X, X])
    z = CM.forecast(model, C1, idx, tgt, rsel(6 * Bc, C1.K, K, g, dev)).view(6, Bc).T
    l1, l2 = pair_joints(z)
    w = 0.5 * (l1.exp() + l2.exp()).detach()
    return (w * (l1 - l2) ** 2).sum((1, 2)).mean()


def order_sums(model, C2, s, K, g, dev, n_orders):
    """Summed log step predictions along each stored order of terminal vectors s -> (len(s), n_orders)."""
    perm = C2.meta['perm'][s][:, :n_orders]                                         # (B,O,30)
    B = len(s)
    O_all = C2.meta['perm'].shape[1]
    base = (s[:, None] * O_all + torch.arange(n_orders, device=dev)[None, :]) * 30   # (B,O)
    idx = (base[:, :, None] + torch.arange(30, device=dev)[None, None, :]).reshape(-1)
    tgt = perm.reshape(-1)
    z = CM.forecast(model, C2, idx, tgt, rsel(len(idx), C2.K, K, g, dev)).view(B, n_orders, 30)
    val = torch.gather(C2.meta['xvec'][s][:, None, :].expand(B, n_orders, 30), 2, perm)
    return F.logsigmoid(z * (2 * val - 1).float()).sum(-1)


def cons_ii(model, C2, g, Bt, dev, anchored):
    K = LEVELS[int(torch.randint(0, len([k for k in LEVELS if k <= C2.K]), (1,), generator=g, device=dev))]
    s = torch.randint(0, C2.meta['xvec'].shape[0], (Bt,), generator=g, device=dev)
    L = order_sums(model, C2, s, K, g, dev, 2)
    if anchored:
        return (L - C2.meta['logRhat'][s][:, None]).flatten().var() / 30
    return ((L[:, 0] - L[:, 1]) ** 2 / 2).mean() / 30


def cons_iii(model, C1, g, Bu, dev, sym):
    lv = int(torch.randint(0, len(LEVELS) - 1, (1,), generator=g, device=dev))
    K, K4 = LEVELS[lv], LEVELS[lv + 1]
    idx = torch.randint(0, C1.N, (Bu,), generator=g, device=dev)
    tgt = rand_target(C1.mask[idx], g)
    sel = rsel(Bu, C1.K, K4, g, dev)
    zK, z4 = CM.forecast(model, C1, idx, tgt, sel[:, :K]), CM.forecast(model, C1, idx, tgt, sel)
    if sym:
        return ((torch.sigmoid(zK) - torch.sigmoid(z4)) * (zK - z4)).mean()
    p4 = torch.sigmoid(z4).detach()
    return F.binary_cross_entropy_with_logits(zK, p4) - F.binary_cross_entropy(p4.clamp(1e-6, 1 - 1e-6), p4)


def cons_synth(model, C1, g, Bu, dev):
    n = C1.N // 5
    K = LEVELS[int(torch.randint(0, len(LEVELS), (1,), generator=g, device=dev))]
    t = torch.randint(0, n, (Bu,), generator=g, device=dev)
    xr = C1.meta['xreal'][t]                                    # the M-sample's own values (not real data)
    X, Y = C1.meta['X'][t], C1.meta['Y'][t]
    which = torch.randint(0, 3, (Bu,), generator=g, device=dev)
    blk = torch.where(which == 0, torch.zeros_like(t), torch.where(which == 1, 1 + xr.gather(1, X[:, None])[:, 0], 3 + xr.gather(1, Y[:, None])[:, 0]))
    idx = blk * n + t
    tgt = rand_target(C1.mask[idx], g)
    z = CM.forecast(model, C1, idx, tgt, rsel(Bu, C1.K, K, g, dev))
    return F.binary_cross_entropy_with_logits(z, xr.gather(1, tgt[:, None])[:, 0].float())


def chunked(pred, D, idx, tgt, sel, bs=4096):
    return torch.cat([pred(D, idx[i:i + bs], tgt[i:i + bs], sel[i:i + bs]) for i in range(0, len(idx), bs)], 0)


@torch.no_grad()
def evaluate(pred, T1, T2, T3, dev):
    out = {}
    for K in LEVELS:
        if K > T1.K:
            continue
        ar = torch.arange(T1.N, device=dev)
        pref = torch.arange(K, device=dev)[None, :].expand(T1.N, K)
        out['t1_K%d' % K] = torch.stack([chunked(pred, T1, ar, torch.full_like(ar, c), pref) for c in range(30)], 1).float().cpu().numpy()
        n = T2.N // 5
        t = torch.arange(n, device=dev)
        X, Y = T2.meta['X'], T2.meta['Y']
        idx = torch.cat([t, t, n + t, 2 * n + t, 3 * n + t, 4 * n + t])
        tgt = torch.cat([X, Y, Y, Y, X, X])
        perm = T3.meta['perm']
        for r in range(2 if 2 * K <= T2.K else 1):
            sel = torch.arange(r * K, (r + 1) * K, device=dev)[None, :]
            out['t2_K%d_r%d' % (K, r)] = chunked(pred, T2, idx, tgt, sel.expand(len(idx), K)).view(6, n).T.float().cpu().numpy()
            i3 = torch.arange(T3.N, device=dev)
            out['t3_K%d_r%d' % (K, r)] = chunked(pred, T3, i3, perm.reshape(-1), sel.expand(T3.N, K)).view(perm.shape).float().cpu().numpy()
    return out


def train(arm, lam, n, d, r, lab, C1, C2, dev, steps, log):
    torch.manual_seed(7919 * r + 101 * d + n)
    model = CM.Agg().to(dev)
    gd, gc = torch.Generator(device=dev), torch.Generator(device=dev)
    gd.manual_seed(31 * r + 7 * d + n + 1)
    gc.manual_seed(977 * r + 13 * d + n + 5)
    fast = [p for nm, p in model.named_parameters() if nm == 'pool_raw.weight']
    slow = [p for nm, p in model.named_parameters() if nm != 'pool_raw.weight']
    opt = torch.optim.AdamW([dict(params=slow, lr=1e-3), dict(params=fast, lr=5e-3, weight_decay=0.0)], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / 200) * (0.5 * (1 + math.cos(math.pi * min(1.0, t / steps))) * 0.98 + 0.02))
    if n > 0:
        tr, va, c = subset(lab, n, d)
        gv = torch.Generator(device=dev)
        gv.manual_seed(4242)
        va = va[torch.randperm(len(va), generator=gv, device=dev)[:N_VAL]]
        vtg = torch.stack([rand_target(lab.mask[va], gv) for _ in range(N_VTG)], 1)   # fixed targets per context
        vy = lab.x[va][:, None, :].expand(len(va), N_VTG, 30).gather(2, vtg[:, :, None])[:, :, 0].float()
    best, best_state, hist = 1e9, None, []
    t0 = time.time()
    for step in range(steps + 1):
        if n > 0 and step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                vl = 0.0
                for K in LEVELS:
                    pref = torch.arange(K, device=dev)[None, :].expand(len(va) * N_VTG, K)
                    z = chunked(lambda *a: CM.forecast(model, *a), lab, va[:, None].expand(len(va), N_VTG).reshape(-1), vtg.reshape(-1), pref)
                    vl += float(F.binary_cross_entropy_with_logits(z, vy.reshape(-1))) / len(LEVELS)
            model.train()
            hist.append((step, vl))
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step == steps:
            break
        loss = torch.zeros((), device=dev)
        if n > 0:
            lv = int(torch.randint(0, len(LEVELS) - 1, (1,), generator=gd, device=dev))
            K, K4 = LEVELS[lv], LEVELS[lv + 1]
            idx = tr[torch.randint(0, len(tr), (B_DATA,), generator=gd, device=dev)]
            tgt = rand_target(lab.mask[idx], gd)
            y = lab.x[idx].gather(1, tgt[:, None])[:, 0].float()
            sel = rsel(B_DATA, lab.K, K4, gd, dev)
            loss = F.binary_cross_entropy_with_logits(CM.forecast(model, lab, idx, tgt, sel[:, :K]), y) \
                + F.binary_cross_entropy_with_logits(CM.forecast(model, lab, idx, tgt, sel), y)
        cl = None
        if arm == 'i':
            cl = cons_i(model, C1, gc, B_PAIR, dev)
        elif arm in ('iio', 'iis'):
            cl = cons_ii(model, C2, gc, B_ORD, dev, arm == 'iis')
        elif arm in ('iii', 'iiisym'):
            cl = cons_iii(model, C1, gc, B_UNL, dev, arm == 'iiisym')
        elif arm == 'synth':
            cl = cons_synth(model, C1, gc, B_UNL, dev)
        if cl is not None:
            loss = loss + (lam if n > 0 else 1.0) * cl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log and step % 250 == 0:
            log.write(json.dumps(dict(step=step, loss=float(loss.detach()), cons=None if cl is None else float(cl.detach()), t=round(time.time() - t0, 1))) + '\n')
            log.flush()
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best, hist


if __name__ == '__main__':
    ddir, odir, joblist, shard, nshards = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 1500
    os.makedirs(odir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cpu':
        torch.set_num_threads(int(os.environ.get('NTHREADS', 1)))
    jobs = [ln.split() for ln in open(joblist) if ln.strip() and not ln.startswith('#')]
    jobs = [j for i, j in enumerate(jobs) if i % nshards == shard]
    sets = {k: CM.CS('%s/%s.pt' % (ddir, k), dev) for k in ('lab', 'c1', 'c2', 't1', 't2', 't3')}
    for k in ('lab', 'c1', 'c2', 't1', 't2', 't3'):
        sets[k].meta = {a: (b.to(dev).long() if torch.is_tensor(b) and b.dtype != torch.float32 else (b.to(dev) if torch.is_tensor(b) else b))
                        for a, b in sets[k].meta.items()}
    for arm_s, n, d, r in jobs:
        n, d, r = int(n), int(d), int(r)
        tag = '%s_n%d_d%d_r%d' % (arm_s.replace(':', '-'), n, d, r)
        if os.path.exists('%s/%s.npz' % (odir, tag)):
            continue
        t0 = time.time()
        if arm_s.startswith('fixed:'):
            nm = arm_s.split(':')[1]
            out = evaluate(lambda *a: CM.fixed_forecast(nm, *a), sets['t1'], sets['t2'], sets['t3'], dev)
            meta = dict(arm=arm_s)
        else:
            arm, lam = (arm_s.split('@') + ['0'])[:2]
            log = open('%s/%s.log' % (odir, tag), 'w')
            model, best, hist = train(arm, float(lam), n, d, r, sets['lab'], sets['c1'], sets['c2'], dev, steps, log)
            out = evaluate(lambda *a: CM.forecast(model, *a), sets['t1'], sets['t2'], sets['t3'], dev)
            meta = dict(arm=arm, lam=float(lam), n=n, d=d, r=r, steps=steps, best_val=best, hist=hist,
                        best_step=int(min(hist, key=lambda h: h[1])[0]) if hist else steps, n_params=CM.n_params(model))
        np.savez_compressed('%s/%s.npz' % (odir, tag), **out)
        meta['secs'] = round(time.time() - t0, 1)
        json.dump(meta, open('%s/%s.json' % (odir, tag), 'w'))
        print(tag, 'val %s' % meta.get('best_val'), 'secs %.0f' % meta['secs'], flush=True)
