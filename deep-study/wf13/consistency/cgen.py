"""cgen.py WORLD CKPT OUTDIR SCALE SIZE PART [NAME=CKPT ...] : data for the any-order amplified predictor (wf13/consistency).

A CONTEXT is a subset S of the 30 checks with values. For each context a searcher (wf10's helpful searcher H: each
assumption drawn from M's inverse number given the context checks and the assumptions filled so far, 20 network calls)
proposes K complete fillings h of the 20 assumptions; along each, M gives (2 calls)
    lu = logit M(x_c = 1 | h)            for all 30 checks   (assumptions first, then deduce)
    lv = logit M(x_c = 1 | h, x_S)       for all 30 checks   (the full-context number; used for c not in S)
plus la = sum_i log M(h_i | nothing) and lq = the searcher's own log probability of h. One pass of M (p_one) per context.
One set of K trajectories serves every target check outside S.
Sets written (one .pt each):
  lab  real check vectors, labelled (the only real outcomes any arm may train on), several random contexts per vector
  c1   UNLABELLED, from M's own samples (h ~ M's empty-context numbers, x ~ M(x | h)): triples (C, X, Y) with the five
       contexts C, C+X=0, C+X=1, C+Y=0, C+Y=1
  c2   UNLABELLED, M's own samples: terminal check vectors, two random orders of revealing all 30 checks (60 prefix
       contexts each), and the score's reference value log R-hat(x) from K_ref trajectories with exact weights
  t1 t2 t3  TEST, fresh real vectors (disjoint seeds; vectors seen in M's or the labelled data are dropped):
       t1 random contexts (all outside targets), t2 triples as c1, t3 terminal vectors with four orders as c2
PART = all | labK/N (shard K of N of the labelled contexts) | c1 | c2 | c2K/N | t12 | t3 | t3K/N : parts run as separate
processes with their own seeds (CPU fallback: one thread each). NO exact inference here. No hand-built parent table: the score is M's empty-context numbers times M(x_c | h).
seq_search and fwd are wf10/aggregator/gen.py's functions.
"""
import json
import os
import sys
import time

import numpy as np
import torch

PW = os.environ.get('PROBWORLD', '/home/agent/workspaces/mats/work/deep-study/wf7/probworld')
sys.path.insert(0, PW)
sys.dont_write_bytecode = True
from world import World, pack, N_H, N_X, N_VAR  # noqa: E402
import model as MM  # noqa: E402
from model import MASK  # noqa: E402

TAGS = dict(lab=31, t1=34, t2=35, t3=36)                  # disjoint from wf7's 1..11 and wf10's 21..24
SCALES = dict(
    full=dict(nv=20480, tiers=[(128, 32), (512, 8), (2048, 4), (8192, 2), (20480, 1)], Klab=64, c1=4096, Kc1=64,
              c2=768, Kc2=16, Kref=256, t1=1024, t2=768, t3=96, Kt=64),
    pilot=dict(nv=4096, tiers=[(128, 32), (512, 8), (2048, 4), (4096, 1)], Klab=64, c1=2048, Kc1=64,
               c2=256, Kc2=16, Kref=256, t1=512, t2=256, t3=32, Kt=64),
    cpu=dict(nv=20480, tiers=[(128, 16), (512, 4), (2048, 2), (20480, 1)], Klab=16, c1=768, Kc1=16,
             c2=128, Kc2=16, Kref=64, t1=512, t2=256, t3=32, Kt=16),
    cpuM=dict(nv=20480, tiers=[(128, 16), (512, 4), (2048, 2), (20480, 1)], Klab=4, c1=768, Kc1=4,
              c2=96, Kc2=4, Kref=128, t1=512, t2=192, t3=24, Kt=8),
    cpupilot=dict(nv=2048, tiers=[(128, 16), (512, 4), (2048, 1)], Klab=16, c1=512, Kc1=16,
                  c2=64, Kc2=16, Kref=64, t1=256, t2=128, t3=16, Kt=16),
    tiny=dict(nv=64, tiers=[(16, 4), (64, 1)], Klab=16, c1=16, Kc1=16, c2=4, Kc2=4, Kref=16, t1=16, t2=8, t3=2, Kt=16))
CALLS = dict(n=0)


def fwd(model, tok, amp, bs=16384, cols=None):
    out = []
    for i in range(0, tok.shape[0], bs):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp and tok.is_cuda):
            lg = model(tok[i:i + bs])
        lg = lg.float()
        out.append(lg if cols is None else lg[:, cols])
    CALLS['n'] += tok.shape[0]
    return torch.cat(out, 0)                                 # LOGITS


@torch.no_grad()
def seq_search(model, ctx, K, gen):
    """Searcher H: h_i ~ M(. | ctx, h filled so far), index order. 20 calls per trajectory."""
    B, dev = ctx.shape[0], ctx.device
    R = B * K
    tok = torch.cat([torch.full((R, N_H), MASK, dtype=torch.long, device=dev),
                     ctx[:, None, :].expand(B, K, N_X).reshape(R, N_X)], 1)
    lq = torch.zeros(R, device=dev, dtype=torch.float64)
    for t in range(N_H):
        p = torch.sigmoid(fwd(model, tok, True, cols=t)).clamp(1e-6, 1 - 1e-6)
        v = torch.rand(R, generator=gen, device=dev) < p
        lq += torch.where(v, p, 1 - p).double().log()
        tok[:, t] = v.long()
    return tok[:, :N_H].view(B, K, N_H), lq.view(B, K).float()


@torch.no_grad()
def run_contexts(model, loga, xc, mask, K, gen, extra=None):
    """xc (N,30) long values, mask (N,30) bool (True = revealed). Trajectories and M's numbers, on CPU."""
    N, dev = xc.shape[0], xc.device
    ctx = torch.where(mask, xc, torch.full_like(xc, MASK))
    out = dict(lu=[], lv=[], la=[], lq=[], p_one=[])
    if extra:
        for nm in extra:
            out['p_one_' + nm] = []
    chunk = max(1, 131072 // K)
    for i in range(0, N, chunk):
        c = ctx[i:i + chunk]
        B = c.shape[0]
        tok1 = torch.cat([torch.full((B, N_H), MASK, dtype=torch.long, device=dev), c], 1)
        out['p_one'].append(torch.sigmoid(fwd(model, tok1, False)[:, N_H:]).cpu())
        if extra:
            for nm, mm in extra.items():
                out['p_one_' + nm].append(torch.sigmoid(fwd(mm, tok1, False)[:, N_H:]).cpu())
        h, lq = seq_search(model, c, K, gen)
        hl = h.reshape(B * K, N_H)
        lu = fwd(model, torch.cat([hl, torch.full((B * K, N_X), MASK, dtype=torch.long, device=dev)], 1), True)[:, N_H:]
        cx = c[:, None, :].expand(B, K, N_X).reshape(B * K, N_X)
        lv = fwd(model, torch.cat([hl, cx], 1), True)[:, N_H:]
        la = torch.where(h.bool(), loga[:, 1][None, None, :], loga[:, 0][None, None, :]).sum(-1)
        out['lu'].append(lu.clamp(-13.8, 13.8).view(B, K, N_X).half().cpu())
        out['lv'].append(lv.clamp(-13.8, 13.8).view(B, K, N_X).half().cpu())
        out['la'].append(la.float().cpu())
        out['lq'].append(lq.cpu())
    return {k: torch.cat(v, 0) for k, v in out.items()}


@torch.no_grad()
def score_ref(model, loga, x, Kref, gen):
    """log R-hat(x): the score's value for a complete check vector, sum over assumption fillings estimated from Kref
    searcher trajectories with exact weights R(h, x) / psi(h). R(h, x) = prod_i M(h_i | nothing) prod_c M(x_c | h)."""
    t = run_contexts(model, loga, x, torch.ones_like(x, dtype=torch.bool), Kref, gen)
    lu = t['lu'].float()
    ell = torch.where(x.cpu()[:, None, :] == 1, torch.nn.functional.logsigmoid(lu), torch.nn.functional.logsigmoid(-lu)).sum(-1)
    lw = (t['la'] + ell - t['lq']).double()
    return (torch.logsumexp(lw, 1) - np.log(Kref)).float()


def rand_mask(rng, N, lo=0, hi=N_X - 1, exclude=None):
    """Random subsets: size uniform in lo..hi, a uniformly random subset of that size (never the excluded columns)."""
    k = rng.integers(lo, hi + 1, size=N)
    r = rng.random((N, N_X))
    if exclude is not None:
        r[np.arange(N)[:, None], exclude] = 2.0
    rank = r.argsort(1).argsort(1)
    return rank < k[:, None]


@torch.no_grad()
def m_samples(model, loga, N, gen, dev):
    """M's own samples in its generative direction: h ~ M's empty-context numbers, x_c ~ M(x_c | h). 1 call each."""
    a1 = loga[:, 1].exp()
    h = (torch.rand(N, N_H, generator=gen, device=dev) < a1[None, :]).long()
    lu = fwd(model, torch.cat([h, torch.full((N, N_X), MASK, dtype=torch.long, device=dev)], 1), True)[:, N_H:]
    return (torch.rand(N, N_X, generator=gen, device=dev) < torch.sigmoid(lu)).long()


def triple_set(model, loga, x, rng, K, gen, dev):
    """x (N,30) long on dev. Triples (C, X, Y): 5N contexts in blocks [C | C+X=0 | C+X=1 | C+Y=0 | C+Y=1]."""
    N = x.shape[0]
    xy = np.stack([rng.permutation(N_X)[:2] for _ in range(N)])
    m = torch.tensor(rand_mask(rng, N, 0, N_X - 2, exclude=xy), device=dev)
    X, Y = torch.tensor(xy[:, 0], device=dev), torch.tensor(xy[:, 1], device=dev)
    xs, ms = [x], [m]
    for col in (X, Y):
        for val in (0, 1):
            xs.append(x.scatter(1, col[:, None], val))
            ms.append(m.scatter(1, col[:, None], True))
    xc, mask = torch.cat(xs, 0), torch.cat(ms, 0)
    t = run_contexts(model, loga, xc, mask, K, gen)
    t.update(x=xc.to(torch.uint8).cpu(), mask=mask.cpu(), X=X.cpu(), Y=Y.cpu(), xreal=x.to(torch.uint8).cpu())
    return t


def order_set(model, loga, x, rng, n_orders, K, gen, dev):
    """Terminal vectors x (N,30): n_orders random orders each; context (s, o, t) = the first t checks of order o."""
    N = x.shape[0]
    perm = np.stack([np.stack([rng.permutation(N_X) for _ in range(n_orders)]) for _ in range(N)])    # (N,O,30)
    rank = perm.argsort(-1)                                   # rank[s,o,c] = position of check c in order o
    mask = rank[:, :, None, :] < np.arange(N_X)[None, None, :, None]                                   # (N,O,30,30)
    mask = torch.tensor(mask.reshape(N * n_orders * N_X, N_X), device=dev)
    xc = x[:, None, None, :].expand(N, n_orders, N_X, N_X).reshape(-1, N_X)
    t = run_contexts(model, loga, xc, mask, K, gen)
    t.update(x=xc.to(torch.uint8).cpu(), mask=mask.cpu(), perm=torch.tensor(perm), xvec=x.to(torch.uint8).cpu())
    return t


def fresh_real(W, ws, tag, n, seen):
    _, x = W.sample(int(n * 1.25) + 80, ws * 1000 + TAGS[tag])
    kx = pack(x)
    bad = np.isin(kx, seen['m_x']) | np.isin(kx, seen['lab_x'])
    keep = np.where(~bad)[0][:n]
    return x[keep], dict(drawn=int(len(kx)), seen_in_M_train=float(np.isin(kx, seen['m_x']).mean()),
                         seen_in_lab=float(np.isin(kx, seen['lab_x']).mean()), kept=int(len(keep)))


if __name__ == '__main__':
    ws, ckpt, outdir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    scale = sys.argv[4] if len(sys.argv) > 4 else 'full'
    size = sys.argv[5] if len(sys.argv) > 5 else 'M'
    part = sys.argv[6] if len(sys.argv) > 6 else 'all'
    extra_args = sys.argv[7:]
    os.makedirs(outdir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cpu':
        torch.set_num_threads(int(os.environ.get('NTHREADS', 1)))
    sc = SCALES[scale]
    W = World(ws)

    def load(path, sz):
        ck = torch.load(path, map_location=dev)
        m = MM.build(sz, N_VAR).to(dev)
        m.load_state_dict(ck['state'])
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m
    model = load(ckpt, size)
    extra = {e.split('=')[0]: load(e.split('=')[1], e.split('=')[0]) for e in extra_args}
    with torch.no_grad():
        a = torch.sigmoid(fwd(model, torch.full((1, N_VAR), MASK, dtype=torch.long, device=dev), False)[0, :N_H].double()).clamp(1e-7, 1 - 1e-7)
    loga = torch.stack([(1 - a).log(), a.log()], -1).float()
    torch.save(dict(loga=loga.cpu()), outdir + '/score.pt')
    info, t0 = dict(world=ws, ckpt=ckpt, scale=scale, part=part), time.time()
    PID = dict(lab=0, c1=1, c2=2, tests=3, t3=4)

    def seeds(name, k=0):
        """Own generators per part and shard, so parts can run as separate processes."""
        gen = torch.Generator(device=dev)
        gen.manual_seed(ws * 100000 + 91 + 1000 * PID[name] + k)
        return gen, np.random.default_rng(ws * 1000 + 913 + 17 * PID[name])

    def done(name, t):
        torch.save(t, '%s/%s.pt' % (outdir, name))
        info[name] = dict(contexts=int(t['mask'].shape[0]), K=int(t['lu'].shape[1]), secs=round(time.time() - t0, 1), calls=CALLS['n'])
        print(name, json.dumps(info[name]), flush=True)

    def shard(p):
        k, n = (p.split('/') + ['1'])[:2] if '/' in p else ('0', '1')
        return int(k), int(n)

    _, xr = W.sample(sc['nv'], ws * 1000 + TAGS['lab'])               # the real labelled vectors (h dropped)
    if part == 'all' or part.startswith('lab'):
        k, n = shard(part[3:]) if part.startswith('lab') else (0, 1)
        gen, rng = seeds('lab', k)
        cnt = np.zeros(sc['nv'], dtype=np.int64)                      # nested tiers: contexts per vector
        for hi, c in sc['tiers']:
            cnt[:hi] = np.maximum(cnt[:hi], c)
        vec = np.concatenate([np.full(cnt[v], v) for v in range(sc['nv'])])
        copy = np.concatenate([np.arange(cnt[v]) for v in range(sc['nv'])])
        mask = rand_mask(rng, len(vec))                               # the same list in every shard; then sliced
        vec, copy, mask = vec[k::n], copy[k::n], mask[k::n]
        x = torch.tensor(xr[vec], dtype=torch.long, device=dev)
        t = run_contexts(model, loga, x, torch.tensor(mask, device=dev), sc['Klab'], gen)
        t.update(x=x.to(torch.uint8).cpu(), mask=torch.tensor(mask), vec=torch.tensor(vec), copy=torch.tensor(copy))
        done('lab' if n == 1 else 'lab_%d' % k, t)
    if part in ('all', 'c1'):
        gen, rng = seeds('c1')
        done('c1', triple_set(model, loga, m_samples(model, loga, sc['c1'], gen, dev), rng, sc['Kc1'], gen, dev))
    if part == 'all' or part.startswith('c2'):
        k, n = shard(part[2:]) if len(part) > 2 else (0, 1)
        gen, rng = seeds('c2', k)
        gen0, _ = seeds('c2', 99)
        xs = m_samples(model, loga, sc['c2'], gen0, dev)[k::n]         # the same samples in every shard; then sliced
        rng = np.random.default_rng(ws * 1000 + 913 + 17 * 2 + 1000 * (k + 1))
        t = order_set(model, loga, xs, rng, 2, sc['Kc2'], gen, dev)
        t['logRhat'] = score_ref(model, loga, xs, sc['Kref'], gen)
        done('c2' if n == 1 else 'c2_%d' % k, t)
    if part in ('all', 'tests', 't12') or part.startswith('t3'):
        seen = dict(m_x=pack(W.sample(int(os.environ.get('N_TRAIN_M', 50000)), ws * 1000 + 1)[1]), lab_x=pack(xr))
    if part in ('all', 'tests', 't12'):
        gen, rng = seeds('tests')
        x1, i1 = fresh_real(W, ws, 't1', sc['t1'], seen)
        x1 = torch.tensor(x1, dtype=torch.long, device=dev)
        m1 = torch.tensor(rand_mask(rng, x1.shape[0]), device=dev)
        t = run_contexts(model, loga, x1, m1, sc['Kt'], gen, extra)
        t.update(x=x1.to(torch.uint8).cpu(), mask=m1.cpu())
        done('t1', t)
        x2, i2 = fresh_real(W, ws, 't2', sc['t2'], seen)
        done('t2', triple_set(model, loga, torch.tensor(x2, dtype=torch.long, device=dev), rng, sc['Kt'], gen, dev))
        info.update(t1_info=i1, t2_info=i2)
    if part in ('all', 'tests') or part.startswith('t3'):
        k, n = shard(part[2:]) if len(part) > 2 else (0, 1)
        gen, _ = seeds('t3', k)
        rng = np.random.default_rng(ws * 1000 + 913 + 17 * 4 + 1000 * (k + 1))
        x3, i3 = fresh_real(W, ws, 't3', sc['t3'], seen)
        x3 = torch.tensor(x3[k::n], dtype=torch.long, device=dev)
        t = order_set(model, loga, x3, rng, 4, sc['Kt'], gen, dev)
        t['logRhat'] = score_ref(model, loga, x3, sc['Kref'], gen)
        done('t3' if n == 1 else 't3_%d' % k, t)
        info.update(t3_info=i3)
    info.update(total_calls=CALLS['n'], secs=round(time.time() - t0, 1))
    json.dump(info, open('%s/gen_info_%s.json' % (outdir, part.replace('/', '-')), 'w'), indent=1)
    print('done', json.dumps(dict(part=part, total_calls=CALLS['n'], secs=info['secs'])), flush=True)
