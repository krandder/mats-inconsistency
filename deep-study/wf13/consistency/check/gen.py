"""gen.py WORLD PART K OUT [N] : checker's own generator of trajectories (independent of cgen.py; same tensor format so
that cmodel.CS can read it). No exact inference. Seeds/tags 41..46 are disjoint from wf7 (1..11), wf10 (21..24) and the
work under check (31, 34..36).
PART: lab (N real vectors, nested tiers 16/4/2 contexts) | c1 (N triples from M's own samples) | c2o (N terminal
M-samples x 2 orders) | c2r (score reference for the same N samples, K = Kref) | t1 (N contexts) | t2 (N triples)."""
import sys, time
import numpy as np, torch
PW = '/home/agent/workspaces/mats/work/deep-study/wf7/probworld'
sys.path.insert(0, PW); sys.dont_write_bytecode = True
from world import World, pack
import model as MM
MASK, NH, NX = 2, 20, 30
ws, part, K, outp = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 0
torch.set_num_threads(2)
W = World(ws)
m = MM.build('M', 50); m.load_state_dict(torch.load('%s/ckpt/M_w%d_M.pt' % (PW, ws), map_location='cpu')['state']); m.eval()
CALLS = [0]


@torch.no_grad()
def net(tok, bs=2048):
    CALLS[0] += len(tok)
    return torch.cat([m(tok[i:i + bs]).float() for i in range(0, len(tok), bs)], 0)


a1 = torch.sigmoid(net(torch.full((1, 50), MASK))[0, :NH].double()).clamp(1e-7, 1 - 1e-7)      # M's empty-context numbers
loga = torch.stack([(1 - a1).log(), a1.log()], -1).float()
PIDX = dict(lab=1, c1=2, c2o=3, c2r=4, t1=5, t2=6, lab2=7)
g = torch.Generator(); g.manual_seed(ws * 7919 + 1000 * PIDX[part] + 77)
rng = np.random.default_rng(ws * 104729 + 31 * PIDX[part] + 5)


def run(x, mask, K):
    """x (N,30) long, mask (N,30) bool. Searcher: h_i ~ M(h_i | revealed checks, h_<i), index order."""
    out = dict(lu=[], lv=[], la=[], lq=[], p_one=[])
    step = max(1, 4096 // K)
    for i in range(0, len(x), step):
        ctx = torch.where(mask[i:i + step], x[i:i + step], torch.full_like(x[i:i + step], MASK))
        B = len(ctx)
        out['p_one'].append(torch.sigmoid(net(torch.cat([torch.full((B, NH), MASK), ctx], 1))[:, NH:]))
        cK = ctx.repeat_interleave(K, 0)
        tok = torch.cat([torch.full((B * K, NH), MASK), cK], 1)
        lq = torch.zeros(B * K, dtype=torch.float64)
        for t in range(NH):
            p = torch.sigmoid(net(tok)[:, t]).clamp(1e-6, 1 - 1e-6)
            v = torch.rand(B * K, generator=g) < p
            lq += torch.where(v, p, 1 - p).double().log()
            tok[:, t] = v.long()
        h = tok[:, :NH]
        lu = net(torch.cat([h, torch.full((B * K, NX), MASK)], 1))[:, NH:]
        lv = net(torch.cat([h, cK], 1))[:, NH:]
        la = torch.where(h.bool(), loga[:, 1][None], loga[:, 0][None]).sum(-1)
        out['lu'].append(lu.clamp(-13.8, 13.8).view(B, K, NX).half()); out['lv'].append(lv.clamp(-13.8, 13.8).view(B, K, NX).half())
        out['la'].append(la.view(B, K)); out['lq'].append(lq.float().view(B, K))
    return {k: torch.cat(v, 0) for k, v in out.items()}


def rmask(n, lo, hi, excl=None):
    k = rng.integers(lo, hi + 1, size=n)
    r = rng.random((n, NX))
    if excl is not None:
        r[np.arange(n)[:, None], excl] = 2.0
    return torch.tensor(r.argsort(1).argsort(1) < k[:, None])


def msamples(n, seed):
    gg = torch.Generator(); gg.manual_seed(seed)
    h = (torch.rand(n, NH, generator=gg) < a1.float()[None]).long()
    lu = net(torch.cat([h, torch.full((n, NX), MASK)], 1))[:, NH:]
    return (torch.rand(n, NX, generator=gg) < torch.sigmoid(lu)).long()


def fresh(tag, n, lab_n=1024):
    seen = np.concatenate([pack(W.sample(50000, ws * 1000 + 1)[1]), pack(W.sample(lab_n, ws * 1000 + 41)[1])])
    x = W.sample(2 * n + 100, ws * 1000 + tag)[1]
    bad = np.isin(pack(x), seen)
    print('fresh tag', tag, 'drawn', len(x), 'seen fraction %.4f' % bad.mean(), flush=True)
    return torch.tensor(x[~bad][:n]).long()


def triples(x, K):
    n = len(x)
    xy = np.stack([rng.permutation(NX)[:2] for _ in range(n)])
    mk = rmask(n, 0, NX - 2, xy)
    X, Y = torch.tensor(xy[:, 0]), torch.tensor(xy[:, 1])
    xs, ms = [x], [mk]
    for col in (X, Y):
        for val in (0, 1):
            xs.append(x.scatter(1, col[:, None], val)); ms.append(mk.scatter(1, col[:, None], True))
    xc, mask = torch.cat(xs), torch.cat(ms)
    t = run(xc, mask, K)
    t.update(x=xc.to(torch.uint8), mask=mask, X=X, Y=Y, xreal=x.to(torch.uint8))
    return t


t0 = time.time()
if part in ('lab', 'lab2'):
    xr = torch.tensor(W.sample(N, ws * 1000 + 41)[1]).long()
    cnt = np.full(N, 2)
    if part == 'lab':
        cnt[:512] = 4; cnt[:128] = 16
    vec = np.concatenate([np.full(cnt[v], v) for v in range(N)]); copy = np.concatenate([np.arange(cnt[v]) for v in range(N)])
    mk = rmask(len(vec), 0, NX - 1)
    x = xr[vec]
    t = run(x, mk, K); t.update(x=x.to(torch.uint8), mask=mk, vec=torch.tensor(vec), copy=torch.tensor(copy))
elif part == 'c1':
    t = triples(msamples(N, ws * 31 + 1), K)
elif part in ('c2o', 'c2r'):
    xs = msamples(N, ws * 31 + 2)
    if part == 'c2r':
        r = run(xs, torch.ones_like(xs, dtype=torch.bool), K)
        ell = torch.nn.functional.logsigmoid(r['lu'].float() * (2 * xs[:, None, :] - 1)).sum(-1)
        t = dict(logRhat=(torch.logsumexp((r['la'] + ell - r['lq']).double(), 1) - np.log(K)).float(), xvec=xs.to(torch.uint8))
    else:
        perm = np.stack([np.stack([rng.permutation(NX) for _ in range(2)]) for _ in range(N)])
        rank = perm.argsort(-1)
        mask = torch.tensor((rank[:, :, None, :] < np.arange(NX)[None, None, :, None]).reshape(-1, NX))
        xc = xs[:, None, None, :].expand(N, 2, NX, NX).reshape(-1, NX)
        t = run(xc, mask, K); t.update(x=xc.to(torch.uint8), mask=mask, perm=torch.tensor(perm), xvec=xs.to(torch.uint8))
elif part == 't1':
    x = fresh(44, N); mk = rmask(N, 0, NX - 1)
    t = run(x, mk, K); t.update(x=x.to(torch.uint8), mask=mk)
elif part == 't2':
    t = triples(fresh(45, N), K)
t['loga'] = loga
torch.save(t, outp)
print('done', part, 'K', K, 'N', N, 'calls', CALLS[0], 'secs', round(time.time() - t0), flush=True)
