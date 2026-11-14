"""tr.py DATADIR OUTDIR JOBFILE SHARD NSHARDS : checker's own training loop and losses for the amplified predictor.
Reuses only the predictor's definition (cmodel.Agg / features / CS: the object under test). Losses, subsets, validation
and evaluation are rewritten from PREREG.md sections 3-4, not copied from ctrain.py. NO exact inference here.
Job line: ARM N D R [LABMIN]. ARM = data | i@W | iio@W | iis@W | synth@W | p1@W (control: anchor to one pass of M). LABMIN: real outcomes used only on contexts
with at least LABMIN revealed checks (the post hoc test 2e)."""
import json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.dont_write_bytecode = True
sys.path.insert(0, '/home/agent/workspaces/mats/work/deep-study/wf13/consistency')
import cmodel as CM
LEV = CM.LEVELS
torch.set_num_threads(1)
ddir, odir, jobfile, shard, nsh = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
SUF = os.environ.get('SUF', '')                      # '' or '_K16'
STEPS = 700
dev = torch.device('cpu')


def load(name):
    D = CM.CS('%s/%s.pt' % (ddir, name), dev)
    D.meta = {k: (v.long() if torch.is_tensor(v) and v.dtype != torch.float32 else v) for k, v in D.meta.items()}
    return D


lab, T1, T2 = load('lab' + SUF), load('t1' + SUF), load('t2' + ('_K32' if SUF else ''))
C1 = load('c1' + SUF) if os.path.exists('%s/c1%s.pt' % (ddir, SUF)) else None
C2 = None
if not SUF:
    C2 = load('c2o')
    C2.meta['logRhat'] = torch.load(ddir + '/c2r.pt')['logRhat']
    assert (torch.load(ddir + '/c2r.pt')['xvec'].long() == C2.meta['xvec']).all()


RESID = int(os.environ.get('RESID', 0))    # checker's control: forecast = one pass of M + a correction that starts at zero


def fc(model, D, idx, tgt, sel):
    z = CM.forecast(model, D, idx, tgt, sel)
    return z + CM.lgt(D.p_one[idx, tgt]) if RESID else z


def pick(B, Kmax, K, g):
    return torch.rand(B, Kmax, generator=g).argsort(1)[:, :K]


def rtarget(mask, g):
    return torch.rand(mask.shape, generator=g).masked_fill(mask, -1.0).argmax(1)


def logp2(z):
    return torch.stack([F.logsigmoid(-z), F.logsigmoid(z)], -1)


def two_orders(z):
    """z (B,6): A(X|C), A(Y|C), A(Y|C,X=0), A(Y|C,X=1), A(X|C,Y=0), A(X|C,Y=1) -> log joints [x,y] in both orders."""
    a = logp2(z[:, 0])[:, :, None] + torch.stack([logp2(z[:, 2]), logp2(z[:, 3])], 1)
    b = logp2(z[:, 1])[:, None, :] + torch.stack([logp2(z[:, 4]), logp2(z[:, 5])], 2)
    return a, b


def triple_logits(model, D, t, K, g, sel=None):
    n = D.N // 5
    X, Y = D.meta['X'][t], D.meta['Y'][t]
    idx = torch.cat([t, t, n + t, 2 * n + t, 3 * n + t, 4 * n + t]); tgt = torch.cat([X, Y, Y, Y, X, X])
    s = pick(len(idx), D.K, K, g) if sel is None else sel.expand(len(idx), -1)
    return fc(model, D, idx, tgt, s).view(6, len(t)).T


def loss_pairs(model, g):
    K = LEV[int(torch.randint(0, len([k for k in LEV if k <= C1.K]), (1,), generator=g))]
    a, b = two_orders(triple_logits(model, C1, torch.randint(0, C1.N // 5, (48,), generator=g), K, g))
    return ((0.5 * (a.exp() + b.exp())).detach() * (a - b) ** 2).sum((1, 2)).mean()


def loss_orders(model, g, anchored):
    K = LEV[int(torch.randint(0, len([k for k in LEV if k <= C2.K]), (1,), generator=g))]
    s = torch.randint(0, C2.meta['xvec'].shape[0], (6,), generator=g)
    perm = C2.meta['perm'][s]                                                    # (6,2,30)
    idx = ((s[:, None] * 2 + torch.arange(2)[None]) * 30)[:, :, None] + torch.arange(30)[None, None]
    z = fc(model, C2, idx.reshape(-1), perm.reshape(-1), pick(idx.numel(), C2.K, K, g)).view(6, 2, 30)
    val = torch.gather(C2.meta['xvec'][s][:, None, :].expand(6, 2, 30), 2, perm)
    L = F.logsigmoid(z * (2 * val - 1).float()).sum(-1)
    if anchored:
        return (L - C2.meta['logRhat'][s][:, None]).flatten().var() / 30
    return ((L[:, 0] - L[:, 1]) ** 2 / 2).mean() / 30


def loss_synth(model, g):
    n = C1.N // 5
    K = LEV[int(torch.randint(0, len([k for k in LEV if k <= C1.K]), (1,), generator=g))]
    t = torch.randint(0, n, (128,), generator=g)
    xr, X, Y = C1.meta['xreal'][t], C1.meta['X'][t], C1.meta['Y'][t]            # the M-sample's own values
    which = torch.randint(0, 3, (128,), generator=g)
    blk = torch.where(which == 0, torch.zeros_like(t), torch.where(which == 1, 1 + xr.gather(1, X[:, None])[:, 0], 3 + xr.gather(1, Y[:, None])[:, 0]))
    idx = blk * n + t
    tgt = rtarget(C1.mask[idx], g)
    return F.binary_cross_entropy_with_logits(fc(model, C1, idx, tgt, pick(128, C1.K, K, g)), xr.gather(1, tgt[:, None])[:, 0].float())


def loss_p1(model, g):
    """CONTROL (checker's, not a consistency term): pull the forecast toward one pass of M on unlabelled contexts."""
    K = LEV[int(torch.randint(0, len([k for k in LEV if k <= C1.K]), (1,), generator=g))]
    idx = torch.randint(0, C1.N, (128,), generator=g)
    tgt = rtarget(C1.mask[idx], g)
    return F.binary_cross_entropy_with_logits(fc(model, C1, idx, tgt, pick(128, C1.K, K, g)), C1.p_one[idx, tgt])


def split(n, d, labmin):
    vec, copy = lab.meta['vec'], lab.meta['copy']
    nv = int(vec.max()) + 1
    lo, hi = (d * n, (d + 1) * n) if 2 * n <= nv else (0, nv)
    c = int(torch.bincount(vec)[lo:hi].min())
    nval = min(n // 4, 1024)
    ok = (vec >= lo) & (vec < hi) & (copy < c)
    if labmin:
        ok &= lab.mask.sum(1) >= labmin
    return torch.where(ok & (vec < hi - nval))[0], torch.where(ok & (vec >= hi - nval))[0]


def run(arm, lam, n, d, r, labmin):
    torch.manual_seed(100003 * r + 1009 * d + n + 17)
    model = CM.Agg()
    if RESID:
        torch.nn.init.zeros_(model.head[-1].weight); torch.nn.init.zeros_(model.head[-1].bias)
    g = torch.Generator(); g.manual_seed(50021 * r + 211 * d + n + 3)
    fast = [p for nm, p in model.named_parameters() if nm == 'pool_raw.weight']; slow = [p for nm, p in model.named_parameters() if nm != 'pool_raw.weight']
    opt = torch.optim.AdamW([dict(params=slow, lr=1e-3), dict(params=fast, lr=5e-3, weight_decay=0.0)], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / 200) * (0.5 * (1 + math.cos(math.pi * min(1.0, t / STEPS))) * 0.98 + 0.02))
    Ktop = max(k for k in LEV if k <= lab.K)
    if n:
        tr, va = split(n, d, labmin)
        gv = torch.Generator(); gv.manual_seed(999)
        va = va[torch.randperm(len(va), generator=gv)[:512]]
        vt = torch.stack([rtarget(lab.mask[va], gv) for _ in range(4)], 1)
        vy = lab.x[va][:, None, :].expand(-1, 4, -1).gather(2, vt[:, :, None])[:, :, 0].float().reshape(-1)
        vi = va[:, None].expand(-1, 4).reshape(-1)
    best, state, hist = 1e9, None, []
    for step in range(STEPS + 1):
        if n and step % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([float(F.binary_cross_entropy_with_logits(fc(model, lab, vi, vt.reshape(-1), torch.arange(K)[None].expand(len(vi), K)), vy))
                                    for K in LEV if K <= lab.K]))
            model.train(); hist.append((step, vl))
            if vl < best:
                best, state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if step == STEPS:
            break
        loss = torch.zeros(())
        if n:
            idx = tr[torch.randint(0, len(tr), (128,), generator=g)]
            tgt = rtarget(lab.mask[idx], g)
            y = lab.x[idx].gather(1, tgt[:, None])[:, 0].float()
            li = int(torch.randint(0, len([k for k in LEV if k <= lab.K]) - 1, (1,), generator=g))
            sel = pick(128, lab.K, LEV[li + 1], g)
            loss = F.binary_cross_entropy_with_logits(fc(model, lab, idx, tgt, sel[:, :LEV[li]]), y) + F.binary_cross_entropy_with_logits(fc(model, lab, idx, tgt, sel), y)
        c = dict(i=lambda: loss_pairs(model, g), iio=lambda: loss_orders(model, g, False), iis=lambda: loss_orders(model, g, True),
                 synth=lambda: loss_synth(model, g), p1=lambda: loss_p1(model, g)).get(arm, lambda: None)()
        if c is not None:
            loss = loss + (lam if n else 1.0) * c
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    out = {}
    with torch.no_grad():
        for K in LEV:
            if K > T1.K:
                continue
            ar = torch.arange(T1.N); pre = torch.arange(K)[None].expand(T1.N, K)
            out['t1_K%d' % K] = torch.stack([fc(model, T1, ar, torch.full_like(ar, c), pre) for c in range(30)], 1).numpy()
            for rr in range(2 if 2 * K <= T2.K else 1):
                out['t2_K%d_r%d' % (K, rr)] = triple_logits(model, T2, torch.arange(T2.N // 5), K, None, torch.arange(rr * K, (rr + 1) * K)[None]).numpy()
    return out, dict(arm=arm, lam=lam, n=n, d=d, r=r, labmin=labmin, best_val=best, best_step=int(min(hist, key=lambda h: h[1])[0]) if hist else STEPS,
                     n_train_ctx=int(len(tr)) if n else 0)


os.makedirs(odir, exist_ok=True)
jobs = [ln.split() for ln in open(jobfile) if ln.strip() and not ln.startswith('#')]
for j in [j for i, j in enumerate(jobs) if i % nsh == shard]:
    arm, lam = (j[0].split('@') + ['0'])[:2]
    n, d, r = int(j[1]), int(j[2]), int(j[3]); labmin = int(j[4]) if len(j) > 4 else 0
    tag = '%s_n%d_d%d_r%d%s' % (j[0], n, d, r, '_ph%d' % labmin if labmin else '')
    if os.path.exists('%s/%s.npz' % (odir, tag)):
        continue
    t0 = time.time()
    out, meta = run(arm, float(lam), n, d, r, labmin)
    np.savez_compressed('%s/%s.npz' % (odir, tag), **out)
    meta['secs'] = round(time.time() - t0, 1)
    json.dump(meta, open('%s/%s.json' % (odir, tag), 'w'))
    print(tag, json.dumps(meta), flush=True)
