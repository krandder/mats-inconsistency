"""P4 stage 3: selection of latent groups on market-resnet bases, then latent-market
fine-tunes, then extrapolation evaluation.

Imports a read-only copy of market-resnet/mresnet.py (MRDIR, default /tmp/p4ft).

  select  : rank 16 groups of 8 h channels (patched at all positions at step t) by
            AS-SOFTFLIP / AS-PAY (m midpoints), FLIP-LOO, GRADxACT, random, and
            (--shapK > 0) sampled FLIP/SOFT/PAY-Shapley with a ladder and a reference.
  finetune: 1000 iterations from a base: base loss (+ arm term), then eval L64/L128/L256.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MRDIR = os.environ.get('MRDIR', '/tmp/p4ft')
sys.path.insert(0, MRDIR)
from mresnet import MarketResNet, prefix_batch, score  # noqa: E402

NG, GW = 16, 8          # 16 groups of 8 channels
CLIP = (0.01, 0.99)
EVAL = {'L64': (64, 64, 1000, 128), 'L128': (128, 128, 500, 256), 'L256': (256, 256, 250, 512),
        'id_8_32': (8, 32, 1000, 24), 'smoke': (8, 16, 20, 24)}


def load(ck, device):
    s = torch.load(ck, map_location=device)
    a = s['args']
    m = MarketResNet(w=a['width']).to(device)
    m.load_state_dict(s['model'])
    return m.eval(), a


# ---------------------------------------------------------------- selection game
class Game:
    """Pairs (x, x') of the same length; state after t steps; continue to T."""

    def __init__(self, model, P, t, T, lmin, lmax, seed, device):
        g = torch.Generator().manual_seed(seed)
        x, y, m = prefix_batch(P, lmin, lmax, g, device)
        xr = torch.randint(0, 2, x.shape, generator=g).to(device)
        xp = torch.where(m > 0, xr, x)
        self.model, self.t, self.T, self.m, self.y = model, t, T, m, y
        with torch.no_grad():
            self.e = model.encode(x)
            h, b = self.run(x, t)
            hp, _ = self.run(xp, t)
            hT, bT = self.cont(h, b, self.e, m, T - t)
        self.h, self.hp, self.b = h, hp, b
        self.s0 = torch.where(bT >= 0, 1.0, -1.0)
        self.ypm = 2 * y - 1
        self.P = P
        G = torch.zeros(NG, model.w, device=device)
        for k in range(NG):
            G[k, k * GW:(k + 1) * GW] = 1
        self.G = G

    def run(self, x, t):
        mk = (x != 2).float()
        e = self.model.encode(x)
        h = torch.zeros(x.shape[0], self.model.w, x.shape[1], device=x.device)
        b = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        return self.cont(h, b, e, mk, t)

    def cont(self, h, b, e, m, steps):
        for _ in range(steps):
            h, b = self.model.step(h, b, e, m)
        return h, b

    def values(self, bT, idx):
        m, nb = self.m[idx], self.m[idx].sum(1)
        flip = (((bT >= 0).float() * 2 - 1 == self.s0[idx]).float() * m).sum(1) / nb
        soft = (torch.sigmoid(bT * self.s0[idx]) * m).sum(1) / nb
        p = torch.sigmoid(bT * self.ypm[idx]).clamp(*CLIP)
        pay = (torch.log(p) * m).sum(1) / nb
        return {'flip': flip, 'soft': soft, 'pay': pay}

    def eval_masks(self, M, idx):
        """M: (R, NG) group masks for pair rows idx (R,). 1 = keep h(x)."""
        cm = (M @ self.G).clamp(max=1)[:, :, None]
        h = self.hp[idx] + cm * (self.h[idx] - self.hp[idx])
        _, bT = self.cont(h, self.b[idx], self.e[idx], self.m[idx], self.T - self.t)
        return self.values(bT, idx)

    def grads(self, ts, idx, games=('soft', 'pay')):
        """sum over midpoints of per-pair group attributions (len(idx), NG)."""
        out = {g: torch.zeros(len(idx), NG, device=self.h.device) for g in games}
        D = self.h[idx] - self.hp[idx]
        for tt in ts:
            h = (self.hp[idx] + tt * D).detach().requires_grad_(True)
            _, bT = self.cont(h, self.b[idx], self.e[idx], self.m[idx], self.T - self.t)
            v = self.values(bT, idx)
            for gi, g in enumerate(games):
                gr, = torch.autograd.grad(v[g].sum(), h, retain_graph=gi < len(games) - 1)
                out[g] += ((gr * D).sum(2) @ self.G.t()).detach()
        return {g: v / len(ts) for g, v in out.items()}


def chunks(P, c):
    for i in range(0, P, c):
        yield torch.arange(i, min(P, i + c))


def select(a):
    dev = a.device
    torch.manual_seed(0)
    model, args = load(a.ckpt, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    res = dict(ckpt=a.ckpt, t=a.t, P=a.P, T=args['T'])
    t0 = time.time()
    game = Game(model, a.P, a.t, args['T'], args['lmin'], args['lmax'], a.seed, dev)
    P, n = a.P, NG
    full = torch.ones(1, n, device=dev)
    rules, per_pair = {}, {}
    with torch.no_grad():
        vC = {g: [] for g in ('flip', 'soft', 'pay')}
        v0 = {g: [] for g in ('flip', 'soft', 'pay')}
        loo = {g: torch.zeros(P, n, device=dev) for g in ('flip', 'soft', 'pay')}
        for idx in chunks(P, a.chunk):
            R = len(idx)
            M = torch.cat([full, torch.zeros(1, n, device=dev), 1 - torch.eye(n, device=dev)])  # (n+2, n)
            Mr = M.repeat_interleave(R, 0)
            v = game.eval_masks(Mr, idx.repeat(n + 2))
            for g in v:
                vv = v[g].reshape(n + 2, R)
                vC[g].append(vv[0]); v0[g].append(vv[1])
                loo[g][idx] = (vv[0][None] - vv[2:]).t()
        for g in vC:
            vC[g], v0[g] = torch.cat(vC[g]), torch.cat(v0[g])
            rules[f'{g.upper()}-LOO'] = loo[g].mean(0)
    res['vC'] = {g: float(v.mean()) for g, v in vC.items()}
    res['v0'] = {g: float(v.mean()) for g, v in v0.items()}
    res['sec_loo'] = round(time.time() - t0, 1)
    # AS ladders and GRADxACT
    for mm in a.m:
        ts = [(k + 0.5) / mm for k in range(mm)]
        acc = {g: torch.zeros(P, n, device=dev) for g in ('soft', 'pay')}
        for idx in chunks(P, a.chunk):
            gr = game.grads(ts, idx)
            for g in gr:
                acc[g][idx] = gr[g]
        for g in acc:
            rules[f'AS-{g.upper()}-m{mm}'] = acc[g].mean(0)
            per_pair[f'AS-{g.upper()}-m{mm}'] = acc[g]
            dv = vC[g] - v0[g]
            res.setdefault('completeness', {})[f'{g}-m{mm}'] = float((acc[g].sum(1) - dv).abs().sum() / dv.abs().sum())
    acc = {g: torch.zeros(P, n, device=dev) for g in ('soft', 'pay')}
    for idx in chunks(P, a.chunk):
        gr = game.grads([1.0], idx)
        for g in gr:
            acc[g][idx] = gr[g]
    for g in acc:
        rules[f'GRADxACT-{g.upper()}'] = acc[g].mean(0)
    res['sec_as'] = round(time.time() - t0, 1)
    rng = np.random.default_rng(99)
    rules['RANDOM'] = torch.tensor(rng.random(n))
    # sampled Shapley: ladder orders [0, Klad) and reference orders [Klad, Klad + Kref), independent
    if a.shapK > 0:
        Klad, Kref = a.shapK, a.refK
        lad = {g: torch.zeros(P, n, device=dev) for g in ('flip', 'soft', 'pay')}
        ref = {g: torch.zeros(P, n, device=dev) for g in ('flip', 'soft', 'pay')}
        ladder_cum = {g: [] for g in ('flip', 'soft', 'pay')}
        Ks = sorted(set([1, 2, 4, 8, 16, 32, 64, 128]) & set(range(1, Klad + 1)))
        gper = torch.Generator().manual_seed(a.seed + 555)
        with torch.no_grad():
            for k in range(Klad + Kref):
                perm = torch.argsort(torch.rand(P, n, generator=gper), 1).to(dev)
                for idx in chunks(P, max(1, a.chunk // (n + 1) * 4)):
                    R = len(idx)
                    pr = perm[idx]
                    M = torch.zeros(n + 1, R, n, device=dev)
                    for j in range(n):
                        M[j + 1] = M[j]
                        M[j + 1, torch.arange(R), pr[:, j]] = 1
                    v = game.eval_masks(M.reshape(-1, n), idx.repeat(n + 1))
                    for g in v:
                        vv = v[g].reshape(n + 1, R)
                        d = vv[1:] - vv[:-1]                       # (n, R) marginal of j-th player
                        phi = torch.zeros(R, n, device=dev)
                        phi.scatter_(1, pr, d.t())
                        if k < Klad:
                            lad[g][idx] += phi
                        else:
                            ref[g][idx] += phi
                if k + 1 in Ks:
                    for g in lad:
                        ladder_cum[g].append(dict(K=k + 1, est=(lad[g] / (k + 1)).mean(0).tolist(),
                                                  cost_pair=(k + 1) * (n - 1) + 2))
        for g in ref:
            rules[f'{g.upper()}-SHAP-ref{Kref}'] = (ref[g] / Kref).mean(0)
            rules[f'{g.upper()}-SHAP-K{Klad}'] = (lad[g] / Klad).mean(0)
        res['shap_ladder'] = ladder_cum
        res['sec_shap'] = round(time.time() - t0, 1)
    res['rules'] = {k: [float(x) for x in v.cpu()] for k, v in rules.items()}
    # picks
    # ties broken by a fixed random permutation (so an all-zero rule picks 2 arbitrary groups, flagged)
    tb = np.random.default_rng(7).permutation(n)
    res['picks'], res['tied'] = {}, {}
    for k, v in res['rules'].items():
        v = np.array(v)
        order = sorted(range(n), key=lambda i: (-v[i], tb[i]))
        res['picks'][k] = [int(i) for i in order[:2]]
        res['tied'][k] = int((np.abs(v - v[order[1]]) <= 1e-9 * max(1e-12, np.abs(v).max())).sum())
    res['sec'] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, 'w'))
    print(json.dumps({k: res[k] for k in ('vC', 'v0', 'completeness', 'sec')}), flush=True)
    for k in res['picks']:
        print(k, res['picks'][k], np.round(res['rules'][k], 4).tolist(), flush=True)


# ---------------------------------------------------------------- fine-tune
def group_score(q, tgt, m):
    """mean over batch of log score of q (B, GW, L) on binary tgt, per bit (masked)."""
    ll = -F.binary_cross_entropy_with_logits(q, tgt, reduction='none') * m[:, None]
    return (ll.sum((1, 2)) / (m.sum(1) * GW)).mean()


def ft_loss(arm, base_arm, model, heads, groups, x, y, m, T, lam):
    bs, _, _, hs = model(x, m, T, want_h=True)
    nbits = m.sum(1)
    S = [score(b, y, m) for b in bs]
    loss = (-S[-1] / nbits).mean()
    logs = {'final': loss.item()}
    if base_arm == 'Bds':
        loss = loss + torch.stack([(-s / nbits).mean() for s in S[1:]]).mean()
    if arm != 'F0':
        nh = [model.norm(h) for h in hs]
        tot = 0.0
        for gi, g in enumerate(groups):
            sl = slice(g * GW, (g + 1) * GW)
            tgt = (nh[-1][:, sl] > 0).float().detach()
            q = torch.zeros_like(tgt)
            terms = []
            for t in range(T):
                dq = heads[gi](nh[t][:, sl]) * m[:, None]
                if arm.endswith('mkt'):
                    pre = q.detach()
                    terms.append(-(group_score(pre + dq, tgt, m) - group_score(pre, tgt, m)))
                else:
                    terms.append(-group_score(q + dq, tgt, m))
                q = q + dq
            tot = tot + torch.stack(terms).mean()
            logs[f'final_q_score_g{g}'] = group_score(q, tgt, m).item()
        loss = loss + lam * tot / len(groups)
        logs['aux'] = float(tot) / len(groups)
    logs['loss'] = loss.item()
    return loss, logs


@torch.no_grad()
def evaluate(model, device, sets, chunk):
    out = {}
    for name in sets:
        lmin, lmax, n, Tc = EVAL[name]
        g = torch.Generator().manual_seed(777 + lmin * 1000 + lmax)
        x, y, m = prefix_batch(n, lmin, lmax, g, device)
        acc, nll = [], []
        for i in range(0, n, chunk):
            xs, ys, ms = x[i:i + chunk], y[i:i + chunk], m[i:i + chunk]
            e = model.encode(xs)
            h = torch.zeros(len(xs), model.w, xs.shape[1], device=device)
            b = torch.zeros(len(xs), xs.shape[1], device=device)
            for _ in range(Tc):
                h, b = model.step(h, b, e, ms)
            acc.append((((b > 0).float() == ys) | (ms == 0)).all(1).float().cpu())
            nll.append((-score(b, ys, ms) / ms.sum(1)).cpu())
        out[name] = dict(acc=float(torch.cat(acc).mean()), nll_bit=float(torch.cat(nll).mean()), T_cap=Tc, n=n)
    return out


@torch.no_grad()
def group_var(model, groups, device, T):
    g = torch.Generator().manual_seed(4242)
    x, y, m = prefix_batch(512, 8, 32, g, device)
    _, _, _, hs = model(x, m, T, want_h=True)
    nh = model.norm(hs[-1])
    out = {}
    for k in range(NG):
        v = nh[:, k * GW:(k + 1) * GW]
        mk = m[:, None].expand_as(v) > 0
        out[k] = float(v[mk].var())
    return out


def finetune(a):
    dev = a.device
    model, args = load(a.ckpt, dev)
    base_arm = args['arm']
    T = args['T']
    groups = [int(g) for g in a.groups.split(',')] if a.groups else []
    torch.manual_seed(a.seed)
    heads = nn.ModuleList([nn.Conv1d(GW, GW, 1) for _ in groups]).to(dev)
    var0 = group_var(model, groups, dev, T)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * i / a.iters)))
    gen = torch.Generator().manual_seed(2000 + a.seed)   # same data across arms
    log, t0 = [], time.time()
    model.train()
    for it in range(a.iters):
        x, y, m = prefix_batch(a.bs, args['lmin'], args['lmax'], gen, dev)
        loss, lg = ft_loss(a.arm, base_arm, model, heads, groups, x, y, m, T, a.lam)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if it % 100 == 0 or it == a.iters - 1:
            lg.update(it=it, sec=round(time.time() - t0, 1))
            log.append(lg)
            print(json.dumps(lg), flush=True)
    model.eval()
    var1 = group_var(model, groups, dev, T)
    ev = evaluate(model, dev, a.sets.split(','), a.eval_chunk)
    res = dict(base=os.path.basename(a.ckpt).replace('.pt', ''), base_arm=base_arm, arm=a.arm, seed=a.seed,
               groups=groups, lam=a.lam, iters=a.iters, lr=a.lr, eval=ev,
               primary=0.5 * (ev['L64']['acc'] + ev['L128']['acc']) if 'L128' in ev else None,
               var_ratio={k: var1[k] / max(var0[k], 1e-12) for k in var0},
               collapse=bool(any(var1[g] < 0.5 * var0[g] for g in groups)),
               log=log, sec=round(time.time() - t0, 1))
    json.dump(res, open(a.out, 'w'))
    print('DONE', json.dumps({k: res[k] for k in ('base', 'arm', 'seed', 'groups', 'primary', 'collapse', 'sec')}),
          json.dumps(ev), flush=True)


def evalbase(a):
    model, args = load(a.ckpt, a.device)
    ev = evaluate(model, a.device, a.sets.split(','), a.eval_chunk)
    json.dump(dict(ckpt=a.ckpt, eval=ev), open(a.out, 'w'))
    print(json.dumps(ev), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['select', 'finetune', 'evalbase'])
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    # select
    p.add_argument('--t', type=int, default=16)
    p.add_argument('--P', type=int, default=256)
    p.add_argument('--m', type=int, nargs='+', default=[5, 10, 20, 50])
    p.add_argument('--shapK', type=int, default=0)
    p.add_argument('--refK', type=int, default=448)
    p.add_argument('--chunk', type=int, default=256)
    # finetune
    p.add_argument('--arm', default='F0')
    p.add_argument('--groups', default='')
    p.add_argument('--iters', type=int, default=1000)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lam', type=float, default=0.1)
    p.add_argument('--sets', default='id_8_32,L64,L128,L256')
    p.add_argument('--eval_chunk', type=int, default=250)
    p.add_argument('--threads', type=int, default=0)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    {'select': select, 'finetune': finetune, 'evalbase': evalbase}[a.cmd](a)
