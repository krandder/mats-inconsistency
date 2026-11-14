"""Tiny set-style PFN on the tiny reliability world: train and evaluate against exact references.

Adapted from ../pfn-dedup/pfn.py (copied): same pre-norm block, no positional encoding, same masks, Gaussian
mean/log-std head. Changes: width/heads/depth/MLP-ratio configurable, optional attention-only blocks, no keys arm,
lr 1e-3 with a final linear cooldown (see PREREG.md).

usage: python pfn.py train --width D --heads H --depth L [--attn_only] --seed s --out DIR
       python pfn.py testset --out DIR            # fixed test set + all references (once)
       python pfn.py eval --run DIR --testset DIR/testset.npz
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import world as W

torch.set_num_threads(1)
VAL_SEED, TEST_SEED = 10_000, 20_000
N_VAL, N_TEST = 4096, 16_000
LR, WD, WARMUP, BATCH = 1e-3, 0.01, 1000, 256
STEPS1, STEPS1_EXT, COOLDOWN, EVAL_EVERY = 40_000, 80_000, 10_000, 1000
CONV_TOL = 0.003
if os.environ.get('TINYPFN_SMOKE'):                                    # pipeline smoke test only
    STEPS1, STEPS1_EXT, COOLDOWN, EVAL_EVERY, N_VAL, N_TEST = 2000, 4000, 1000, 500, 512, 1000


class Block(nn.Module):
    def __init__(self, d, h, mlp_ratio=2, attn_only=False):
        super().__init__()
        self.h = h; self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.attn_only = attn_only
        if not attn_only:
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, x, allow, want_attn=False):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, -1)
        sp = lambda t: t.view(B, T, self.h, d // self.h).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        s = q @ k.transpose(-1, -2) / math.sqrt(d // self.h)
        s = s.masked_fill(~allow[:, None], float('-inf'))
        a = s.softmax(-1)
        x = x + self.o((a @ v).transpose(1, 2).reshape(B, T, d))
        if not self.attn_only:
            x = x + self.mlp(self.ln2(x))
        return x, (a if want_attn else None)


class PFN(nn.Module):
    def __init__(self, fin, depth, d, h, attn_only=False, mlp_ratio=2, mixk=0):
        super().__init__()
        self.mixk = mixk                                       # 0 = Gaussian head; K > 0 = K-component mixture
        self.inp = nn.Linear(fin, d)
        self.blocks = nn.ModuleList([Block(d, h, mlp_ratio, attn_only) for _ in range(depth)])
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, 2 if mixk == 0 else 3 * mixk)

    def forward(self, X, valid, want=False):
        B, T, _ = X.shape
        qi = T - 1
        ctx = valid.clone(); ctx[:, qi] = False
        allow = ctx[:, None, :].expand(B, T, T).clone()      # context tokens see all valid context tokens
        allow[:, qi, qi] = True                              # query sees context + itself; nobody sees the query
        eye = torch.eye(T, dtype=torch.bool, device=X.device)[None]
        allow = allow | (eye & ~valid[:, :, None])
        x = self.inp(X); hs, attns = [x], []
        for blk in self.blocks:
            x, a = blk(x, allow, want)
            hs.append(x); attns.append(a)
        y = self.out(self.ln(x[:, qi]))
        if self.mixk == 0:
            head = (y[:, 0], y[:, 1].clamp(-6, 3))                       # (mu, logs)
        else:
            K = self.mixk                                               # (mu_k, logs_k, logits_k), each (B, K)
            head = (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
        return (*head, hs, attns) if want else head


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def nll_t(mu, logs, y):
    return logs + 0.5 * math.log(2 * math.pi) + 0.5 * ((y - mu) / logs.exp()) ** 2


def head_nll_t(head, y):
    if len(head) == 2:
        return nll_t(head[0], head[1], y)
    mu, logs, logits = head
    lp = torch.log_softmax(logits, -1) - logs - 0.5 * math.log(2 * math.pi) - 0.5 * ((y[:, None] - mu) / logs.exp()) ** 2
    return -torch.logsumexp(lp, -1)


def head_nll_np(head, y):
    if len(head) == 2:
        return nll_np(head[0], head[1], y)
    mu, logs, logits = head
    lw = logits - logits.max(-1, keepdims=True); lw = lw - np.log(np.exp(lw).sum(-1, keepdims=True))
    lp = lw - logs - 0.5 * math.log(2 * math.pi) - 0.5 * ((y[:, None] - mu) / np.exp(logs)) ** 2
    mx = lp.max(-1, keepdims=True)
    return -(mx[:, 0] + np.log(np.exp(lp - mx).sum(-1)))


def head_moments(head):
    if len(head) == 2:
        return head[0], np.exp(2 * head[1])
    mu, logs, logits = head
    w = np.exp(logits - logits.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    m = (w * mu).sum(-1)
    return m, (w * (np.exp(2 * logs) + mu ** 2)).sum(-1) - m ** 2


def kl_grid(T, head, ngrid=2001, chunk=1000):
    """KL(Bayes || model) per episode by quadrature on a per-episode grid spanning +-9 sd of every Bayes
    component and +-9 sd of every model component."""
    means, vars_, w = T['comp_means'], T['comp_vars'], T['w_bayes']
    if len(head) == 2:
        mm, ll, lg = head[0][:, None], head[1][:, None], np.zeros((len(w), 1))
    else:
        mm, ll, lg = head[0], head[1], head[2]
    sd = np.sqrt(vars_); out = np.empty(len(w))
    for i in range(0, len(w), chunk):
        sl = slice(i, i + chunk)
        lo = np.minimum((means[sl] - 9 * sd).min(1), (mm[sl] - 9 * np.exp(ll[sl])).min(1))
        hi = np.maximum((means[sl] + 9 * sd).max(1), (mm[sl] + 9 * np.exp(ll[sl])).max(1))
        g = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, ngrid)[None]
        lpc = np.log(w[sl])[:, None, :] - 0.5 * np.log(2 * np.pi * vars_) - 0.5 * (g[:, :, None] - means[sl][:, None]) ** 2 / vars_
        lqc = (lg[sl] - np.log(np.exp(lg[sl] - lg[sl].max(1, keepdims=True)).sum(1, keepdims=True)) - lg[sl].max(1, keepdims=True))[:, None, :] \
            - ll[sl][:, None] - 0.5 * np.log(2 * np.pi) - 0.5 * (g[:, :, None] - mm[sl][:, None]) ** 2 / np.exp(2 * ll[sl])[:, None]
        lp = np.logaddexp.reduce(lpc, axis=2); lq = np.logaddexp.reduce(lqc, axis=2)
        out[sl] = np.trapezoid(np.exp(lp) * (lp - lq), g, axis=1)
    return out


def nll_np(mu, logs, y):
    return logs + 0.5 * math.log(2 * math.pi) + 0.5 * ((y - mu) / np.exp(logs)) ** 2


def fin_of(ds):
    return 2 * ds + 4


def fixed_set(seed, n, ds):
    rng = np.random.default_rng(seed)
    d = W.sample_np(n, rng)                    # world draws come first: episodes are identical for every ds
    X, m = W.tokens_np(d, rng, ds)
    return d, X, m


def predict(model, X, m, bs=2048):
    outs = []
    was = model.training; model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append([t.numpy() for t in model(torch.from_numpy(X[i:i + bs]), torch.from_numpy(m[i:i + bs]))])
    model.train(was)
    return tuple(np.concatenate([o[j] for o in outs]) for j in range(len(outs[0])))


def cfg_name(a):
    return (f"w{a.width}_h{a.heads}_L{a.depth}" + ("_attn" if a.attn_only else "") + (f"_ds{a.ds}" if a.ds != W.DS else "")
            + (f"_mix{a.mixk}" if a.mixk else "") + ("_long" if a.phase1_fixed else "") + ("_ws" if a.init_from else ""))


def build(meta):
    return PFN(fin_of(meta['ds']), meta['depth'], meta['width'], meta['heads'], meta['attn_only'], mixk=meta.get('mixk', 0))


WS_OFFSETS = (-0.15, -0.05, 0.05, 0.15)


def warm_start(model, ckpt, seed):
    """Copy every body weight from a Gaussian-head checkpoint; initialise the K-component mixture head around the
    Gaussian output: component means = Gaussian mean + fixed offsets (WS_OFFSETS, nats-free units of y) with N(0, 1e-3)
    weight jitter, component log-stds = Gaussian log-std, equal logits (zero weights and biases)."""
    sd = torch.load(ckpt)
    body = {k: v for k, v in sd.items() if not k.startswith('out.')}
    missing, unexpected = model.load_state_dict(body, strict=False)
    assert set(missing) == {'out.weight', 'out.bias'} and not unexpected, (missing, unexpected)
    K = model.mixk; Wg, bg = sd['out.weight'], sd['out.bias']
    assert K == len(WS_OFFSETS) and Wg.shape[0] == 2
    g = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        model.out.weight.zero_(); model.out.bias.zero_()
        for k in range(K):
            model.out.weight[K + k] = Wg[0] + 1e-3 * torch.randn(Wg.shape[1], generator=g)
            model.out.bias[K + k] = bg[0] + WS_OFFSETS[k]
            model.out.weight[2 * K + k] = Wg[1]
            model.out.bias[2 * K + k] = bg[1]


def conv_rule(curve, end):
    """Parent rule on the constant-lr phase ending at `end`: improvement of best val NLL over the last 25%."""
    last = [v for s, v in curve if 0.75 * end <= s <= end]
    before = [v for s, v in curve if s < 0.75 * end]
    imp = min(before) - min(last)
    return imp, bool(imp < CONV_TOL)


def train(a):
    torch.manual_seed(a.seed)
    cid = sum(ord(c) * (i + 1) for i, c in enumerate(cfg_name(a)))
    rng = np.random.default_rng([a.seed, cid, 7])                     # training stream, disjoint from val/test ints
    meta = dict(width=a.width, heads=a.heads, depth=a.depth, attn_only=a.attn_only, ds=a.ds, seed=a.seed, cfg=cfg_name(a),
                mixk=a.mixk, phase1_fixed=a.phase1_fixed)
    model = build(meta)
    meta['params'] = n_params(model)
    if a.init_from:
        warm_start(model, a.init_from, a.seed)
        meta['init_from'] = a.init_from
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    vd, vX, vm = fixed_set(VAL_SEED, N_VAL, a.ds)
    vy = vd['y'][:, W.QL]
    _, vnll, _, _ = W.references(vd)
    os.makedirs(a.out, exist_ok=True)
    if a.init_from:
        meta['init_val'] = float(head_nll_np(predict(model, vX, vm), vy).mean())
        meta['init_val_gap'] = meta['init_val'] - float(vnll['bayes'].mean())
        print(f"[{cfg_name(a)} s{a.seed}] warm start from {a.init_from}: val {meta['init_val']:.4f} gap {meta['init_val_gap']:+.4f}", flush=True)
    curve, best, best_step, t0 = [], 1e9, -1, time.time()
    best1, best1_step = 1e9, -1
    end1 = a.phase1_fixed or STEPS1; conv = None; step = 0; total = None

    def lr_at(s):
        if total is None or s <= end1:
            return LR * min(1.0, s / WARMUP)
        return LR * max(0.0, (total - s) / COOLDOWN)

    while True:
        step += 1
        for g in opt.param_groups: g['lr'] = lr_at(step)
        d = W.sample_np(BATCH, rng)
        X, m = W.tokens_np(d, rng, a.ds)
        head = model(torch.from_numpy(X), torch.from_numpy(m))
        loss = head_nll_t(head, torch.from_numpy(d['y'][:, W.QL]).float()).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % EVAL_EVERY == 0:
            v = float(head_nll_np(predict(model, vX, vm), vy).mean())
            curve.append((step, v))
            if v < best:
                best, best_step = v, step
                torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
            if total is None and v < best1:
                best1, best1_step = v, step
                torch.save(model.state_dict(), os.path.join(a.out, 'model_phase1.pt'))
            print(f'[{cfg_name(a)} s{a.seed}] step {step} lr {lr_at(step):.2e} val {v:.4f} gap {v - vnll["bayes"].mean():+.4f} '
                  f'best {best:.4f}@{best_step} ({time.time() - t0:.0f}s)', flush=True)
            if a.phase1_fixed and total is None and step in (STEPS1, STEPS1_EXT) and step < end1:
                imp, c = conv_rule(curve, step)                    # recorded only; the fixed phase continues
                meta.setdefault('conv_checks', []).append(dict(end=step, improvement=imp, converged=c))
            if total is None and step == end1:
                imp, conv = conv_rule(curve, end1)
                meta.setdefault('conv_checks', []).append(dict(end=end1, improvement=imp, converged=conv))
                if conv or a.phase1_fixed or end1 >= STEPS1_EXT:
                    total = end1 + COOLDOWN
                else:
                    end1 = STEPS1_EXT
            if total is not None and step >= total:
                break
    meta.update(phase1_steps=end1, total_steps=total, converged=conv, best_val=best, best_step=best_step,
                best_val_phase1=best1, best_step_phase1=best1_step, val_bayes=float(vnll['bayes'].mean()),
                seconds=time.time() - t0, curve=curve)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


def testset(a):
    d, X, m = fixed_set(TEST_SEED, N_TEST, a.ds)
    out, nll, mom, w = W.references(d)
    means, vars_, _ = out['bayes']
    H = W.mix_entropy(means, vars_, w)
    mb, vb = mom['bayes']
    floor = 0.5 * np.log(2 * np.pi * np.e * vb) - H                  # KL(bayes || moment-matched Gaussian)
    sp = W.source_posterior(d)                                        # labelled-only per-source posterior
    wl = out['lab'][2]
    R = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]
    os.makedirs(a.out, exist_ok=True)
    fn = 'testset.npz' if a.ds == W.DS else f'testset_ds{a.ds}.npz'
    np.savez_compressed(os.path.join(a.out, fn), X=X, ds=a.ds, valid=m, o=d['o'], y=d['y'], v=d['v'], vidx=d['vidx'],
                        ystar=d['y'][:, W.QL], comb=W.COMB, vlev=W.VLEV,
                        mean_a=out['a'][0], var_a=out['a'][1], mean_b=out['b'][0], var_b=out['b'][1],
                        comp_means=means, comp_vars=vars_, w_bayes=w, w_lab=wl,
                        mean_bayes=mb, var_bayes=vb, mean_lab=mom['lab'][0], var_lab=mom['lab'][1],
                        w_qonly=out['qonly'][2], mean_qonly=mom['qonly'][0], var_qonly=mom['qonly'][1],
                        nll_a=nll['a'], nll_b=nll['b'], nll_bayes=nll['bayes'], nll_lab=nll['lab'], nll_qonly=nll['qonly'],
                        entropy_bayes=H, gauss_floor=floor, src_post_lab=sp, ss_resid=(R ** 2).sum(2))
    print('testset', {k: round(float(v.mean()), 4) for k, v in nll.items()}, 'floor', round(float(floor.mean()), 4))


def evaluate(a):
    meta = json.load(open(os.path.join(a.run, 'meta.json')))
    T = np.load(a.testset)
    assert int(T['ds']) == meta['ds'], 'test set token dim does not match the model'
    res = {}
    for tag, f in (('best', 'model.pt'), ('phase1', 'model_phase1.pt')):
        model = build(meta); model.load_state_dict(torch.load(os.path.join(a.run, f)))
        head = predict(model, T['X'], T['valid'])
        ys = T['ystar']
        nll = head_nll_np(head, ys)
        mu, s2 = head_moments(head); ls = 0.5 * np.log(s2); mb, vb = T['mean_bayes'], T['var_bayes']
        # Gaussian head: exact KL = floor + mean part + scale part. Mixture head: KL by quadrature; the 'mean'/'scale'
        # parts are then those of KL(moment-matched Bayes || moment-matched model), a diagnostic only.
        kl_mean = (mb - mu) ** 2 / (2 * s2)
        kl_scale = ls - 0.5 * np.log(vb) + vb / (2 * s2) - 0.5
        kl = T['gauss_floor'] + kl_mean + kl_scale if len(head) == 2 else kl_grid(T, head)
        res[tag] = dict(nll=float(nll.mean()), gap=float((nll - T['nll_bayes']).mean()),
                        gap_se=float((nll - T['nll_bayes']).std() / math.sqrt(len(ys))),
                        kl=float(kl.mean()), kl_mean=float(kl_mean.mean()), kl_scale=float(kl_scale.mean()))
        if tag == 'best':
            extra = {} if len(head) == 2 else dict(comp_mu=head[0], comp_logstd=head[1], comp_logits=head[2])
            np.savez_compressed(os.path.join(a.run, 'test_preds.npz'), mu=mu, logstd=ls, nll=nll, kl=kl,
                                kl_mean=kl_mean, kl_scale=kl_scale, **extra)
    res['meta'] = {k: v for k, v in meta.items() if k != 'curve'}
    json.dump(res, open(os.path.join(a.run, 'eval.json'), 'w'))
    print(meta['cfg'], meta['seed'], json.dumps({k: v for k, v in res.items() if k != 'meta'}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--width', type=int, default=32); ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--depth', type=int, default=2); ap.add_argument('--attn_only', action='store_true')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='out')
    ap.add_argument('--init_from', default='')
    ap.add_argument('--mixk', type=int, default=0); ap.add_argument('--phase1_fixed', type=int, default=0)
    ap.add_argument('--ds', type=int, default=W.DS); ap.add_argument('--run', default=''); ap.add_argument('--testset', default='')
    a = ap.parse_args()
    {'train': train, 'testset': testset, 'eval': evaluate}[a.cmd](a)
