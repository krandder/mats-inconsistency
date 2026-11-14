"""Noisy fixed-norm tiny PFN (phase B, 2026-10-28).

Architecture = tiny-pfn/pfn.py (pre-LN set transformer, mixture head) with two changes:
  * every written activation is projected to the sphere of radius sqrt(dim) (RMS 1 per coordinate, no learned gain),
  * then i.i.d. N(0, sigma^2) noise is added.
Noisy sites (per token): emb (after the input projection), a{l} (residual after attention sublayer l), m{l} (MLP hidden
units of block l, 2d wide), r{l} (residual after the MLP sublayer of block l). Readers (LN1, LN2, final LN) are unchanged.
Each site is a Gaussian channel with a hard per-token power constraint ||s||^2 = dim:
    I(X; H_site) <= T * (dim/2) * log2(1 + 1/sigma^2) bits   (T = 16 tokens; per token (dim/2) log2(1 + 1/sigma^2)).
Markov chain over residual cuts: X -> emb -> a1 -> r1 -> a2 -> r2 -> ... (m_l hangs off a_l: X -> a_l -> m_l).

usage: python nnoise.py train --init RUN --sigma S --out DIR [--steps 25000 --cooldown 10000]
       python nnoise.py eval --run DIR [--kdraw 16]
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

TP = os.path.dirname(os.path.abspath(__file__))  # gate-entropy: local copies of pfn.py, world.py
sys.path.insert(0, TP)
import world as W                      # noqa: E402
import pfn as P                        # noqa: E402

torch.set_num_threads(1)


def sphere(v):
    return v * (math.sqrt(v.shape[-1]) / v.norm(dim=-1, keepdim=True).clamp_min(1e-6))


class NBlock(nn.Module):
    def __init__(self, d, h, dh=None):
        super().__init__()
        self.h = h; self.dh = dh or d // h
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * h * self.dh); self.o = nn.Linear(h * self.dh, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def attn_out(self, x, allow):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(self.h * self.dh, -1)
        sp = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        s = (q @ k.transpose(-1, -2) / math.sqrt(self.dh)).masked_fill(~allow[:, None], float('-inf'))
        return self.o((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, self.h * self.dh))

    def hidden_pre(self, x):                        # MLP hidden before sphere+noise
        return self.mlp[1](self.mlp[0](self.ln2(x)))

    def write_mlp(self, x, m):
        return x + self.mlp[2](m)


class NoisyPFN(nn.Module):
    def __init__(self, fin, depth, d, h, mixk=4, sigma=0.0):
        super().__init__()
        self.mixk, self.sigma, self.depth, self.d = mixk, sigma, depth, d
        self.inp = nn.Linear(fin, d)
        self.blocks = nn.ModuleList([NBlock(d, h) for _ in range(depth)])
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, 3 * mixk)

    @staticmethod
    def allow_mask(valid):
        B, T = valid.shape
        qi = T - 1
        ctx = valid.clone(); ctx[:, qi] = False
        allow = ctx[:, None, :].expand(B, T, T).clone(); allow[:, qi, qi] = True
        eye = torch.eye(T, dtype=torch.bool, device=valid.device)[None]
        return allow | (eye & ~valid[:, :, None])

    def site_names(self):
        n = ['emb']
        for l in range(1, self.depth + 1):
            n += [f'a{l}', f'm{l}', f'r{l}']
        return n

    def noise(self, s, sigma, gen):
        if sigma == 0:
            return s
        z = torch.randn(s.shape, generator=gen) if gen is not None else torch.randn_like(s)
        return s + sigma * z

    # ---- step functions: each maps the previous *residual* state (and for r, the m state) to the next noisy state
    def f_emb(self, X):
        return sphere(self.inp(X))

    def f_a(self, l, x, allow):
        b = self.blocks[l - 1]
        return sphere(x + b.attn_out(x, allow))

    def f_m(self, l, xa):
        return sphere(self.blocks[l - 1].hidden_pre(xa))

    def f_r(self, l, xa, m):
        return sphere(self.blocks[l - 1].write_mlp(xa, m))

    def head(self, x):
        y = self.out(self.ln(x[:, -1]))
        K = self.mixk
        return y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K]

    def forward(self, X, valid, want=False, sigma=None, gen=None):
        sg = self.sigma if sigma is None else sigma
        allow = self.allow_mask(valid)
        st = {}
        x = self.noise(self.f_emb(X), sg, gen); st['emb'] = x
        for l in range(1, self.depth + 1):
            xa = self.noise(self.f_a(l, x, allow), sg, gen); st[f'a{l}'] = xa
            m = self.noise(self.f_m(l, xa), sg, gen); st[f'm{l}'] = m
            x = self.noise(self.f_r(l, xa, m), sg, gen); st[f'r{l}'] = x
        hd = self.head(x)
        return (*hd, st) if want else hd


def load_clean_into(model, ckpt):
    sd = torch.load(ckpt, map_location='cpu')
    sd = {k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)


def pad_load(model, ckpt, d0, h):
    """Warm start a wider model: copy the base weights into the leading block (respecting the head layout of qkv / o);
    every other weight entry keeps its default init scaled by 0.1 (LN gains 1, biases 0). Not function-preserving."""
    src = torch.load(ckpt, map_location='cpu'); new = model.state_dict(); d1 = model.d
    dh0, dh1 = d0 // h, d1 // h
    for k, v in src.items():
        t = new[k].clone()
        if t.dim() == 2: t.mul_(0.1)
        if k.endswith('qkv.weight'):
            t.view(3, h, dh1, d1)[:, :, :dh0, :d0] = v.view(3, h, dh0, d0)
        elif k.endswith('qkv.bias'):
            t.view(3, h, dh1)[:, :, :dh0] = v.view(3, h, dh0)
        elif k.endswith('.o.weight'):
            t.view(d1, h, dh1)[:d0, :, :dh0] = v.view(d0, h, dh0)
        else:
            t[tuple(slice(0, n) for n in v.shape)] = v
        new[k] = t
    model.load_state_dict(new)


def build(meta):
    return NoisyPFN(P.fin_of(meta['ds']), meta['depth'], meta['width'], meta['heads'], meta['mixk'], meta['sigma'])


def load_run(run):
    meta = json.load(open(os.path.join(run, 'meta.json')))
    m = build(meta); m.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu')); m.eval()
    return m, meta


def predict_mix(model, X, m, kdraw, seed, bs=2000):
    """Noise-marginalised predictive as one mixture: components of all kdraw noise draws, logits shifted by -log kdraw.
    Also returns the per-draw heads for the single-draw expected KL."""
    g = torch.Generator().manual_seed(seed)
    mus, lss, lgs = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            Xi = torch.from_numpy(X[i:i + bs]); mi = torch.from_numpy(m[i:i + bs])
            a, b, c = [], [], []
            for k in range(kdraw):
                mu, ls, lg = model(Xi, mi, gen=g)
                a.append(mu); b.append(ls); c.append(torch.log_softmax(lg, -1))
            mus.append(torch.stack(a, 1)); lss.append(torch.stack(b, 1)); lgs.append(torch.stack(c, 1))
    return torch.cat(mus).numpy(), torch.cat(lss).numpy(), torch.cat(lgs).numpy()       # (n, kdraw, K)


def mix_nll(mu, ls, lg, y):
    """mu, ls, lg (n, R, K) -> NLL of the equal-weight mixture over R of the K-mixtures."""
    n, R, K = mu.shape
    lw = lg - math.log(R)
    lp = lw - ls - 0.5 * math.log(2 * math.pi) - 0.5 * ((y[:, None, None] - mu) / np.exp(ls)) ** 2
    lp = lp.reshape(n, -1); mx = lp.max(1, keepdims=True)
    return -(mx[:, 0] + np.log(np.exp(lp - mx).sum(1)))


def kl_mix_grid(A, Bm, ngrid=1201, chunk=200):
    """Per-episode KL(A || B) for two Gaussian mixtures given as (mu, logstd, log-weights) arrays (n, K), by quadrature."""
    out = np.empty(len(A[0]))
    def lpdf(g, mu, ls, lw):
        lw = lw - np.logaddexp.reduce(lw, axis=1, keepdims=True)
        return np.logaddexp.reduce(lw[:, None] - ls[:, None] - 0.5 * math.log(2 * math.pi) - 0.5 * ((g[:, :, None] - mu[:, None]) / np.exp(ls)[:, None]) ** 2, axis=2)
    for i in range(0, len(out), chunk):
        a = [t[i:i + chunk] for t in A]; b = [t[i:i + chunk] for t in Bm]
        lo = np.minimum((a[0] - 9 * np.exp(a[1])).min(1), (b[0] - 9 * np.exp(b[1])).min(1))
        hi = np.maximum((a[0] + 9 * np.exp(a[1])).max(1), (b[0] + 9 * np.exp(b[1])).max(1))
        g = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, ngrid)[None]
        la = lpdf(g, *a); lb = lpdf(g, *b)
        out[i:i + chunk] = np.trapezoid(np.exp(la) * (la - lb), g, axis=1)
    return out


def train(a):
    torch.manual_seed(a.seed)
    src = json.load(open(os.path.join(a.init, 'meta.json')))
    meta = dict(width=a.width or src['width'], heads=src['heads'], depth=src['depth'], ds=src['ds'], mixk=src['mixk'], sigma=a.sigma,
                seed=a.seed, init_from=a.init, steps=a.steps, cooldown=a.cooldown, ramp=a.ramp, lr=a.lr)
    model = build(meta)
    if meta['width'] == src['width']:
        load_clean_into(model, os.path.join(a.init, 'model.pt'))
    else:
        pad_load(model, os.path.join(a.init, 'model.pt'), src['width'], src['heads'])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=P.WD)
    vd, vX, vm = P.fixed_set(P.VAL_SEED, P.N_VAL, meta['ds'])
    vy = vd['y'][:, W.QL]; _, vnll, _, _ = W.references(vd); vb = float(vnll['bayes'].mean())
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng([a.seed, int(a.sigma * 1000), 11])
    total = a.steps + a.cooldown
    best, curve, t0 = 1e9, [], time.time()

    def val():
        model.eval()
        mu, ls, lg = predict_mix(model, vX, vm, 4, 777)
        model.train()
        return float(mix_nll(mu, ls, lg, vy).mean())

    v0 = val(); meta['init_val_gap'] = v0 - vb
    print(f'[sigma {a.sigma}] init (with noise, before finetune) gap {v0 - vb:+.4f}', flush=True)
    for step in range(1, total + 1):
        lr = a.lr * min(1.0, step / 500) if step <= a.steps else a.lr * (total - step) / a.cooldown
        for g in opt.param_groups: g['lr'] = lr
        model.sigma = a.sigma * min(1.0, step / a.ramp)
        d = W.sample_np(P.BATCH, rng); X, m = W.tokens_np(d, rng, meta['ds'])
        head = model(torch.from_numpy(X), torch.from_numpy(m))
        loss = P.head_nll_t(head, torch.from_numpy(d['y'][:, W.QL]).float()).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            model.sigma = a.sigma
            v = val(); curve.append((step, v))
            if step >= a.ramp and v < best:
                best = v; torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
            print(f'[sigma {a.sigma}] step {step} lr {lr:.1e} val gap {v - vb:+.4f} best {best - vb:+.4f} ({time.time() - t0:.0f}s)', flush=True)
    meta.update(best_val=best, val_bayes=vb, curve=curve, seconds=time.time() - t0)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


def evaluate(a):
    model, meta = load_run(a.run)
    T = np.load(os.path.join(TP, 'runs/testset.npz'))
    X, m, ys = T['X'][:a.n], T['valid'][:a.n], T['ystar'][:a.n]
    Tn = {k: T[k][:a.n] for k in ('comp_means', 'w_bayes', 'nll_bayes')}; Tn['comp_vars'] = T['comp_vars']
    mu, ls, lg = predict_mix(model, X, m, a.kdraw, 20_001)
    n, R, K = mu.shape
    nll = mix_nll(mu, ls, lg, ys)
    kl_marg = P.kl_grid(Tn, (mu.reshape(n, -1), ls.reshape(n, -1), lg.reshape(n, -1)), ngrid=1201, chunk=200)
    kl_single = np.mean([P.kl_grid(Tn, (mu[:, r], ls[:, r], lg[:, r]), ngrid=1201) for r in range(min(R, 4))], 0)
    res = dict(sigma=meta['sigma'], depth=meta['depth'], n=int(n), kdraw=R,
               gap=float((nll - Tn['nll_bayes']).mean()), gap_se=float((nll - Tn['nll_bayes']).std() / math.sqrt(n)),
               kl_marg=float(kl_marg.mean()), kl_marg_se=float(kl_marg.std() / math.sqrt(n)),
               kl_single=float(kl_single.mean()), kl_single_se=float(kl_single.std() / math.sqrt(n)))
    ref = os.path.join(os.path.dirname(os.path.abspath(a.run)), f"L{meta['depth']}_s0.0")      # noiseless base model
    if meta['sigma'] > 0 and os.path.exists(os.path.join(ref, 'model.pt')):
        rm, _ = load_run(ref)
        r_mu, r_ls, r_lg = predict_mix(rm, X, m, 1, 1)
        agree = kl_mix_grid((r_mu[:, 0], r_ls[:, 0], r_lg[:, 0]), (mu.reshape(n, -1), ls.reshape(n, -1), lg.reshape(n, -1)))
        res.update(agree_kl=float(agree.mean()), agree_kl_se=float(agree.std() / math.sqrt(n)))
    json.dump(res, open(os.path.join(a.run, 'eval.json'), 'w'))
    np.savez_compressed(os.path.join(a.run, 'test_kl.npz'), kl_marg=kl_marg, kl_single=kl_single)
    print(json.dumps(res))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--init', default=''); ap.add_argument('--sigma', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='')
    ap.add_argument('--steps', type=int, default=25_000); ap.add_argument('--cooldown', type=int, default=10_000)
    ap.add_argument('--ramp', type=int, default=5_000); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--width', type=int, default=0); ap.add_argument('--run', default=''); ap.add_argument('--kdraw', type=int, default=16); ap.add_argument('--n', type=int, default=8000)
    a = ap.parse_args()
    {'train': train, 'eval': evaluate}[a.cmd](a)
