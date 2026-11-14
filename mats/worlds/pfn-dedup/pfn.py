"""Set-style PFN on the source world: train, evaluate against exact references, mechanism probes.

usage: python pfn.py train --depth L --keys 0|1 --seed s --out DIR [--steps N]
       python pfn.py eval  --ckpt DIR/model.pt --out DIR [--mech]
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import world as W

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(1)
VAL_SEED, TEST_SEED, MECH_SEED = 10_000, 20_000, 30_000


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h; self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, allow, want_attn=False):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, -1)
        sp = lambda t: t.view(B, T, self.h, d // self.h).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        s = q @ k.transpose(-1, -2) / math.sqrt(d // self.h)
        s = s.masked_fill(~allow[:, None], float('-inf'))
        a = s.softmax(-1)
        x = x + self.o((a @ v).transpose(1, 2).reshape(B, T, d))
        x = x + self.mlp(self.ln2(x))
        return (x, a) if want_attn else (x, None)


class PFN(nn.Module):
    def __init__(self, fin, depth, d=96, h=4):
        super().__init__()
        self.inp = nn.Linear(fin, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(depth)])
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, 2)

    def forward(self, X, valid, ablate=None, want=False):
        B, T, _ = X.shape
        qi = T - 1
        ctx = valid.clone(); ctx[:, qi] = False
        allow = ctx[:, None, :].expand(B, T, T).clone()
        allow[:, qi, qi] = True
        eye = torch.eye(T, dtype=torch.bool, device=X.device)[None]
        allow = allow | (eye & ~valid[:, :, None])           # padded rows see themselves (no NaN)
        self_only = eye.expand(B, T, T).clone()
        self_only[:, qi] = allow[:, qi]                      # the query keeps its normal view
        x = self.inp(X); hs, attns = [x], []
        for li, blk in enumerate(self.blocks):
            x, a = blk(x, self_only if ablate == li else allow, want)
            hs.append(x); attns.append(a)
        y = self.out(self.ln(x[:, qi]))
        mu, logs = y[:, 0], y[:, 1].clamp(-6, 3)
        return (mu, logs, hs, attns) if want else (mu, logs)


def nll(mu, logs, y):
    return (logs + 0.5 * math.log(2 * math.pi) + 0.5 * ((y - mu) / logs.exp()) ** 2)


def fixed_set(seed, n, keys):
    rng = np.random.default_rng(seed)
    d = W.sample_np(n, rng)
    X, m = W.tokens_np(d, rng, keys)
    return d, X, m


def to_t(X, m, y):
    return (torch.tensor(X, device=DEV), torch.tensor(m, device=DEV),
            torch.tensor(y, dtype=torch.float32, device=DEV))


def predict(model, X, m, bs=1024, ablate=None):
    mus, ss = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb, mb = torch.tensor(X[i:i + bs], device=DEV), torch.tensor(m[i:i + bs], device=DEV)
            mu, logs = model(xb, mb, ablate=ablate)
            mus.append(mu.cpu().numpy()); ss.append(logs.cpu().numpy())
    model.train()
    return np.concatenate(mus), np.concatenate(ss)


def train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(1000 + a.seed * 7 + a.depth * 31 + a.keys * 101)
    fin = W.DS * (3 if a.keys else 2) + 4
    model = PFN(fin, a.depth).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    vd, vX, vm = fixed_set(VAL_SEED, 4096, a.keys)
    vy = vd['y'][:, W.QL]
    os.makedirs(a.out, exist_ok=True)
    curve, best, best_step, t0 = [], 1e9, -1, time.time()
    for step in range(1, a.steps + 1):
        lr = 3e-4 * min(1.0, step / 1000)
        for g in opt.param_groups: g['lr'] = lr
        d = W.sample_np(256, rng)
        X, m = W.tokens_np(d, rng, a.keys)
        xb, mb, yb = to_t(X, m, d['y'][:, W.QL])
        mu, logs = model(xb, mb)
        loss = nll(mu, logs, yb).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0 or step == a.steps:
            mu, ls = predict(model, vX, vm)
            v = float(nll(torch.tensor(mu), torch.tensor(ls), torch.tensor(vy, dtype=torch.float32)).mean())
            curve.append((step, v))
            if v < best:
                best, best_step = v, step
                torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
            print(f'[L{a.depth} k{a.keys} s{a.seed}] step {step} val {v:.4f} best {best:.4f}@{best_step} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    last_q = [v for s, v in curve if s >= 0.75 * a.steps]
    before = [v for s, v in curve if s < 0.75 * a.steps]
    improvement = (min(before) - min(last_q)) if before else float('nan')
    meta = dict(depth=a.depth, keys=a.keys, seed=a.seed, steps=a.steps, best_val=best, best_step=best_step,
                last_quarter_improvement=improvement, converged=bool(improvement < 0.003),
                seconds=time.time() - t0, curve=curve, fin=fin)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


def splits(d):
    sizes, has_dup = W.redundancy_type(d)
    return {
        'all': np.ones(len(sizes), bool),
        'dup_singletons': has_dup & (sizes == 1),
        'singletons_nodup': (~has_dup) & (sizes == 1),
        'group2': sizes == 2, 'group3': sizes == 3, 'group4plus': sizes >= 4,
        'shared_any': sizes >= 2,
    }


def evaluate(a):
    meta = json.load(open(os.path.join(os.path.dirname(a.ckpt), 'meta.json')))
    keys = meta['keys']
    model = PFN(meta['fin'], meta['depth']).to(DEV)
    model.load_state_dict(torch.load(a.ckpt, map_location=DEV))
    d, X, m = fixed_set(TEST_SEED, 8000, keys)
    out, rn, (md, vd), w = W.references(d)
    ys = d['y'][:, W.QL]
    mu, ls = predict(model, X, m)
    mnll = (ls + 0.5 * math.log(2 * math.pi) + 0.5 * ((ys - mu) / np.exp(ls)) ** 2)
    bayes = rn['c'] if keys else rn['d']
    bayes_mean = out['c'][0] if keys else md
    res = {'meta': {k: meta[k] for k in ('depth', 'keys', 'seed', 'best_step', 'converged', 'last_quarter_improvement')}}
    for name, sel in splits(d).items():
        if sel.sum() == 0: continue
        g = mnll[sel] - bayes[sel]
        r = {'n': int(sel.sum()), 'gap_model': float(g.mean()), 'se_model': float(g.std() / math.sqrt(sel.sum()))}
        for kk in 'abcd':
            r[f'gap_{kk}'] = float((rn[kk][sel] - bayes[sel]).mean())
        r['mse_to_bayes_mean'] = float(((mu - bayes_mean)[sel] ** 2).mean())
        for kk in 'abc':
            r[f'mse_to_{kk}'] = float(((mu - out[kk][0])[sel] ** 2).mean())
        r['mse_to_d'] = float(((mu - md)[sel] ** 2).mean())
        res[name] = r
    if a.mech:
        res['mech'] = mechanism(model, meta, keys)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'eval.json'), 'w'))
    print(json.dumps({k: (v if k in ('meta', 'mech') else {kk: round(vv, 4) if isinstance(vv, float) else vv
                                                           for kk, vv in v.items()}) for k, v in res.items()}),
          flush=True)


def ridge_r2(H, t, lam=1e-2):
    n = len(H) // 2
    Ha = np.concatenate([H, np.ones((len(H), 1))], 1)
    A, b = Ha[:n], t[:n]
    wv = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
    pred = Ha[n:] @ wv
    return float(1 - ((t[n:] - pred) ** 2).mean() / t[n:].var())


def mechanism(model, meta, keys):
    d, X, m = fixed_set(MECH_SEED, 2000, keys)
    out, rn, (md, vd), w = W.references(d)
    src, valid = W.rows_of(d)
    R = W.S + W.MAXDUP
    xb, mb = torch.tensor(X, device=DEV), torch.tensor(m, device=DEV)
    model.eval()
    with torch.no_grad():
        mu, logs, hs, attns = model(xb, mb, want=True)
    res = {'probe_residual': [], 'probe_groupsize': [], 'probe_query_bayesmean': [], 'ablation': [],
           'query_attn': []}
    # token-level targets: labelled-question tokens of the 6 original rows (r < S, q < QL)
    rows = np.arange(W.S); qs = np.arange(W.QL)
    tok = (rows[:, None] * W.Q + qs[None, :]).reshape(-1)
    resid = (d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]).reshape(len(d['k']), -1)
    gsize = np.stack([np.bincount(l)[l] for l in d['lab']])            # (n, S) size of each source's group
    gs_tok = np.repeat(gsize[:, :, None], W.QL, 2).reshape(len(d['k']), -1)
    bmean = out['c'][0] if keys else md
    for li, h in enumerate(hs):
        H = h.cpu().numpy()
        Ht = H[:, tok].reshape(-1, H.shape[-1])
        res['probe_residual'].append(ridge_r2(Ht, resid.reshape(-1)))
        res['probe_groupsize'].append(ridge_r2(Ht, gs_tok.reshape(-1).astype(float)))
        res['probe_query_bayesmean'].append(ridge_r2(H[:, -1], bmean))
    ys = d['y'][:, W.QL]
    for li in range(len(model.blocks)):
        mu_a, ls_a = predict(model, X, m, ablate=li)
        nl = (ls_a + 0.5 * math.log(2 * math.pi) + 0.5 * ((ys - mu_a) / np.exp(ls_a)) ** 2)
        bayes = rn['c'] if keys else rn['d']
        res['ablation'].append({'layer': li, 'gap': float((nl - bayes).mean()),
                                **{f'mse_to_{k}': float(((mu_a - out[k][0]) ** 2).mean()) for k in 'abc'},
                                'mse_to_d': float(((mu_a - md) ** 2).mean())})
    # query attention per source on query-question tokens, per layer, averaged over heads
    qtok = np.arange(R) * W.Q + W.QL
    gls_w = W.WVEC[d['pidx']]; gls_w = gls_w / gls_w.sum(1, keepdims=True)  # per distinct source
    nodup = d['k'] == 0
    for li, a in enumerate(attns):
        A = a.mean(1)[:, -1].cpu().numpy()                   # (n, T) query row
        mass_q = A[:, qtok]                                   # (n, R) per row
        frac_on_query_q = float(mass_q.sum(1).mean())
        per_src = np.zeros((len(A), W.S))
        for r in range(R):
            np.add.at(per_src, (np.arange(len(A)), src[:, r]), mass_q[:, r] * valid[:, r])
        per_src_n = per_src / np.maximum(per_src.sum(1, keepdims=True), 1e-12)
        sel = nodup
        corr = float(np.corrcoef(per_src_n[sel].ravel(), gls_w[sel].ravel())[0, 1])
        by_m = {int(mm): float(per_src_n[sel][gsize[sel] == mm].mean()) for mm in range(1, 7)
                if (gsize[sel] == mm).any()}
        gls_by_m = {int(mm): float(gls_w[sel][gsize[sel] == mm].mean()) for mm in range(1, 7)
                    if (gsize[sel] == mm).any()}
        res['query_attn'].append({'layer': li, 'frac_on_query_question': frac_on_query_q,
                                  'corr_with_gls_weight_nodup': corr, 'mass_by_groupsize': by_m,
                                  'gls_by_groupsize': gls_by_m})
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--depth', type=int, default=2); ap.add_argument('--keys', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--out', default='out'); ap.add_argument('--ckpt', default=''); ap.add_argument('--mech', action='store_true')
    a = ap.parse_args()
    train(a) if a.cmd == 'train' else evaluate(a)
