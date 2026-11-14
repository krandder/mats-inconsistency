"""E4: World FA binding with routing structure. Flat dense attention failed to bind history columns to query traders
(world-fa A4/A6: 0% of the history value). Here the same world (prior f, one anti-correlated pair, N = 4, T = 32,
report tokens: one per (trader, round), random 32-d trader and round ids) is given to sfmodels.TokTF with per-layer
routing: told grouping (attention restricted to the same trader's tokens), learned grouping (routing from the id
features only), or flat dense. fa_core_copy.py is an unchanged copy of work/world-fa/fa_core.py.

usage: python e4fa.py mkdata --split val|test --n 2048
       python e4fa.py train --attn dense:all,dense:all,dense:all,dense:all --seed 0 --out DIR [--device cuda]
       python e4fa.py eval --run DIR [--split test]
Read-out: fraction of the history value recovered at T = 32: (NLL_T0bayes - NLL_net) / (NLL_T0bayes - NLL_bayes).
"""
import argparse, json, math, os, time
import numpy as np
import torch
import fa_core_copy as C
import sfmodels as M

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = C.full_cfg({"prior": "f", "f_templates": [{"blocks": [[2, 0.8, 1]], "marg": 1.0}], "tok_mode": "report",
                  "id_ortho": False})
N, T = 4, 32
L_TOK = T * N + N + 1
EVAL_STEPS = (500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 15000, 20000, 25000, 30000)


def index_tensors(dev):
    k = torch.arange(L_TOK, device=dev)
    src = torch.where(k < T * N, k % N, k - T * N); src[-1] = -1
    qq = torch.where(k < T * N, k // N, torch.full_like(k, T)); qq[-1] = -1
    return src, qq


def mix_nll_np(w, m, v, y):
    lp = np.log(np.maximum(w, 1e-300)) - 0.5 * np.log(2 * np.pi * v) - 0.5 * (y[:, None] - m) ** 2 / v
    return -np.logaddexp.reduce(lp, axis=1)


def mkdata(a):
    torch.manual_seed({'val': 101, 'test': 202}[a.split])
    b = C.gen(a.n, N, T, CFG)
    X, Mk = C.tokens(b, CFG)
    assert bool(Mk.all())
    y = b['v'].double().numpy()
    w, m, v = [t.numpy() for t in C.bayes(b, CFG)]
    w0, m0, v0 = [t.numpy() for t in C.bayes(b, CFG, use_hist=False)]
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, 'data', f'fa_{a.split}.npz'), X=X.numpy(), y=y,
                        nll_bayes=mix_nll_np(w, m, v, y), nll_T0=mix_nll_np(w0, m0, v0, y))
    z = np.load(os.path.join(HERE, 'data', f'fa_{a.split}.npz'))
    print(a.split, 'bayes', z['nll_bayes'].mean(), 'T0', z['nll_T0'].mean(), 'history value',
          (z['nll_T0'] - z['nll_bayes']).mean())


def build(a):
    attn = a.attn.split(',')
    return M.TokTF(C.feat_dim(CFG), d=a.d, L=len(attn), H=a.H, K=4, attn=attn, nid=CFG['d_id'] + CFG['d_round'])


def predict(model, X, dev, bs=512):
    src, qq = index_tensors(dev)
    out = []
    tf32 = torch.backends.cuda.matmul.allow_tf32                # evaluation always in full FP32
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.from_numpy(X[i:i + bs]).to(dev)
            n = len(x)
            mu, ls, lg = model(x, src=src[None].expand(n, -1), qq=qq[None].expand(n, -1))
            out.append((mu.cpu().double().numpy(), ls.cpu().double().numpy(), lg.cpu().double().numpy()))
    model.train()
    torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
    return tuple(np.concatenate([o[j] for o in out]) for j in range(3))


def nll_head(mu, ls, lg, y):
    lw = lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)
    lp = lw - ls - 0.5 * math.log(2 * math.pi) - 0.5 * ((y[:, None] - mu) / np.exp(ls)) ** 2
    return -np.logaddexp.reduce(lp, axis=1)


def frac(z, nll):
    hv = z['nll_T0'] - z['nll_bayes']
    return float(((z['nll_T0'] - nll).mean()) / hv.mean()), float((nll - z['nll_bayes']).mean())


def train(a):
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    model = build(a).to(dev)
    if dev.type == "cuda":                          # TF32 matmuls for training steps only (evaluation in FP32)
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    vz = np.load(os.path.join(HERE, 'data', 'fa_val.npz'))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    src, qq = index_tensors(dev)
    os.makedirs(a.out, exist_ok=True)
    meta = dict(vars(a)); meta['params'] = M.n_params(model)
    curve, best, t0, tl = [], -1e9, time.time(), []
    run = torch.zeros((), device=dev)
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 1000) * (1.0 if step <= a.steps - a.cool else (a.steps - step) / a.cool)
        for g in opt.param_groups: g['lr'] = lr
        b = C.gen(a.bs, N, T, CFG, dev)
        X, _ = C.tokens(b, CFG)
        y = b['v'].float()
        n = len(X)
        loss = M.head_nll(model(X, src=src[None].expand(n, -1), qq=qq[None].expand(n, -1)), y).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        run += loss.detach()
        if step % 100 == 0:
            tl.append((step, float(run) / 100)); run.zero_()
        if step in EVAL_STEPS or step == a.steps:
            nll = nll_head(*predict(model, vz['X'], dev), vz['y'])
            f, gap = frac(vz, nll)
            curve.append((step, f, gap))
            if f > best:
                best = f; torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
            print(f'[{a.attn} s{a.seed}] step {step} frac {f:+.3f} gap {gap:.4f} ({time.time() - t0:.0f}s)', flush=True)
    meta.update(curve=curve, train_loss=tl, best_frac=best, seconds=time.time() - t0)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


def evaluate(a):
    meta = json.load(open(os.path.join(a.run, 'meta.json')))
    for k in ('attn', 'd', 'H'):
        setattr(a, k, meta[k])
    dev = torch.device(a.device)
    model = build(a).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, 'model.pt'), map_location=dev))
    z = np.load(os.path.join(HERE, 'data', f'fa_{a.split}.npz'))
    nll = nll_head(*predict(model, z['X'], dev), z['y'])
    f, gap = frac(z, nll)
    d = nll - z['nll_bayes']
    res = dict(frac=f, gap=gap, gap_se=float(d.std() / math.sqrt(len(d))),
               hv=float((z['nll_T0'] - z['nll_bayes']).mean()), n=len(d))
    json.dump(res, open(os.path.join(a.run, f'eval_{a.split}.json'), 'w'))
    print(a.run, json.dumps(res))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--split', default='val'); ap.add_argument('--n', type=int, default=2048)
    ap.add_argument('--attn', default='dense:all,dense:all,dense:all,dense:all')
    ap.add_argument('--d', type=int, default=64); ap.add_argument('--H', type=int, default=4)
    ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--cool', type=int, default=5000)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=''); ap.add_argument('--run', default='')
    a = ap.parse_args()
    {'mkdata': mkdata, 'train': train, 'eval': evaluate}[a.cmd](a)
