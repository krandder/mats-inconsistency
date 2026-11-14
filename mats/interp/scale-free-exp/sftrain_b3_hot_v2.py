"""Train / fine-tune / evaluate nets on the scale-free worlds (2026-10-29).

usage:
  python sftrain.py mkdata --world indep --N 12 [--rho R --gsize G --a1 --a2] --split val|test --n 4096
  python sftrain.py train  --arch tree --m 8 --world indep --N 12 --seed 0 --out runs/X [--device cuda]
  python sftrain.py eval   --run runs/X [--split test]
Data files: data/<world key>_<split>.npz (episodes + exact Bayes NLL/mean/var [+ grid/logp for test]).
Training draws fresh episodes every step (numpy generator seeded by --seed); val/test episodes are fixed.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import sfworld as W
import sfmodels as M

HERE = os.path.dirname(os.path.abspath(__file__))
DS = 16                                   # id-embedding dim for token nets (source, question, group)
EVAL_STEPS = sorted(set([250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000,
                         20000, 25000, 30000, 35000, 40000, 50000, 60000, 80000, 100000]))


def wkey(a):
    k = f"{a.world}_N{a.N}"
    if a.world in ('groups', 'pairs'):
        k += f"_r{a.rho:g}"
    if a.world == 'groups':
        k += f"_g{a.gsize}"
    if a.world == 'pairs':
        k += f"_a{a.a1:g}-{a.a2:g}"
    return k


def wkw(a):
    return dict(mode=a.world, rho=a.rho, gsize=a.gsize, a1=a.a1, a2=a.a2)


# ----------------------------------------------------------------------------------------------- inputs
def build_inputs(d, arch, rng, device, want_group=False):
    """Returns dict of torch tensors for the given arch. rng draws the random id vectors (fresh per episode)."""
    n, N = d['o'].shape[:2]
    xs = W.src_features(d)
    if want_group and d['mode'] == 'groups':
        G = N // d['gsize']
        gemb = (rng.standard_normal((n, G, DS)) / 4).astype(np.float32)
        ge = np.take_along_axis(gemb, d['grp'][:, :, None].repeat(DS, 2), 1)
    else:
        ge = None
    if arch in ('tree', 'hier'):
        return dict(xs=torch.from_numpy(xs).to(device))
    if arch == 'set':
        if ge is not None:
            xs = np.concatenate([xs, ge], 2)
        return dict(xs=torch.from_numpy(xs).to(device))
    if arch == 'tok':
        semb = (rng.standard_normal((n, N, DS)) / 4).astype(np.float32)
        qemb = (rng.standard_normal((n, W.Q, DS)) / 4).astype(np.float32)
        T = N * W.Q + 1
        F = 2 * DS + 4 + (DS if ge is not None else 0)
        X = np.zeros((n, T, F), dtype=np.float32)
        Xs = X[:, :T - 1].reshape(n, N, W.Q, F)
        Xs[:, :, :, :DS] = semb[:, :, None, :]
        Xs[:, :, :, DS:2 * DS] = qemb[:, None, :, :]
        Xs[:, :, :, 2 * DS] = d['o']
        Xs[:, :, :W.QL, 2 * DS + 1] = d['y'][:, None, :W.QL]
        Xs[:, :, W.QL, 2 * DS + 2] = 1.0
        if ge is not None:
            Xs[:, :, :, 2 * DS + 4:] = ge[:, :, None, :]
        X[:, :T - 1] = Xs.reshape(n, N * W.Q, F)
        X[:, T - 1, DS:2 * DS] = qemb[:, W.QL]
        X[:, T - 1, 2 * DS + 3] = 1.0
        src = np.concatenate([np.repeat(np.arange(N), W.Q), [-1]])
        qq = np.concatenate([np.tile(np.arange(W.Q), N), [-1]])
        gs = d['gsize'] if d['mode'] == 'groups' else 1
        grp = np.concatenate([np.repeat(np.arange(N) // gs, W.Q), [-1]])
        return dict(tok=torch.from_numpy(X).to(device),
                    src=torch.from_numpy(src).to(device)[None].expand(n, T),
                    qq=torch.from_numpy(qq).to(device)[None].expand(n, T),
                    grp=torch.from_numpy(grp).to(device)[None].expand(n, T))
    raise ValueError(arch)


def fin_of(a, world):
    if a.arch in ('tree', 'hier'):
        return 9
    if a.arch == 'set':
        return 9 + (DS if (world == 'groups' and a.gid) else 0)
    if a.arch == 'tok':
        return 2 * DS + 4 + (DS if (world == 'groups' and a.gid) else 0)


def build_model(a, world):
    fin = fin_of(a, world)
    if a.arch == 'tree':
        return M.TreeNet(fin, a.m, h=a.h, K=a.K, pool=a.pool)
    if a.arch == 'hier':
        return M.HierNet(fin, a.gsize, m1=a.m1, m2=a.m, h=a.h, K=a.K, inner=a.inner)
    if a.arch == 'set':
        return M.SetTF(fin, d=a.d, L=a.L, H=a.H, K=a.K, cross=not a.nocross)
    if a.arch == 'tok':
        attn = a.attn.split(',') if a.attn else None
        ck = None
        if a.cut_k != '':
            ck = [None if c == 'n' else int(c) for c in a.cut_k.split(',')]
            if len(ck) == 1:
                ck = ck * a.L
        return M.TokTF(fin, d=a.d, L=a.L, H=a.H, K=a.K, attn=attn, cut=a.cut or None, cut_k=ck,
                       nid=(3 if (world == 'groups' and a.gid) else 2) * DS)
    raise ValueError(a.arch)


# ----------------------------------------------------------------------------------------------- data
def data_path(key, split):
    return os.path.join(HERE, 'data', f'{key}_{split}.npz')


def mkdata(a):
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    seed = {'val': 10_000, 'test': 20_000, 'test2': 30_000}[a.split]
    rng = np.random.default_rng([seed, a.N, int(a.rho * 1000), a.gsize])
    d = W.sample(a.n, a.N, rng, **wkw(a))
    t0 = time.time()
    B = W.bayes(d)
    extra = {}
    if a.world != 'indep':
        I = W.indep_bayes(d)
        extra = dict(nll_indep=I['nll'], mean_indep=I['mean'], var_indep=I['var'])
    if a.split != 'val':
        extra.update(grid=B['grid'].astype(np.float32), logp=B['logp'].astype(np.float32))
    np.savez_compressed(data_path(wkey(a), a.split), o=d['o'], y=d['y'], v=d['v'], vidx=d['vidx'], blk=d['blk'],
                        grp=d['grp'], nll_bayes=B['nll'], mean_bayes=B['mean'], var_bayes=B['var'], **extra)
    print(wkey(a), a.split, a.n, f'bayes nll {B["nll"].mean():.4f}', f'({time.time() - t0:.0f}s)', flush=True)


def load_data(a, split):
    z = np.load(data_path(wkey(a), split))
    d = dict(o=z['o'], y=z['y'], v=z['v'], vidx=z['vidx'], blk=z['blk'], grp=z['grp'], N=a.N, mode=a.world,
             rho=a.rho, gsize=a.gsize, a1=a.a1, a2=a.a2)
    return d, z


def predict(model, d, a, device, seed=777, bs=1024):
    rng = np.random.default_rng(seed)
    outs = []
    tf32 = torch.backends.cuda.matmul.allow_tf32                # evaluation always in full FP32
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model.eval()
    with torch.no_grad():
        for i in range(0, len(d['y']), bs):
            sl = {k: (v[i:i + bs] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
            inp = build_inputs(sl, a.arch, rng, device, want_group=a.gid)
            mu, ls, lg = model(**inp)
            outs.append((mu.cpu().numpy(), ls.cpu().numpy(), lg.cpu().numpy()))
    model.train()
    torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
    return tuple(np.concatenate([o[j] for o in outs]).astype(np.float64) for j in range(3))


def nll_np(mu, ls, lg, y):
    lw = lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)
    lp = lw - ls - 0.5 * math.log(2 * math.pi) - 0.5 * ((y[:, None] - mu) / np.exp(ls)) ** 2
    return -np.logaddexp.reduce(lp, axis=1)


# ----------------------------------------------------------------------------------------------- train
def train(a):
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    torch.set_num_threads(1)
    model = build_model(a, a.world).to(dev)
    if dev.type == "cuda":                          # hot patch 20:05Z: TF32 matmuls for training steps only
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if a.init:
        sd = torch.load(os.path.join(a.init, 'model.pt'), map_location=dev)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        print('init from', a.init, 'new params:', missing, flush=True)
        if a.arch == 'tok' and a.cut:                  # PCA init of the rank-k cut maps from the trained net's states
            r0 = np.random.default_rng(5)
            d0 = W.sample(512, a.N, r0, **wkw(a))
            inp = build_inputs(d0, a.arch, r0, dev, want_group=a.gid)
            with torch.no_grad():
                _, hs, _ = model(**inp, want=True)
            for l, b in enumerate(model.blocks):
                if b.cut_k:
                    b.init_cut_pca(hs[l], None)
    meta = dict(vars(a)); meta['params'] = M.n_params(model); meta['wkey'] = wkey(a)
    vd, vz = load_data(a, 'val')
    vy = vd['y'][:, W.QL]; vb = vz['nll_bayes']
    os.makedirs(a.out, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    rng = np.random.default_rng([a.seed, 4242, a.N])
    total = a.steps
    cool0 = total - a.cool

    def lr_at(s):
        if a.init and a.warm == 0:
            w = 1.0
        else:
            w = min(1.0, s / max(1, a.warm))
        return a.lr * w * (1.0 if s <= cool0 else max(0.0, (total - s) / a.cool))

    def val():
        mu, ls, lg = predict(model, vd, a, dev)
        g = nll_np(mu, ls, lg, vy) - vb
        return float(g.mean()), float(g.std() / math.sqrt(len(g)))

    curve, tl, best, t0 = [], [], 1e9, time.time()
    if a.init:
        g0 = val(); curve.append((0, g0[0], g0[1])); print('init val gap', g0, flush=True)
    evals = set(s for s in EVAL_STEPS if s <= total) | {total}
    run_loss = torch.zeros((), device=dev)
    if a.tdata:                                         # on-device sampler (same worlds; test_sfgpu.py)
        import sfgpu as SG
        tgen = torch.Generator(device=dev); tgen.manual_seed(1000003 * a.seed + 4242 + a.N)
    for step in range(1, total + 1):
        for g in opt.param_groups: g['lr'] = lr_at(step)
        if a.tdata:
            d = SG.sample_t(a.bs, a.N, tgen, dev, **wkw(a))
            inp = SG.build_inputs_t(d, a.arch, tgen, dev, want_group=a.gid)
            y = d['y'][:, W.QL]
        else:
            d = W.sample(a.bs, a.N, rng, **wkw(a))
            inp = build_inputs(d, a.arch, rng, dev, want_group=a.gid)
            y = torch.from_numpy(d['y'][:, W.QL]).float().to(dev)
        loss = M.head_nll(model(**inp), y).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        run_loss += loss.detach()
        if step % 100 == 0:
            tl.append((step, float(run_loss) / 100)); run_loss.zero_()
        if step in evals:
            g, se = val(); curve.append((step, g, se))
            if g < best:
                best = g; torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
            print(f'[{a.arch} {wkey(a)} s{a.seed}] step {step} val gap {g:+.4f} (se {se:.4f}) best {best:+.4f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    torch.save(model.state_dict(), os.path.join(a.out, 'model_final.pt'))
    meta.update(curve=curve, train_loss=tl, best_val_gap=best, final_val_gap=curve[-1][1], seconds=time.time() - t0,
                val_bayes=float(vb.mean()))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


def evaluate(a):
    meta = json.load(open(os.path.join(a.run, 'meta.json')))
    for k, v in meta.items():
        if k not in ('run', 'split', 'cmd', 'device', 'which'):
            setattr(a, k, v)
    dev = torch.device(a.device)
    model = build_model(a, a.world).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, a.which), map_location=dev))
    d, z = load_data(a, a.split)
    mu, ls, lg = predict(model, d, a, dev, seed=999)
    ys = d['y'][:, W.QL]
    gap = nll_np(mu, ls, lg, ys) - z['nll_bayes']
    B = dict(grid=z['grid'].astype(np.float64), logp=z['logp'].astype(np.float64))
    kl = W.kl_bayes_model(B, mu, ls, lg)
    w = np.exp(lg - np.logaddexp.reduce(lg, axis=1, keepdims=True))
    mean = (w * mu).sum(1)
    res = dict(gap=float(gap.mean()), gap_se=float(gap.std() / math.sqrt(len(gap))), kl=float(kl.mean()),
               kl_se=float(kl.std() / math.sqrt(len(kl))), mean_mse_vs_bayes=float(((mean - z['mean_bayes']) ** 2).mean()),
               bayes_var=float(z['var_bayes'].mean()), n=len(gap), which=a.which)
    if 'nll_indep' in z:
        res['indep_gap'] = float((z['nll_indep'] - z['nll_bayes']).mean())
    json.dump(res, open(os.path.join(a.run, f'eval_{a.split}_{a.which.replace(".pt", "")}.json'), 'w'))
    np.savez_compressed(os.path.join(a.run, f'pred_{a.split}.npz'), mu=mu, ls=ls, lg=lg, kl=kl, gap=gap)
    print(a.run, json.dumps(res), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--world', default='indep'); ap.add_argument('--N', type=int, default=3)
    ap.add_argument('--rho', type=float, default=0.0); ap.add_argument('--gsize', type=int, default=1)
    ap.add_argument('--a1', type=float, default=1.0); ap.add_argument('--a2', type=float, default=1.0)
    ap.add_argument('--split', default='val'); ap.add_argument('--n', type=int, default=4096)
    ap.add_argument('--arch', default='tree'); ap.add_argument('--m', type=int, default=8)
    ap.add_argument('--m1', type=int, default=16); ap.add_argument('--inner', default='sum')
    ap.add_argument('--h', type=int, default=64); ap.add_argument('--pool', default='sum')
    ap.add_argument('--d', type=int, default=32); ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--H', type=int, default=2); ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--nocross', action='store_true'); ap.add_argument('--gid', action='store_true')
    ap.add_argument('--attn', default=''); ap.add_argument('--cut', default=''); ap.add_argument('--cut_k', default='')
    ap.add_argument('--steps', type=int, default=10000); ap.add_argument('--cool', type=int, default=2500)
    ap.add_argument('--warm', type=int, default=250); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bs', type=int, default=1024); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--init', default='')
    ap.add_argument('--tdata', action='store_true')
    ap.add_argument('--out', default=''); ap.add_argument('--run', default=''); ap.add_argument('--which', default='model.pt')
    a = ap.parse_args()
    {'mkdata': mkdata, 'train': train, 'eval': evaluate}[a.cmd](a)
