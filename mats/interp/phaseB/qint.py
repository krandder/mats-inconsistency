"""E7.1 (PREREG-B amendment 14): interventions on the readout path for the systematic spread (log-sd) error.

Per net: a pool of 65,536 fresh fit episodes traced once (one noise draw) gives a2/r2 query states for the decoders (I1,
I1c) and the retrained readouts (I2 head on r2, I3 MLP on a2); then one branched evaluation over K = 64 noise draws on the
first 4,000 test episodes computes the baseline and I1, I1c, I2, I3, I4a, I4b, I6 (I5 = one noise-free pass), the
single-draw diagnostic I7 and the A/B decomposition of the baseline's log-sd deviation.
usage: python qint.py RUN_DIR [--smoke] -> RUN_DIR/qint.npz (per-episode deviations) + RUN_DIR/qint.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
import nnoise as N
import world as W
import devpred as DP
import qw
import qsys as QS

torch.set_num_threads(1)
SMOKE = '--smoke' in sys.argv
FEATS = ('mean', 'logsd', 'kl')


def mix_head(lin_out, K=4):
    return lin_out[:, K:2 * K], lin_out[:, 2 * K:].clamp(-6, 3), lin_out[:, :K]


class NewHead(nn.Module):
    """Same form as the network's head: LayerNorm + linear -> 4-component mixture."""
    def __init__(self, d):
        super().__init__(); self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, 12)
    def forward(self, x): return mix_head(self.out(self.ln(x)))


class MLPReadout(nn.Module):
    def __init__(self, d, h=64):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 12))
    def forward(self, x): return mix_head(self.net(x))


def nll_t(head, y):
    mu, ls, lg = head
    lp = torch.log_softmax(lg, -1) - ls - 0.5 * math.log(2 * math.pi) - 0.5 * ((y[:, None] - mu) / torch.exp(ls)) ** 2
    return -torch.logsumexp(lp, -1)


def train_readout(mod, Xs, ys, steps, seed=0):
    opt = torch.optim.Adam(mod.parameters(), lr=3e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed); n = len(Xs)
    for s in range(steps):
        idx = torch.randint(n, (256,), generator=g); loss = nll_t(mod(Xs[idx]), ys[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return mod


def forward_layers(model, X, allow, sigma, nz, upto_layer):
    """Residual after layer upto_layer - 1 (or emb) with noise nz (dict of (n,T,D) tensors)."""
    x = model.f_emb(X) + sigma * nz['emb']
    for l in range(1, upto_layer):
        xa = model.f_a(l, x, allow) + sigma * nz[f'a{l}']; m = model.f_m(l, xa) + sigma * nz[f'm{l}']
        x = model.f_r(l, xa, m) + sigma * nz[f'r{l}']
    return x


def run(run_dir):
    t0 = time.time(); model, meta = N.load_run(run_dir); model.requires_grad_(False); sg = meta['sigma']; L = meta['depth']; d = model.d
    n_pool, n_test, K, steps = (2048, 300, 8, 200) if SMOKE else (65536, 4000, 64, 6000)
    # ---- pool of fresh fit episodes (one noise draw each)
    rng = np.random.default_rng(97_000); dp = W.sample_np(n_pool, rng); Xp, vp = W.tokens_np(dp, rng, meta['ds'])
    _, _, momp, _ = W.references(dp); mbp, lsbp = momp['bayes'][0], 0.5 * np.log(momp['bayes'][1])
    g = torch.Generator().manual_seed(97_001); A2, R2 = [], []
    with torch.no_grad():
        for i in range(0, n_pool, 2048):
            Xi = torch.from_numpy(Xp[i:i + 2048]); al = model.allow_mask(torch.from_numpy(vp[i:i + 2048]))
            nz = {s: torch.randn((len(Xi), 16, (2 * d if s[0] == 'm' else d)), generator=g) for s in model.site_names()}
            x = forward_layers(model, Xi, al, sg, nz, L)
            xa = model.f_a(L, x, al) + sg * nz[f'a{L}']; m = model.f_m(L, xa) + sg * nz[f'm{L}']; r = model.f_r(L, xa, m) + sg * nz[f'r{L}']
            A2.append(xa[:, -1]); R2.append(r[:, -1])
    A2, R2 = torch.cat(A2), torch.cat(R2); ys = torch.as_tensor(dp['y'][:, W.QL], dtype=torch.float32)
    # decoders (I1: ridge; I1c: reverse-regression calibrated: fit the ridge direction, then rescale so E[decode | target] = target)
    A2n = A2.numpy().astype(np.float64)
    dec = {}
    for nm, tgt in (('mean', mbp), ('logsd', lsbp)):
        rg = Ridge(alpha=1.0).fit(A2n, tgt); p = rg.predict(A2n)
        slope = float(np.cov(p, tgt)[0, 1] / np.var(tgt)); icpt = float(p.mean() - slope * tgt.mean())   # p ~ icpt + slope * tgt
        dec[nm] = (rg, slope, icpt, float(1 - ((tgt - p) ** 2).mean() / np.var(tgt)))
    # retrained readouts
    torch.manual_seed(0); head2 = train_readout(NewHead(d), R2, ys, steps, 1)
    torch.manual_seed(0); mlp3 = train_readout(MLPReadout(d), A2, ys, steps, 2)
    # mean MLP2 write over the pool (for I4a)
    b = model.blocks[L - 1]
    with torch.no_grad():
        mm = N.sphere(b.hidden_pre(A2[:4096]))
        mean_write = b.mlp[2](mm).mean(0)
    print('pool + decoders + readouts %.0fs' % (time.time() - t0), {k: round(v[3], 3) for k, v in dec.items()}, flush=True)
    # ---- branched evaluation over K draws on the test episodes
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz'))
    Tt = {k: Tz[k][:n_test] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    Xt, vt = torch.from_numpy(Tz['X'][:n_test]), torch.from_numpy(Tz['valid'][:n_test]); al = model.allow_mask(vt)
    names = ['base', 'I1', 'I1c', 'I2', 'I3', 'I4a', 'I4b', 'I6']
    heads = {nm: ([], [], []) for nm in names}
    single = []                                                      # per-draw (mean, var) of the baseline single-draw predictive
    gk = torch.Generator().manual_seed(1)
    def add(nm, h):
        mu, ls, lg = h; heads[nm][0].append(mu); heads[nm][1].append(ls); heads[nm][2].append(torch.log_softmax(lg, -1) - math.log(K))
    with torch.no_grad():
        for k in range(K):
            nz = {s: torch.randn((n_test, 16, (2 * d if s[0] == 'm' else d)), generator=gk) for s in model.site_names()}
            x = forward_layers(model, Xt, al, sg, nz, L)
            pa = model.f_a(L, x, al); xa = pa + sg * nz[f'a{L}']
            m = model.f_m(L, xa) + sg * nz[f'm{L}']; r = model.f_r(L, xa, m) + sg * nz[f'r{L}']
            hb = model.head(r); add('base', hb)
            mu, ls, lg = hb; w = torch.softmax(lg, -1); m1 = (w * mu).sum(1); v1 = (w * (torch.exp(2 * ls) + mu ** 2)).sum(1) - m1 ** 2
            single.append(torch.stack([m1, v1], 1))
            aq = xa[:, -1].numpy().astype(np.float64)
            for nm, cal in (('I1', False), ('I1c', True)):
                mhat = dec['mean'][0].predict(aq); lhat = dec['logsd'][0].predict(aq)
                if cal:
                    mhat = (mhat - dec['mean'][2]) / dec['mean'][1]; lhat = (lhat - dec['logsd'][2]) / dec['logsd'][1]
                mu_ = torch.as_tensor(mhat, dtype=torch.float32)[:, None].repeat(1, 4); ls_ = torch.as_tensor(lhat, dtype=torch.float32)[:, None].repeat(1, 4)
                add(nm, (mu_, ls_, torch.zeros(n_test, 4)))
            add('I2', head2(r[:, -1])); add('I3', mlp3(xa[:, -1]))
            r4a = N.sphere(xa + mean_write) + sg * nz[f'r{L}']; add('I4a', model.head(r4a))
            add('I4b', model.head(xa))
            m6 = model.f_m(L, pa); r6 = model.f_r(L, pa, m6); add('I6', model.head(r6))
        # I5: noise off everywhere, one deterministic pass
        z0 = {s: torch.zeros((n_test, 16, (2 * d if s[0] == 'm' else d))) for s in model.site_names()}
        x5 = forward_layers(model, Xt, al, 0.0, z0, L + 1); h5 = model.head(x5)
    devs = {}
    for nm in names:
        mu, ls, lg = [torch.cat(t, 1).numpy() for t in heads[nm]]; devs[nm] = qw.deviations_t((mu, ls, lg), Tt)
    devs['I5'] = qw.deviations_t(tuple(t.numpy() for t in h5), Tt)
    S = torch.stack(single, 0).numpy().astype(np.float64)              # (K, n, 2)
    lsb = 0.5 * np.log(Tt['var_bayes'])
    Ev = S[..., 1].mean(0); Vm = S[..., 0].var(0)
    A = 0.5 * np.log(Ev) - lsb; Bj = 0.5 * np.log(1 + Vm / Ev)
    devs['I7'] = dict(mean=(S[..., 0].mean(0) - Tt['mean_bayes']), logsd=(0.5 * np.log(S[..., 1]) - lsb).mean(0), kl=np.zeros(n_test))
    np.savez(os.path.join(run_dir, 'qint_smoke.npz' if SMOKE else 'qint.npz'), **{f'{nm}_{k}': devs[nm][k] for nm in devs for k in FEATS}, A=A, B=Bj)
    out = dict(decoder_R2={k: v[3] for k, v in dec.items()}, decoder_slope={k: v[1] for k, v in dec.items()}, n_test=n_test, K=K,
               nll_heads=None, seconds=time.time() - t0)
    json.dump(out, open(os.path.join(run_dir, 'qint_smoke.json' if SMOKE else 'qint.json'), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
