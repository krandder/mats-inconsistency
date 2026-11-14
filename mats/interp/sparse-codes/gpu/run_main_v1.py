"""Main per-net driver of the sparse-codes lane: E1 dictionaries in bits, E2 one-step higher-order structure carried by
latent-conditional Gaussians (sparse SAE latent vs dense PCA latent), E3 representation at a cut pushed through the
real remaining layers (ARC nets). See PREREG.md for the definitions; this file is the implementation.

usage: run_main.py OUT FAMILY N SIGMA SEED [--layers 1,2,...] [--fmults 1,2,4,8,16] [--epochs 20] [--e3 4,8,16,24]
                   [--ntrain 18] [--ne2 18] [--n3 20] [--ntruth 24] [--save-sae 4,8,16,24]
Outputs: OUT/<net>/e1_L<l>.json, e2_L<l>.json, e3.json, truth.npz, sae_L<l>_F<F>.pt (subset), log.txt
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import nets, sae   # noqa: E402

LN2 = math.log(2.0)
p = argparse.ArgumentParser()
p.add_argument('out'); p.add_argument('family'); p.add_argument('n', type=int); p.add_argument('sigma', type=float)
p.add_argument('seed', type=int)
p.add_argument('--layers', default=None)
p.add_argument('--fmults', default='1,2,4,8,16')
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--batch', type=int, default=2048)
p.add_argument('--ntrain', type=int, default=18)
p.add_argument('--ntest', type=int, default=15)
p.add_argument('--ne2', type=int, default=18)
p.add_argument('--n3', type=int, default=20)
p.add_argument('--ntruth', type=int, default=24)
p.add_argument('--e3', default='')
p.add_argument('--save-sae', default='')
p.add_argument('--skip-e1-train', action='store_true')
p.add_argument('--long-control', default='')       # layers at which F=4n is also trained with 2x epochs
args = p.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu':
    torch.set_num_threads(int(os.environ.get('NT', '4')))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
net = nets.SNet(args.family, args.n, args.seed, args.sigma, device=dev, dtype=torch.float32)
n = args.n
DEF_LAYERS = {'arc': [1, 2, 4, 8, 16, 24, 31], 'tord': [1, 2, 4, 8, 31], 'tcha': [1, 8, 31]}
layers = [int(x) for x in args.layers.split(',')] if args.layers else DEF_LAYERS[args.family]
fmults = [int(x) for x in args.fmults.split(',')]
KS = [n // 16, n // 8, n // 4, n // 2]
KD = sorted(set([1, 2, 4, 8, 16, 32, 64, n // 4, n // 2]) & set(range(1, n)))
e3_cuts = [int(x) for x in args.e3.split(',')] if args.e3 else []
save_layers = set(int(x) for x in args.save_sae.split(',')) if args.save_sae else set()
long_layers = set(int(x) for x in args.long_control.split(',')) if args.long_control else set()
OUT = os.path.join(args.out, net.name()); os.makedirs(OUT, exist_ok=True)
LOGF = open(os.path.join(OUT, 'log.txt'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()


def dump(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), 'w'), indent=1)


log(f"== {net.name()} dev={dev} layers={layers} fmults={fmults} KS={KS} KD={KD} e3={e3_cuts} "
    f"s={[round(net.noise_sd(l), 6) for l in layers]}")
T0 = time.time()


# ---------------------------------------------------------------------------------------------------------------- utils
class Moments:
    """Per-column power sums around a fixed shift, float64: mean, var, third and fourth cumulants."""
    def __init__(self):
        self.n = 0; self.S = None; self.shift = None

    def add(self, z):
        z = z.double()
        if self.shift is None:
            self.shift = z.mean(0)
            self.S = [torch.zeros_like(self.shift) for _ in range(4)]
        c = z - self.shift
        c2 = c * c
        self.S[0] += c.sum(0); self.S[1] += c2.sum(0); self.S[2] += (c2 * c).sum(0); self.S[3] += (c2 * c2).sum(0)
        self.n += z.shape[0]

    def get(self):
        N = self.n
        m1, m2, m3, m4 = (s / N for s in self.S)
        mean = self.shift + m1
        var = m2 - m1 ** 2
        k3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
        mu4 = m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4
        k4 = mu4 - 3 * var ** 2
        return mean, var, k3, k4


def gh(nq=96):
    x, w = np.polynomial.hermite_e.hermegauss(nq)
    return torch.tensor(x, device=dev, dtype=torch.float64), torch.tensor(w / w.sum(), device=dev, dtype=torch.float64)


GHX, GHW = gh(96)


def npdf(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def ncdf(x):
    return 0.5 * torch.erfc(-x / math.sqrt(2))


def readout(mean, var, g3=None, g4=None, order=0):
    """E[act(z)] per neuron for z with the given mean/var and (Edgeworth) standardized cumulants g3, g4.
    order 0: Gaussian; 3: + g3 He3 term; 4: + g4 He4 + g3^2 He6 terms. ReLU closed form; tanh by Gauss-Hermite."""
    sd = var.clamp_min(1e-300).sqrt()
    if net.kind == 'relu':
        m = mean / sd
        val = m * ncdf(m) + npdf(m)
        if order >= 3:
            val = val + (g3 / 6) * (-m) * npdf(m)
        if order >= 4:
            val = val + (g4 / 24) * (m * m - 1) * npdf(m) + (g3 * g3 / 72) * (m ** 4 - 6 * m * m + 3) * npdf(m)
        return sd * val
    x = GHX[None, :]
    wts = GHW[None, :].expand(mean.shape[0], -1)
    if order >= 3:
        he3 = x ** 3 - 3 * x
        wts = wts * (1 + (g3[:, None] / 6) * he3) if order == 3 else wts
    if order >= 4:
        he3 = x ** 3 - 3 * x; he4 = x ** 4 - 6 * x ** 2 + 3; he6 = x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15
        wts = GHW[None, :] * (1 + (g3[:, None] / 6) * he3 + (g4[:, None] / 24) * he4 + (g3[:, None] ** 2 / 72) * he6)
    return (torch.tanh(mean[:, None] + sd[:, None] * x) * wts).sum(1)


def pre_act(l1, a, eps):
    """z_{l1} = W_{l1} (a + s_{l1-1} eps) + b_{l1}."""
    return net.pre(l1, a + net.noise_sd(l1 - 1) * eps)


def elem_act(z):
    return torch.relu(z) if net.kind == 'relu' else torch.tanh(z)


def nmse(est, truth_mean, truth_var):
    return float(((est - truth_mean) ** 2).mean() / truth_var.mean())


def kurt_excess(x):
    """Per-column excess kurtosis (float64)."""
    x = x.double(); c = x - x.mean(0)
    v = (c * c).mean(0)
    return (c ** 4).mean(0) / v.clamp_min(1e-300) ** 2 - 3.0, v


def quad_feats(y):
    k = y.shape[1]
    iu = torch.triu_indices(k, k, device=y.device)
    return torch.cat([y, (y[:, :, None] * y[:, None, :])[:, iu[0], iu[1]]], 1)


def DQ_fit(A, mu, Uk):
    """Residual of the top-k PCA latent regressed on [y, y_i y_j]; returns the model pieces (float64)."""
    c = A - mu; y = c @ Uk; r = c - y @ Uk.T
    psi = quad_feats(y); mp = psi.mean(0); P = psi - mp
    Cpp = P.T @ P / A.shape[0]; Crp = r.T @ P / A.shape[0]
    lp, Vp = torch.linalg.eigh(Cpp); keep = lp > lp.max() * 1e-12
    B = Crp @ ((Vp[:, keep] / lp[keep][None]) @ Vp[:, keep].T)
    e = r - P @ B.T
    Ce = e.T @ e / A.shape[0]; Ce = 0.5 * (Ce + Ce.T)
    return dict(mu=mu, Uk=Uk, mp=mp, B=B, Ce=Ce, Le=sym_sqrt_psd(Ce))


def DQ_apply(A, q, g):
    c = A - q['mu']; y = c @ q['Uk']
    psi = quad_feats(y) - q['mp']
    xi = torch.randn(A.shape[0], A.shape[1], generator=g, device=A.device, dtype=A.dtype)
    return q['mu'] + y @ q['Uk'].T + psi @ q['B'].T + xi @ q['Le'].T.to(A.dtype)


def gen(seed):
    return torch.Generator(device=dev).manual_seed(seed)


def sym_sqrt_psd(C, floor=0.0):
    lam, U = torch.linalg.eigh(C)
    lam = lam.clamp_min(floor)
    return U * lam.sqrt()[None, :]          # C ~ (U sqrt(lam)) (U sqrt(lam))^T


# ---------------------------------------------------------------------------------------------------------------- data
NTR, NTE, NE2 = 1 << args.ntrain, 1 << args.ntest, 1 << args.ne2
data_seed = 1_000 * args.seed + 17
t = time.time()
Xall = net.sample(NTR + NTE, gen(data_seed), layers)
XE2 = net.sample(NE2, gen(data_seed + 1), layers)
log(f"sampled train/test {NTR}+{NTE} and E2 {NE2} in {time.time() - t:.0f}s")

E2_CACHE = {}
for l in layers:
    s = net.noise_sd(l)
    Xtr, Xte = Xall[l][:NTR], Xall[l][NTR:]
    A2 = XE2[l]
    # ------------------------------------------------------------------------------------------------ E1 baselines
    mu = A2.double().mean(0); C = A2.double() - mu
    Sig = C.T @ C / NE2
    lam, U = torch.linalg.eigh(Sig); lam = lam.flip(0).clamp_min(0); U = U.flip(1)
    tail = torch.cat([lam.flip(0).cumsum(0).flip(0), torch.zeros(1, device=dev, dtype=torch.float64)])
    s2 = s * s
    e1 = dict(net=net.name(), layer=l, s=s, n=n, r2=float((A2.double() ** 2).mean()), var_per_coord=float(lam.sum() / n),
              PR=float(lam.sum() ** 2 / (lam ** 2).sum()), eig_above_noise=int((lam > s2).sum()),
              eig=lam.tolist())
    e1['pca'] = {int(k): dict(noise_units=float(tail[k] / (n * s2)), bits=float(tail[k] / (2 * s2) / LN2))
                 for k in sorted(set([0] + KS + KD + [n]))}
    e1['k_pca_below_noise'] = int(next(k for k in range(n + 1) if float(tail[k]) < n * s2))
    e1['k_pca_1bit'] = int(next(k for k in range(n + 1) if float(tail[k]) / (2 * s2) / LN2 < 1.0))
    # neuron basis k-sparse code around the mean (held-out): keep the k largest |a_i - mu_i|
    dev_ = (Xte.double() - mu).abs()
    srt = dev_.sort(dim=1, descending=True).values ** 2
    cum = srt.cumsum(1); tot = srt.sum(1)
    e1['neuron_topk'] = {int(k): dict(noise_units=float((tot - cum[:, k - 1]).mean() / (n * s2)),
                                      bits=float((tot - cum[:, k - 1]).mean() / (2 * s2) / LN2)) for k in KS + [n]}
    if net.kind == 'relu':
        srt0 = Xte.double().sort(dim=1, descending=True).values ** 2
        cum0 = srt0.cumsum(1); tot0 = srt0.sum(1)
        e1['neuron_topk_zero'] = {int(k): dict(noise_units=float((tot0 - cum0[:, k - 1]).mean() / (n * s2)),
                                               bits=float((tot0 - cum0[:, k - 1]).mean() / (2 * s2) / LN2))
                                  for k in KS + [n]}
        act = (Xte > 0).double().sum(1)
        e1['active'] = dict(mean=float(act.mean()), sd=float(act.std()), max=int(act.max()),
                            ever_active=int((Xte > 0).any(0).sum()))
        # zero-based neuron code: the exact active set (L0 = #active, error 0); recorded for reference
    # ------------------------------------------------------------------------------------------------ E1 SAEs
    e1['sae'] = []
    saes = {}
    for fm in fmults:
        F = fm * n
        t = time.time()
        model, c, hist = sae.train_batched(Xtr, F, KS, epochs=args.epochs, batch=args.batch, device=dev,
                                           seed=args.seed * 100 + l * 7 + fm)
        res = sae.evaluate(model, c, Xte, s, chunk=2048)
        for r in res:
            r['train_secs'] = time.time() - t
            e1['sae'].append(r)
        log(f"L{l} F={F} ({time.time() - t:.0f}s): " + ' | '.join(
            f"k{r['k']} nu={r['noise_units']:.3g} rf={r['noise_units_refit']:.3g} dead={r['dead_frac']:.2f}" for r in res))
        saes[fm] = (model, c)
        if l in save_layers and fm in (1, 4, 16):
            torch.save(dict(state={k: v.half().cpu() for k, v in model.state_dict().items()}, c=c, ks=KS, F=F, n=n),
                       os.path.join(OUT, f'sae_L{l}_F{F}.pt'))
        if l in long_layers and fm == 4:
            t = time.time()
            m2_, c2_, _ = sae.train_batched(Xtr, F, KS, epochs=2 * args.epochs, batch=args.batch, device=dev,
                                            seed=args.seed * 100 + l * 7 + fm + 1)
            res2 = sae.evaluate(m2_, c2_, Xte, s, chunk=2048)
            e1['long_control'] = res2
            log(f"L{l} F={F} 2x epochs ({time.time() - t:.0f}s): " + ' | '.join(
                f"k{r['k']} nu={r['noise_units']:.3g} rf={r['noise_units_refit']:.3g}" for r in res2))
            del m2_
    dump(f'e1_L{l}.json', e1)

    # ------------------------------------------------------------------------------------------------ E2 (l < L)
    if l >= net.L:
        del Xall[l], XE2[l]
        continue
    l1 = l + 1
    eps = torch.randn(A2.shape, generator=gen(data_seed + 1000 + l), device=dev, dtype=A2.dtype)
    eps2 = torch.randn(A2.shape, generator=gen(data_seed + 2000 + l), device=dev, dtype=A2.dtype)
    CH = min(1 << 15, NE2)

    def stats_of(states_fn, noise):
        """states_fn(i0, i1) -> model states for rows i0:i1; returns per-neuron cumulants of z' and mean/var of act(z')."""
        Mz, Ma = Moments(), Moments()
        for i0 in range(0, NE2, CH):
            a = states_fn(i0, min(NE2, i0 + CH))
            z = pre_act(l1, a, noise[i0:i0 + a.shape[0]])
            Mz.add(z); Ma.add(elem_act(z))
        mz, vz, k3, k4 = Mz.get(); ma, va, _, _ = Ma.get()
        return dict(mz=mz, vz=vz, k3=k3, k4=k4, ma=ma, va=va)

    T = stats_of(lambda i0, i1: A2[i0:i1], eps)
    # independent-truth control (fresh true states and noise): the metrics' sampling floor for independent samples
    A2b = net.sample(NE2, gen(data_seed + 3000 + l), [l])[l]
    T2 = stats_of(lambda i0, i1: A2b[i0:i1], eps2)
    del A2b

    def metrics(Mst, name, extra=None):
        d = dict(model=name)
        d['cap3'] = float(1 - ((Mst['k3'] - T['k3']) ** 2).sum() / (T['k3'] ** 2).sum())
        d['cap4'] = float(1 - ((Mst['k4'] - T['k4']) ** 2).sum() / (T['k4'] ** 2).sum())
        d['mean_nmse'] = nmse(Mst['ma'], T['ma'], T['va'])
        d['var_relerr_med'] = float(((Mst['va'] / T['va']) - 1).abs().median())
        d['zvar_relerr_med'] = float(((Mst['vz'] / T['vz']) - 1).abs().median())
        if extra:
            d.update(extra)
        return d

    rows = []
    rows.append(metrics(T2, 'TRUE_indep'))
    # Gaussian closure (K=2 one step) and Edgeworth readouts with the true per-neuron cumulants (oracle per-neuron)
    g3 = T['k3'] / T['vz'] ** 1.5; g4 = T['k4'] / T['vz'] ** 2
    for name, order in (('G', 0), ('EW3_oracle', 3), ('EW4_oracle', 4)):
        ma = readout(T['mz'], T['vz'], g3, g4, order)
        rows.append(dict(model=name, cap3=0.0 if order < 3 else 1.0, cap4=0.0 if order < 4 else 1.0,
                         mean_nmse=nmse(ma, T['ma'], T['va'])))
    log(f"E2 L{l}: z' median |g3|={float(g3.abs().median()):.3g} median g4={float(g4.median()):.3g}; "
        + ' '.join(f"{r['model']}:{r['mean_nmse']:.2e}" for r in rows))
    # dense latent: top-k PCA coordinates exact, the rest independent Gaussian (full covariance matched)
    A2f = A2.double()
    for k in KD:
        Uk, Ur = U[:, :k], U[:, k:]
        sr = lam[k:].sqrt()
        gk = gen(data_seed + 4000 + l * 100 + k)

        def st_dense(i0, i1, Uk=Uk, Ur=Ur, sr=sr, gk=gk):
            c = A2f[i0:i1] - mu
            y = c @ Uk
            xi = torch.randn(i1 - i0, Ur.shape[1], generator=gk, device=dev, dtype=torch.float64)
            return (mu + y @ Uk.T + (xi * sr[None]) @ Ur.T).float()
        Ms = stats_of(st_dense, eps)
        rows.append(metrics(Ms, f'DENSE{k}', dict(k=k, latent_dim=k, params_map=n * k)))
        if k <= 16:
            # DENSEQ: residual mean quadratic in the latent (regressors [y, y_i y_j]), Gaussian residual (Schur
            # complement covariance): a latent-conditional Gaussian with a nonlinear conditional mean
            qa = DQ_fit(A2f, mu, Uk)
            gq = gen(data_seed + 4500 + l * 100 + k)

            def st_denseq(i0, i1, qa=qa, gq=gq):
                return DQ_apply(A2f[i0:i1], qa, gq).float()
            Ms = stats_of(st_denseq, eps)
            rows.append(metrics(Ms, f'DENSEQ{k}', dict(k=k, latent_dim=k, params_map=n * k + n * k * (k + 3) // 2,
                                                         resid_noise_units=float(torch.trace(qa['Ce']) / (n * s2)))))
    # sparse latent (SAE code exact) + linear-Gaussian residual given the reconstruction (full covariance matched)
    for fm, (model, c) in saes.items():
        F = fm * n
        with torch.no_grad():
            recs = []
            for i0 in range(0, NE2, 4096):
                f, _, _, _ = model.encode(A2[i0:i0 + 4096] / c)
                recs.append((model.decode(f) * c).double())      # (S, B, n)
            Rec = torch.cat(recs, 1)                              # (S, NE2, n)
            del recs
        for si, k in enumerate(KS):
            ah = Rec[si]
            r = A2f - ah
            mh = ah.mean(0); mr = r.mean(0)
            Ch = (ah - mh).T @ (ah - mh) / NE2; Cr = (r - mr).T @ (r - mr) / NE2; Crh = (r - mr).T @ (ah - mh) / NE2
            lh, Vh = torch.linalg.eigh(Ch)
            keep = lh > lh.max() * 1e-10
            Chp = (Vh[:, keep] / lh[keep][None]) @ Vh[:, keep].T     # pseudo-inverse
            B = Crh @ Chp
            Ce = Cr - B @ Crh.T
            Ce = 0.5 * (Ce + Ce.T)
            Le = sym_sqrt_psd(Ce)
            gk = gen(data_seed + 5000 + l * 100 + fm * 10 + si)

            def st_sparse(i0, i1, ah=ah, mh=mh, mr=mr, B=B, Le=Le, gk=gk):
                xi = torch.randn(i1 - i0, n, generator=gk, device=dev, dtype=torch.float64)
                return (ah[i0:i1] + mr + (ah[i0:i1] - mh) @ B.T + xi @ Le.T).float()
            Ms = stats_of(st_sparse, eps)
            # residual (given the features) Gaussianity
            e_true = r - mr - (ah - mh) @ B.T
            ke, ve = kurt_excess(e_true)
            sel = ve > ve.max() * 1e-6
            kp, _ = kurt_excess(e_true.float() @ net.W[l1 - 1].T)
            # conditional variance by regression of e^2 on the (sparse) code, fitted on half, scored on the other half
            ext = dict(k=k, F=F, latent_dim=k, params_map=n * F, resid_noise_units=float(torch.trace(Ce) / (n * s2)),
                       resid_kurt_med=float(ke[sel].median()), resid_kurt_p90=float(ke[sel].quantile(0.9)),
                       resid_proj_kurt_med=float(kp.median()), resid_proj_kurt_p90=float(kp.quantile(0.9)))
            try:
                with torch.no_grad():
                    NR = min(NE2, 1 << 16); h = NR // 2               # 2^16 rows, the 512 most active features
                    fs = []
                    for i0 in range(0, NR, 4096):
                        f, _, _, _ = model.encode(A2[i0:i0 + 4096] / c)
                        fs.append(f[si])
                    Fc = torch.cat(fs)                                # (NR, F) sparse code
                    del fs
                    top = (Fc[:NR] > 0).float().mean(0).topk(min(512, F)).indices
                    Fs = Fc[:NR, top]
                    Phi = torch.cat([torch.ones(NR, 1, device=dev), Fs, (Fs > 0).float()], 1)
                    del Fc
                    et = e_true[:NR].float()
                    for tag, E in (('neuron', et), ('proj', et @ net.W[l1 - 1].T)):
                        Y = E * E
                        G_ = Phi[:h].T @ Phi[:h]
                        G_ += 1e-3 * torch.diag(G_).mean() * torch.eye(G_.shape[0], device=dev)
                        beta = torch.linalg.solve(G_.double(), (Phi[:h].T @ Y[:h]).double()).float()
                        v = (Phi[h:] @ beta).clamp_min(1e-3 * Y[:h].mean(0, keepdim=True))
                        kc, _ = kurt_excess(E[h:] / v.sqrt())
                        ext[f'resid_{tag}_condkurt_med'] = float(kc.median())
                        ext[f'resid_{tag}_condkurt_p90'] = float(kc.quantile(0.9))
                    del Phi, Fs
            except RuntimeError as ex:                                 # memory: skip the conditional check
                ext['condkurt_error'] = str(ex)[:120]
            rows.append(metrics(Ms, f'SPARSE_F{F}_k{k}', ext))
        del Rec
    # dense residual kurtosis for reference (k in KS)
    for k in KS:
        Uk = U[:, :k]
        rres = (A2f - mu) - ((A2f - mu) @ Uk) @ Uk.T
        ke, ve = kurt_excess(rres); sel = ve > ve.max() * 1e-6
        kp, _ = kurt_excess(rres.float() @ net.W[l1 - 1].T)
        NR = min(NE2, 1 << 16); h = NR // 2
        y = ((A2f[:NR] - mu) @ Uk).float()
        Phi = torch.cat([torch.ones(NR, 1, device=dev), quad_feats(y) if k <= 16 else torch.cat([y, y * y], 1)], 1)
        ext = {}
        rr_ = rres[:NR].float()
        for tag, E in (('neuron', rr_), ('proj', rr_ @ net.W[l1 - 1].T)):
            Y = E * E
            G_ = Phi[:h].T @ Phi[:h]; G_ += 1e-6 * torch.diag(G_).mean() * torch.eye(G_.shape[0], device=dev)
            beta = torch.linalg.solve(G_.double(), (Phi[:h].T @ Y[:h]).double()).float()
            v = (Phi[h:] @ beta).clamp_min(1e-3 * Y[:h].mean(0, keepdim=True))
            kc, _ = kurt_excess(E[h:] / v.sqrt())
            ext[f'{tag}_condkurt_med'] = float(kc.median()); ext[f'{tag}_condkurt_p90'] = float(kc.quantile(0.9))
        for rr in rows:
            if rr['model'] == f'DENSE{k}':
                rr.update(dict(resid_kurt_med=float(ke[sel].median()), resid_kurt_p90=float(ke[sel].quantile(0.9)),
                               resid_proj_kurt_med=float(kp.median()), resid_proj_kurt_p90=float(kp.quantile(0.9)),
                               **{f'resid_{a}': b for a, b in ext.items()}))
    del A2f
    dump(f'e2_L{l}.json', dict(net=net.name(), layer=l, s=s, rows=rows,
                               z_g3_med=float(g3.abs().median()), z_g4_med=float(g4.median()),
                               k3_T=T['k3'].tolist(), k4_T=T['k4'].tolist()))
    log(f"E2 L{l} done ({time.time() - T0:.0f}s): " + ' '.join(
        f"{r['model']}:c3={r['cap3']:.2f},c4={r['cap4']:.2f},m={r['mean_nmse']:.1e}" for r in rows))
    if l in e3_cuts:
        E2_CACHE[l] = dict(mu=mu.float(), U=U.float(), lam=lam.float(), saes=saes, models={},
                           dq={k: DQ_fit(A2.double(), mu, U[:, :k]) for k in KD if k <= 16})
        # residual-model parameters for the sparse latents at this cut (refit on the E2 set, stored for E3)
        for fm, (model, c) in saes.items():
            with torch.no_grad():
                recs = []
                for i0 in range(0, NE2, 4096):
                    f, _, _, _ = model.encode(A2[i0:i0 + 4096] / c)
                    recs.append((model.decode(f) * c).double())
                Rec = torch.cat(recs, 1)
            for si, k in enumerate(KS):
                ah = Rec[si]; r = A2.double() - ah
                mh = ah.mean(0); mr = r.mean(0)
                Ch = (ah - mh).T @ (ah - mh) / NE2; Cr = (r - mr).T @ (r - mr) / NE2; Crh = (r - mr).T @ (ah - mh) / NE2
                lh, Vh = torch.linalg.eigh(Ch); keep = lh > lh.max() * 1e-10
                B = Crh @ ((Vh[:, keep] / lh[keep][None]) @ Vh[:, keep].T)
                Ce = Cr - B @ Crh.T; Ce = 0.5 * (Ce + Ce.T)
                E2_CACHE[l]['models'][(fm, si)] = dict(mh=mh.float(), mr=mr.float(), B=B.float(), Le=sym_sqrt_psd(Ce).float())
            del Rec
    else:
        del saes
    del Xall[l], XE2[l]
    torch.cuda.empty_cache() if dev == 'cuda' else None

# -------------------------------------------------------------------------------------------------------------- E3
if e3_cuts and net.kind == 'relu':
    NT = 1 << args.ntruth
    t = time.time()
    S1 = torch.zeros(net.L, n, dtype=torch.float64, device=dev); S2 = torch.zeros_like(S1)
    g = gen(data_seed + 7)
    CHT = min(1 << 16, NT)
    for i0 in range(0, NT, CHT):
        h = torch.randn(CHT, n, generator=g, device=dev)
        for l in range(1, net.L + 1):
            a = net.act(l, net.pre(l, h))
            ad = a.double(); S1[l - 1] += ad.sum(0); S2[l - 1] += (ad * ad).sum(0)
            h = a + net.noise_sd(l) * torch.randn(a.shape, generator=g, device=dev)
    tmean = S1 / NT; tvar = S2 / NT - tmean ** 2
    np.savez(os.path.join(OUT, 'truth.npz'), S1=S1.cpu().numpy(), S2=S2.cpu().numpy(), N=NT)
    log(f"truth {NT} paths in {time.time() - t:.0f}s")
    OUTS = {4: [8], 8: [16, 32], 16: [32], 24: [32]}
    N3 = 1 << args.n3
    CH3 = min(1 << 16, N3)
    e3 = []
    for m in e3_cuts:
        cache = E2_CACHE[m]
        outs = [L for L in OUTS.get(m, [32]) if L > m]
        models = [('TRUE', None), ('G', None)] + [(f'DENSE{k}', k) for k in KD] + \
                 [(f'DENSEQ{k}', k) for k in KD if k <= 16] + \
                 [(f'SPARSE_F{fm * n}_k{KS[si]}', (fm, si)) for fm in cache['saes'] for si in range(len(KS))]
        acc = {name: {L: torch.zeros(n, dtype=torch.float64, device=dev) for L in outs} for name, _ in models}
        gm = gen(data_seed + 9000 + m)
        Lg = sym_sqrt_psd((cache['U'] * cache['lam'][None]) @ cache['U'].T)
        t = time.time()
        for i0 in range(0, N3, CH3):
            base = net.sample(CH3, gm, [m])[m]                    # fresh true states at the cut (shared: CRN)
            AH = {}
            with torch.no_grad():
                for fm, (model, c) in cache['saes'].items():
                    AH[fm] = torch.cat([model.decode(model.encode(base[j:j + 4096] / c)[0]) * c
                                        for j in range(0, CH3, 4096)], 1)       # (S, CH3, n)
            for name, spec in models:
                if name == 'TRUE':
                    a = base
                elif name == 'G':
                    a = cache['mu'] + torch.randn(CH3, n, generator=gm, device=dev) @ Lg.T
                elif name.startswith('DENSEQ'):
                    a = DQ_apply(base.double(), cache['dq'][spec], gm).float()
                elif name.startswith('DENSE'):
                    k = spec; Uk, Ur = cache['U'][:, :k], cache['U'][:, k:]
                    y = (base - cache['mu']) @ Uk
                    xi = torch.randn(CH3, n - k, generator=gm, device=dev) * cache['lam'][k:].sqrt()[None]
                    a = cache['mu'] + y @ Uk.T + xi @ Ur.T
                else:
                    fm, si = spec
                    pm = cache['models'][(fm, si)]
                    ah = AH[fm][si]
                    a = ah + pm['mr'] + (ah - pm['mh']) @ pm['B'].T + torch.randn(CH3, n, generator=gm, device=dev) @ pm['Le'].T
                x = a
                for j in range(m, max(outs)):
                    hj = x + net.noise_sd(j) * torch.randn(x.shape, generator=gm, device=dev)
                    x = net.act(j + 1, net.pre(j + 1, hj))
                    if (j + 1) in acc[name]:
                        acc[name][j + 1] += x.double().sum(0)
        for name, _ in models:
            for L in outs:
                est = acc[name][L] / N3
                err = float(((est - tmean[L - 1]) ** 2).mean() / tvar[L - 1].mean()) - 1.0 / N3 - 1.0 / NT
                e3.append(dict(cut=m, L=L, model=name, nmse=err))
        log(f"E3 cut {m} ({time.time() - t:.0f}s): " + ' '.join(f"{d['model']}@{d['L']}:{d['nmse']:.2e}" for d in e3 if d['cut'] == m))
    dump('e3.json', dict(net=net.name(), N3=N3, NT=NT, rows=e3))
log(f"ALL DONE {time.time() - T0:.0f}s")
