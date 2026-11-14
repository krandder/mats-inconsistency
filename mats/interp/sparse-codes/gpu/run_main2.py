"""v2 of the main driver (after the GPT-6 design critique, PREREG amendment 2). Changes from v1 (gpu/run_main_v1.py):
- E1: PCA fitted on the SAE training set and scored held out; held-out learning curves for every SAE; controls at the
  --controls layers: same-seed 2x-epoch continuation (F = 4n) and zero-origin identity-initialised SAEs (F = n, 4n).
- E2: separate FIT / SEL / CONF sets (fits on FIT; selection on SEL; verdicts on CONF). Readouts Rao-Blackwellised:
  given the latent, a model's next pre-activation is Gaussian, so E[ReLU(z')] (or E[tanh(z')]) is integrated per state;
  cumulants kappa3/kappa4 of z' come from the deterministic (latent) part, to which the Gaussian parts add nothing.
  Debiased nMSE with an 8-block variance correction and a block jackknife SE. Residual diagnostics held out (signed and
  absolute skewness and excess kurtosis, marginal and conditional-variance-normalised; R^2 of the variance regression).
  Support entropy of the sparse latents. tanh: post-LN next-state mean by sampling (common random numbers) as well.
- E3: common random numbers across models (residual draws and every downstream noise), debiasing with each model's own
  output variance.
usage: run_main2.py OUT FAMILY N SIGMA SEED [--layers ..] [--fmults ..] [--e3 4,8,16,24] [--controls 8,24]
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
p.add_argument('--nfit', type=int, default=18)
p.add_argument('--nscore', type=int, default=17)
p.add_argument('--n3', type=int, default=20)
p.add_argument('--ntruth', type=int, default=24)
p.add_argument('--e3', default='')
p.add_argument('--controls', default='')
p.add_argument('--save-sae', default='')
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
ctrl_layers = set(int(x) for x in args.controls.split(',')) if args.controls else set()
save_layers = set(int(x) for x in args.save_sae.split(',')) if args.save_sae else set()
assert all(m in layers for m in e3_cuts), 'E3 cuts must be analysed layers'
OUT = os.path.join(args.out, net.name()); os.makedirs(OUT, exist_ok=True)
LOGF = open(os.path.join(OUT, 'log.txt'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()


def dump(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), 'w'), indent=1)


def gen(seed):
    return torch.Generator(device=dev).manual_seed(int(seed))


log(f"== v2 {net.name()} dev={dev} layers={layers} fmults={fmults} KS={KS} KD={KD} e3={e3_cuts} "
    f"s={[round(net.noise_sd(l), 6) for l in layers]}")
T0 = time.time()


# ---------------------------------------------------------------------------------------------------------------- utils
class Moments:
    def __init__(self):
        self.n = 0; self.S = None; self.shift = None

    def add(self, z):
        z = z.double()
        if self.shift is None:
            self.shift = z.mean(0); self.S = [torch.zeros_like(self.shift) for _ in range(4)]
        c = z - self.shift; c2 = c * c
        self.S[0] += c.sum(0); self.S[1] += c2.sum(0); self.S[2] += (c2 * c).sum(0); self.S[3] += (c2 * c2).sum(0)
        self.n += z.shape[0]

    def get(self):
        N = self.n
        m1, m2, m3, m4 = (s / N for s in self.S)
        var = m2 - m1 ** 2
        k3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
        k4 = m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4 - 3 * var ** 2
        return self.shift + m1, var, k3, k4


GHX, GHW = [torch.tensor(v, device=dev, dtype=torch.float64) for v in np.polynomial.hermite_e.hermegauss(32)]
GHW = GHW / GHW.sum()


def npdf(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def ncdf(x):
    return 0.5 * torch.erfc(-x / math.sqrt(2))


def rb_act(m, v):
    """E[act(z)] and E[act(z)^2] for z ~ N(m, v) elementwise (m: (B, n) float64, v: (n,))."""
    sd = v.clamp_min(1e-300).sqrt()[None]
    if net.kind == 'relu':
        t = m / sd
        e1 = m * ncdf(t) + sd * npdf(t)
        e2 = (m * m + sd * sd) * ncdf(t) + m * sd * npdf(t)
        return e1, e2
    o1, o2 = [], []
    for i0 in range(0, m.shape[0], 4096):
        a = torch.tanh(m[i0:i0 + 4096, :, None] + sd[..., None] * GHX)
        o1.append((a * GHW).sum(-1)); o2.append((a * a * GHW).sum(-1))
    return torch.cat(o1), torch.cat(o2)


def quad_feats(y):
    k = y.shape[1]
    iu = torch.triu_indices(k, k, device=y.device)
    return torch.cat([y, (y[:, :, None] * y[:, None, :])[:, iu[0], iu[1]]], 1)


def pinv_psd(C, rel=1e-10):
    lam, V = torch.linalg.eigh(C); keep = lam > lam.max() * rel
    return (V[:, keep] / lam[keep][None]) @ V[:, keep].T


def sqrt_psd(C):
    lam, U = torch.linalg.eigh(0.5 * (C + C.T))
    return U * lam.clamp_min(0).sqrt()[None, :], float((lam.clamp_max(0) ** 2).sum().sqrt() / (lam ** 2).sum().sqrt())


def mom_stats(x):
    """per-column skewness and excess kurtosis (float64)."""
    x = x.double(); c = x - x.mean(0); v = (c * c).mean(0).clamp_min(1e-300)
    return (c ** 3).mean(0) / v ** 1.5, (c ** 4).mean(0) / v ** 2 - 3.0


# ---------------------------------------------------------------------------------------------------------------- data
NTR, NTE, NFIT, NSC = 1 << args.ntrain, 1 << args.ntest, 1 << args.nfit, 1 << args.nscore
data_seed = 1_000 * args.seed + 17
t = time.time()
Xall = net.sample(NTR + NTE, gen(data_seed), layers)                 # same stream as v1 (E1 reproducible)
XF = net.sample(NFIT, gen(data_seed + 11), layers)                   # FIT
XS = net.sample(NSC, gen(data_seed + 12), layers)                    # SEL
XC = net.sample(NSC, gen(data_seed + 13), layers)                    # CONF
log(f"sampled train/test {NTR}+{NTE}, fit {NFIT}, sel/conf {NSC} in {time.time() - t:.0f}s")
E3_CACHE = {}

for l in layers:
    s = net.noise_sd(l); s2 = s * s
    Xtr, Xte = Xall[l][:NTR], Xall[l][NTR:]
    # ------------------------------------------------------------------------------------------------ E1 baselines
    mu_tr = Xtr.double().mean(0); Ctr = Xtr.double() - mu_tr
    lam_tr, U_tr = torch.linalg.eigh(Ctr.T @ Ctr / NTR); lam_tr = lam_tr.flip(0).clamp_min(0); U_tr = U_tr.flip(1)
    del Ctr
    cte = Xte.double() - mu_tr
    proj = cte @ U_tr                                                    # held-out PCA coordinates
    tot_te = (cte * cte).sum(1).mean()
    cum_te = torch.cat([torch.zeros(1, device=dev, dtype=torch.float64), (proj * proj).mean(0).cumsum(0)])
    e1 = dict(net=net.name(), layer=l, s=s, n=n, r2=float((Xte.double() ** 2).mean()),
              var_per_coord=float(lam_tr.sum() / n), PR=float(lam_tr.sum() ** 2 / (lam_tr ** 2).sum()),
              eig_above_noise=int((lam_tr > s2).sum()), eig=lam_tr.tolist())
    pca_err = lambda k: float((tot_te - cum_te[k]) / (n * s2))           # noqa: E731
    e1['pca'] = {int(k): dict(noise_units=pca_err(k), bits=pca_err(k) * n / 2 / LN2)
                 for k in sorted(set([0] + KS + KD + [n]))}
    e1['k_pca_below_noise'] = int(next((k for k in range(n + 1) if pca_err(k) < 1.0), n))
    e1['k_pca_1bit'] = int(next((k for k in range(n + 1) if pca_err(k) * n / 2 / LN2 < 1.0), n))
    dev_ = cte.abs(); srt = dev_.sort(dim=1, descending=True).values ** 2; cum = srt.cumsum(1); tot = srt.sum(1)
    e1['neuron_topk'] = {int(k): dict(noise_units=float((tot - cum[:, k - 1]).mean() / (n * s2))) for k in KS + [n]}
    del dev_, srt, cum, tot, cte, proj
    if net.kind == 'relu':
        srt0 = Xte.double().sort(dim=1, descending=True).values ** 2
        cum0 = srt0.cumsum(1); tot0 = srt0.sum(1)
        e1['neuron_topk_zero'] = {int(k): dict(noise_units=float((tot0 - cum0[:, k - 1]).mean() / (n * s2))) for k in KS + [n]}
        act = (Xte > 0).double().sum(1)
        e1['active'] = dict(mean=float(act.mean()), sd=float(act.std()), max=int(act.max()), min=int(act.min()),
                            ever_active=int((Xte > 0).any(0).sum()),
                            hist=torch.bincount(act.long(), minlength=n + 1).tolist())
        del srt0, cum0, tot0
    # ------------------------------------------------------------------------------------------------ E1 SAEs
    e1['sae'] = []; e1['curves'] = {}
    saes = {}
    Xval = Xte[:1 << 13]
    VARIANTS = [('mean', False)] + ([('zero', True)] if net.kind == 'relu' else [])
    for origin, idinit in VARIANTS:
        for fm in (fmults if origin == 'mean' else [f for f in fmults if f in (1, 4)]):
            F = fm * n
            t = time.time()
            model, c, hist = sae.train_batched(Xtr, F, KS, epochs=args.epochs, batch=args.batch, device=dev,
                                               seed=args.seed * 100 + l * 7 + fm, curve=(Xval, s, 10),
                                               origin=origin, init_identity=idinit)
            res = sae.evaluate(model, c, Xte, s, chunk=2048)
            for r in res:
                r['train_secs'] = time.time() - t; r['origin'] = origin
                e1['sae'].append(r)
            e1['curves'][f'{origin}_F{F}'] = hist
            log(f"L{l} {origin} F={F} ({time.time() - t:.0f}s): " + ' | '.join(
                f"k{r['k']} nu={r['noise_units']:.3g} rf={r['noise_units_refit']:.3g} dead={r['dead_frac']:.2f}" for r in res))
            if origin == 'mean' or fm in (1, 4):
                saes[(origin, fm)] = (model, c)
            if l in save_layers and fm in (1, 4, 16):
                torch.save(dict(state={k: v.cpu() for k, v in model.state_dict().items()}, c=c, ks=KS, F=F, n=n, origin=origin),
                           os.path.join(OUT, f'sae_L{l}_{origin}_F{F}.pt'))
    if l in ctrl_layers:
        ctrl = {}
        t = time.time()
        for origin, idinit in VARIANTS:
            m2_, c2_, h2 = sae.train_batched(Xtr, 4 * n, KS, epochs=2 * args.epochs, batch=args.batch, device=dev,
                                             seed=args.seed * 100 + l * 7 + 4, curve=(Xval, s, 10), origin=origin,
                                             init_identity=idinit)
            ctrl[f'same_seed_2x_epochs_{origin}_F4n'] = dict(res=sae.evaluate(m2_, c2_, Xte, s, chunk=2048), curve=h2)
            del m2_
            m3_, c3_, h3 = sae.train_batched(Xtr, 4 * n, KS, epochs=args.epochs, batch=args.batch, device=dev,
                                             seed=args.seed * 100 + l * 7 + 4 + 1000, curve=(Xval, s, 10), origin=origin,
                                             init_identity=idinit)
            ctrl[f'restart_{origin}_F4n'] = dict(res=sae.evaluate(m3_, c3_, Xte, s, chunk=2048), curve=h3)
            del m3_
        e1['controls'] = ctrl
        log(f"L{l} controls ({time.time() - t:.0f}s): " + ' ; '.join(
            f"{name}: " + ' '.join(f"k{r['k']}:{r['noise_units']:.3g}/{r['noise_units_refit']:.3g}" for r in v['res'])
            for name, v in ctrl.items()))
    dump(f'e1_L{l}.json', e1)

    # ------------------------------------------------------------------------------------------------ E2 (l < L)
    if l >= net.L:
        del saes
        continue
    l1 = l + 1
    Wn = net.W[l1 - 1].double()
    bn = net.b[l1 - 1].double() if net.b[l1 - 1] is not None else torch.zeros(n, device=dev, dtype=torch.float64)
    AF = XF[l].double()
    mu = AF.mean(0); CF = AF - mu
    Sig = CF.T @ CF / NFIT
    lam, U = torch.linalg.eigh(Sig); lam = lam.flip(0).clamp_min(0); U = U.flip(1)
    del CF
    # ----- model definitions: det(A) -> deterministic part (float64), Sres -> residual covariance (constant)
    MODELS = {}
    for k in KD:
        Uk = U[:, :k]
        MODELS[f'DENSE{k}'] = dict(kind='dense', k=k, det=(lambda A, Uk=Uk, mu=mu: mu + ((A - mu) @ Uk) @ Uk.T),
                                   Sres=(U[:, k:] * lam[k:][None]) @ U[:, k:].T, latent=lambda A, Uk=Uk, mu=mu: (A - mu) @ Uk)
        if k <= 16:
            y = (AF - mu) @ Uk; r = (AF - mu) - y @ Uk.T
            psi = quad_feats(y); mp = psi.mean(0); P_ = psi - mp
            B = (r.T @ P_ / NFIT) @ pinv_psd(P_.T @ P_ / NFIT, 1e-12)
            eres = r - P_ @ B.T; Se = eres.T @ eres / NFIT
            MODELS[f'DENSEQ{k}'] = dict(kind='denseq', k=k,
                                        det=(lambda A, Uk=Uk, mp=mp, B=B, mu=mu: mu + ((A - mu) @ Uk) @ Uk.T + (quad_feats((A - mu) @ Uk) - mp) @ B.T),
                                        Sres=0.5 * (Se + Se.T), latent=lambda A, Uk=Uk, mu=mu: quad_feats((A - mu) @ Uk))
            del y, r, psi, P_, eres

    def sae_recon(model, c, A, si=None):
        out = []
        with torch.no_grad():
            for i0 in range(0, A.shape[0], 4096):
                f, _, _, _ = model.encode(A[i0:i0 + 4096].float() / c)
                rec = model.decode(f) * c
                out.append(rec.double() if si is None else rec[si].double())
        return torch.cat(out, 1 if si is None else 0)

    def sae_codes(model, c, A, si, rows):
        out = []
        with torch.no_grad():
            for i0 in range(0, rows, 4096):
                f, _, _, _ = model.encode(A[i0:i0 + 4096].float() / c)
                out.append(f[si])
        return torch.cat(out)

    for (origin, fm), (model, c) in saes.items():
        Rec = sae_recon(model, c, XF[l])                                   # (S, NFIT, n)
        tagm = 'SPARSE' if origin == 'mean' else 'SPARSEZ'
        for si, k in enumerate(KS):
            ah = Rec[si]; r = AF - ah
            mh = ah.mean(0); mr = r.mean(0)
            Ch = (ah - mh).T @ (ah - mh) / NFIT; Crh = (r - mr).T @ (ah - mh) / NFIT; Cr = (r - mr).T @ (r - mr) / NFIT
            B = Crh @ pinv_psd(Ch)
            Se = Cr - B @ Crh.T
            MODELS[f'{tagm}_F{fm * n}_k{k}'] = dict(
                kind='sparse', k=k, F=fm * n, fm=(origin, fm), si=si, mh=mh, mr=mr, B=B, Sres=0.5 * (Se + Se.T),
                det=(lambda A, model=model, c=c, si=si, mh=mh, mr=mr, B=B: (lambda ah: ah + mr + (ah - mh) @ B.T)(sae_recon(model, c, A, si))),
                det_ah=(lambda ah, mh=mh, mr=mr, B=B: ah + mr + (ah - mh) @ B.T))
        del Rec
    del AF
    # ----- scoring on SEL and CONF
    NB = 8
    rows = {}

    def score_set(A, tag):
        """Returns per-model dict of metrics on this set (A: (NSC, n) float32 true states)."""
        Ad = A.double()
        nb = A.shape[0] // NB
        # truth: RB readout given the true state (only the next noise is integrated)
        vT = (s2 * (Wn * Wn).sum(1))
        mT = Ad @ Wn.T + bn
        e1T, e2T = rb_act(mT, vT)
        tm_b = torch.stack([e1T[b * nb:(b + 1) * nb].mean(0) for b in range(NB)])          # (NB, n)
        tvar = (e2T.mean(0) - e1T.mean(0) ** 2)                                            # Var(act) per neuron
        V = float(tvar.mean())
        MT = Moments(); MT.add(mT); _, _, k3T, k4T = MT.get()
        res = {'_V': V, '_k3T_norm': float((k3T ** 2).sum().sqrt()), '_k4T_norm': float((k4T ** 2).sum().sqrt())}

        def nmse_blocks(mod_b):
            d_b = mod_b - tm_b                                                             # (NB, n)
            d = d_b.mean(0); var = d_b.var(0) / NB
            est = float(((d * d - var).mean()) / V)
            jk = []
            for b in range(NB):
                keep = [i for i in range(NB) if i != b]
                db = d_b[keep]; dd = db.mean(0); vv = db.var(0) / (NB - 1)
                jk.append(float(((dd * dd - vv).mean()) / V))
            jk = np.array(jk); se = float(np.sqrt((NB - 1) / NB * ((jk - jk.mean()) ** 2).sum()))
            return est, se, float((d * d).mean() / V)
        # G: analytic
        mG = (Wn @ mu + bn)[None]; vG = ((Wn @ (Sig + s2 * torch.eye(n, device=dev, dtype=torch.float64))) * Wn).sum(1)
        g1, _ = rb_act(mG, vG)
        est, se, raw = nmse_blocks(g1.expand(NB, -1))
        res['G'] = dict(mean_nmse=est, se=se, raw=raw, cap3=0.0, cap4=0.0)
        if net.kind == 'tanh':
            # post-LN next-state mean by sampling (CRN): truth and models share eps
            ge = gen(data_seed + 6000 + l + (0 if tag == 'sel' else 50))
            eps = torch.randn(A.shape, generator=ge, device=dev, dtype=torch.float32)
            xi = torch.randn(A.shape, generator=ge, device=dev, dtype=torch.float32)
            lnT = net.act(l1, net.pre(l1, A + s * eps)).double()
            lnT_b = torch.stack([lnT[b * nb:(b + 1) * nb].mean(0) for b in range(NB)]); Vln = float(lnT.var(0).mean())
            LgG, _ = sqrt_psd(Sig)
            aG = (mu + xi.double() @ LgG.T).float()
            lnG = net.act(l1, net.pre(l1, aG + s * eps)).double()
            dG = torch.stack([lnG[b * nb:(b + 1) * nb].mean(0) for b in range(NB)]) - lnT_b
            res['G']['postLN_nmse'] = float(((dG.mean(0) ** 2) - dG.var(0) / NB).mean() / Vln)
        Ct = torch.cov(Ad.T)
        REC = {}
        for name, M in MODELS.items():
            if M['kind'] == 'sparse':
                if M['fm'] not in REC:
                    REC = {M['fm']: sae_recon(saes[M['fm']][0], saes[M['fm']][1], A)}       # (S, NSC, n), one F at a time
                det = M['det_ah'](REC[M['fm']][M['si']])
            else:
                det = M['det'](Ad)                                                         # (NSC, n) float64
            md = det @ Wn.T + bn
            vM = ((Wn @ (M['Sres'] + s2 * torch.eye(n, device=dev, dtype=torch.float64))) * Wn).sum(1)
            m1, _ = rb_act(md, vM)
            mod_b = torch.stack([m1[b * nb:(b + 1) * nb].mean(0) for b in range(NB)])
            est, se, raw = nmse_blocks(mod_b)
            MM = Moments(); MM.add(md); _, _, k3M, k4M = MM.get()
            d = dict(mean_nmse=est, se=se, raw=raw,
                     cap3=float(1 - ((k3M - k3T) ** 2).sum() / (k3T ** 2).sum()),
                     cap4=float(1 - ((k4M - k4T) ** 2).sum() / (k4T ** 2).sum()),
                     k3_abs_err=float(((k3M - k3T) ** 2).sum().sqrt()), k4_abs_err=float(((k4M - k4T) ** 2).sum().sqrt()))
            # held-out covariance match (model total covariance vs truth on this set)
            Cm = torch.cov(det.T) + M['Sres']
            d['cov_relerr'] = float((Cm - Ct).norm() / Ct.norm())
            if net.kind == 'tanh':
                Lr, _ = sqrt_psd(M['Sres'])
                aM = (det + xi.double() @ Lr.T).float()
                lnM = net.act(l1, net.pre(l1, aM + s * eps)).double()
                dM = torch.stack([lnM[b * nb:(b + 1) * nb].mean(0) for b in range(NB)]) - lnT_b
                d['postLN_nmse'] = float(((dM.mean(0) ** 2) - dM.var(0) / NB).mean() / Vln)
            if tag == 'conf':
                # residual diagnostics: e = a - det (held out); conditional variance regressed on the latent, fitted on FIT
                e = Ad - det
                for space, E in (('neuron', e), ('proj', e @ Wn.T)):
                    sk, ku = mom_stats(E)
                    sel = E.var(0) > E.var(0).max() * 1e-6
                    d[f'resid_{space}_skew_absmed'] = float(sk[sel].abs().median())
                    d[f'resid_{space}_kurt_med'] = float(ku[sel].median())
                    d[f'resid_{space}_kurt_absmed'] = float(ku[sel].abs().median())
                    d[f'resid_{space}_kurt_absp90'] = float(ku[sel].abs().quantile(0.9))
                d['resid_noise_units'] = float((e * e).sum(1).mean() / (n * s2))
                if M['kind'] == 'sparse':
                    fmodel, fc = saes[M['fm']]
                    fr = (sae_codes(fmodel, fc, A, M['si'], A.shape[0]) > 0).double().mean(0)
                    h = -(fr * torch.log2(fr.clamp_min(1e-30)) + (1 - fr) * torch.log2((1 - fr).clamp_min(1e-30)))
                    d['support_H_indep_bits'] = float(h.sum())
                    d['support_log2_binom_bits'] = float((math.lgamma(M['F'] + 1) - math.lgamma(M['k'] + 1) - math.lgamma(M['F'] - M['k'] + 1)) / LN2)
                    d['L0'] = float(fr.sum())
                    d['alive'] = int((fr > 0).sum())
            res[name] = d
            del det, md
        del REC
        return res

    def cond_diag(name, M):
        """Conditional-variance-normalised residual diagnostics: regression of e^2 on latent features fitted on FIT (2^16 rows),
        applied to CONF (2^16 rows)."""
        NR = min(1 << 16, NFIT, NSC)
        AFr = XF[l][:NR].double(); ACr = XC[l][:NR].double()
        if M['kind'] == 'sparse':
            fmodel, fc = saes[M['fm']]
            fF = sae_codes(fmodel, fc, XF[l], M['si'], NR); fC = sae_codes(fmodel, fc, XC[l], M['si'], NR)
            top = (fF > 0).float().mean(0).topk(min(512, fF.shape[1])).indices
            PhiF = torch.cat([torch.ones(NR, 1, device=dev), fF[:, top], (fF[:, top] > 0).float()], 1).double()
            PhiC = torch.cat([torch.ones(NR, 1, device=dev), fC[:, top], (fC[:, top] > 0).float()], 1).double()
        else:
            k = M['k']; Uk = U[:, :k]
            yF = (AFr - mu) @ Uk; yC = (ACr - mu) @ Uk
            fF = quad_feats(yF) if k <= 16 else torch.cat([yF, yF * yF], 1)
            fC = quad_feats(yC) if k <= 16 else torch.cat([yC, yC * yC], 1)
            PhiF = torch.cat([torch.ones(NR, 1, device=dev, dtype=torch.float64), fF], 1)
            PhiC = torch.cat([torch.ones(NR, 1, device=dev, dtype=torch.float64), fC], 1)
        eF = AFr - M['det'](AFr); eC = ACr - M['det'](ACr)
        out = {}
        for space, EF, EC in (('neuron', eF, eC), ('proj', eF @ Wn.T, eC @ Wn.T)):
            Y = EF * EF
            G_ = PhiF.T @ PhiF; G_ += 1e-3 * G_.diagonal().mean() * torch.eye(G_.shape[0], device=dev, dtype=torch.float64)
            beta = torch.linalg.solve(G_, PhiF.T @ Y)
            vC = (PhiC @ beta).clamp_min(0.1 * Y.mean(0, keepdim=True))
            YC = EC * EC
            r2 = 1 - ((YC - vC) ** 2).sum(0) / ((YC - YC.mean(0)) ** 2).sum(0)
            sk, ku = mom_stats(EC / vC.sqrt())
            sel = EC.var(0) > EC.var(0).max() * 1e-6
            out[f'resid_{space}_cond_skew_absmed'] = float(sk[sel].abs().median())
            out[f'resid_{space}_cond_kurt_med'] = float(ku[sel].median())
            out[f'resid_{space}_cond_kurt_absmed'] = float(ku[sel].abs().median())
            out[f'resid_{space}_cond_kurt_absp90'] = float(ku[sel].abs().quantile(0.9))
            out[f'resid_{space}_var_R2_med'] = float(r2[sel].median())
        return out

    t = time.time()
    RS = score_set(XS[l], 'sel'); RC = score_set(XC[l], 'conf')
    for name, M in MODELS.items():
        if name.startswith('SPARSE') and (M['F'] != 4 * n or M['k'] not in (n // 16, n // 8, n // 4)):
            continue                                                       # conditional diagnostics for a subset
        if name.startswith('DENSE') and M['k'] not in KS:
            continue
        try:
            RC[name].update(cond_diag(name, M))
        except RuntimeError as ex:
            RC[name]['cond_diag_error'] = str(ex)[:160]
    out_rows = []
    for name in ['G'] + list(MODELS.keys()):
        M = MODELS.get(name, {})
        out_rows.append(dict(model=name, k=M.get('k'), F=M.get('F'), sel=RS[name], conf=RC[name]))
    dump(f'e2_L{l}.json', dict(net=net.name(), layer=l, s=s, V_sel=RS['_V'], V_conf=RC['_V'],
                               k3T_norm=RC['_k3T_norm'], k4T_norm=RC['_k4T_norm'], rows=out_rows))
    log(f"E2 L{l} ({time.time() - t:.0f}s): " + ' '.join(
        f"{r['model']}:{r['conf']['mean_nmse']:.1e}+-{r['conf']['se']:.0e},c3={r['conf']['cap3']:.2f}" for r in out_rows[:1] + out_rows[1::3]))
    if l in e3_cuts:
        E3_CACHE[l] = dict(mu=mu, U=U, lam=lam, Sig=Sig, MODELS={k_: v for k_, v in MODELS.items()}, saes=saes)
    else:
        del saes, MODELS
    del Xall[l], XF[l], XS[l], XC[l]
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
            a = net.act(l, net.pre(l, h)); ad = a.double(); S1[l - 1] += ad.sum(0); S2[l - 1] += (ad * ad).sum(0)
            h = a + net.noise_sd(l) * torch.randn(a.shape, generator=g, device=dev)
    tmean = S1 / NT; tvar = S2 / NT - tmean ** 2
    np.savez(os.path.join(OUT, 'truth.npz'), S1=S1.cpu().numpy(), S2=S2.cpu().numpy(), N=NT)
    log(f"truth {NT} paths in {time.time() - t:.0f}s")
    OUTS = {4: [8], 8: [16, 32], 16: [32], 24: [32]}
    N3 = 1 << args.n3
    CH3 = min(1 << 16, N3)
    e3 = []
    for m in e3_cuts:
        C3 = E3_CACHE[m]
        outs = [L for L in OUTS.get(m, [32]) if L > m]
        names = ['TRUE', 'G'] + list(C3['MODELS'].keys())
        acc1 = {nm_: {L: torch.zeros(n, dtype=torch.float64, device=dev) for L in outs} for nm_ in names}
        acc2 = {nm_: {L: torch.zeros(n, dtype=torch.float64, device=dev) for L in outs} for nm_ in names}
        LG, _ = sqrt_psd(C3['Sig'])
        LR = {nm_: sqrt_psd(M['Sres'])[0].float() for nm_, M in C3['MODELS'].items()}
        t = time.time()
        for ci, i0 in enumerate(range(0, N3, CH3)):
            base = net.sample(CH3, gen(data_seed + 9000 + 1000 * m + ci), [m])[m]
            xi = torch.randn(CH3, n, generator=gen(data_seed + 19000 + 1000 * m + ci), device=dev)  # shared residual draw
            based = base.double()
            AH = {fm: sae_recon(mc[0], mc[1], base) for fm, mc in C3['saes'].items()}   # (S, CH3, n) per F
            for nm_ in names:
                if nm_ == 'TRUE':
                    a = base
                elif nm_ == 'G':
                    a = (C3['mu'] + xi.double() @ LG.T).float()
                else:
                    M = C3['MODELS'][nm_]
                    det = M['det_ah'](AH[M['fm']][M['si']]) if M['kind'] == 'sparse' else M['det'](based)
                    a = det.float() + xi @ LR[nm_].T
                x = a
                for j in range(m, max(outs)):
                    eps = torch.randn(x.shape, generator=gen(data_seed + 29000 + 100000 * m + 100 * ci + j), device=dev)
                    x = net.act(j + 1, net.pre(j + 1, x + net.noise_sd(j) * eps))
                    if (j + 1) in acc1[nm_]:
                        xd = x.double(); acc1[nm_][j + 1] += xd.sum(0); acc2[nm_][j + 1] += (xd * xd).sum(0)
        for nm_ in names:
            for L in outs:
                est = acc1[nm_][L] / N3; mv = acc2[nm_][L] / N3 - est ** 2
                raw = float(((est - tmean[L - 1]) ** 2).mean() / tvar[L - 1].mean())
                err = raw - float(mv.mean() / tvar[L - 1].mean()) / N3 - 1.0 / NT
                e3.append(dict(cut=m, L=L, model=nm_, nmse=err, raw=raw, var_ratio=float(mv.mean() / tvar[L - 1].mean())))
        log(f"E3 cut {m} ({time.time() - t:.0f}s): " + ' '.join(f"{d['model']}@{d['L']}:{d['nmse']:.2e}" for d in e3 if d['cut'] == m)[:3000])
    dump('e3.json', dict(net=net.name(), N3=N3, NT=NT, rows=e3))
log(f"ALL DONE {time.time() - T0:.0f}s")
