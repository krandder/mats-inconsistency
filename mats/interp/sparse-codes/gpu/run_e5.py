"""E5 (PREREG amendment 4): does a feature family with exact marginals and the full covariance, but nothing more, carry
what the exact sparse latent carries? The dependence risk, asked in the covariance-carrying frame.
At cuts m in {8, 16, 24} of an ARC net, from TopK SAEs trained here (F = n, 4n; origins mean and zero; k = n/16, n/8,
n/4), with the decoded state a = D f + b_dec and the v2 residual model (linear-Gaussian given the reconstruction):
  SPARSE      exact latent f (from true states) + residual                         (v2's model; oracle latent law)
  SPARSECOP   f from a Gaussian copula: exact zero-inflated marginals + normal-scores correlation, + residual,
              recoloured to the exact mean and covariance of the state
  SPARSEIND   f with independent exact marginals + residual, recoloured to the exact mean and covariance
  SPARSEIND0  the same without recolouring (the independent-feature product; coordinator's analytic case)
  NEUCOP      neurons with exact marginals + Gaussian copula, recoloured
  G           Gaussian with the exact mean and covariance
  DENSE(k)    v2's dense latent (top-k PCA coordinates exact + independent Gaussian rest)
Recolouring: a -> mu + S^1/2 S_M^-1/2 (a - mu_M), S_M and mu_M estimated from 2^17 model samples (FIT); so every model
except SPARSEIND0 has the exact mean and covariance up to sampling, and they differ only beyond second order.
Scores: one step (Rao-Blackwellised E[ReLU(z')] on 2^17 states, 8-block debiased nMSE + jackknife SE, against the
truth's RB readout on independent true states), and the representation at the cut pushed through the real layers to
depth 32 (and 16 from cut 8): 2^20 states, common downstream noise across models, debiased with each model's variance.
usage: run_e5.py OUT N SIGMA SEED [--cuts 8,16,24] [--epochs 30]
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import nets, sae   # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('out'); p.add_argument('n', type=int); p.add_argument('sigma', type=float); p.add_argument('seed', type=int)
p.add_argument('--cuts', default='8,16,24')
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--batch', type=int, default=2048)
p.add_argument('--ntrain', type=int, default=18)
p.add_argument('--nfit', type=int, default=16)
p.add_argument('--nscore', type=int, default=17)
p.add_argument('--n3', type=int, default=20)
p.add_argument('--ntruth', type=int, default=24)
args = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu':
    torch.set_num_threads(int(os.environ.get('NT', '4')))
torch.backends.cuda.matmul.allow_tf32 = False
net = nets.SNet('arc', args.n, args.seed, args.sigma, device=dev)
n = args.n
cuts = [int(x) for x in args.cuts.split(',')]
KS = [n // 16, n // 8, n // 4]
OUT = os.path.join(args.out, net.name()); os.makedirs(OUT, exist_ok=True)
LOGF = open(os.path.join(OUT, 'e5_log.txt'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()


def gen(s):
    return torch.Generator(device=dev).manual_seed(int(s))


def npdf(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def ncdf(x):
    return 0.5 * torch.erfc(-x / math.sqrt(2))


def nicdf(u):
    return math.sqrt(2) * torch.erfinv(2 * u - 1)


def sqrtm_psd(C, inv=False, rel=1e-10):
    lam, U = torch.linalg.eigh(0.5 * (C + C.T)); lam = lam.clamp_min(lam.max() * rel)
    return (U * (lam ** (-0.5 if inv else 0.5))[None]) @ U.T


def rb_relu(m, v):
    sd = v.clamp_min(1e-300).sqrt()[None]; t = m / sd
    return m * ncdf(t) + sd * npdf(t)


class Copula:
    """Zero-inflated marginals (point mass at 0 + empirical quantiles of the positive part) + Gaussian copula with the
    normal-scores correlation (zeros scored at E[Z | Z < c]). X: (N, F) nonnegative codes (float32)."""
    def __init__(self, X, nq=1025, indep=False):
        N, F = X.shape
        on = X > 0
        q = 1 - on.double().mean(0)                                     # P(f = 0)
        self.alive = (q < 1 - 1.0 / N)
        Xa = X[:, self.alive]; ona = on[:, self.alive]; qa = q[self.alive]
        self.q = qa; self.F = F; self.Fa = int(self.alive.sum())
        # quantile tables of the positive parts
        grid = torch.linspace(0, 1, nq, device=X.device, dtype=torch.float64)
        Q = torch.empty(self.Fa, nq, device=X.device, dtype=torch.float64)
        for j0 in range(0, self.Fa, 64):
            blk = Xa[:, j0:j0 + 64]
            big = torch.where(blk > 0, blk, torch.full_like(blk, float('inf')))
            srt = big.sort(0).values.double()
            cnt = (blk > 0).sum(0)
            del big
            for jj in range(blk.shape[1]):
                c_ = int(cnt[jj]); v = srt[:c_, jj]
                if c_ == 0:
                    Q[j0 + jj] = 0.0; continue
                pos = grid * (c_ - 1); lo = pos.floor().long(); hi = pos.ceil().long(); w = pos - lo
                Q[j0 + jj] = v[lo] * (1 - w) + v[hi] * w
        self.Qt = Q
        self.indep = indep
        if not indep:
            c = nicdf(qa.clamp(1e-12, 1 - 1e-12))
            zero_score = -npdf(c) / ncdf(c).clamp_min(1e-300)
            Z = torch.empty(N, self.Fa, device=X.device, dtype=torch.float64)
            for j0 in range(0, self.Fa, 64):
                blk = Xa[:, j0:j0 + 64].double(); onb = blk > 0
                ranks = torch.where(onb, blk, torch.full_like(blk, -1.0)).argsort(0).argsort(0).double()   # zeros first
                nz = (~onb).sum(0).double(); npos = onb.sum(0).double().clamp_min(1)
                u = qa[j0:j0 + 64][None] + (1 - qa[j0:j0 + 64][None]) * (ranks - nz[None] + 0.5) / npos[None]
                z = nicdf(u.clamp(1e-12, 1 - 1e-12))
                Z[:, j0:j0 + 64] = torch.where(onb, z, zero_score[j0:j0 + 64][None].expand_as(z))
            Z = Z - Z.mean(0); Cz = Z.T @ Z / N; sd = Cz.diagonal().clamp_min(1e-300).sqrt()
            R = Cz / (sd[:, None] * sd[None, :])
            self.L = sqrtm_psd(R).float()                            # symmetric square root

    def sample(self, M, g):
        xi = torch.randn(M, self.Fa, generator=g, device=self.Qt.device)
        z = xi if self.indep else xi @ self.L.T
        u = ncdf(z)                                                  # float32 (memory)
        qa = self.q[None].float()
        upos = ((u - qa) / (1 - qa)).clamp(0, 1)
        nq = self.Qt.shape[1]
        pos = upos * (nq - 1); lo = pos.floor().long().clamp(0, nq - 1); hi = (lo + 1).clamp(max=nq - 1); w = pos - lo
        Qt = self.Qt.T                                               # (nq, Fa)
        Qt = Qt.float()
        val = torch.gather(Qt, 0, lo) * (1 - w) + torch.gather(Qt, 0, hi) * w
        f = torch.where(u > qa, val, torch.zeros_like(val))
        del u, val
        out = torch.zeros(M, self.F, device=self.Qt.device, dtype=torch.float32)
        out[:, self.alive] = f.float()
        return out


T0 = time.time()
# ------------------------------------------------------------------------------------------------------------- truth
NT = 1 << args.ntruth
S1 = torch.zeros(net.L, n, dtype=torch.float64, device=dev); S2 = torch.zeros_like(S1)
g = gen(1000 * args.seed + 24)                       # = run_main(2)'s truth stream (data_seed + 7)
CHT = min(1 << 16, NT)
for i0 in range(0, NT, CHT):
    h = torch.randn(CHT, n, generator=g, device=dev)
    for l in range(1, net.L + 1):
        a = net.act(l, net.pre(l, h)); ad = a.double(); S1[l - 1] += ad.sum(0); S2[l - 1] += (ad * ad).sum(0)
        h = a + net.noise_sd(l) * torch.randn(a.shape, generator=g, device=dev)
tmean = S1 / NT; tvar = S2 / NT - tmean ** 2
log(f"== {net.name()} E5 truth {NT} paths ({time.time() - T0:.0f}s)")

ds = 1000 * args.seed + 517
NTR, NFIT, NSC, N3 = 1 << args.ntrain, 1 << args.nfit, 1 << args.nscore, 1 << args.n3
XTR = net.sample(NTR, gen(ds), cuts)
XF = net.sample(NFIT, gen(ds + 1), cuts)
XC = net.sample(NSC, gen(ds + 2), cuts)
res = dict(net=net.name(), NT=NT, N3=N3, cuts={})
for m in cuts:
    t = time.time()
    s = net.noise_sd(m); s2 = s * s
    Wn = net.W[m].double()                                            # W_{m+1}
    AF = XF[m].double(); mu = AF.mean(0); Sig = torch.cov(AF.T)
    lam, U = torch.linalg.eigh(Sig); lam = lam.flip(0).clamp_min(0); U = U.flip(1)
    SQ = sqrtm_psd(Sig)
    gen_m = lambda tag: gen(ds + 100000 * m + tag)                     # noqa: E731
    # ---------------- model samplers: f(M, g) -> (M, n) float64 states; 'from_true' models take true states instead
    MOD = {}
    MOD['G'] = dict(kind='sample', fn=lambda M, g: mu + torch.randn(M, n, generator=g, device=dev, dtype=torch.float64) @ SQ.T)
    for k in KS:
        Uk, Ur, sr = U[:, :k], U[:, k:], lam[k:].sqrt()
        MOD[f'DENSE{k}'] = dict(kind='true', fn=lambda A, g, Uk=Uk, Ur=Ur, sr=sr: mu + ((A - mu) @ Uk) @ Uk.T +
                                (torch.randn(A.shape[0], Ur.shape[1], generator=g, device=dev, dtype=torch.float64) * sr) @ Ur.T)
    # neuron copula (zero-inflated ReLU marginals)
    ncop = Copula(XF[m])
    MOD['NEUCOP_raw'] = dict(kind='sample', fn=lambda M, g, c=ncop: c.sample(M, g).double())
    # SAEs and their latent families
    for origin, idinit in (('mean', False), ('zero', True)):
        for fm in (1, 4):
            F = fm * n
            model, c, _ = sae.train_batched(XTR[m], F, KS, epochs=args.epochs, batch=args.batch, device=dev,
                                            seed=args.seed * 100 + m * 7 + fm + (0 if origin == 'mean' else 50),
                                            origin=origin, init_identity=idinit)
            for si, k in enumerate(KS):
                with torch.no_grad():
                    fF = torch.cat([model.encode(XF[m][i:i + 4096] / c)[0][si] for i in range(0, NFIT, 4096)]) * c   # (NFIT, F)
                Dd = (model.W_dec[si].detach() * c).double(); bd = (model.b_dec[si].detach() * c).double()
                ah = fF.double() @ Dd.T + bd; r = AF - ah
                mh = ah.mean(0); mr = r.mean(0)
                Ch = torch.cov(ah.T); Crh = (r - mr).T @ (ah - mh) / NFIT; Cr = torch.cov(r.T)
                lh, Vh = torch.linalg.eigh(Ch); keep = lh > lh.max() * 1e-10
                B = Crh @ ((Vh[:, keep] / lh[keep][None]) @ Vh[:, keep].T)
                Le = sqrtm_psd(Cr - B @ Crh.T)

                def resid(ah_, g, mh=mh, mr=mr, B=B, Le=Le):
                    return ah_ + mr + (ah_ - mh) @ B.T + torch.randn(ah_.shape[0], n, generator=g, device=dev, dtype=torch.float64) @ Le.T

                def enc(A, model=model, c=c, si=si, Dd=Dd, bd=bd):
                    with torch.no_grad():
                        f = torch.cat([model.encode(A[i:i + 4096].float() / c)[0][si] for i in range(0, A.shape[0], 4096)])
                    return (f * c).double() @ Dd.T + bd
                bn = 'SPARSE' if origin == 'mean' else 'SPARSEZ'
                MOD[f'{bn}_F{F}_k{k}'] = dict(kind='true', fn=lambda A, g, enc=enc, resid=resid: resid(enc(A), g))
                cop = Copula(fF); ind = Copula(fF, indep=True)
                MOD[f'{bn}COP_raw_F{F}_k{k}'] = dict(kind='sample', fn=lambda M, g, cop=cop, Dd=Dd, bd=bd, resid=resid:
                                                     resid(cop.sample(M, g).double() @ Dd.T + bd, g))
                MOD[f'{bn}IND_raw_F{F}_k{k}'] = dict(kind='sample', fn=lambda M, g, ind=ind, Dd=Dd, bd=bd, resid=resid:
                                                     resid(ind.sample(M, g).double() @ Dd.T + bd, g))
            del fF, model
            torch.cuda.empty_cache() if dev == 'cuda' else None
    # recoloured versions of the sampled families (exact mean and covariance up to sampling)
    for name in [k_ for k_ in list(MOD) if k_.endswith('_raw') or '_raw_' in k_]:
        base = MOD[name]['fn']
        Ms = torch.cat([base(1 << 15, gen_m(7000 + j)) for j in range(4)])     # 2^17 model samples
        muM = Ms.mean(0); TM = SQ @ sqrtm_psd(torch.cov(Ms.T), inv=True)
        cov_err = float((torch.cov(Ms.T) - Sig).norm() / Sig.norm())
        MOD[name]['cov_relerr_raw'] = cov_err
        MOD[name.replace('_raw', '')] = dict(kind='sample', fn=lambda M, g, base=base, muM=muM, TM=TM: mu + (base(M, g) - muM) @ TM.T,
                                             cov_relerr_raw=cov_err)
    log(f"cut {m}: {len(MOD)} models built ({time.time() - t:.0f}s)")
    torch.cuda.empty_cache() if dev == 'cuda' else None
    # ---------------- one step (RB readout), 8 blocks; truth on CONF states
    NB = 8; nb = NSC // NB
    vN = s2 * (Wn * Wn).sum(1)
    AC = XC[m].double()
    tr = rb_relu(AC @ Wn.T, vN)
    tm_b = torch.stack([tr[b * nb:(b + 1) * nb].mean(0) for b in range(NB)])
    V = float(((rb_relu(AC @ Wn.T, vN) ** 2).mean(0) - tr.mean(0) ** 2).mean())   # approx Var(act) scale (RB part)
    one = {}
    for name, M in MOD.items():
        gm = gen_m(8000 + abs(hash(name)) % 100000)
        A = M['fn'](AC, gm) if M['kind'] == 'true' else torch.cat([M['fn'](min(1 << 15, NSC), gm) for _ in range(max(1, NSC >> 15))])
        r1 = rb_relu(A @ Wn.T, vN)
        d_b = torch.stack([r1[b * nb:(b + 1) * nb].mean(0) for b in range(NB)]) - tm_b
        d = d_b.mean(0); var = d_b.var(0) / NB
        est = float((d * d - var).mean() / V)
        jk = []
        for b in range(NB):
            db = d_b[[i for i in range(NB) if i != b]]; dd = db.mean(0); vv = db.var(0) / (NB - 1)
            jk.append(float((dd * dd - vv).mean() / V))
        jk = np.array(jk)
        one[name] = dict(nmse=est, se=float(np.sqrt((NB - 1) / NB * ((jk - jk.mean()) ** 2).sum())),
                         cov_relerr=float((torch.cov(A.T) - torch.cov(AC.T)).norm() / torch.cov(AC.T).norm()),
                         cov_relerr_raw=M.get('cov_relerr_raw'))
    log(f"cut {m} one-step: " + ' '.join(f"{k_}:{v['nmse']:.1e}" for k_, v in one.items())[:2500])
    # ---------------- representation at the cut pushed through the real layers (CRN downstream noise)
    outs = [16, 32] if m == 8 else [32]
    acc1 = {k_: {L: torch.zeros(n, dtype=torch.float64, device=dev) for L in outs} for k_ in MOD}
    acc2 = {k_: {L: torch.zeros(n, dtype=torch.float64, device=dev) for L in outs} for k_ in MOD}
    CH3 = min(1 << 15, N3)
    for ci in range(N3 // CH3):
        base_states = net.sample(CH3, gen_m(20000 + ci), [m])[m].double()
        for name, M in MOD.items():
            gm = gen_m(30000 + 1000 * ci + abs(hash(name)) % 997)
            A = M['fn'](base_states, gm) if M['kind'] == 'true' else M['fn'](CH3, gm)
            x = A.float()
            for j in range(m, max(outs)):
                eps = torch.randn(x.shape, generator=gen_m(40000 + 100 * ci + j), device=dev)
                x = net.act(j + 1, net.pre(j + 1, x + net.noise_sd(j) * eps))
                if (j + 1) in acc1[name]:
                    xd = x.double(); acc1[name][j + 1] += xd.sum(0); acc2[name][j + 1] += (xd * xd).sum(0)
    rep = {}
    for name in MOD:
        rep[name] = {}
        for L in outs:
            est = acc1[name][L] / N3; mv = acc2[name][L] / N3 - est ** 2
            raw = float(((est - tmean[L - 1]) ** 2).mean() / tvar[L - 1].mean())
            rep[name][L] = raw - float(mv.mean() / tvar[L - 1].mean()) / N3 - 1.0 / NT
    log(f"cut {m} representation: " + ' '.join(f"{k_}@32:{v[32]:.1e}" for k_, v in rep.items())[:2500])
    res['cuts'][m] = dict(one_step=one, rep=rep)
    json.dump(res, open(os.path.join(OUT, 'e5.json'), 'w'), indent=1)
    del MOD
    torch.cuda.empty_cache() if dev == 'cuda' else None
log(f"E5 done {time.time() - T0:.0f}s")
