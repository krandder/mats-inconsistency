"""E4: transcoder chain on ARC nets (PREREG.md). One transcoder per layer l = 1..32, h_{l-1} -> a_l, trained on real
paths (TopK during training; per-feature JumpReLU thresholds at inference, quantile-matched to the TopK firing rates).
Reports per setting: per-layer error in bits on real inputs (path KL of the replacement), the chain's own E[a_L] by
sampling (L = 8, 16, 32), and a deductive feature-level Gaussian closure through the chain (mean + full feature
covariance, closed-form per-feature moments, Mehler/Hermite off-diagonals), against the truth.
Code check: the identity transcoder (F = n, E = W_l, D = I, theta = 0) = neuron-level Gaussian closure (K=2).
usage: run_e4.py OUT N SIGMA SEED [--ntrain 17] [--epochs 30] [--settings n:8,n:4,4n:8,4n:4] [--nchain 18] [--ntruth 24]
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import nets   # noqa: E402

LN2 = math.log(2.0)
p = argparse.ArgumentParser()
p.add_argument('out'); p.add_argument('n', type=int); p.add_argument('sigma', type=float); p.add_argument('seed', type=int)
p.add_argument('--ntrain', type=int, default=17)
p.add_argument('--ntest', type=int, default=14)
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--batch', type=int, default=2048)
p.add_argument('--fmults', default='1,4')
p.add_argument('--kdivs', default='8,4,2')          # k = n / kdiv
p.add_argument('--nchain', type=int, default=18)
p.add_argument('--ntruth', type=int, default=24)
p.add_argument('--hermite', type=int, default=60)
args = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu':
    torch.set_num_threads(int(os.environ.get('NT', '4')))
torch.backends.cuda.matmul.allow_tf32 = False
net = nets.SNet('arc', args.n, args.seed, args.sigma, device=dev)
n, Ldep = args.n, net.L
OUT = os.path.join(args.out, net.name()); os.makedirs(OUT, exist_ok=True)
LOGF = open(os.path.join(OUT, 'e4_log.txt'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()


def gen(s):
    return torch.Generator(device=dev).manual_seed(s)


fmults = [int(x) for x in args.fmults.split(',')]
ks = [n // int(x) for x in args.kdivs.split(',')]
T0 = time.time()


# ------------------------------------------------------------------------------------------- transcoder (batched over k)
class BTC(torch.nn.Module):
    def __init__(self, n, F, ks, mu_in, mu_out, g):
        super().__init__()
        S = len(ks)
        self.S, self.F, self.n = S, F, n
        self.register_buffer('ks', torch.tensor(ks, device=dev)); self.kmax = max(ks)
        Wd = torch.randn(S, n, F, device=dev, generator=g); Wd = Wd / Wd.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(Wd)
        self.W_enc = torch.nn.Parameter(torch.randn(S, F, n, device=dev, generator=g) / math.sqrt(n))
        self.b_enc = torch.nn.Parameter(torch.zeros(S, F, device=dev))
        self.b_pre = torch.nn.Parameter(mu_in[None].repeat(S, 1).clone())
        self.b_dec = torch.nn.Parameter(mu_out[None].repeat(S, 1).clone())
        self.register_buffer('theta', torch.zeros(S, F, device=dev))

    def u(self, h):
        return torch.baddbmm(self.b_enc[:, None, :], h[None] - self.b_pre[:, None, :], self.W_enc.transpose(1, 2))

    def f_topk(self, h):
        p = torch.relu(self.u(h))
        v, i = p.topk(self.kmax, dim=-1)
        mask = (torch.arange(self.kmax, device=dev)[None, None, :] < self.ks[:, None, None]).to(v.dtype)
        return torch.zeros_like(p).scatter_(-1, i, v * mask), p

    def f_jump(self, h):
        u = self.u(h)
        return u * (u > self.theta[:, None, :]).to(u.dtype)

    def dec(self, f):
        return torch.bmm(f, self.W_dec.transpose(1, 2)) + self.b_dec[:, None, :]


def train_tc(H, A, F, ks, seed):
    N = H.shape[0]
    g = gen(seed)
    ch = float(((H - H.mean(0)) ** 2).mean().sqrt()); ca = float(((A - A.mean(0)) ** 2).mean().sqrt()) + 1e-12
    Hs, As = H / ch, A / ca
    m = BTC(n, F, ks, Hs.mean(0), As.mean(0), g)
    lr = 2e-3 / math.sqrt(max(1.0, F / (4 * n)))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    steps = max(1, args.epochs * N // args.batch); dstart = int(0.6 * steps)
    last = torch.zeros(len(ks), F, device=dev)
    perm = torch.randperm(N, device=dev, generator=g); pos = 0
    for t in range(steps):
        if pos + args.batch > N:
            perm = torch.randperm(N, device=dev, generator=g); pos = 0
        idx = perm[pos:pos + args.batch]; pos += args.batch
        f, pp = m.f_topk(Hs[idx]); out = m.dec(f)
        err = out - As[idx][None]
        loss = (err ** 2).sum(-1).mean(-1).sum()
        with torch.no_grad():
            fired = (f > 0).any(1).float(); last.mul_(1 - fired).add_(1); dead = last > max(50, (1 << 16) // args.batch)
        if dead.any():
            pd = pp.masked_fill(~dead[:, None, :], 0.0); ka = min(n // 2, int(dead.sum(1).max()))
            v, ii = pd.topk(ka, dim=-1); fa = torch.zeros_like(pd).scatter_(-1, ii, v)
            ea = torch.bmm(fa, m.W_dec.transpose(1, 2)); res = (As[idx][None] - out).detach()
            loss = loss + (((ea - res) ** 2).sum(-1).mean(-1) / (res ** 2).sum(-1).mean(-1).clamp_min(1e-12)).sum() / 32
        opt.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            Wd = m.W_dec; gr = Wd.grad
            gr.sub_(Wd * (gr * Wd).sum(1, keepdim=True))
        if t >= dstart:
            for pg in opt.param_groups:
                pg['lr'] = lr * max(0.0, (steps - t) / (steps - dstart))
        opt.step()
        with torch.no_grad():
            m.W_dec.div_(m.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-12))
    # per-feature JumpReLU thresholds: theta_j = (1 - p_j) quantile of u_j, p_j = TopK firing rate (on 2^15 rows)
    with torch.no_grad():
        M = min(N, 1 << 15)
        f, _ = m.f_topk(Hs[:M]); u = m.u(Hs[:M])
        pr = (f > 0).float().mean(1)                                        # (S, F)
        us, _ = u.sort(dim=1)                                                # (S, M, F)
        pos_ = ((1 - pr) * (M - 1)).round().long().clamp(0, M - 1)           # (S, F)
        th = torch.gather(us, 1, pos_[:, None, :])[:, 0, :]
        th = torch.where(pr > 0, th, torch.full_like(th, 1e9))
        m.theta.copy_(th)
    return m, ch, ca


# ------------------------------------------------------------------------------------------------ closed-form moments
def npdf(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def nsurv(x):
    return 0.5 * torch.erfc(x / math.sqrt(2))


def jump_hermite(mu, sd, th, K):
    """c_k = E[f(mu + sd xi) He_k(xi)], f(u) = u 1[u > th], k = 0..K (float64). Also E[f^2]."""
    t = ((th - mu) / sd).clamp(-38.0, 38.0)        # beyond |t| = 38, phi(t) underflows to 0 and Q(t) is 0 or 1
    ph = npdf(t); Q = nsurv(t)
    He = [torch.ones_like(t), t.clone()]
    for k in range(1, K + 1):
        He.append(t * He[k] - k * He[k - 1])                 # He_0..He_{K+1}
    I = [Q] + [He[m - 1] * ph for m in range(1, K + 1)]       # I_m = int_t^inf He_m phi
    c = [mu * Q + sd * ph]
    for k in range(1, K + 1):
        Jk = He[k] * ph + k * I[k - 1]                       # int_t^inf xi He_k phi
        c.append(mu * I[k] + sd * Jk)
    Ef2 = (mu * mu + sd * sd) * Q + sd * ph * (mu + th)
    return torch.stack(c, 1), Ef2


def closure_layer(mu_h, S_h, We, be, bpre, th, Wd, bd, K):
    """Feature-level Gaussian closure of one transcoder layer. Inputs float64. Returns mean/cov of a_hat."""
    mu_u = We @ (mu_h - bpre) + be
    S_u = We @ S_h @ We.T
    sd = S_u.diagonal().clamp_min(1e-30).sqrt()
    c, Ef2 = jump_hermite(mu_u, sd, th, K)                   # (F, K+1)
    mu_f = c[:, 0]
    rho = (S_u / (sd[:, None] * sd[None, :])).clamp(-1, 1)
    cov = torch.zeros_like(S_u); rk = torch.ones_like(S_u); fact = 1.0
    for k in range(1, K + 1):
        rk = rk * rho; fact *= k
        cov += (c[:, k:k + 1] * c[None, :, k]) * rk / fact
    cov.diagonal().copy_(Ef2 - mu_f ** 2)
    mu_a = Wd @ mu_f + bd
    S_a = Wd @ cov @ Wd.T
    return mu_a, S_a


# --------------------------------------------------------------------------------------------------------------- truth
# The transcoders are trained on real paths from the standard input law, so their biases carry the real per-layer means
# of that law (sample information). The honest test of a description is a question it was not fitted to: the same
# closure run from other input laws (shift, aniso; lib/nets.input_law), against truths for those laws.
LAWS = ('std', 'shift', 'aniso')
NT = 1 << args.ntruth
OUTL = (8, 16, 32)
LAW = {}
for li, law in enumerate(LAWS):
    t = time.time()
    m0, L0 = nets.input_law(law, n, args.seed)
    m0 = torch.tensor(m0, device=dev, dtype=torch.float64); L0 = torch.tensor(L0, device=dev, dtype=torch.float64)
    S1 = torch.zeros(Ldep, n, dtype=torch.float64, device=dev); S2 = torch.zeros_like(S1)
    g = gen(1000 * args.seed + 7 + 100 * li)    # std: the same stream as run_main.py's E3 truth (data_seed + 7)
    CHT = min(1 << 16, NT)
    for i0 in range(0, NT, CHT):
        h = torch.randn(CHT, n, generator=g, device=dev)
        if law != 'std':
            h = (m0 + h.double() @ L0.T).float()
        for l in range(1, Ldep + 1):
            a = net.act(l, net.pre(l, h)); ad = a.double(); S1[l - 1] += ad.sum(0); S2[l - 1] += (ad * ad).sum(0)
            h = a + net.noise_sd(l) * torch.randn(a.shape, generator=g, device=dev)
    LAW[law] = dict(m0=m0, L0=L0, tmean=S1 / NT, tvar=S2 / NT - (S1 / NT) ** 2)
    np.savez(os.path.join(OUT, f'truth_{law}.npz'), S1=S1.cpu().numpy(), S2=S2.cpu().numpy(), N=NT)
    log(f"== {net.name()} E4 truth law={law} {NT} paths in {time.time() - t:.0f}s")


def score(est, L, law='std'):
    T = LAW[law]
    return float(((est.detach() - T['tmean'][L - 1]) ** 2).mean() / T['tvar'][L - 1].mean()) - 1.0 / NT


res = dict(net=net.name(), NT=NT, laws=LAWS, settings=[])
# how far each law moves the answer: the std-law truth used as the estimate for the other laws
res['std_mean_as_estimate'] = {law: {L: score(LAW['std']['tmean'][L - 1], L, law) for L in OUTL} for law in LAWS}
log("std-law mean used for other laws: " + str(res['std_mean_as_estimate']))

# ---------------------------------------------------- identity transcoder = neuron-level Gaussian closure (code check)
I = torch.eye(n, dtype=torch.float64, device=dev); z0 = torch.zeros(n, dtype=torch.float64, device=dev)
res['identity_closure'] = {}; res['identity_means'] = {}
for law in LAWS:
    mu_h = LAW[law]['m0'].clone(); S_h = LAW[law]['L0'] @ LAW[law]['L0'].T
    out = {}
    for l in range(1, Ldep + 1):
        mu_a, S_a = closure_layer(mu_h, S_h, net.W64[l - 1].to(dev), z0, z0, z0, I, z0, args.hermite)
        if l in OUTL:
            out[l] = score(mu_a, l, law)
            if law == 'std':
                res['identity_means'][l] = mu_a.cpu().tolist()
        mu_h, S_h = mu_a, S_a + net.noise_sd(l) ** 2 * I
    res['identity_closure'][law] = out
    log(f"identity transcoder closure (neuron K=2) law={law}: " + ' '.join(f"L{l}:{v:.3e}" for l, v in out.items()))

# ------------------------------------------------------------------------------------------ train the chain, per F
NTR, NTE = 1 << args.ntrain, 1 << args.ntest
for fm in fmults:
    F = fm * n
    t = time.time()
    g = gen(1000 * args.seed + 300 + fm)
    H = torch.randn(NTR + NTE, n, generator=g, device=dev)        # real h_{l-1}
    chain = []
    per_layer = []
    for l in range(1, Ldep + 1):
        A = net.act(l, net.pre(l, H))
        m, ch, ca = train_tc(H[:NTR], A[:NTR], F, ks, 1000 * args.seed + 10 * l + fm)
        with torch.no_grad():
            Hs = H[NTR:] / ch
            fj = m.f_jump(Hs); outj = m.dec(fj) * ca
            ftk, _ = m.f_topk(Hs); outk = m.dec(ftk) * ca
            s = net.noise_sd(l)
            ej = ((outj - A[NTR:][None]) ** 2).sum(-1).mean(-1)
            ek = ((outk - A[NTR:][None]) ** 2).sum(-1).mean(-1)
            l0j = (fj > 0).float().sum(-1).mean(-1)
        per_layer.append(dict(layer=l, s=s, bits_jump=(ej / (2 * s * s) / LN2).tolist(),
                              bits_topk=(ek / (2 * s * s) / LN2).tolist(),
                              noise_units_jump=(ej / (n * s * s)).tolist(), L0_jump=l0j.tolist()))
        chain.append((m, ch, ca))
        H = A + net.noise_sd(l) * torch.randn(A.shape, generator=g, device=dev)
        if l in (1, 8, 16, 24, 32):
            log(f"F={F} L{l}: bits(jump)={[round(x, 1) for x in per_layer[-1]['bits_jump']]} "
                f"nu={[round(x, 2) for x in per_layer[-1]['noise_units_jump']]} L0={[round(x, 1) for x in l0j.tolist()]}")
    log(f"F={F} chain trained in {time.time() - t:.0f}s")
    torch.save([dict(state={k_: v.cpu() for k_, v in m_.state_dict().items()}, ch=ch_, ca=ca_) for (m_, ch_, ca_) in chain],
               os.path.join(OUT, f'chain_F{F}.pt'))
    # chain's own E[a_L] by sampling (JumpReLU transcoders, noise as in the net), under each input law
    NC = 1 << args.nchain
    CH = min(1 << 15, NC)
    for si, k in enumerate(ks):
        row = dict(F=F, k=k, path_kl_bits_l1_31=sum(pl['bits_jump'][si] for pl in per_layer[:-1]),
                   bits_per_layer=[pl['bits_jump'][si] for pl in per_layer],
                   noise_units_per_layer=[pl['noise_units_jump'][si] for pl in per_layer],
                   L0_per_layer=[pl['L0_jump'][si] for pl in per_layer],
                   bits_topk_per_layer=[pl['bits_topk'][si] for pl in per_layer],
                   chain_sampled={}, chain_closure={}, closure_vs_chain={})
        for li, law in enumerate(LAWS):
            gch = gen(1000 * args.seed + 500 + 10 * fm + li)
            acc = {L: torch.zeros(n, dtype=torch.float64, device=dev) for L in OUTL}
            with torch.no_grad():
                for i0 in range(0, NC, CH):
                    h = torch.randn(CH, n, generator=gch, device=dev)
                    if law != 'std':
                        h = (LAW[law]['m0'] + h.double() @ LAW[law]['L0'].T).float()
                    for l in range(1, Ldep + 1):
                        m, ch, ca = chain[l - 1]
                        u = torch.addmm(m.b_enc[si], h / ch - m.b_pre[si], m.W_enc[si].T)
                        f = u * (u > m.theta[si]).to(u.dtype)
                        a = (f @ m.W_dec[si].T + m.b_dec[si]) * ca
                        if l in OUTL:
                            acc[l] += a.double().sum(0)
                        h = a + net.noise_sd(l) * torch.randn(a.shape, generator=gch, device=dev)
            chain_mean = {L: acc[L] / NC for L in OUTL}
            row['chain_sampled'][law] = {L: score(chain_mean[L], L, law) - 1.0 / NC for L in OUTL}
            # deductive feature-level Gaussian closure through the chain from this law's input moments
            mu_h = LAW[law]['m0'].clone(); S_h = LAW[law]['L0'] @ LAW[law]['L0'].T
            clo, cvc = {}, {}
            with torch.no_grad():
                for l in range(1, Ldep + 1):
                    m, ch, ca = chain[l - 1]
                    We = (m.W_enc[si] / ch).double(); bpre = (m.b_pre[si] * ch).double(); be = m.b_enc[si].double()
                    Wd = (m.W_dec[si] * ca).double(); bd = (m.b_dec[si] * ca).double(); th = m.theta[si].double()
                    mu_a, S_a = closure_layer(mu_h, S_h, We, be, bpre, th, Wd, bd, args.hermite)
                    if l in OUTL:
                        clo[l] = score(mu_a, l, law)
                        T = LAW[law]
                        cvc[l] = float(((mu_a - chain_mean[l]) ** 2).mean() / T['tvar'][l - 1].mean()) - 1.0 / NC
                    mu_h, S_h = mu_a, S_a + net.noise_sd(l) ** 2 * I
            row['chain_closure'][law] = clo; row['closure_vs_chain'][law] = cvc
            log(f"F={F} k={k} law={law}: chain sampled " + ' '.join(f"L{L}:{v:.3e}" for L, v in row['chain_sampled'][law].items())
                + "; chain closure " + ' '.join(f"L{L}:{v:.3e}" for L, v in clo.items())
                + "; closure vs own chain " + ' '.join(f"L{L}:{v:.3e}" for L, v in cvc.items()))
        res['settings'].append(row)
        log(f"F={F} k={k}: path KL (l=1..31) {row['path_kl_bits_l1_31']:.4g} bits")
    json.dump(res, open(os.path.join(OUT, 'e4.json'), 'w'), indent=1)
    del chain, H
json.dump(res, open(os.path.join(OUT, 'e4.json'), 'w'), indent=1)
log(f"E4 done {time.time() - T0:.0f}s")
