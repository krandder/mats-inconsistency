"""E3 rerun for the dense latents only (TRUE, G, DENSE(k), DENSEQ(k)), after the late-binding bug in run_main2.py v2:
its DENSE/DENSEQ closures read `mu` when called, so in E3 (after the layer loop) they used layer 31's mean at every
cut. Same data streams, fits and common random numbers as run_main2.py, so TRUE and G reproduce its E3 rows exactly.
usage: run_e3dense.py OUT N SIGMA SEED [--n3 20] [--ntruth 24]   -> OUT/<net>/e3dense.json"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import nets   # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('out'); p.add_argument('n', type=int); p.add_argument('sigma', type=float); p.add_argument('seed', type=int)
p.add_argument('--nfit', type=int, default=18)
p.add_argument('--n3', type=int, default=20)
p.add_argument('--ntruth', type=int, default=24)
args = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False
net = nets.SNet('arc', args.n, args.seed, args.sigma, device=dev)
n = args.n
OUT = os.path.join(args.out, net.name()); os.makedirs(OUT, exist_ok=True)
layers = [1, 2, 4, 8, 16, 24, 31]
cuts = [4, 8, 16, 24]
KD = sorted(set([1, 2, 4, 8, 16, 32, 64, n // 4, n // 2]) & set(range(1, n)))


def gen(s):
    return torch.Generator(device=dev).manual_seed(int(s))


def quad_feats(y):
    k = y.shape[1]
    iu = torch.triu_indices(k, k, device=y.device)
    return torch.cat([y, (y[:, :, None] * y[:, None, :])[:, iu[0], iu[1]]], 1)


def pinv_psd(C, rel=1e-10):
    lam, V = torch.linalg.eigh(C); keep = lam > lam.max() * rel
    return (V[:, keep] / lam[keep][None]) @ V[:, keep].T


def sqrt_psd(C):
    lam, U = torch.linalg.eigh(0.5 * (C + C.T))
    return U * lam.clamp_min(0).sqrt()[None, :]


T0 = time.time()
data_seed = 1_000 * args.seed + 17
NFIT = 1 << args.nfit
XF = net.sample(NFIT, gen(data_seed + 11), cuts, upto=max(layers))  # = run_main2's FIT set (same stream, cut layers kept)
NT = 1 << args.ntruth
S1 = torch.zeros(net.L, n, dtype=torch.float64, device=dev); S2 = torch.zeros_like(S1)
g = gen(data_seed + 7)
CHT = min(1 << 16, NT)
for i0 in range(0, NT, CHT):
    h = torch.randn(CHT, n, generator=g, device=dev)
    for l in range(1, net.L + 1):
        a = net.act(l, net.pre(l, h)); ad = a.double(); S1[l - 1] += ad.sum(0); S2[l - 1] += (ad * ad).sum(0)
        h = a + net.noise_sd(l) * torch.randn(a.shape, generator=g, device=dev)
tmean = S1 / NT; tvar = S2 / NT - tmean ** 2
print(f"== {net.name()} e3dense truth {NT} ({time.time() - T0:.0f}s)", flush=True)
OUTS = {4: [8], 8: [16, 32], 16: [32], 24: [32]}
N3 = 1 << args.n3; CH3 = min(1 << 16, N3)
rows = []
for m in cuts:
    AF = XF[m].double(); mu = AF.mean(0); CF = AF - mu
    Sig = CF.T @ CF / NFIT
    del CF
    lam, U = torch.linalg.eigh(Sig); lam = lam.flip(0).clamp_min(0); U = U.flip(1)
    MOD = {}
    for k in KD:
        Uk = U[:, :k]
        MOD[f'DENSE{k}'] = (lambda A, Uk=Uk, mu=mu: mu + ((A - mu) @ Uk) @ Uk.T, (U[:, k:] * lam[k:][None]) @ U[:, k:].T)
        if k <= 16:
            y = (AF - mu) @ Uk; r = (AF - mu) - y @ Uk.T
            psi = quad_feats(y); mp = psi.mean(0); P_ = psi - mp
            B = (r.T @ P_ / NFIT) @ pinv_psd(P_.T @ P_ / NFIT, 1e-12)
            eres = r - P_ @ B.T; Se = eres.T @ eres / NFIT
            del y, r, psi, P_, eres
            MOD[f'DENSEQ{k}'] = (lambda A, Uk=Uk, mp=mp, B=B, mu=mu: mu + ((A - mu) @ Uk) @ Uk.T + (quad_feats((A - mu) @ Uk) - mp) @ B.T,
                                 0.5 * (Se + Se.T))
    del AF
    torch.cuda.empty_cache() if dev == 'cuda' else None
    outs = [L for L in OUTS[m] if L > m]
    names = ['TRUE', 'G'] + list(MOD)
    acc1 = {k_: {L: torch.zeros(n, dtype=torch.float64, device=dev) for L in outs} for k_ in names}
    acc2 = {k_: {L: torch.zeros(n, dtype=torch.float64, device=dev) for L in outs} for k_ in names}
    LG = sqrt_psd(Sig).float()
    LR = {k_: sqrt_psd(v[1]).float() for k_, v in MOD.items()}
    t = time.time()
    for ci in range(N3 // CH3):
        base = net.sample(CH3, gen(data_seed + 9000 + 1000 * m + ci), [m])[m]
        xi = torch.randn(CH3, n, generator=gen(data_seed + 19000 + 1000 * m + ci), device=dev)
        based = base.double()
        for nm_ in names:
            if nm_ == 'TRUE':
                a = base
            elif nm_ == 'G':
                a = (mu + xi.double() @ LG.double().T).float()
            else:
                a = MOD[nm_][0](based).float() + xi @ LR[nm_].T
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
            rows.append(dict(cut=m, L=L, model=nm_, nmse=err, raw=raw))
    print(f"cut {m} ({time.time() - t:.0f}s): " + ' '.join(f"{d['model']}@{d['L']}:{d['nmse']:.1e}" for d in rows if d['cut'] == m), flush=True)
json.dump(dict(net=net.name(), N3=N3, NT=NT, rows=rows), open(os.path.join(OUT, 'e3dense.json'), 'w'), indent=1)
print(f"E3DENSE DONE {time.time() - T0:.0f}s", flush=True)
