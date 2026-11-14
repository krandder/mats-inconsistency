"""Latent description size at noise precision, for the information-matched comparison asked by the design critique
(finding 7): sparse latent bits = H(support) [upper bound: sum of per-feature binary entropies]
+ sum_j p_j * 1/2 log2(1 + Var(f_j | f_j > 0) / s^2); dense latent bits = sum_{i<=k} 1/2 log2(1 + lambda_i / s^2).
Uses saved SAEs (v2 runs) at layers 8/16/24. usage: latent_bits.py RUNDIR N SEED [layers]"""
import glob, json, math, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import nets, sae
torch.set_num_threads(int(os.environ.get('NT', '2')))
rundir, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
layers = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [8, 16, 24]
net = nets.SNet('arc', n, seed, 0.01)
X = net.sample(1 << 15, torch.Generator().manual_seed(98765), layers)
out = []
for l in layers:
    A = X[l].double(); s = net.noise_sd(l); s2 = s * s
    lam = torch.linalg.eigvalsh(torch.cov(A.T)).flip(0).clamp_min(0)
    cum = torch.cumsum(0.5 * torch.log2(1 + lam / s2), 0)
    dense = {int(k): float(cum[k - 1]) for k in (1, 2, 4, 8, 16, 32, 64, n // 4, n // 2) if k <= n}
    out.append(dict(layer=l, kind='dense', bits=dense))
    print(l, 'dense', {k: round(v, 1) for k, v in dense.items()}, flush=True)
    for p in sorted(glob.glob(os.path.join(rundir, f'sae_L{l}_*F*.pt'))):
        ck = torch.load(p)
        F = ck['F']; origin = ck.get('origin', 'mean')
        model = sae.BatchedTopK(n, F, ck['ks'], torch.zeros(n), 'cpu', origin=origin)
        model.load_state_dict({k: v.float() for k, v in ck['state'].items()})
        with torch.no_grad():
            f, _, _, _ = model.encode(X[l] / ck['c'])
        for si, k in enumerate(ck['ks']):
            fz = (f[si] * ck['c']).double()
            on = fz > 0
            pj = on.double().mean(0)
            h = -(pj * torch.log2(pj.clamp_min(1e-30)) + (1 - pj) * torch.log2((1 - pj).clamp_min(1e-30)))
            cnt = on.double().sum(0).clamp_min(1)
            m1 = fz.sum(0) / cnt; m2 = (fz * fz).sum(0) / cnt
            vj = (m2 - m1 * m1).clamp_min(0)
            amp = float((pj * 0.5 * torch.log2(1 + vj / s2)).sum())
            row = dict(layer=l, kind='sparse', origin=origin, F=F, k=k, H_support_indep=float(h.sum()),
                       log2_binom=(math.lgamma(F + 1) - math.lgamma(k + 1) - math.lgamma(F - k + 1)) / math.log(2),
                       amp_bits=amp, total_bits=float(h.sum()) + amp, L0=float(on.double().sum(1).mean()))
            out.append(row)
            print(l, origin, F, k, {kk: round(vv, 1) for kk, vv in row.items() if isinstance(vv, float)}, flush=True)
json.dump(out, open(os.path.join(HERE, '..', 'results', f'latent_bits_n{n}_seed{seed}.json'), 'w'), indent=1)
