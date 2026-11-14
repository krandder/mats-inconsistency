"""Amendment 2, test T5b (Kelvin via main, 29 Oct ~18:10Z): finite-time Lyapunov spectra of the downstream Jacobian.
For a path at cut l, J(l, L') = d a_{l+L'} / d z_l = D_{l+L'} W_{l+L'} ... D_{l+1} W_{l+1} D_l (D = gate masks of that
path, noise included in the path). Per path: singular values s_1 >= s_2 >= ... of J (float64, renormalised every
layer). Reported per (n, net, cut, L'), averaged over paths:
- lambda_k - lambda_1 = (log s_k - log s_1) / L' for k = 1..min(n, 128) (the finite-time Lyapunov gaps per layer);
- N_surv(tau) = number of k with s_k / s_1 >= tau, tau = e^-1, e^-2, e^-3 (directions that survive the horizon);
- the rank (number of s_k > 1e-12 s_1) and PR of the s_k^2;
- the Gaussian-product prediction 2n/L' (Newman 1986: gaps about (k-1)/(2n) per layer) for comparison.
python scratch/lyap.py --n 256 --L 64 --nets 113,114 --cuts 16,32,48 --Lps 4,8,16,32 [--paths 128] --out runs/a2"""
import argparse, json, math, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import bf, ng

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=256); ap.add_argument('--L', type=int, default=64)
ap.add_argument('--nets', required=True); ap.add_argument('--cuts', required=True); ap.add_argument('--Lps', required=True)
ap.add_argument('--paths', type=int, default=128); ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--sigma', type=float, default=0.001); ap.add_argument('--out', default='runs/a2')
ap.add_argument('--threads', type=int, default=int(os.environ.get('TH', '3')))
A = ap.parse_args()
torch.set_num_threads(A.threads)
dev = ng.device()
if dev.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False
os.chdir(os.path.join(HERE, '..')); os.makedirs(A.out, exist_ok=True)
Lps = [int(x) for x in A.Lps.split(',')]
res = []
for seed in [int(s) for s in A.nets.split(',')]:
    net = bf.get_net(seed, A.n, A.L, A.sigma); dn = ng.DNet(net, dev)
    W64 = [w.double() for w in dn.W]
    for l in [int(c) for c in A.cuts.split(',')]:
        t0 = time.time()
        g = torch.Generator(device=dev).manual_seed(180_000 + 97 * seed + l)
        acc = {Lp: [] for Lp in Lps if l + Lp <= A.L}
        done = 0
        while done < A.paths:
            B = min(A.batch, A.paths - done)
            with torch.no_grad():
                h = torch.randn(B, dn.n, generator=g, device=dev)
                for k in range(1, l + 1):
                    z = h @ dn.W[k - 1].T
                    if k < l:
                        h = torch.relu(z) + dn.s[k - 1] * torch.randn(B, dn.n, generator=g, device=dev)
                J = torch.diag_embed((z > 0).double())                         # D_l, (B, n, n)
                logscale = torch.zeros(B, dtype=ng.F64, device=dev)
                h = torch.relu(z) + dn.s[l - 1] * torch.randn(B, dn.n, generator=g, device=dev)
                for k in range(l + 1, max(acc) + l + 1):
                    z = h @ dn.W[k - 1].T
                    J = (z > 0).double()[:, :, None] * (W64[k - 1] @ J)
                    nrm = J.flatten(1).norm(dim=1).clamp_min(1e-300)
                    J = J / nrm[:, None, None]; logscale += nrm.log()
                    if k - l in acc:
                        s = torch.linalg.svdvals(J)                             # (B, n), descending
                        acc[k - l].append(torch.cat([logscale[:, None], s], 1).cpu())
                    if k < l + max(acc):
                        h = torch.relu(z) + dn.s[k - 1] * torch.randn(B, dn.n, generator=g, device=dev)
            done += B
        for Lp, parts in acc.items():
            T = torch.cat(parts, 0); s = T[:, 1:]
            ratio = s / s[:, :1]
            lg = torch.log(ratio.clamp_min(1e-300)) / Lp
            kk = min(A.n, 128)
            rec = dict(seed=seed, n=A.n, L=A.L, cut=l, Lp=Lp, paths=int(T.shape[0]),
                       gap_per_layer=[float(x) for x in lg[:, :kk].mean(0)],
                       n_surv={str(t): float((ratio >= math.exp(-t)).double().sum(1).mean()) for t in (1, 2, 3)},
                       rank=float((ratio > 1e-12).double().sum(1).mean()),
                       pr=float(((s ** 2).sum(1) ** 2 / (s ** 4).sum(1)).mean()),
                       lambda1=float(((T[:, 0] + s[:, 0].log()) / Lp).mean()),
                       pred_gauss=2 * A.n / Lp)
            res.append(rec)
            print(f"n {A.n} net {seed} cut {l} L' {Lp}: N_surv e^-1/e^-2/e^-3 {rec['n_surv']['1']:.1f}/"
                  f"{rec['n_surv']['2']:.1f}/{rec['n_surv']['3']:.1f} (Gaussian-product 2n/L' {2 * A.n / Lp:.0f}); "
                  f"PR {rec['pr']:.2f}; rank {rec['rank']:.0f}; gap k=2,4,8,16: "
                  + ' '.join(f"{rec['gap_per_layer'][k - 1]:+.3f}" for k in (2, 4, 8, 16) if k <= kk)
                  + f"; lambda1 {rec['lambda1']:+.3f}  {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(f'{A.out}/lyap_{A.n}_{A.L}.json', 'w'), indent=1)
print('done', flush=True)
