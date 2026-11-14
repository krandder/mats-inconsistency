"""Amendment 2, test T5 (Kelvin, 29 Oct ~18:00Z): where does the downstream amplification live?
At a cut l of a net (n, L), for output layers Lo in {l+4, l+8, l+16, L} (those <= L):
  M = E[J^T J], J = d a_Lo / d z_l (per path, with the path's noise), estimated as E[g g^T], g = J^T r, r ~ N(0, I);
  C = Cov(z_l); P = I - u u^T with u the mean direction (the radial direction is exact by homogeneity);
  S = (P C P)^1/2 M (P C P)^1/2  (sensitivity weighted by the state's own spread, radial direction removed).
Reported per (net, cut, Lo): for S and for P M P: participation ratio, k90 and k99 (number of eigen-directions holding
90% / 99% of the trace), and, for S, the share of trace in its top 1/4/16/64 directions. Two independent halves of
the paths give the estimates' stability (k90 / k99 of each half).
python scratch/amp.py --n 256 --L 64 --nets 113,114 --cuts 32,48 [--ns 16] --out runs/a2"""
import argparse, json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import bf, ng

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=256); ap.add_argument('--L', type=int, default=64)
ap.add_argument('--nets', required=True); ap.add_argument('--cuts', required=True)
ap.add_argument('--ns', type=int, default=16); ap.add_argument('--nc', type=int, default=18)
ap.add_argument('--sigma', type=float, default=0.001); ap.add_argument('--out', default='runs/a2')
ap.add_argument('--threads', type=int, default=int(os.environ.get('TH', '3')))
A = ap.parse_args()
torch.set_num_threads(A.threads)
dev = ng.device()
if dev.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False
os.chdir(os.path.join(HERE, '..')); os.makedirs(A.out, exist_ok=True)
NS, NC = 1 << A.ns, 1 << A.nc


def spec_stats(S):
    ev = torch.linalg.eigvalsh(0.5 * (S + S.T)).flip(0).clamp_min(0)
    tr = ev.sum(); c = torch.cumsum(ev, 0) / tr
    k90 = int((c < 0.9).sum()) + 1; k99 = int((c < 0.99).sum()) + 1
    pr = float(tr ** 2 / (ev ** 2).sum())
    top = {t: float(c[t - 1]) for t in (1, 4, 16, 64) if t <= len(ev)}
    return dict(pr=pr, k90=k90, k99=k99, top=top, trace=float(tr))


def psd_sqrt(C):
    ev, V = torch.linalg.eigh(0.5 * (C + C.T))
    return (V * ev.clamp_min(0).sqrt()) @ V.T


@torch.no_grad()
def paths_at(dn, N, l, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    out = torch.empty(N, dn.n, device=dev); done = 0
    while done < N:
        b = min(ng.CH, N - done)
        h = torch.randn(b, dn.n, generator=g, device=dev)
        for k in range(1, l + 1):
            z = h @ dn.W[k - 1].T
            if k == l:
                break
            h = torch.relu(z) + dn.s[k - 1] * torch.randn(b, dn.n, generator=g, device=dev)
        out[done:done + b] = z; done += b
    return out


def sens(dn, Z, l, Lo, seed):
    """E[g g^T] over the rows of Z, g = J^T r, with fresh noise and r per path; returns M (float64) per half."""
    g = torch.Generator(device=dev).manual_seed(seed)
    Ms = []
    half = Z.shape[0] // 2
    for part in (Z[:half], Z[half:]):
        M = torch.zeros(dn.n, dn.n, dtype=ng.F64, device=dev)
        for a in range(0, part.shape[0], 4096):
            z = part[a:a + 4096].clone().requires_grad_(True)
            with torch.enable_grad():
                h = z
                for k in range(l, Lo + 1):
                    if k > l:
                        h = h @ dn.W[k - 1].T
                    h = torch.relu(h)
                    if k < Lo:
                        h = h + dn.s[k - 1] * torch.randn(h.shape, generator=g, device=dev)
                r = torch.randn(h.shape, generator=g, device=dev)
                (h * r).sum().backward()
            G = z.grad.double(); M += G.T @ G
        Ms.append(M / part.shape[0])
    return Ms


res = []
for seed in [int(s) for s in A.nets.split(',')]:
    net = bf.get_net(seed, A.n, A.L, A.sigma); dn = ng.DNet(net, dev)
    for l in [int(c) for c in A.cuts.split(',')]:
        t0 = time.time()
        Zc = paths_at(dn, NC, l, 150_000 + seed)
        m, C = ng.mean_cov(Zc)
        u = m / m.norm(); P = torch.eye(dn.n, dtype=ng.F64, device=dev) - torch.outer(u, u)
        R = psd_sqrt(P @ C @ P)
        Zs = paths_at(dn, NS, l, 160_000 + seed)
        for Lo in sorted(set(x for x in (l + 4, l + 8, l + 16, A.L) if x <= A.L)):
            M1, M2 = sens(dn, Zs, l, Lo, 170_000 + 31 * seed + Lo)
            M = 0.5 * (M1 + M2)
            rec = dict(seed=seed, n=A.n, L=A.L, cut=l, Lo=Lo,
                       S=spec_stats(R @ M @ R), PMP=spec_stats(P @ M @ P),
                       S_halves=[spec_stats(R @ Mh @ R) for Mh in (M1, M2)],
                       C_pr=spec_stats(P @ C @ P)['pr'])
            res.append(rec)
            s = rec['S']
            print(f"n {A.n} net {seed} cut {l} Lo {Lo}: S pr {s['pr']:.1f} k90 {s['k90']} k99 {s['k99']} "
                  f"(halves k90 {rec['S_halves'][0]['k90']}/{rec['S_halves'][1]['k90']}, k99 "
                  f"{rec['S_halves'][0]['k99']}/{rec['S_halves'][1]['k99']}) top1/4/16 "
                  f"{s['top'][1]:.2f}/{s['top'][4]:.2f}/{s['top'][16]:.2f} | PMP pr {rec['PMP']['pr']:.1f} "
                  f"k90 {rec['PMP']['k90']} k99 {rec['PMP']['k99']} | state PR {rec['C_pr']:.1f}  {time.time() - t0:.0f}s",
                  flush=True)
        json.dump(res, open(f'{A.out}/amp_{A.n}_{A.L}.json', 'w'), indent=1)
print('done', flush=True)
