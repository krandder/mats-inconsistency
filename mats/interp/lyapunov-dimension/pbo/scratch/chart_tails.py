"""First-order tails in the families' own chart (answers GPT-6's finding 3): the harness's sphere chart, the
rho-weighted covariance C of w = z/|z| (mean direction projected out, P C P), the harness's sensitivity M (ng2.sensitivity,
2^14 paths), and the blocks exactly as built by the families: PB_k (top-k weighted PCs of w), PBW_k (orth(R V_k)),
PBO_k (orth(Pi M R V_k)). tail(U) = tr(M Schur_U) / tr(M P C P), Schur_U the residual covariance of P(w - m) after linear
regression on U^T (w - m).
python chart_tails.py --nets 113,114,115,116 --cuts 32,48 [--nfit 18]"""
import argparse, json, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import bf, ng, ng2, ng3

ap = argparse.ArgumentParser()
ap.add_argument('--nets', default='113,114,115,116'); ap.add_argument('--cuts', default='32,48')
ap.add_argument('--nfit', type=int, default=18); ap.add_argument('--nsm', type=int, default=14)
ap.add_argument('--threads', type=int, default=2)
A = ap.parse_args()
torch.set_num_threads(A.threads)
dev = torch.device('cpu')
KS = [8, 16, 24, 32, 48, 64]
out = {}
for seed in [int(s) for s in A.nets.split(',')]:
    net = bf.get_net(seed, 256, 64, 0.001); dn = ng.DNet(net, dev)
    g = torch.Generator().manual_seed(7700 + seed)
    N = 1 << A.nfit
    X = torch.randn(N, 256, generator=g)
    for l in range(1, max(int(c) for c in A.cuts.split(',')) + 1):
        Z = X @ dn.W[l - 1].T
        if l in [int(c) for c in A.cuts.split(',')]:
            rho, Wd = ng.sphere(Z)
            m, C = ng.mean_cov(Wd, rho)
            u = m / m.norm(); P = torch.eye(256, dtype=ng.F64) - torch.outer(u, u)
            PCP = P @ C @ P
            M = ng2.sensitivity(dn, Z, l, 64, torch.Generator().manual_seed(8800 + seed + l), ns=1 << A.nsm)
            den = float(torch.trace(M @ PCP))
            ev, V = ng.eig_desc(C)

            def tail(U):
                Cyy = U.T @ C @ U; Cry = C @ U
                Sch = C - Cry @ torch.linalg.pinv(Cyy) @ Cry.T
                Sch = P @ Sch @ P
                return float(torch.trace(M @ Sch)) / den
            rec = {}
            for k in KS:
                rec[f'PB{k}'] = tail(V[:, :k])
                rec[f'PBW{k}'] = tail(ng3.blocks_from_M(Wd, rho, M, k, 'W'))
                rec[f'PBO{k}'] = tail(ng3.blocks_from_M(Wd, rho, M, k, 'O'))
            out[f'{l}/{seed}'] = rec
            print(f"net {seed} cut {l}: " + "; ".join(f"{f} " + " ".join(f"{rec[f'{f}{k}']:.3f}" for k in KS)
                                                     for f in ('PB', 'PBW', 'PBO')), flush=True)
        X = torch.relu(Z) + dn.s[l - 1] * torch.randn(N, 256, generator=g)
json.dump(out, open(os.path.join(HERE, '..', 'runs', 'chart_tails.json'), 'w'), indent=1)
