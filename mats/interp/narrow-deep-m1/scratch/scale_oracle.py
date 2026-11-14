"""Scaling of the structure with width (coordinator's scaling arm, oracle only): plain nets, width N, depth L = 32 N.
usage: python scale_oracle.py N SEED SIGMA OUTJSON
- downstream-distinct classes (noiseless future gate patterns, 2^14 inputs; 99%-mass counts) at cuts 1..L/4;
- k*_F(S_c; 1e-2) and k*_F(M_c; 1e-2) at cuts (Jacobians of 2^11 noisy paths);
- projected meet (top-k S directions, dropped part by regression; input cut: dropped part 0), nMSE, k in {1,2,4,8,..};
- ARC kprop K = 2 and K2X: nMSE of E[a_c], c = 1..8 (truth: 2^18 paths);
- GHM(1e-3): components, cost (paths of this net) and nMSE of E[h_c] at c = 1..4.
"""
import sys, os, json, math, time
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import nd, k2, ghm

torch.set_num_threads(1)
N, SEED, SIG, OUT = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
L = 32 * N
DT = nd.DT
t0 = time.time()
net = nd.Net('plain', SEED, N, L, SIG)
g = torch.Generator().manual_seed(17_000_001 + SEED + 1000 * N)
res = dict(n=N, L=L, seed=SEED, sigma=SIG, flops_path=net.flops_path())
CUTS = sorted(set([c for c in (0, 2, 4, 8, 16, 32, 64, 128, 256) if c <= L // 4]))


def eps(l, P):
    gg = torch.Generator().manual_seed(23_000_003 + 7919 * SEED + l + 100000 * N)
    return torch.randn(P, N, generator=gg, dtype=DT)


def run_from(h, l_from):
    for l in range(l_from, L):
        a = torch.relu(h @ net.W[l].T)
        h = a + eps(l, h.shape[0]) * net.s[l] if (l < L - 1 and SIG > 0) else a
    return h


# truth-ish statistics of the layers (2^18 paths) and noiseless dead fraction
P = 1 << 14
x = torch.randn(P, N, generator=g, dtype=DT)
h = x; Hc = {0: x.clone()}
for l in range(L):
    a = torch.relu(h @ net.W[l].T)
    h = a + eps(l, P) * net.s[l] if (l < L - 1 and SIG > 0) else a
    if (l + 1) in CUTS:
        Hc[l + 1] = h.clone()
out = h
mean = out.mean(0); trv = float(((out - mean) ** 2).sum(1).mean())
res['trvar'] = trv
# projected meet and k*_F
pm = {}
for c in CUTS:
    if c >= L:
        continue
    hc = Hc[c]
    sub = hc[:1 << 10]
    Ms = torch.zeros(N, N, dtype=DT)
    for hv in sub:
        hh = hv.clone(); J = torch.eye(N, dtype=DT)
        for l in range(c, L):
            z = net.W[l] @ hh; D = (z > 0).to(DT); hh = torch.relu(z); J = D[:, None] * (net.W[l] @ J)
            nj = J.norm()
            if nj > 0: J = J / nj
        Ms += J.T @ J
    Ms = Ms / Ms.trace().clamp_min(1e-300)
    C = torch.cov(hc.T) if c > 0 else torch.eye(N, dtype=DT)
    ev, U = torch.linalg.eigh(0.5 * (C + C.T)); Ch = U @ torch.diag(ev.clamp_min(0).sqrt()) @ U.T
    S = Ch @ Ms @ Ch; eS, VS = torch.linalg.eigh(0.5 * (S + S.T)); eS, VS = eS.flip(0), VS.flip(1)
    eM = torch.linalg.eigvalsh(Ms).flip(0)
    V = torch.linalg.qr(Ch @ VS)[0]
    row = dict(kS=nd.kstar_F(eS.numpy(), 1e-2), kM=nd.kstar_F(eM.numpy(), 1e-2), kS3=nd.kstar_F(eS.numpy(), 1e-3), proj={})
    for k in [kk for kk in (1, 2, 4, 8, 12, 16) if kk < N]:
        Uk = V[:, :k]; y = hc @ Uk
        if c == 0:
            ht = y @ Uk.T
        else:
            perp = hc - y @ Uk.T
            X = torch.cat([y, torch.ones(P, 1, dtype=DT)], 1)
            B = torch.linalg.lstsq(X, perp).solution
            ht = y @ Uk.T + X @ B
        o2 = run_from(ht, c)
        d = (o2 - out).mean(0)
        row['proj'][k] = float(d.pow(2).sum() / trv)
    pm[c] = row
    print(f"n={N} c={c}: kS(1e-2)={row['kS']} kM={row['kM']} kS(1e-3)={row['kS3']} proj={ {k: f'{v:.2g}' for k, v in row['proj'].items()} } ({time.time()-t0:.0f}s)", flush=True)
res['cuts'] = pm
# classes (noiseless)
net0 = nd.Net('plain', SEED, N, L, 0.0)
P2 = 1 << 14
h = x[:P2]; codes = []
pw = torch.tensor([1 << i for i in range(N)], dtype=torch.long) if N <= 62 else None
MOD = (1 << 61) - 1
for l in range(L):
    z = h @ net0.W[l].T
    codes.append(((z > 0).long() * pw).sum(1) % MOD)
    h = torch.relu(z)
w0 = h.norm(dim=1)
codes = torch.stack(codes, 1)
cls = {}
for c in [cc for cc in CUTS if 0 < cc < L]:
    k = torch.zeros(P2, dtype=torch.long)
    for j in range(c, L):
        k = (k * 1_000_003 + codes[:, j] * 7_919 + j) % MOD
    u, inv, cnt = torch.unique(k, return_inverse=True, return_counts=True)
    ws = torch.zeros(len(u), dtype=DT).index_add_(0, inv, w0).sort(descending=True).values
    cum = torch.cumsum(ws, 0) / ws.sum().clamp_min(1e-300)
    cls[c] = dict(distinct=int(len(u)), w99=int((cum < 0.99).sum()) + 1)
print('classes', cls, flush=True)
res['classes'] = cls
# kprop K=2 and K2X vs sampled E[a_c], c = 1..8; GHM(1e-3) at c = 1..4
import kprop_noisy as KN
h = torch.randn(1 << 18, N, generator=g, dtype=DT); tm = []; tv = []
for l in range(8):
    a = torch.relu(h @ net.W[l].T); tm.append(a.mean(0)); tv.append(float(a.var(0).sum()))
    h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if SIG > 0 else a
means, fl_kp, _ = KN.kprop_noisy(net, 8, 2, keep=True)
res['kp2_err'] = [float((means[c] - tm[c]).pow(2).sum() / tv[c]) for c in range(8)]
mu = torch.zeros(N, dtype=DT); C = torch.eye(N, dtype=DT); k2e = []
for l in range(8):
    m = net.W[l] @ mu; Cz = net.W[l] @ C @ net.W[l].T
    mu, C, _ = k2.relu_mom(m, Cz, True)
    k2e.append(float((mu - tm[l]).pow(2).sum() / tv[l]))
    if SIG > 0: C = C + torch.eye(N, dtype=DT) * float(net.s[l]) ** 2
res['k2x_err'] = k2e
print('kp2', [f'{e:.2g}' for e in res['kp2_err']], 'k2x', [f'{e:.2g}' for e in k2e], flush=True)
try:
    outg, _ = ghm.ghm_forward(net, 4, 1e-3, max_comp=400_000)
    gr = {}
    for c in range(1, 5):
        w, mu_, C_, fc = outg[c]
        gr[c] = dict(ncomp=int(len(w)), cost_paths=fc / net.flops_path(),
                     err_a=float((ghm.mixture_mean(w, mu_) - tm[c - 1]).pow(2).sum() / tv[c - 1]))
    res['ghm'] = gr
    print('ghm', gr, flush=True)
except Exception as e:
    res['ghm'] = dict(error=str(e)[:200])
res['sec'] = time.time() - t0
json.dump(res, open(OUT, 'w'), indent=1)
