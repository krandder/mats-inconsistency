"""Oracle gate mixture (added after review 2, finding 1; not pre-registered).
usage: python p5_oracle.py KIND NPZ LOCATEJSON OUTJSON [LOG2_PATHS=20]
The split closure of p5_split re-Gaussianises each piece before the next split, so it may lose what an accurately
conditioned gate mixture would keep. This oracle removes that: after l0 layers (or steps), condition the sampled
paths on the joint on/off pattern of the k units of the last layer with the largest one-step gate error there
(k = 1, 2, 4, 8), take each pattern's EXACT conditional mean and covariance of the state (sampled), run K2 (exact
ReLU moments) from each component to the output, and mix with the exact pattern probabilities.
Compared, on the same paths, with the single Gaussian restart (exact pooled moments, then K2: E_out(l0) of
p5_locate) and with K2 from the input. Patterns with fewer than MINC paths are pooled into one component (exact
moments of their union), so the mixture's total mass is exact."""
import sys, os, json, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import tnets, k2

torch.set_num_threads(1)
DT = torch.float64
KIND, NPZ, LOC, OUT = sys.argv[1:5]
LOG2 = int(sys.argv[5]) if len(sys.argv) > 5 else 20
MINC = 64
net = tnets.RNNNet(NPZ) if KIND == 'rnn' else tnets.MLPNet(NPZ)
depth = net.T if KIND == 'rnn' else net.L
w = net.d if KIND == 'rnn' else net.n
nin = net.n
loc = json.load(open(LOC))
p_one = torch.tensor(loc['p_one'], dtype=DT); p_true = torch.tensor(loc['p_true'], dtype=DT)
E1 = (p_one - p_true).abs()                                    # (depth, w) one-step gate error
L0S = [l0 for l0 in (2, 4, 8, 16, 32, 48, 56) if l0 < depth]
UNITS = {l0: [int(i) for i in E1[l0 - 1].argsort(descending=True)[:8]] for l0 in L0S}
K8 = 8
t0 = time.time()
g = torch.Generator().manual_seed(61_000_029)

cnt = {l0: torch.zeros(1 << K8, dtype=DT) for l0 in L0S}
s1 = {l0: torch.zeros(1 << K8, w, dtype=DT) for l0 in L0S}
s2 = {l0: torch.zeros(1 << K8, w, w, dtype=DT) for l0 in L0S}
So = 0; Soo = 0; NT = 0
CH = 1 << 15
bits = (2 ** torch.arange(K8)).to(torch.long)
for it in range(1 << max(LOG2 - 15, 0)):
    x = torch.randn(CH, nin, generator=g, dtype=DT)
    h = torch.zeros(CH, w, dtype=DT) if KIND == 'rnn' else x
    for l in range(depth):
        if KIND == 'rnn':
            z = h @ net.W_hh.T + x[:, l:l + 1] * net.W_hi
            a = torch.relu(z)
            h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if net.sigma > 0 else a
        else:
            z = h @ net.W[l].T
            a = torch.relu(z)
            e = torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if net.sigma > 0 else 0.0
            h = h + (a + e) @ net.V[l].T
        l0 = l + 1
        if l0 in cnt:
            pat = ((z[:, UNITS[l0]] > 0).to(torch.long) * bits).sum(1)
            cnt[l0].index_add_(0, pat, torch.ones(CH, dtype=DT))
            s1[l0].index_add_(0, pat, h)
            s2[l0].index_add_(0, pat, h[:, :, None] * h[:, None, :])
    out = h @ net.W_oh.T if KIND == 'rnn' else h
    So = So + out.sum(0); Soo = Soo + (out * out).sum(0); NT += CH
mo = So / NT; trv = float((Soo / NT - mo ** 2).sum())


def outmean(mu):
    return net.W_oh @ mu if KIND == 'rnn' else mu


def k2_from(mu, C, l0):
    if KIND == 'rnn':
        m, _, _ = k2.rnn_prop(net, mu, C, l0, depth, True)
    else:
        m, _, _ = k2.res_prop(net, mu, C, l0, depth, True)
    return outmean(m)


def moments(c, a, b):
    mu = a / c
    C = b / c - torch.outer(mu, mu)
    return mu, 0.5 * (C + C.T)


if KIND == 'rnn':
    mu0, _, _ = k2.rnn_prop(net, torch.zeros(w, dtype=DT), torch.zeros(w, w, dtype=DT), 0, depth, True)
else:
    mu0, _, _ = k2.res_prop(net, torch.zeros(nin, dtype=DT), torch.eye(nin, dtype=DT), 0, depth, True)
k2_in = float(((outmean(mu0) - mo) ** 2).sum() / trv)
res = dict(npz=NPZ, kind=KIND, paths=NT, trvar=trv, k2_from_input=k2_in, minc=MINC, units=UNITS, results={})
print(f"{os.path.basename(NPZ)}: K2 from the input nMSE {k2_in:.4g}  ({time.time()-t0:.0f}s)", flush=True)
for l0 in L0S:
    c8, a8, b8 = cnt[l0], s1[l0], s2[l0]
    mu, C = moments(c8.sum(), a8.sum(0), b8.sum(0))
    e_single = float(((k2_from(mu, C, l0) - mo) ** 2).sum() / trv)
    row = dict(single=e_single, mix={})
    for k in (1, 2, 4, 8):
        # marginalise the 8-bit pattern onto the first k units (nested top-k sets)
        idx = torch.arange(1 << K8) % (1 << k)
        ck = torch.zeros(1 << k, dtype=DT).index_add_(0, idx, c8)
        ak = torch.zeros(1 << k, w, dtype=DT).index_add_(0, idx, a8)
        bk = torch.zeros(1 << k, w, w, dtype=DT).index_add_(0, idx, b8)
        big = ck >= MINC
        om = torch.zeros(len(mo), dtype=DT); ncomp = 0
        for p in torch.nonzero(big).flatten().tolist():
            m_p, C_p = moments(ck[p], ak[p], bk[p])
            om = om + (ck[p] / NT) * k2_from(m_p, C_p, l0); ncomp += 1
        small = (~big) & (ck > 0)
        if small.any():
            m_p, C_p = moments(ck[small].sum(), ak[small].sum(0), bk[small].sum(0))
            om = om + (ck[small].sum() / NT) * k2_from(m_p, C_p, l0); ncomp += 1
        e_mix = float(((om - mo) ** 2).sum() / trv)
        row['mix'][k] = dict(nmse=e_mix, ncomp=ncomp, gain_vs_single=e_single / e_mix, gain_vs_k2=k2_in / e_mix)
    res['results'][l0] = row
    print(f"  l0={l0}: single {e_single:.4g}; mixture k=1/2/4/8: " +
          ' '.join(f"{v['nmse']:.4g} ({v['gain_vs_single']:.2f}x vs single, {v['gain_vs_k2']:.2f}x vs K2, {v['ncomp']} comps)"
                   for v in row['mix'].values()) + f"  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(OUT, 'w'))
json.dump(res, open(OUT, 'w'))
