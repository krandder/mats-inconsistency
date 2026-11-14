"""Follow-up step 1: where does K2 go wrong on trained narrow nets? (PREREG Part 2)
usage: python p5_locate.py KIND NPZ OUTJSON [LOG2_PATHS=20]      (KIND = rnn | mlp; mlp = trained residual sorter)
Truth (sampled, noise on): per layer/step l and unit i the pre-activation mean, variance and gate-on frequency
P(z_li > 0); the state mean and covariance; the output mean.
K2 (exact ReLU moments) from the input: the same quantities, layer by layer ("accumulated").
One-step: K2 applied to the exact mean and covariance of the previous state ("local"); its predicted gate
probability is Phi(m/s) with the exact m, s of z_li, so the one-step gate error is the non-Gaussianity of z_li.
Suffix: K2 started from the exact Gaussian moment-match of the state at layer l0, run to the output: nMSE vs truth.
Concentration: share of the summed |gate error| in the worst k (layer, unit) pairs, per-layer and per-unit sums.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import tnets, k2
import gauss_relu as GR

torch.set_num_threads(1)
DT = torch.float64
KIND, NPZ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LOG2 = int(sys.argv[4]) if len(sys.argv) > 4 else 20
net = tnets.RNNNet(NPZ) if KIND == 'rnn' else tnets.MLPNet(NPZ)
depth = net.T if KIND == 'rnn' else net.L
w = net.d if KIND == 'rnn' else net.n
nin = net.n
t0 = time.time()
g = torch.Generator().manual_seed(41_000_003)

# ---------------- truth pass
Sz = torch.zeros(depth, w, dtype=DT); Szz = torch.zeros(depth, w, dtype=DT); Gon = torch.zeros(depth, w, dtype=DT)
Sh = torch.zeros(depth, w, dtype=DT); Shh = torch.zeros(depth, w, w, dtype=DT)
So = 0; Soo = 0; NT = 0
CH = 1 << 16
for it in range(1 << max(LOG2 - 16, 0)):
    x = torch.randn(CH, nin, generator=g, dtype=DT)
    if KIND == 'rnn':
        h = torch.zeros(CH, w, dtype=DT)
        for t in range(depth):
            z = h @ net.W_hh.T + x[:, t:t + 1] * net.W_hi
            a = torch.relu(z)
            h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[t] if net.sigma > 0 else a
            Sz[t] += z.sum(0); Szz[t] += (z * z).sum(0); Gon[t] += (z > 0).double().sum(0)
            Sh[t] += h.sum(0); Shh[t] += h.T @ h
        out = h @ net.W_oh.T
    else:
        h = x
        for l in range(depth):
            z = h @ net.W[l].T
            a = torch.relu(z)
            e = torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if net.sigma > 0 else 0.0
            h = h + (a + e) @ net.V[l].T
            Sz[l] += z.sum(0); Szz[l] += (z * z).sum(0); Gon[l] += (z > 0).double().sum(0)
            Sh[l] += h.sum(0); Shh[l] += h.T @ h
        out = h
    So = So + out.sum(0); Soo = Soo + (out * out).sum(0); NT += CH
mz_t = Sz / NT; vz_t = Szz / NT - mz_t ** 2; p_t = Gon / NT
mh_t = Sh / NT; Ch_t = Shh / NT - mh_t[:, :, None] * mh_t[:, None, :]
mo_t = So / NT; trv = float((Soo / NT - mo_t ** 2).sum())


def outmean(mu):
    return net.W_oh @ mu if KIND == 'rnn' else mu


def prop(mu, C, lo, hi):
    if KIND == 'rnn':
        return k2.rnn_prop(net, mu, C, lo, hi, True)
    return k2.res_prop(net, mu, C, lo, hi, True)


# ---------------- K2 accumulated, layer by layer
mu = torch.zeros(w if KIND == 'rnn' else nin, dtype=DT)
C = torch.zeros(w, w, dtype=DT) if KIND == 'rnn' else torch.eye(nin, dtype=DT)
mz_k = torch.zeros(depth, w, dtype=DT); vz_k = torch.zeros(depth, w, dtype=DT); state_err = []
for l in range(depth):
    if KIND == 'rnn':
        m = net.W_hh @ mu; Cz = net.W_hh @ C @ net.W_hh.T + torch.outer(net.W_hi, net.W_hi)
    else:
        m = net.W[l] @ mu; Cz = net.W[l] @ C @ net.W[l].T
    mz_k[l] = m; vz_k[l] = Cz.diagonal()
    mu, C, _ = prop(mu, C, l, l + 1)
    tr = float(torch.trace(Ch_t[l]))
    state_err.append(float(((mu - mh_t[l]) ** 2).sum() / max(tr, 1e-300)))
k2_out = float(((outmean(mu) - mo_t) ** 2).sum() / trv)
p_k = GR.ncdf(mz_k / vz_k.clamp_min(1e-300).sqrt())

# ---------------- one-step from exact moments
p_1 = GR.ncdf(mz_t / vz_t.clamp_min(1e-300).sqrt())
one_err = []
for l in range(depth):
    if l == 0:
        mu0 = torch.zeros(w if KIND == 'rnn' else nin, dtype=DT)
        C0 = torch.zeros(w, w, dtype=DT) if KIND == 'rnn' else torch.eye(nin, dtype=DT)
    else:
        mu0 = mh_t[l - 1]; C0 = Ch_t[l - 1]
    m1, _, _ = prop(mu0, C0, l, l + 1)
    tr = float(torch.trace(Ch_t[l]))
    one_err.append(float(((m1 - mh_t[l]) ** 2).sum() / max(tr, 1e-300)))

# ---------------- suffix output error E_out(l0)
suffix = {}
for l0 in sorted(set([0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, depth - 4, depth - 2, depth - 1])):
    if l0 >= depth: continue
    if l0 == 0:
        mu0 = torch.zeros(w if KIND == 'rnn' else nin, dtype=DT)
        C0 = torch.zeros(w, w, dtype=DT) if KIND == 'rnn' else torch.eye(nin, dtype=DT)
    else:
        mu0 = mh_t[l0 - 1]; C0 = Ch_t[l0 - 1]
    mL, _, _ = prop(mu0, C0, l0, depth)
    suffix[l0] = float(((outmean(mL) - mo_t) ** 2).sum() / trv)

# ---------------- concentration of gate errors
def conc(E):
    flat = E.flatten().abs()
    tot = float(flat.sum()); srt = flat.sort(descending=True)
    shares = {k: float(srt.values[:k].sum() / tot) for k in (1, 2, 4, 8, 16, 32, 64, 128)}
    top = [(int(i) // w, int(i) % w, float(flat[i])) for i in srt.indices[:16]]
    return dict(total=tot, mean=float(flat.mean()), max=float(flat.max()), shares=shares, top=top,
                per_layer=E.abs().sum(1).tolist(), per_unit=E.abs().sum(0).tolist())


Ea = (p_k - p_t); E1 = (p_1 - p_t)
res = dict(npz=NPZ, kind=KIND, depth=depth, width=w, paths=NT, trvar=trv, k2_out_nmse=k2_out,
           state_err_acc=state_err, state_err_onestep=one_err, suffix=suffix,
           gate_acc=conc(Ea), gate_one=conc(E1),
           mean_err_acc=((mz_k - mz_t).abs() / vz_t.clamp_min(1e-300).sqrt()).tolist(),
           sd_ratio_acc=(vz_k.clamp_min(1e-300).sqrt() / vz_t.clamp_min(1e-300).sqrt()).tolist(),
           p_true=p_t.tolist(), p_k2=p_k.tolist(), p_one=p_1.tolist(), sec=time.time() - t0)
json.dump(res, open(OUT, 'w'))
ga, g1 = res['gate_acc'], res['gate_one']
print(f"{os.path.basename(NPZ)}: K2 output nMSE {k2_out:.3g}; gate |err| accumulated: mean {ga['mean']:.3g} max {ga['max']:.3g}, "
      f"top-1/8/64 share {ga['shares'][1]:.3f}/{ga['shares'][8]:.3f}/{ga['shares'][64]:.3f}; one-step: mean {g1['mean']:.3g} "
      f"max {g1['max']:.3g}, top-1/8/64 share {g1['shares'][1]:.3f}/{g1['shares'][8]:.3f}/{g1['shares'][64]:.3f}", flush=True)
print('   suffix E_out(l0): ' + ' '.join(f"{k}:{v:.2g}" for k, v in suffix.items()), flush=True)
print('   accumulated state error at l=1,2,4,8,16,32,L: ' + ' '.join(f"{state_err[l-1]:.2g}" for l in (1, 2, 4, 8, 16, 32, depth)), flush=True)
print('   one-step state error at l=1,2,4,8,16,32,L: ' + ' '.join(f"{one_err[l-1]:.2g}" for l in (1, 2, 4, 8, 16, 32, depth)), flush=True)
print(f"   top accumulated (layer, unit, |dp|): {ga['top'][:8]}", flush=True)
print(f"   top one-step   (layer, unit, |dp|): {g1['top'][:8]}  ({time.time()-t0:.0f}s)", flush=True)
