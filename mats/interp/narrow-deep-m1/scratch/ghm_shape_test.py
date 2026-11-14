"""Dev test (coordinator suggestion 1, shape hypothesis): GHM from layer 2 on, started from the EXACT first two moments of
each layer-1 gate-pattern piece (oracle: 2^22 sampled paths grouped by the layer-1 pattern), against GHM started from
the input. If the error stays, the floor is the Gaussian shape of pieces at later boundaries, not the layer-1 moments.
Also times a deterministic 8-D orthant probability (Genz transformation on a fixed Korobov lattice) to price exact
pieces."""
import sys, json, torch, math, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib'); sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, ghm
from scipy.stats import norm
torch.set_num_threads(1)
DT = torch.float64
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 103
T = json.load(open(f'runs/truth_s{SEED}_sig0.001.json'))
net = nd.Net('plain', SEED, 8, 256, 1e-3, r=T['r']); n = 8
fp = net.flops_path()
g = torch.Generator().manual_seed(21)
# exact layer-1 piece statistics (oracle)
P = 1 << 22
W1 = net.W[0]
pw = torch.tensor([1 << i for i in range(n)], dtype=torch.long)
sw = torch.zeros(256, dtype=DT); sm = torch.zeros(256, n, dtype=DT); ss = torch.zeros(256, n, n, dtype=DT)
for b in range(P >> 16):
    x = torch.randn(1 << 16, n, generator=g, dtype=DT)
    z = x @ W1.T
    code = ((z > 0).long() * pw).sum(1)
    h = torch.relu(z) + torch.randn(z.shape, generator=g, dtype=DT) * net.s[0]
    sw.index_add_(0, code, torch.ones(len(code), dtype=DT)); sm.index_add_(0, code, h); ss.index_add_(0, code, h[:, :, None] * h[:, None, :])
nz = sw > 0
w1 = sw[nz] / P; m1 = sm[nz] / sw[nz][:, None]; C1 = ss[nz] / sw[nz][:, None, None] - m1[:, :, None] * m1[:, None, :]
print(f"layer-1 pieces: {int(nz.sum())}", flush=True)
for tau in (1e-3, 1e-4):
    fl = ghm.Flops()
    w, mu, C = w1.clone(), m1.clone(), C1.clone()
    row = []
    for l in range(1, 8):
        w, mu, C = ghm.relu_layer(w, mu, C, net.W[l], float(net.s[l]) ** 2, tau, fl)
        c = l + 1
        mm = ghm.mixture_mean(w, mu)
        e = float((mm - torch.tensor(T['layers'][str(c)]['mean_h'], dtype=DT)).pow(2).sum() / torch.tensor(T['layers'][str(c)]['cov_h']).trace())
        row.append(f"c{c}:{len(w)}/{e:.1g}")
    print(f"GHM tau={tau:g} from EXACT layer-1 pieces: " + ' '.join(row), flush=True)
    out, _ = ghm.ghm_forward(net, 8, tau)
    row = []
    for c in range(2, 9):
        w, mu, C, fc = out[c]
        e = float((ghm.mixture_mean(w, mu) - torch.tensor(T['layers'][str(c)]['mean_h'], dtype=DT)).pow(2).sum() / torch.tensor(T['layers'][str(c)]['cov_h']).trace())
        row.append(f"c{c}:{len(w)}/{e:.1g}")
    print(f"GHM tau={tau:g} from the input:            " + ' '.join(row), flush=True)


# price of an exact piece: deterministic Genz (separation of variables) on a fixed Korobov lattice, 8-D orthant
def genz_orthant(Sig, Npts, gen_vec):
    """P(z > 0), z ~ N(0, Sig) via Genz's transformation; deterministic rank-1 lattice (no random shift)."""
    d = Sig.shape[0]
    Lc = np.linalg.cholesky(Sig)
    k = np.arange(Npts)[:, None]
    U = np.mod(k * gen_vec[None, :d - 1] / Npts + 0.5 / Npts, 1.0)
    U = np.abs(2 * U - 1)                                       # baker transform (periodizing)
    e = np.ones(Npts) * norm.cdf(0 / Lc[0, 0])                  # P(z_0 > 0) = P(-z_0 < 0): use upper limits on -z
    f = e.copy()
    y = np.zeros((Npts, d))
    # upper-limit form: P(-z <= 0); a = -inf, b = 0 for -z with chol of Sig
    b0 = 0.0
    e = norm.cdf(b0 / Lc[0, 0]) * np.ones(Npts); f = e.copy()
    for i in range(1, d):
        y[:, i - 1] = norm.ppf(np.clip(U[:, i - 1] * e, 1e-300, 1 - 1e-16))
        s = (y[:, :i] @ Lc[i, :i])
        e = norm.cdf((0.0 - s) / Lc[i, i])
        f = f * e
    return f.mean()


rng = np.random.RandomState(0)
Wn = W1.numpy()
Sig = Wn @ Wn.T
gv = np.array([1, 182667, 469891, 498753, 110745, 446247, 250185, 118627], dtype=float)
for Npts in (1 << 10, 1 << 14, 1 << 18):
    t0 = time.time()
    p = genz_orthant(Sig, Npts, gv)
    print(f"8-D orthant P(W1 x > 0) by deterministic lattice, N={Npts}: {p:.6g} ({time.time()-t0:.2f}s; ~{Npts*8*(200+16):.2g} FLOPs = {Npts*8*(200+16)/fp:.2g} paths)", flush=True)
xs = torch.randn(1 << 22, n, generator=g, dtype=DT)
print("MC check of that orthant (2^22):", float(((xs @ W1.T) > 0).all(1).double().mean()), flush=True)
