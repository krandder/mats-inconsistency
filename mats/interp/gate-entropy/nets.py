"""Network adapters for the gate-entropy measurement (2026-10-29). All read weights from other lanes' files; nothing
is written outside work/gate-entropy. Each adapter exposes:
  layers                  list of layer ids (int) analysed
  m[l]                    number of gates at layer l
  sample(N, seed)         -> state dict with stored randomness, pre-activations and gates for every analysed layer
  gates(st, l)            -> (N, m_l) bool gate pattern
  preact(st, l)           -> (N, k) float pre-activations the gates threshold (for the subspace measurement)
  sub(st, l, idx, Gnew)   -> answers of samples idx when layer l's gates are set to Gnew (linear extension of the
                             piece selected by each gate), all other randomness held fixed (common random numbers)
  changed(a0, a1)         -> bool per sample: the answer changed beyond the question's tolerance
Gate definitions:
  ReLU: g = 1[z > 0]; a = g z (flip = linear extension of the other piece).
  tanh: the kinks of hardtanh(z) = z - ReLU(z - 1) + ReLU(-z - 1): two gates per neuron, up = 1[z > 1], dn = 1[z < -1];
        a substituted gate adds hardtanh_new(z) - hardtanh_actual(z) to tanh(z) (continuous at the kink).
  GELU: g = 1[u > 0] (the middle of GELU's switching region); GELU(u) = u Phi(u); a flipped gate uses u Phi(-u)
        (the soft gate reflected about 1/2), which equals the ReLU flip far from 0 and vanishes at u = 0.
"""
import math, os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'ext'))
torch.set_num_threads(1)
F32 = torch.float32


# ----------------------------------------------------------------------------------------------- AlgZoo noisy RNN
class RNNNet:
    """h_0 = 0; z_t = W_hh h_{t-1} + W_hi x_t; a_t = ReLU(z_t); h_t = a_t + sigma r_t eps_t (every step incl. last);
    logits = W_oh h_n. Answer per sample: 1[argmax logits == 2nd argmax of x] (the question 'is the output the 2nd
    argmax?'). Layers are steps t = 1..n."""

    kind = 'rnn'

    def __init__(self, path, name=None):
        z = np.load(path)
        self.W_hi = torch.tensor(z['W_hi'], dtype=F32); self.W_hh = torch.tensor(z['W_hh'], dtype=F32)
        self.W_oh = torch.tensor(z['W_oh'], dtype=F32); self.r = torch.tensor(z['r'], dtype=F32)
        self.sigma = float(z['sigma']); self.d = self.W_hi.shape[0]; self.n = self.W_oh.shape[0]
        self.layers = list(range(1, self.n + 1)); self.m = {l: self.d for l in self.layers}
        self.name = name or os.path.basename(path)
        self.capacity_bits = self.d / 2 * math.log2(1 + 1 / self.sigma ** 2)

    def sample(self, N, seed, keep_noise=True, x=None):
        g = torch.Generator().manual_seed(seed)
        xr = torch.randn(N, self.n, generator=g)
        x = xr if x is None else x
        eps = torch.randn(self.n, N, self.d, generator=g)
        h = torch.zeros(N, self.d); Z = []; A = []
        for t in range(self.n):
            z = h @ self.W_hh.T + x[:, t:t + 1] * self.W_hi
            a = torch.relu(z); Z.append(z); A.append(a)
            h = a + self.sigma * self.r[t] * eps[t]
        lo = h @ self.W_oh.T
        st = dict(N=N, x=x, eps=eps, Z=Z, A=A, ans=self._ans(lo, x), logits=lo)
        return st

    @staticmethod
    def _ans(lo, x):
        return (lo.argmax(-1) == x.argsort(-1)[:, -2])

    def gates(self, st, l):
        return (st['Z'][l - 1] > 0).numpy()

    def preact(self, st, l):
        return st['Z'][l - 1].numpy()

    def input_to(self, st, l):
        """h_{l-1} (noisy state feeding step l) and x_l; used for capacity bounds."""
        if l == 1:
            return torch.zeros(st['N'], self.d)
        return st['A'][l - 2] + self.sigma * self.r[l - 2] * st['eps'][l - 2]

    def sub(self, st, l, idx, Gnew):
        """Gnew (k, d) bool for samples idx (k,), possibly repeated."""
        t = l - 1
        idx = torch.as_tensor(idx)
        z = st['Z'][t][idx]; x = st['x'][idx]
        a = torch.as_tensor(Gnew, dtype=F32) * z
        h = a + self.sigma * self.r[t] * st['eps'][t][idx]
        for s in range(t + 1, self.n):
            a = torch.relu(h @ self.W_hh.T + x[:, s:s + 1] * self.W_hi)
            h = a + self.sigma * self.r[s] * st['eps'][s][idx]
        return self._ans(h @ self.W_oh.T, x)

    def changed(self, a0, a1):
        return (a0 != a1)

    def answer_summary(self, st):
        return dict(acc=float(st['ans'].float().mean()))


# ----------------------------------------------------------------------------------------------- ARC random ReLU MLP
class ArcNet:
    """ARC's family (work/arc-noisy/anet.py, copied): x ~ N(0, I_n); z_l = W_l h_{l-1}; a_l = ReLU(z_l);
    h_l = a_l + s_l eps_l for l < L; output a_L. W_l iid N(0, 2/n) from RandomState(1_000_003 + 7919 seed);
    s_l = sigma r_l, r_l from the seeded 2^12-path pilot. Question: E[a_L]; per-sample answer a_L(x, eps).
    changed: ||delta a_L||_2 > tol * sqrt(sum_j Var a_L,j)."""

    kind = 'arc'

    def __init__(self, n, L, sigma, seed, tol=0.1, layers=None):
        import anet_copy as A
        self.n, self.L, self.sigma, self.seed, self.tol = n, L, sigma, seed, tol
        W = A.make_weights(seed, n, L)
        r = A.pilot_rms(W, sigma, seed=seed) if sigma > 0 else [1.0] * L
        self.W = [w.float() for w in W]; self.s = [sigma * ri for ri in r]
        self.layers = layers or list(range(1, L + 1)); self.m = {l: n for l in self.layers}
        self.name = f'arc_n{n}_L{L}_s{sigma:g}_seed{seed}'
        self.capacity_bits = n / 2 * math.log2(1 + 1 / sigma ** 2) if sigma > 0 else float('inf')
        self.scale = None

    def sample(self, N, seed, store=True, x=None):
        g = torch.Generator().manual_seed(10_000 + seed)
        h = torch.randn(N, self.n, generator=g)
        h = h if x is None else x
        Z, E = [], []
        for l in range(self.L):
            z = h @ self.W[l].T
            a = torch.relu(z)
            Z.append(z)
            if l < self.L - 1:
                e = torch.randn(N, self.n, generator=g); E.append(e)
                h = a + self.s[l] * e
        st = dict(N=N, Z=Z, E=E, ans=a)
        if self.scale is None:
            self.scale = float(a.var(0).sum().sqrt())
        return st

    def gates(self, st, l):
        return (st['Z'][l - 1] > 0).numpy()

    def preact(self, st, l):
        return st['Z'][l - 1].numpy()

    def input_to(self, st, l):
        if l == 1:
            return None
        return torch.relu(st['Z'][l - 2]) + self.s[l - 2] * st['E'][l - 2]

    def noise_sd_into(self, l):
        return self.s[l - 2] if l >= 2 else 0.0

    def sub(self, st, l, idx, Gnew):
        idx = torch.as_tensor(idx)
        a = torch.as_tensor(Gnew, dtype=F32) * st['Z'][l - 1][idx]
        for k in range(l, self.L):            # 0-based next layer index k = l .. L-1
            h = a + self.s[k - 1] * st['E'][k - 1][idx]
            a = torch.relu(h @ self.W[k].T)
        return a

    def changed(self, a0, a1):
        return ((a1 - a0).norm(dim=-1) > self.tol * self.scale)

    def bias_nmse(self, a0, a1, N_total):
        """normalized squared bias of the mean caused by the substitution (ARC's nMSE units): sum over the substituted
        samples / N_total, squared, over mean_j Var(a_L,j)."""
        d = (a1 - a0).double().sum(0) / N_total
        return float((d ** 2).mean() / (self.scale ** 2 / self.n))

    def answer_summary(self, st):
        return dict(scale=self.scale, mean_norm=float(st['ans'].mean(0).norm()))


# ----------------------------------------------------------------------------------------------- tanh LN MLP (phase R)
class TanhNet:
    """Phase R's net (work/cm-interp-R/rnet.py, copied): x ~ N(0, I_d); z_l = W_l h_{l-1} + b_l, W_l iid N(0, g^2/d),
    b_l iid N(0, bias^2); s_l = LN(tanh(z_l)) (no affine); h_l = s_l + sigma eps_l. Output y = W_out s_L (last noise
    integrated; mean-zero). Question: E[y]. Gates: up = 1[z > 1], dn = 1[z < -1] (hardtanh kinks), 2d per layer."""

    kind = 'tanh'

    def __init__(self, d, L, g, sigma, seed, bias=0.5, tol=0.1, layers=None):
        import rnet_copy as R
        net = R.RNet(d, L, 'tanh', g, sigma, seed, bias=bias)
        self.W = [w.float() for w in net.W]; self.b = [b.float() for b in net.b]; self.Wout = net.Wout.float()
        self.d, self.L, self.g, self.sigma, self.seed, self.tol = d, L, g, sigma, seed, tol
        self.layers = layers or list(range(1, L + 1)); self.m = {l: 2 * d for l in self.layers}
        self.name = f'tanh_g{g:g}_d{d}_L{L}_s{sigma:g}_seed{seed}'
        self.sqd = math.sqrt(d)
        self.capacity_bits = (d - 1) / 2 * math.log2(1 + 1 / sigma ** 2)
        self.scale = None

    def ln(self, a):
        c = a - a.mean(-1, keepdim=True)
        return c * (self.sqd / c.norm(dim=-1, keepdim=True).clamp_min(1e-30))

    def sample(self, N, seed, x=None):
        g = torch.Generator().manual_seed(20_000 + seed)
        h = torch.randn(N, self.d, generator=g)
        h = h if x is None else x
        Z, E = [], []
        for l in range(self.L):
            z = h @ self.W[l].T + self.b[l]
            s = self.ln(torch.tanh(z)); Z.append(z)
            if l < self.L - 1:
                e = torch.randn(N, self.d, generator=g); E.append(e)
                h = s + self.sigma * e
        y = s @ self.Wout.T
        st = dict(N=N, Z=Z, E=E, ans=y)
        if self.scale is None:
            self.scale = float(y.var(0).sum().sqrt())
        return st

    def gates(self, st, l):
        z = st['Z'][l - 1]
        return torch.cat([z > 1, z < -1], 1).numpy()

    def preact(self, st, l):
        return st['Z'][l - 1].numpy()

    def input_to(self, st, l):
        if l == 1:
            return None
        return self.ln(torch.tanh(st['Z'][l - 2])) + self.sigma * st['E'][l - 2]

    def noise_sd_into(self, l):
        return self.sigma if l >= 2 else 0.0

    def sub(self, st, l, idx, Gnew):
        idx = torch.as_tensor(idx)
        z = st['Z'][l - 1][idx]
        Gn = torch.as_tensor(Gnew, dtype=F32); d = self.d
        u, dn = (z > 1).float(), (z < -1).float()
        a = torch.tanh(z) - (Gn[:, :d] - u) * (z - 1) + (Gn[:, d:] - dn) * (-z - 1)
        s = self.ln(a)
        for k in range(l, self.L):
            h = s + self.sigma * st['E'][k - 1][idx]
            s = self.ln(torch.tanh(h @ self.W[k].T + self.b[k]))
        return s @ self.Wout.T

    def changed(self, a0, a1):
        return ((a1 - a0).norm(dim=-1) > self.tol * self.scale)

    def bias_nmse(self, a0, a1, N_total):
        d = (a1 - a0).double().sum(0) / N_total
        return float((d ** 2).mean() / (self.scale ** 2 / self.d))

    def answer_summary(self, st):
        return dict(scale=self.scale, mean_norm=float(st['ans'].mean(0).norm()))


# ----------------------------------------------------------------------------------------------- phase B noisy PFN
class PFNNet:
    """Phase B noisy tiny PFN (work/cm-interp-B/nnoise.py, copied): 2 blocks, d = 32, 2 heads, MLP hidden 64 (GELU),
    16 tokens; every written activation projected to the sphere, then N(0, sigma^2) noise (sites emb, a1, m1, r1, a2,
    m2, r2). Gates: g = 1[u > 0] for the GELU pre-activations u = Linear(LN2(a_l)), per token and unit (16 x 64 per
    block). Question: the predictive mean of the query (mixture mean of the head); its deviation from exact Bayes is the
    same number minus a constant per episode. changed: |delta mean| > tol_abs."""

    kind = 'pfn'

    def __init__(self, run, tol_abs=None):
        import nnoise_copy as NN
        import pfn as P
        import world as W
        self.NN, self.P, self.Wd = NN, P, W
        meta = json.load(open(os.path.join(run, 'meta.json')))
        self.model = NN.build(meta); self.model.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu'))
        self.model.eval()
        self.sigma = meta['sigma']; self.T = 16; self.dm = 2 * meta['width']; self.dw = meta['width']
        self.layers = [1, 2]; self.m = {1: self.T * self.dm, 2: self.T * self.dm}
        self.name = 'pfn_' + os.path.basename(run.rstrip('/'))
        self.tol_abs = tol_abs
        self.capacity_bits = self.T * self.dw / 2 * math.log2(1 + 1 / self.sigma ** 2)     # residual cut, all tokens

    @torch.no_grad()
    def sample(self, N, seed):
        rng = np.random.default_rng(30_000 + seed)
        d = self.Wd.sample_np(N, rng)
        X, valid = self.Wd.tokens_np(d, rng, 16)
        ref, _, mom, _ = self.Wd.references(d)
        X = torch.from_numpy(X); valid = torch.from_numpy(valid)
        g = torch.Generator().manual_seed(40_000 + seed)
        M = self.model; sg = self.sigma
        allow = M.allow_mask(valid)
        nz = lambda shape: torch.randn(shape, generator=g)
        E = {}
        x = M.f_emb(X); E['emb'] = nz(x.shape); x = x + sg * E['emb']
        U = {}; st = dict(N=N, allow=allow, bayes_mean=torch.from_numpy(mom['bayes'][0]).float())
        for l in (1, 2):
            xa = M.f_a(l, x, allow); E[f'a{l}'] = nz(xa.shape); xa = xa + sg * E[f'a{l}']
            b = M.blocks[l - 1]
            u = b.mlp[0](b.ln2(xa)); U[l] = u
            mm = self.NN.sphere(b.mlp[1](u)); E[f'm{l}'] = nz(mm.shape); mm = mm + sg * E[f'm{l}']
            x = M.f_r(l, xa, mm); E[f'r{l}'] = nz(x.shape); x = x + sg * E[f'r{l}']
            st[f'xa{l}'] = xa
            if l == 1: st['r1'] = x
        st.update(E=E, U=U, ans=self._mean(M.head(x)))
        return st

    @staticmethod
    def _mean(hd):
        mu, ls, lg = hd
        return (torch.softmax(lg, -1) * mu).sum(-1)

    def gates(self, st, l):
        return (st['U'][l] > 0).reshape(st['N'], -1).numpy()

    def preact(self, st, l):
        return st['U'][l].reshape(st['N'], -1).numpy()

    def input_to(self, st, l):
        return None

    @torch.no_grad()
    def sub(self, st, l, idx, Gnew):
        """Gnew (k, T*64) bool. Recomputes everything downstream of block l's MLP for the query output."""
        M = self.model; sg = self.sigma; E = st['E']
        idx = torch.as_tensor(idx)
        k = len(idx)
        u = st['U'][l][idx]
        G0 = (u > 0)
        Gn = torch.as_tensor(Gnew).reshape(k, self.T, self.dm)
        phi = 0.5 * (1 + torch.erf(u / math.sqrt(2)))
        gate = torch.where(Gn == G0, phi, 1 - phi)
        b = M.blocks[l - 1]
        mm = self.NN.sphere(u * gate) + sg * E[f'm{l}'][idx]
        xa = st[f'xa{l}'][idx]
        if l == 1:
            r1 = M.f_r(1, xa, mm) + sg * E['r1'][idx]
            allow = st['allow'][idx]
            # block 2, query row only
            b2 = M.blocks[1]
            xa2 = M.f_a(2, r1, allow)[:, -1] + sg * E['a2'][idx][:, -1]
            u2 = b2.mlp[0](b2.ln2(xa2))
            m2 = self.NN.sphere(b2.mlp[1](u2)) + sg * E['m2'][idx][:, -1]
            r2 = self.NN.sphere(xa2 + b2.mlp[2](m2)) + sg * E['r2'][idx][:, -1]
        else:
            r2 = M.f_r(2, xa, mm)[:, -1] + sg * E['r2'][idx][:, -1]
        return self._mean(M.head(r2[:, None, :]))

    def changed(self, a0, a1):
        return ((a1 - a0).abs() > self.tol_abs)

    def bias_abs(self, a0, a1, N_total):
        return float((a1 - a0).double().sum() / N_total)

    def answer_summary(self, st):
        dev = st['ans'] - st['bayes_mean']
        return dict(rms_dev_bayes=float(dev.pow(2).mean().sqrt()), sd_mean=float(st['ans'].std()), tol_abs=self.tol_abs)
