"""Batched TopK sparse autoencoders (Gao et al. 2024 style) and evaluation in bits relative to the noise.

S SAEs with the same dictionary size F and different k are trained together (one batched forward):
    pre_s = ReLU(W_enc_s (x - b_dec_s) + b_enc_s);  f_s = TopK_{k_s}(pre_s);  xhat_s = W_dec_s f_s + b_dec_s
Unit-norm decoder columns (renormalised after every step), tied init, b_dec = data mean, Adam with linear decay to 0
over the last 40% of steps, AuxK loss (k_aux = n/2 dead latents reconstruct the residual, weight 1/32).
Data are scaled by one global scalar c (so the centred data have unit mean per-coordinate variance); all errors are
reported in the original units.
Error in bits relative to the noise of the layer (s = noise sd added after this layer):
    bits = E||a - ahat||^2 / (2 s^2) / ln 2      (= the path KL of replacing a by ahat before the noise, per layer)
    noise units per coordinate = E||a - ahat||^2 / (n s^2)   (< 1 means "below the noise")
LS refit: given the TopK support, the coefficients are refitted by least squares (removes the encoder's amplitude error;
the dictionary's capability on that support).
"""
import math
import torch

LN2 = math.log(2.0)


class BatchedTopK(torch.nn.Module):
    def __init__(self, n, F, ks, mean, device, gen=None):
        super().__init__()
        S = len(ks)
        self.n, self.F, self.S = n, F, S
        self.register_buffer('ks', torch.tensor(ks, device=device))
        self.kmax = max(ks)
        Wd = torch.randn(S, n, F, device=device, generator=gen)
        Wd = Wd / Wd.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(Wd)                          # (S, n, F)
        self.W_enc = torch.nn.Parameter(Wd.transpose(1, 2).clone())  # (S, F, n)
        self.b_enc = torch.nn.Parameter(torch.zeros(S, F, device=device))
        self.b_dec = torch.nn.Parameter(mean[None].repeat(S, 1).clone())
        self.register_buffer('last_fire', torch.zeros(S, F, device=device))

    def pre(self, x):
        xc = x[None] - self.b_dec[:, None, :]                        # (S, B, n)
        return torch.baddbmm(self.b_enc[:, None, :], xc, self.W_enc.transpose(1, 2))   # (S, B, F)

    def encode(self, x):
        p = torch.relu(self.pre(x))
        vals, idx = p.topk(self.kmax, dim=-1)                        # (S, B, kmax), sorted
        mask = (torch.arange(self.kmax, device=x.device)[None, None, :] < self.ks[:, None, None]).to(vals.dtype)
        vals = vals * mask
        f = torch.zeros_like(p).scatter_(-1, idx, vals)
        return f, p, idx, mask

    def decode(self, f):
        return torch.bmm(f, self.W_dec.transpose(1, 2)) + self.b_dec[:, None, :]   # (S, B, n)

    @torch.no_grad()
    def renorm(self):
        nrm = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.W_dec.div_(nrm)


def train_batched(X, F, ks, steps=None, epochs=20, batch=2048, lr=None, device='cpu', seed=0, log=None,
                  aux_k=None, dead_steps=None):
    """X: (N, n) float32 (already on device). Returns (model, scale c, history)."""
    N, n = X.shape
    gen = torch.Generator(device=device).manual_seed(seed)
    mu = X.mean(0)
    c = float(((X - mu) ** 2).mean().sqrt())
    Xs = X / c
    model = BatchedTopK(n, F, ks, mu / c, device, gen)
    if lr is None:
        lr = 2e-3 / math.sqrt(max(1.0, F / (4 * n)))
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    steps = steps or max(1, epochs * N // batch)
    aux_k = aux_k or max(1, n // 2)
    dead_steps = dead_steps or max(50, (1 << 16) // batch)   # dead = not fired in the last 2^16 samples
    decay_start = int(0.6 * steps)
    hist = []
    perm = torch.randperm(N, device=device, generator=gen)
    pos = 0
    for t in range(steps):
        if pos + batch > N:
            perm = torch.randperm(N, device=device, generator=gen); pos = 0
        xb = Xs[perm[pos:pos + batch]]; pos += batch
        f, p, idx, mask = model.encode(xb)
        xh = model.decode(f)
        err = xh - xb[None]
        mse = (err ** 2).sum(-1).mean(-1)                            # (S,)
        loss = mse.sum()
        # AuxK: dead latents reconstruct the residual
        with torch.no_grad():
            fired = (f > 0).any(1).to(model.last_fire.dtype)            # (S, F)
            model.last_fire.mul_(1 - fired).add_(1)                   # steps since last fire (reset to 1)
            dead = model.last_fire > dead_steps                       # (S, F)
        if dead.any():
            pd = p.masked_fill(~dead[:, None, :], 0.0)
            ka = min(aux_k, int(dead.sum(1).max()))
            v, ii = pd.topk(ka, dim=-1)
            fa = torch.zeros_like(pd).scatter_(-1, ii, v)
            ea = torch.bmm(fa, model.W_dec.transpose(1, 2))
            resid = (xb[None] - xh).detach()
            aux = ((ea - resid) ** 2).sum(-1).mean(-1) / (resid ** 2).sum(-1).mean(-1).clamp_min(1e-12)
            loss = loss + aux.sum() / 32.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # remove the gradient component along each decoder column (keeps unit norm to first order)
        with torch.no_grad():
            Wd = model.W_dec; g = Wd.grad
            if g is not None:
                g.sub_(Wd * (g * Wd).sum(1, keepdim=True))
        if t >= decay_start:
            for pg in opt.param_groups:
                pg['lr'] = lr * max(0.0, (steps - t) / (steps - decay_start))
        opt.step()
        model.renorm()
        if log is not None and (t % max(1, steps // 10) == 0 or t == steps - 1):
            hist.append((t, [float(m) for m in mse.detach() / n]))
            log(f"  step {t}/{steps} fvu-ish(per coord, scaled) " + ' '.join(f"{float(m)/n:.3e}" for m in mse.detach()))
    return model, c, hist


@torch.no_grad()
def evaluate(model, c, X, s, chunk=4096, refit=True, ridge=1e-6):
    """X: (M, n) held-out, s: noise sd. Returns per-SAE dict of metrics (original units)."""
    M, n = X.shape
    S = model.S
    se = torch.zeros(S, dtype=torch.float64, device=X.device)
    se_rf = torch.zeros(S, dtype=torch.float64, device=X.device)
    l0 = torch.zeros(S, dtype=torch.float64, device=X.device)
    fires = torch.zeros(S, model.F, dtype=torch.float64, device=X.device)
    for i in range(0, M, chunk):
        xb = X[i:i + chunk] / c
        f, p, idx, mask = model.encode(xb)
        xh = model.decode(f)
        se += ((xh - xb[None]) ** 2).sum(-1).sum(-1).double()
        act = f > 0
        l0 += act.sum(-1).sum(-1).double()
        fires += act.sum(1).double()
        if refit:
            # least squares on the support (the first k_s indices of idx for SAE s)
            for si in range(S):
                k = int(model.ks[si])
                ii = idx[si, :, :k]                                    # (B, k)
                Dsub = model.W_dec[si].T[ii]                           # (B, k, n)
                y = xb - model.b_dec[si][None]                         # (B, n)
                G = Dsub @ Dsub.transpose(1, 2)                        # (B, k, k)
                G = G + ridge * torch.eye(k, device=X.device)[None]
                rhs = (Dsub @ y[:, :, None])                           # (B, k, 1)
                coef = torch.linalg.solve(G.double(), rhs.double()).to(y.dtype)
                yh = (Dsub.transpose(1, 2) @ coef)[:, :, 0]
                se_rf[si] += ((yh - y) ** 2).sum().double()
    mse = se / M * c * c                       # E||a - ahat||^2 (original units)
    out = []
    for si in range(S):
        r = dict(k=int(model.ks[si]), F=model.F, L0=float(l0[si] / M),
                 err_sq=float(mse[si]), noise_units=float(mse[si] / (n * s * s)),
                 bits=float(mse[si] / (2 * s * s) / LN2),
                 dead_frac=float((fires[si] == 0).double().mean()),
                 alive=int((fires[si] > 0).sum()))
        if refit:
            m2 = float(se_rf[si] / M * c * c)
            r.update(err_sq_refit=m2, noise_units_refit=m2 / (n * s * s), bits_refit=m2 / (2 * s * s) / LN2)
        out.append(r)
    return out
