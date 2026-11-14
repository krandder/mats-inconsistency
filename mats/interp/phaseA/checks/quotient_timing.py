"""Pre-method sanity checks (2026-10-28): (1) LayerNorm quotient of G at the block-1 cut of the 2-layer model;
(2) cost of a network forward from the cut and of the G quadrature (existing kl_grid vs Bayes-support-only grid)."""
import sys, time, math
import numpy as np, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import pfn as P
torch.set_num_threads(1)
RUN = '/home/agent/workspaces/mats/work/tiny-pfn/runs/w32_h2_L2_mix4_ws_s0'
ck = torch.load(RUN + '/model.pt', map_location='cpu')
import json
meta = json.load(open(RUN + '/meta.json'))
model = P.build(meta) if 'width' in meta else None
sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
model.load_state_dict(sd); model.eval()
T = dict(np.load('/home/agent/workspaces/mats/work/tiny-pfn/runs/testset.npz'))
n = 4000
X = torch.from_numpy(T['X'][:n]); V = torch.from_numpy(T['valid'][:n])

def downstream_from1(h1, valid):
    B, Tn, _ = h1.shape; qi = Tn - 1
    ctx = valid.clone(); ctx[:, qi] = False
    allow = ctx[:, None, :].expand(B, Tn, Tn).clone(); allow[:, qi, qi] = True
    eye = torch.eye(Tn, dtype=torch.bool)[None]; allow = allow | (eye & ~valid[:, :, None])
    x, _ = model.blocks[1](h1, allow)
    y = model.out(model.ln(x[:, qi])); K = model.mixk
    return (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])

with torch.no_grad():
    mu, ls, lg, hs, _ = model(X, V, want=True)
    h1 = hs[1]
    base = downstream_from1(h1, V)
    rng = torch.Generator().manual_seed(0)
    shift = h1 + torch.randn(n, 16, 1, generator=rng) * 3.0          # c*1 per token, all tokens
    scale = h1.clone(); a = torch.rand(n, 15, 1, generator=rng) * 2 + 0.5
    m = h1[:, :15].mean(-1, keepdim=True)
    scale[:, :15] = m + a * (h1[:, :15] - m)                           # positive rescale of centred context tokens
    scaleQT = h1.clone(); mq = h1[:, 15:].mean(-1, keepdim=True)
    scaleQT[:, 15:] = mq + 1.7 * (h1[:, 15:] - mq)                      # rescale QT (should NOT be invariant)
    for name, hh in (('shift all tokens', shift), ('scale context tokens', scale), ('scale QT (control)', scaleQT)):
        o = downstream_from1(hh, V)
        print(name, 'max|d mu|', float((o[0] - base[0]).abs().max()), 'max|d logits|', float((o[2] - base[2]).abs().max()))
    t0 = time.time()
    for _ in range(5): downstream_from1(h1, V)
    print('downstream-from-block1 forward: %.2e s/episode' % ((time.time() - t0) / 5 / n))
head = tuple(t.numpy().astype(np.float64) for t in base)
Ts = {k: T[k][:n] for k in ('comp_means', 'comp_vars', 'w_bayes')}; Ts['comp_vars'] = T['comp_vars']
t0 = time.time(); kl1 = P.kl_grid(Ts, head); t1 = time.time()
print('kl_grid(2001): %.2e s/episode, mean KL %.5f' % ((t1 - t0) / n, kl1.mean()))

def kl_bayes_grid(Ts, head, ngrid):
    means, vars_, w = Ts['comp_means'], Ts['comp_vars'], Ts['w_bayes']
    mm, ll, lg = head; sd = np.sqrt(vars_)
    lo = (means - 9 * sd).min(1); hi = (means + 9 * sd).max(1)
    g = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, ngrid)[None]
    lp = np.logaddexp.reduce(np.log(w)[:, None, :] - 0.5 * np.log(2 * np.pi * vars_) - 0.5 * (g[:, :, None] - means[:, None]) ** 2 / vars_, axis=2)
    lgn = lg - lg.max(1, keepdims=True); lgn = lgn - np.log(np.exp(lgn).sum(1, keepdims=True))
    lq = np.logaddexp.reduce(lgn[:, None, :] - ll[:, None] - 0.5 * np.log(2 * np.pi) - 0.5 * (g[:, :, None] - mm[:, None]) ** 2 / np.exp(2 * ll)[:, None], axis=2)
    return np.trapezoid(np.exp(lp) * (lp - lq), g, axis=1), lp, g

for ng in (2001, 401, 201):
    t0 = time.time(); kl2, _, _ = kl_bayes_grid(Ts, head, ng); dt = time.time() - t0
    print('bayes-support grid %d: %.2e s/episode, max|diff vs kl_grid| %.2e' % (ng, dt / n, np.abs(kl2 - kl1).max()))
