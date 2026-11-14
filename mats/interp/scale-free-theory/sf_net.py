"""Re-implemented forward for the clean tiny-PFN nets (work/tiny-pfn/pfn.py architecture) with hooks for the
scale-free-theory addendum (M3 attention substitutions, M4 rank-k bottleneck, M5 attention read-out).
Nets are read from work/scale-free-theory/nets/ (copies). Read-only use of work/tiny-pfn/world.py and pfn.py.
"""
import json, math, os, sys
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import world as W
import pfn as P

NETS = '/home/agent/workspaces/mats/work/scale-free-theory/nets'
torch.set_num_threads(1)
YG = np.linspace(-6, 6, 1201)


def load_clean(name):
    rd = os.path.join(NETS, 'clean', name)
    meta = json.load(open(os.path.join(rd, 'meta.json')))
    m = P.build(meta); m.load_state_dict(torch.load(os.path.join(rd, 'model.pt'), map_location='cpu')); m.eval()
    return m, meta


def allow_mask(valid):
    B, T = valid.shape
    qi = T - 1
    ctx = valid.clone(); ctx[:, qi] = False
    allow = ctx[:, None, :].expand(B, T, T).clone()
    allow[:, qi, qi] = True
    eye = torch.eye(T, dtype=torch.bool)[None]
    return allow | (eye & ~valid[:, :, None])


def block_forward(blk, x, allow, a_override=None):
    """Returns (x_out, attention). a_override: (B, H, T, T) probabilities replacing softmax(scores)."""
    B, T, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
    sp = lambda t: t.view(B, T, blk.h, d // blk.h).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    s = q @ k.transpose(-1, -2) / math.sqrt(d // blk.h)
    s = s.masked_fill(~allow[:, None], float('-inf'))
    a = s.softmax(-1) if a_override is None else a_override
    x = x + blk.o((a @ v).transpose(1, 2).reshape(B, T, d))
    if not blk.attn_only:
        x = x + blk.mlp(blk.ln2(x))
    return x, a


def forward(model, X, valid, overrides=None, bottleneck=None):
    """overrides: {layer_index: a_override}. bottleneck: callable applied to the context tokens' residual stream after
    block 1 (input to block 2). Returns head tensors and the list of attentions."""
    allow = allow_mask(valid)
    x = model.inp(X); attns = []
    for li, blk in enumerate(model.blocks):
        x, a = block_forward(blk, x, allow, None if overrides is None else overrides.get(li))
        attns.append(a)
        if bottleneck is not None and li == 0 and len(model.blocks) > 1:
            x = torch.cat([bottleneck(x[:, :-1]), x[:, -1:]], 1)
    y = model.out(model.ln(x[:, -1]))
    if model.mixk == 0:
        head = (y[:, 0], y[:, 1].clamp(-6, 3))
    else:
        K = model.mixk
        head = (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
    return head, attns


def head_logpdf(head, yg=YG):
    """log density of the predictive on grid yg: (B, G) numpy."""
    yt = torch.as_tensor(yg, dtype=torch.float32)
    if len(head) == 2:
        mu, ls = head
        lp = -ls[:, None] - 0.5 * math.log(2 * math.pi) - 0.5 * ((yt[None] - mu[:, None]) / ls.exp()[:, None]) ** 2
    else:
        mu, ls, lg = head
        lw = torch.log_softmax(lg, -1)
        lp = lw[:, None, :] - ls[:, None, :] - 0.5 * math.log(2 * math.pi) - 0.5 * ((yt[None, :, None] - mu[:, None, :]) / ls.exp()[:, None, :]) ** 2
        lp = torch.logsumexp(lp, -1)
    return lp.detach().numpy().astype(np.float64)


def bayes_logpdf(d, yg=YG):
    out, nll, mom, w = W.references(d)
    means, vars_, wts = out['bayes']
    comp = -0.5 * np.log(2 * np.pi * vars_)[None, None] - 0.5 * (yg[None, :, None] - means[:, None, :]) ** 2 / vars_[None, None]
    mx = comp.max(2, keepdims=True)
    return mx[:, :, 0] + np.log((wts[:, None, :] * np.exp(comp - mx)).sum(2))


def kl_grid(lp_p, lp_q, yg=YG):
    """Mean over rows of KL(p || q) for log densities on a uniform grid (renormalised on the grid)."""
    dy = yg[1] - yg[0]
    lp_p = lp_p - np.log(np.exp(lp_p - lp_p.max(1, keepdims=True)).sum(1, keepdims=True) * dy) - lp_p.max(1, keepdims=True)
    lp_q = lp_q - np.log(np.exp(lp_q - lp_q.max(1, keepdims=True)).sum(1, keepdims=True) * dy) - lp_q.max(1, keepdims=True)
    p = np.exp(lp_p)
    return (p * (lp_p - lp_q)).sum(1) * dy


def episodes(n, seed, ds):
    rng = np.random.default_rng(seed)
    d = W.sample_np(n, rng)
    rid = np.random.default_rng(seed + 7)
    semb = rid.standard_normal((n, W.S, ds)) / 4
    qemb = rid.standard_normal((n, W.Q, ds)) / 4
    X, m = W.assemble(semb, qemb, d['o'], d['y'])
    return d, torch.from_numpy(X), torch.from_numpy(m)


def plumbing(name):
    model, meta = load_clean(name)
    d, X, m = episodes(256, 123, meta['ds'])
    with torch.no_grad():
        ref = model(X, m)
        mine, _ = forward(model, X, m)
        diff = max(float((a - b).abs().max()) for a, b in zip(ref, mine))
        # permutation equivariance: permute the 15 context tokens; the output must not change
        perm = torch.randperm(15)
        Xp = X.clone(); Xp[:, :15] = X[:, perm]
        mp = m.clone(); mp[:, :15] = m[:, perm]
        outp, _ = forward(model, Xp, mp)
        pdiff = max(float((a - b).abs().max()) for a, b in zip(mine, outp))
    return diff, pdiff


if __name__ == '__main__':
    for nm in ['w32_h2_L2_s0', 'w32_h2_L2_mix4_s0', 'w32_h2_L1_s0']:
        print(nm, 'reimplementation max abs diff %.2e, permutation change %.2e' % plumbing(nm))
