"""E63 exploration (seventh agent; NOT registered results; disclosed in amendment 71): on nets that the registered E63 tests exclude
(E43's newinit ctrl nets: new initialisations, same recipe), how large are (1) the cross-source leakage of the summary tokens' block-1
attention, (2) the code-induced variation of the LayerNorm norm of the labelled tokens, and (3) the KL change of two evaluation-time
surgeries in block 1: S1 = the summary rows restricted to their own source's tokens (renormalised; 'own+self' keeps the summary's
attention to itself, 'own' only its four labelled tokens, as pfn_own), and S2 = block 1's LayerNorm denominator replaced by its
expectation over the identity codes (exact: E|P(Wx+b)|^2 over semb, qemb ~ N(0, I/16), numerators unchanged).
usage: python e63x_explore.py RUN [RUN ...] -> prints; results/e63x_explore.json
"""
import json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_own as PO  # noqa: E402

P2.build = lambda meta: PO.build_any(meta, P2, P2.fin_of(meta['ds']))
import e26_run as RU  # noqa: E402
import e30_eval as E30  # noqa: E402
import e48_fmt as F  # noqa: E402
import pfn_uni as PU  # noqa: E402

E30.F = F
torch.set_num_threads(1)
S, Q, QL, T = 3, 5, 4, 16
SUMM = [s * Q + QL for s in range(S)]
DS = 16


def base_allow(B):
    valid = torch.ones(B, T, dtype=torch.bool)
    qi = T - 1
    ctx = valid.clone(); ctx[:, qi] = False
    allow = ctx[:, None, :].expand(B, T, T).clone(); allow[:, qi, qi] = True
    return allow


def ln_s2(blk_ln, x, W, X):
    """LayerNorm with the variance replaced by its expectation over the identity codes (numerator unchanged)."""
    mu = x.mean(-1, keepdim=True); xc = x - mu
    X0 = X.clone(); X0[..., :2 * DS] = 0
    x0 = X0 @ W.T + blk_bias[0]
    x0c = x0 - x0.mean(-1, keepdim=True)
    d = x.shape[-1]
    C = torch.eye(d) - torch.full((d, d), 1.0 / d)
    Ws, Wq = W[:, :DS], W[:, DS:2 * DS]
    ts = torch.trace(Ws.T @ C @ Ws) / 16.0; tq = torch.trace(Wq.T @ C @ Wq) / 16.0
    has_s = torch.ones(T); has_s[T - 1] = 0.0                     # the query token carries no source code
    ev = ((x0c ** 2).sum(-1) + has_s[None] * ts + tq) / d
    return blk_ln.weight * xc / torch.sqrt(ev[..., None] + blk_ln.eps) + blk_ln.bias


blk_bias = [None]


class Wrap(torch.nn.Module):
    def __init__(self, m, mode):
        super().__init__(); self.m = m; self.mode = mode

    def forward(self, X, valid, want=False):
        m = self.m; B = X.shape[0]
        allow = base_allow(B)
        allow0 = allow.clone()
        if self.mode in ('s1own', 's1self'):
            for s in range(S):
                allow0[:, SUMM[s], :] = False
                allow0[:, SUMM[s], s * Q:s * Q + QL] = True
                if self.mode == 's1self':
                    allow0[:, SUMM[s], SUMM[s]] = True
        x = m.inp(X)
        for i, blk in enumerate(m.blocks):
            if i == 0 and self.mode == 's2':
                blk_bias[0] = m.inp.bias
                h = ln_s2(blk.ln1, x, m.inp.weight, X)
                qq, kk, vv = blk.qkv(h).split(blk.h * blk.dh, -1)
                sp = lambda t: t.view(B, T, blk.h, blk.dh).transpose(1, 2)
                qq, kk, vv = sp(qq), sp(kk), sp(vv)
                sc = (qq @ kk.transpose(-1, -2) / math.sqrt(blk.dh)).masked_fill(~allow[:, None], float('-inf'))
                a = sc.softmax(-1)
                x = x + blk.o((a @ vv).transpose(1, 2).reshape(B, T, blk.h * blk.dh))
                x = x + blk.mlp(blk.ln2(x))
            else:
                x, _ = PU.block_fwd(blk, x, allow0 if i == 0 else allow, None, False)
        y = m.out(m.ln(x[:, T - 1])); K = m.mixk
        return (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])


def attn1(m, X):
    with torch.no_grad():
        x = m.inp(torch.from_numpy(X).float()); blk = m.blocks[0]; B = X.shape[0]
        qq, kk, _ = blk.qkv(blk.ln1(x)).split(blk.h * blk.dh, -1)
        sp = lambda t: t.view(B, T, blk.h, blk.dh).transpose(1, 2)
        sc = (sp(qq) @ sp(kk).transpose(-1, -2) / math.sqrt(blk.dh)).masked_fill(~base_allow(B)[:, None], float('-inf'))
        return sc.softmax(-1).numpy()                                      # (B, H, T, T)


def main():
    z = np.load('results/e27_data.npz'); NB = 1000; sl = slice(0, NB)
    g = z['bmean'][sl, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][sl], z['bvars'], z['bwts'][sl], g)
    se = z['semb'][sl]; qe = z['qemb'][sl]
    res = {}
    for run in sys.argv[1:]:
        model, a, meta = E30.load(run)
        X, v = F.make_X(z['o'][sl, 0], z['y'][sl], se, qe, a)
        o = {}
        for mode in ('base', 's1self', 's1own', 's2'):
            h = RU.heads(Wrap(model, mode), 'clean', X, v)
            kl, _, _ = RU.set_stats([h], lpB, g); o['kl_' + mode] = float(kl.mean())
        A = attn1(model, X)                                                # (B, H, T, T)
        own, selfm, leak = [], [], []
        for s in range(S):
            row = A[:, :, SUMM[s], :]                                      # (B, H, T)
            own.append(row[:, :, s * Q:s * Q + QL].sum(-1)); selfm.append(row[:, :, SUMM[s]])
            leak.append(1 - row[:, :, s * Q:s * Q + Q].sum(-1))
        own, selfm, leak = np.stack(own, 1), np.stack(selfm, 1), np.stack(leak, 1)   # (B, S, H)
        lk = leak.mean(-1).ravel()
        o.update(own_mass=float(own.mean()), self_mass=float(selfm.mean()), leak_mean=float(lk.mean()),
                 leak_q90=float(np.quantile(lk, 0.9)), leak_q99=float(np.quantile(lk, 0.99)), leak_max=float(lk.max()),
                 leak_by_head=[float(leak[:, :, hh].mean()) for hh in range(leak.shape[2])])
        cosm = []
        for s in range(S):
            cs = [np.sum(se[:, s] * se[:, t], 1) / np.linalg.norm(se[:, s], axis=1) / np.linalg.norm(se[:, t], axis=1) for t in range(S) if t != s]
            cosm.append(np.max(np.stack(cs, 1), 1))
        cosm = np.stack(cosm, 1).ravel()
        o['spearman_leak_maxcos'] = float(np.corrcoef(np.argsort(np.argsort(lk)), np.argsort(np.argsort(cosm)))[0, 1])
        W = model.inp.weight.detach().numpy().astype(np.float64); b = model.inp.bias.detach().numpy().astype(np.float64)
        Xl = X[:, [s * Q + q for s in range(S) for q in range(QL)], :].astype(np.float64)
        X0 = Xl.copy(); X0[..., 32] = X0[..., 33]                         # r = 0
        def nrm(XX):
            e = XX @ W.T + b; e = e - e.mean(-1, keepdims=True); return np.linalg.norm(e, axis=-1)
        Afull = nrm(X0)
        Xn = X0.copy(); Xn[..., :2 * DS] = 0; Anocode = nrm(Xn)
        Xs = X0.copy(); Xs[..., DS:2 * DS] = 0; Asonly = nrm(Xs)               # source code kept, question code removed
        o.update(A_cv=float(Afull.std() / Afull.mean()), A_nocode_cv=float(Anocode.std() / Anocode.mean()),
                 A_srconly_cv=float(Asonly.std() / Asonly.mean()))
        # per source: the mean A over its four labelled tokens, its across-episode CV (the per-source common factor)
        As = Afull.reshape(NB, S, QL).mean(-1); Asn = Anocode.reshape(NB, S, QL).mean(-1)
        o['A_src_cv'] = float(As.std() / As.mean()); o['A_src_nocode_cv'] = float(Asn.std() / Asn.mean())
        res[os.path.basename(run)] = o
        print(os.path.basename(run), json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items()}), flush=True)
    json.dump(res, open('results/e63x_explore.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
