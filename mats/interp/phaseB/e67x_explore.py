"""E67 exploration (seventh agent; NOT registered results; to be disclosed in the E67 amendment): world 2 (in-context regression with
an unknown noise level), block 1 of a few E40 ctrl nets that the registered tests will exclude. Per head: the query token's attention
mass on the examples and on itself; the context tokens' attention on themselves; the pre-LN logit decomposition for the query row over
example keys, logit_j = (alpha + beta.(x_j, y_j)) / n_j + const with alpha = sqrt(d / dh) q.W_K(gamma * a) (a = P b, the same for every
example in the ctrl format) and beta the coefficients on (x, y): L = alpha / |a| and K = beta / |a| (y and x parts), the norm term at
a large example (|y| = 3, x = 0); and the KL change when the signed terms are removed from the query row of one head.
usage: python e67x_explore.py RUN [RUN ...]
"""
import json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import t2_world as T  # noqa: E402
import e40_eval as E40  # noqa: E402

torch.set_num_threads(1)


class Surg(torch.nn.Module):
    def __init__(self, m, zero, D):
        super().__init__(); self.m = m; self.zero = list(zero); self.D = D

    def forward(self, X, valid, want=False):
        m = self.m; B, Tn, _ = X.shape; qi = Tn - 1
        ctx = valid.clone(); ctx[:, qi] = False
        allow = ctx[:, None, :].expand(B, Tn, Tn).clone(); allow[:, qi, qi] = True
        x = m.inp(X); blk = m.blocks[0]; ln = blk.ln1
        qq, kk, vv = blk.qkv(ln(x)).split(blk.h * blk.dh, -1)
        sp = lambda t: t.view(B, Tn, blk.h, blk.dh).transpose(1, 2)
        qq, kk, vv = sp(qq), sp(kk), sp(vv)
        sc = qq @ kk.transpose(-1, -2) / math.sqrt(blk.dh)
        if self.zero:
            Wt = m.inp.weight; Wo = Wt[:, :self.D + 1] - Wt[:, :self.D + 1].mean(0, keepdim=True)     # centred columns of x and y
            Wk = blk.qkv.weight[blk.h * blk.dh:2 * blk.h * blk.dh]
            xc = x - x.mean(-1, keepdim=True); sig = torch.sqrt((xc ** 2).mean(-1) + ln.eps)
            odd = (X[:, :, :self.D + 1] @ (ln.weight[:, None] * Wo).T)                             # (B, T, d): gamma * P W_odd u_j
            ko = (odd @ Wk.T).view(B, Tn, blk.h, blk.dh) / sig[:, :, None, None]                    # its key contribution
            sc = sc.clone()
            for hh in self.zero:
                sc[:, hh, qi, :qi] -= (qq[:, hh, qi, None, :] * ko[:, :qi, hh, :]).sum(-1) / math.sqrt(blk.dh)
        sc = sc.masked_fill(~allow[:, None], float('-inf'))
        a = sc.softmax(-1)
        x = x + blk.o((a @ vv).transpose(1, 2).reshape(B, Tn, blk.h * blk.dh)); x = x + blk.mlp(blk.ln2(x))
        for b2 in m.blocks[1:]:
            x, _ = b2(x, allow)
        y = m.out(m.ln(x[:, qi])); K = m.mixk
        return (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])


def main():
    for run in sys.argv[1:]:
        model, meta = E40.load(run); D, N = meta['D'], meta['N']
        rng = np.random.default_rng(67900); d = T.sample(2000, rng, D, N); X, v = T.tokens(d, meta['fmt'])
        bay = T.bayes(d); Xt, vt = torch.from_numpy(X), torch.from_numpy(v)
        blk = model.blocks[0]; B = len(X); Tn = N + 1; qi = N
        with torch.no_grad():
            x = model.inp(Xt); h = blk.ln1(x)
            q, k, _ = blk.qkv(h).split(blk.h * blk.dh, -1)
            sp = lambda t: t.view(B, Tn, blk.h, blk.dh).transpose(1, 2)
            qh, kh = sp(q), sp(k)
            allow = torch.ones(B, Tn, Tn, dtype=torch.bool); allow[:, :, qi] = False; allow[:, qi, qi] = True
            A = (qh @ kh.transpose(-1, -2) / math.sqrt(blk.dh)).masked_fill(~allow[:, None], float('-inf')).softmax(-1).numpy()
        o = dict(q_on_examples=[float(A[:, hh, qi, :qi].sum(-1).mean()) for hh in range(blk.h)],
                 q_self=[float(A[:, hh, qi, qi].mean()) for hh in range(blk.h)],
                 ctx_self=[float(np.mean([A[:, hh, j, j] for j in range(N)])) for hh in range(blk.h)])
        W = model.inp.weight.detach().numpy().astype(np.float64); bb = model.inp.bias.detach().numpy().astype(np.float64)
        gam = blk.ln1.weight.detach().numpy().astype(np.float64); Wk = blk.qkv.weight.detach().numpy().astype(np.float64)[blk.h * blk.dh:2 * blk.h * blk.dh]
        dd = W.shape[0]; a = bb - bb.mean(); Aa = np.linalg.norm(a)
        Wo = W[:, :D + 1] - W[:, :D + 1].mean(0, keepdims=True)
        qn = qh.numpy()[:, :, qi, :]                                                               # (B, H, dh)
        ka = (Wk @ (gam * a)).reshape(blk.h, blk.dh); ko = (Wk @ (gam[:, None] * Wo)).reshape(blk.h, blk.dh, D + 1)
        L = math.sqrt(dd / blk.dh) * np.einsum('bhe,he->bh', qn, ka) / Aa
        K = math.sqrt(dd / blk.dh) * np.einsum('bhe,hek->bhk', qn, ko) / Aa
        n3 = np.linalg.norm(a + 3.0 * Wo[:, D]); r3 = Aa / n3
        o.update(L=[float(L[:, hh].mean()) for hh in range(blk.h)], L_sd=[float(L[:, hh].std()) for hh in range(blk.h)],
                 K_y=[float(K[:, hh, D].mean()) for hh in range(blk.h)], K_y_rms=[float(np.sqrt((K[:, hh, D] ** 2).mean())) for hh in range(blk.h)],
                 K_x_rms=[float(np.sqrt((K[:, hh, :D] ** 2).mean())) for hh in range(blk.h)], dlogit_y3=[float(L[:, hh].mean() * (r3 - 1)) for hh in range(blk.h)])
        g, lpB = E40.family_grid(bay, [E40.heads(model, X, v)])
        kls = {}
        for name, zero in (('base', []), ('zero0', [0]), ('zero1', [1]), ('zeroall', [0, 1])):
            net = Surg(model, zero, D); net.eval()
            hd = E40.heads(net, X, v)
            g2, lpB2 = E40.family_grid(bay, [hd])
            mu, ls, lg = hd
            lq = np.logaddexp.reduce(lg[:, None, :] - ls[:, None, :] - 0.5 * math.log(2 * math.pi) - 0.5 * ((g2[:, :, None] - mu[:, None, :]) / np.exp(ls)[:, None, :]) ** 2, axis=2)
            kls[name] = float(np.trapezoid(np.exp(lpB2) * (lpB2 - lq), g2, axis=1).mean())
        o['kl'] = kls
        print(os.path.basename(run), json.dumps({k: ([round(x, 3) for x in v] if isinstance(v, list) else ({kk: round(vv, 5) for kk, vv in v.items()} if isinstance(v, dict) else v)) for k, v in o.items()}), flush=True)


if __name__ == '__main__':
    main()
