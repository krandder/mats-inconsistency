"""E67 (PREREG-B amendment 75; CPU, existing nets): the head-level account in world 2 (in-context regression with an unknown noise
level), where nothing has to be routed: every context token is an example of the one task.
Nets: E40's ctrl and canon nets and E48's sgnprod nets at D = 1 and 2 (best-validation model.pt; acquired = registered gap < 0.10),
except the three explored before amendment 75 (E40 d2ctrl s0, s1, d1ctrl s0).
Per net, on 2,000 episodes (seed 67,001) in the net's format, for the query token's two block-1 heads:
  attention mass on the examples and on itself;
  the pre-LN logit decomposition over example keys: P x_j = a_j + o_j, with o_j the contribution of the signed raw columns (x_j, y_j;
  in canon the folded values) and a_j the rest (the bias, and in sgnprod the products); logit_j = (alpha_j + beta_j) / n_j + const with
  alpha_j = sqrt(d / dh) q.W_K(gamma * a_j), beta_j = sqrt(d / dh) q.W_K(gamma * o_j), n_j = |a_j + o_j| (epsilon-free);
  the norm-scaled logit L = mean over examples of alpha_j / |a_j|; the norm term at a large example, L (|a| / |a + 3 P w_y| - 1) with
  a the mean a_j (|y| = 3, x = 0); the signed coefficients K = sqrt(d / dh) q.W_K(gamma * P W_odd) / |a| (y part, x part RMS);
  KL (per-episode grid, as e40_eval.family_grid) with the signed terms removed from the query row's logits over example keys, exactly
  (logit_j - q.W_K(gamma * o_j) / (sigma_j sqrt(dh)), sigma_j the LayerNorm standard deviation with eps), in one head and in both.
usage: python e67_run.py COHORT (d1ctrl | d2ctrl | d1canon | d2canon | d1sgnprod | d2sgnprod) -> results/e67/<cohort>.json
"""
import glob, json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import t2_world as T  # noqa: E402
import t2lin as TL  # noqa: E402
import e40_eval as E40  # noqa: E402

torch.set_num_threads(1)
EXPLORED = {'e40_d2ctrl_s0', 'e40_d2ctrl_s1', 'e40_d1ctrl_s0'}
COH = dict(d1ctrl=('runs/e40/e40_d1ctrl_s*', 'results/e40'), d2ctrl=('runs/e40/e40_d2ctrl_s*', 'results/e40'),
           d1canon=('runs/e40/e40_d1canon_s*', 'results/e40'), d2canon=('runs/e40/e40_d2canon_s*', 'results/e40'),
           d1sgnprod=('runs/e48t2/e48_d1sgnprod_s*', 'results/e48t2'), d2sgnprod=('runs/e48t2/e48_d2sgnprod_s*', 'results/e48t2'),
           plumb=('runs/e40/e40_d[12]ctrl_s[01]', 'results/e40'))                                   # plumbing only: the explored nets
NB = 2000


def load(run):
    meta = json.load(open(os.path.join(run, 'meta.json')))
    m = TL.build(meta, P2); m.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu')); m.eval(); m.requires_grad_(False)
    return m, meta


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
            Wt = m.inp.weight; Wo = Wt[:, :self.D + 1] - Wt[:, :self.D + 1].mean(0, keepdim=True)
            Wk = blk.qkv.weight[blk.h * blk.dh:2 * blk.h * blk.dh]
            xc = x - x.mean(-1, keepdim=True); sig = torch.sqrt((xc ** 2).mean(-1) + ln.eps)
            odd = X[:, :, :self.D + 1] @ (ln.weight[:, None] * Wo).T
            ko = (odd @ Wk.T).view(B, Tn, blk.h, blk.dh) / sig[:, :, None, None]
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


def kl_of(net, X, v, bay):
    hd = E40.heads(net, X, v); g, lpB = E40.family_grid(bay, [hd]); mu, ls, lg = hd
    lq = np.logaddexp.reduce(lg[:, None, :] - ls[:, None, :] - 0.5 * math.log(2 * math.pi) - 0.5 * ((g[:, :, None] - mu[:, None, :]) / np.exp(ls)[:, None, :]) ** 2, axis=2)
    return float(np.trapezoid(np.exp(lpB) * (lpB - lq), g, axis=1).mean())


def main():
    name = sys.argv[1]; rpat, rdir = COH[name]
    os.makedirs('results/e67', exist_ok=True)
    res = {}
    for run in sorted(d for d in glob.glob(rpat) if os.path.isdir(d)):
        tag = os.path.basename(run)
        if (tag in EXPLORED) != (name == 'plumb'):
            continue
        evf = os.path.join(rdir, tag + '.npz')
        if not os.path.exists(evf) or float(np.load(evf, allow_pickle=True)['gap']) >= 0.10:
            continue
        model, meta = load(run); D, N, fmt = meta['D'], meta['N'], meta['fmt']
        d = T.sample(NB, np.random.default_rng(67_001), D, N); X, v = TL.tokens(d, fmt); bay = T.bayes(d)
        Xt = torch.from_numpy(X); blk = model.blocks[0]; B = len(X); Tn = N + 1; qi = N
        with torch.no_grad():
            x = model.inp(Xt); q, k, _ = blk.qkv(blk.ln1(x)).split(blk.h * blk.dh, -1)
            sp = lambda t: t.view(B, Tn, blk.h, blk.dh).transpose(1, 2)
            qh, kh = sp(q), sp(k)
            allow = torch.ones(B, Tn, Tn, dtype=torch.bool); allow[:, :, qi] = False; allow[:, qi, qi] = True
            A = (qh @ kh.transpose(-1, -2) / math.sqrt(blk.dh)).masked_fill(~allow[:, None], float('-inf')).softmax(-1).numpy()
        H = blk.h
        o = dict(fmt=fmt, D=D, gap=float(np.load(evf, allow_pickle=True)['gap']),
                 q_on_examples=[float(A[:, hh, qi, :qi].sum(-1).mean()) for hh in range(H)], q_self=[float(A[:, hh, qi, qi].mean()) for hh in range(H)])
        W = model.inp.weight.detach().numpy().astype(np.float64); bb = model.inp.bias.detach().numpy().astype(np.float64)
        gam = blk.ln1.weight.detach().numpy().astype(np.float64); Wk = blk.qkv.weight.detach().numpy().astype(np.float64)[H * blk.dh:2 * H * blk.dh]
        dd = W.shape[0]
        Xc = X[:, :N, :].astype(np.float64); Xe = Xc.copy(); Xe[..., :D + 1] = 0                             # the even part (bias, products)
        a = Xe @ W.T + bb; a = a - a.mean(-1, keepdims=True); An = np.linalg.norm(a, axis=-1)                  # (B, N, d), (B, N)
        Wo = W[:, :D + 1] - W[:, :D + 1].mean(0, keepdims=True)
        qn = qh.numpy()[:, :, qi, :]
        ka = np.einsum('ed,bjd->bje', Wk, gam[None, None] * a).reshape(B, N, H, blk.dh)
        L = math.sqrt(dd / blk.dh) * np.einsum('bhe,bjhe->bhj', qn, ka) / An[:, None, :]
        ko = (Wk @ (gam[:, None] * Wo)).reshape(H, blk.dh, D + 1)
        K = math.sqrt(dd / blk.dh) * np.einsum('bhe,hek->bhk', qn, ko) / An.mean(1)[:, None, None]
        am = a.mean(1); Am = np.linalg.norm(am, axis=-1); n3 = np.linalg.norm(am + 3.0 * Wo[:, D][None], axis=-1); r3 = Am / n3
        Lm = L.mean(2)
        o.update(L=[float(Lm[:, hh].mean()) for hh in range(H)], K_y=[float(K[:, hh, D].mean()) for hh in range(H)],
                 K_y_rms=[float(np.sqrt((K[:, hh, D] ** 2).mean())) for hh in range(H)],
                 K_x_rms=[float(np.sqrt((K[:, hh, :D] ** 2).mean())) for hh in range(H)],
                 dlogit_y3=[float((Lm[:, hh] * (r3 - 1.0)).mean()) for hh in range(H)])
        o['kl'] = {nm: kl_of(Surg(model, z, D).eval(), X, v, bay) for nm, z in (('base', []), ('zero0', [0]), ('zero1', [1]), ('zeroall', [0, 1]))}
        res[tag] = o
        print(name, tag, json.dumps({k2: ([round(x2, 3) for x2 in v2] if isinstance(v2, list) else ({kk: round(vv, 5) for kk, vv in v2.items()} if isinstance(v2, dict) else v2)) for k2, v2 in o.items()}), flush=True)
    json.dump(res, open(('/tmp/e67_%s.json' if name == 'plumb' else 'results/e67/%s.json') % name, 'w'), indent=1)


if __name__ == '__main__':
    main()
