"""Analyses of trained token nets (Amendment 6; theory note 6.3), W0. For one run:
  perm : outputs before vs after a per-episode permutation of the context tokens (no positional encoding).
  M2   : per layer and head, routing classes (greedy clustering of context attention rows, TV <= 0.05 / 0.1 / 0.2),
         per-key column share of row-centred log-attention, label R^2 (held-out half) of the remaining pair part on
         pair-class one-hots, and in the last layer the within-episode correlation of the query's log mass on each
         source's query reading with that source's log E[1/v | its labelled residuals].
  M3   : substitute layer-l attention (after a per-episode permutation): (ii) per-episode pair-class mean logits,
         (iii) another episode's attention, (iv) the batch-averaged attention; KL to the net and to Bayes.
  M5   : last layer, the query token's attention to itself vs the net's and Bayes's predictive variance (R^2).
  M6   : effective rounds: change of source 0's query-question token after layer 1 when its own errors are redrawn vs
         when the other sources' errors are redrawn (mean norm ratio other / own).
usage: python tokan.py RUN [RUN ...] --n 2000 --out results/tokan.jsonl"""
import argparse, json, math, os
import numpy as np
import torch
import sfworld as W
import sftrain as T
import sfmodels as M

torch.set_num_threads(1)


def load(run):
    meta = json.load(open(os.path.join(run, 'meta.json')))
    a = argparse.Namespace(**meta)
    for k, v in dict(gid=False, attn='', cut='', cut_k='', nocross=False, cut_kv='kv').items():
        if not hasattr(a, k):
            setattr(a, k, v)
    model = T.build_model(a, a.world)
    model.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu')); model.eval()
    return model, a


def inputs(d, a, seed=7, perm=None):
    inp = T.build_inputs(d, 'tok', np.random.default_rng(seed), 'cpu', want_group=False)
    if perm is not None:                                   # per-episode permutation of the context tokens
        n, Tt = inp['tok'].shape[:2]
        idx = torch.cat([torch.from_numpy(perm), torch.full((n, 1), Tt - 1)], 1).long()
        inp = dict(tok=torch.gather(inp['tok'], 1, idx[:, :, None].expand(-1, -1, inp['tok'].shape[2])),
                   src=torch.gather(inp['src'], 1, idx), qq=torch.gather(inp['qq'], 1, idx),
                   grp=torch.gather(inp['grp'], 1, idx))
    return inp


def mixture_kl_grid(p, q, grid):
    """KL between two K-mixtures (mu, ls, lg) per episode on given grids (n, G)."""
    lp = W.mix_logpdf(grid, *p); lq = W.mix_logpdf(grid, *q)
    w = np.exp(lp)
    return np.trapezoid(w * (lp - lq), grid, axis=1)


def forward_sub(model, inp, layer, mode, attn_cache=None):
    """Forward with layer `layer`'s attention replaced. mode in {'class', 'other', 'avg'}."""
    allows, _ = model.masks(inp['src'], inp['qq'], inp['grp'], model.maskt, None)
    x = model.inp(inp['tok'])
    for l, b in enumerate(model.blocks):
        allow = allows[model.maskt[l]]
        B_, Tt, d = x.shape
        q, k, v = b.qkv(b.ln1(x)).split(d, -1)
        sp = lambda t: t.view(B_, Tt, b.h, b.dh).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        s = (q @ k.transpose(-1, -2) / math.sqrt(b.dh)).masked_fill(~allow[:, None], float('-inf'))
        a = s.softmax(-1)
        if l == layer:
            if mode == 'other':
                a = torch.roll(a, 1, 0)
            elif mode == 'avg':
                a = a.mean(0, keepdim=True).expand_as(a)
            elif mode == 'class':
                src, qq = inp['src'], inp['qq']
                cls = ((src[:, :, None] == src[:, None, :]).long() * 1 + (qq[:, :, None] == qq[:, None, :]).long() * 2
                       + (qq[:, :, None] == 4).long() * 4 + (qq[:, None, :] == 4).long() * 8
                       + (torch.arange(Tt)[None, :, None] == torch.arange(Tt)[None, None, :]).long() * 16
                       + (src[:, :, None] < 0).long() * 32 + (src[:, None, :] < 0).long() * 64)
                sf = torch.where(allow[:, None], s, torch.zeros_like(s))
                out = torch.empty_like(s)
                for c in torch.unique(cls):
                    m = (cls == c)[:, None].expand_as(s) & allow[:, None]
                    cnt = m.sum((2, 3), keepdim=True).clamp_min(1)
                    mean = (sf * m).sum((2, 3), keepdim=True) / cnt
                    out = torch.where(m, mean.expand_as(s), out)
                out = out.masked_fill(~allow[:, None], float('-inf'))
                a = out.softmax(-1)
        x = x + b.o((a @ v).transpose(1, 2).reshape(B_, Tt, d))
        x = x + b.mlp(b.ln2(x))
    return M.split_head(model.out(model.ln(x[:, -1])), model.K)


def greedy_classes(rows, tau):
    """rows (R, K) distributions; greedy clustering by total variation <= tau. Returns number of clusters."""
    reps = []
    for r in rows:
        if not any(0.5 * np.abs(r - c).sum() <= tau for c in reps):
            reps.append(r)
    return len(reps)


def analyse(run, n=2000, seed=0):
    model, a = load(run)
    N = a.N
    rng = np.random.default_rng([seed, 555, N])
    d = W.sample(n, N, rng)
    B = W.bayes(d)
    res = dict(run=run, N=N, attn=a.attn or 'dense:all x%d' % a.L)
    Tt = N * W.Q + 1
    with torch.no_grad():
        inp = inputs(d, a)
        (mu, ls, lg), hs, attns = model(**inp, want=True)
        base = [t.double().numpy() for t in (mu, ls, lg)]
        # ---- permutation equivariance
        perm = np.argsort(rng.random((n, Tt - 1)), 1)
        inp_p = inputs(d, a, perm=perm)
        mu2, ls2, lg2 = model(**inp_p)
        res['perm_maxdiff'] = float(max((mu - mu2).abs().max(), (ls - ls2).abs().max(), (lg - lg2).abs().max()))
        kl_net_bayes = W.kl_bayes_model(B, *base)
        res['kl_bayes'] = float(kl_net_bayes.mean())
        # ---- M2
        src = inp['src'][0].numpy(); qq = inp['qq'][0].numpy()
        ctx = np.arange(Tt - 1)
        m2 = []
        half = n // 2
        for l, A in enumerate(attns):
            A = A.double().numpy()                                   # (n, H, T, T)
            for h in range(A.shape[1]):
                rows = A[:, h][:, ctx][:, :, ctx]                    # context rows over context keys (n, T-1, T-1)
                C = {tau: float(np.mean([greedy_classes(rows[e], tau) for e in range(min(n, 200))])) for tau in (0.05, 0.1, 0.2)}
                La = np.log(np.maximum(rows, 1e-30))
                Lr = La - La.mean(2, keepdims=True)                  # row-centred
                colm = Lr.mean(1, keepdims=True)                     # per-key column means (weighting w_j)
                tot = ((Lr - Lr.mean((1, 2), keepdims=True)) ** 2).sum()
                col_share = float(((colm - colm.mean((1, 2), keepdims=True)) ** 2).sum() * Lr.shape[1] / tot)
                Rm = (Lr - colm).reshape(n, -1)
                ii, jj = np.meshgrid(ctx, ctx, indexing='ij'); ii = ii.reshape(-1); jj = jj.reshape(-1)
                X = np.stack([src[ii] == src[jj], qq[ii] == qq[jj], qq[ii] == 4, qq[jj] == 4, ii == jj,
                              (src[ii] == src[jj]) & (qq[jj] == 4), (src[ii] == src[jj]) & (qq[ii] == 4)], 1).astype(float)
                X = np.concatenate([X, np.ones((len(X), 1))], 1)
                ytr = Rm[:half].mean(0); coef = np.linalg.lstsq(X, ytr, rcond=None)[0]
                pred = X @ coef
                yte = Rm[half:]
                r2 = float(1 - ((yte - pred[None]) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
                m2.append(dict(layer=l + 1, head=h, C=C, col_share=col_share, label_r2=r2))
        res['M2'] = m2
        # last layer: query row mass on each source's query reading vs log E[1/v | residuals]
        R = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]
        ssr = (R ** 2).sum(2)
        llv = -ssr[:, :, None] / (2 * W.VLEV) - 0.5 * W.QL * np.log(W.VLEV)  # labelled-residual likelihood per level
        llv -= llv.max(2, keepdims=True); post = np.exp(llv); post /= post.sum(2, keepdims=True)  # (n, N, 3)
        logprec = np.log((post / W.VLEV[None, None]).sum(2))               # log E[1/v | residuals]
        qr = [s * W.Q + 4 for s in range(N)]
        A = attns[-1].double().numpy()
        rs = []
        for h in range(A.shape[1]):
            lm = np.log(np.maximum(A[:, h, Tt - 1][:, qr], 1e-30))
            x1 = lm - lm.mean(1, keepdims=True); x2 = logprec - logprec.mean(1, keepdims=True)
            rs.append(float((x1 * x2).sum() / math.sqrt((x1 ** 2).sum() * (x2 ** 2).sum())))
        res['M2_query_logmass_vs_logprec_r'] = rs
        # ---- M5: query self-attention in the last layer vs predictive variance
        w = np.exp(base[2] - np.logaddexp.reduce(base[2], axis=1, keepdims=True))
        mnet = (w * base[0]).sum(1); vnet = (w * (np.exp(2 * base[1]) + base[0] ** 2)).sum(1) - mnet ** 2
        m5 = []
        for h in range(A.shape[1]):
            a0 = A[:, h, Tt - 1, Tt - 1]
            r2n = float(np.corrcoef(a0, vnet)[0, 1] ** 2); r2b = float(np.corrcoef(a0, B['var'])[0, 1] ** 2)
            m5.append(dict(head=h, r2_net_var=r2n, r2_bayes_var=r2b, mean_a0=float(a0.mean())))
        res['M5'] = m5
        # ---- M6: effective rounds (source 0's query-question token after layer 1)
        h1 = hs[1][:, 4].double().numpy()
        e = d['o'] - d['y'][:, None, :]
        fresh_v = W.VLEV[rng.integers(0, 3, (n, N))]
        fresh_e = rng.standard_normal((n, N, W.Q)) * np.sqrt(fresh_v)[:, :, None]
        dd_own = dict(d); eo = e.copy(); eo[:, 0] = fresh_e[:, 0]; dd_own['o'] = d['y'][:, None, :] + eo
        dd_oth = dict(d); et = e.copy(); et[:, 1:] = fresh_e[:, 1:]; dd_oth['o'] = d['y'][:, None, :] + et
        _, hs_o, _ = model(**inputs(dd_own, a), want=True); _, hs_t, _ = model(**inputs(dd_oth, a), want=True)
        own = np.linalg.norm(hs_o[1][:, 4].double().numpy() - h1, axis=1).mean()
        oth = np.linalg.norm(hs_t[1][:, 4].double().numpy() - h1, axis=1).mean()
        res['M6_other_over_own'] = float(oth / own)
        # ---- M3: substitutions after a per-episode permutation
        m3 = {}
        for l in range(len(model.blocks)):
            for mode in ('class', 'other', 'avg'):
                p = [t.double().numpy() for t in forward_sub(model, inp_p, l, mode)]
                kl_net = mixture_kl_grid(base, p, B['grid'])
                kl_b = W.kl_bayes_model(B, *p)
                m3[f'L{l + 1}_{mode}'] = dict(kl_to_net=float(kl_net.mean()), kl_to_bayes=float(kl_b.mean()))
        res['M3'] = m3
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+'); ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--out', default='results/tokan.jsonl')
    args = ap.parse_args()
    for r in args.runs:
        res = analyse(r, args.n)
        print(json.dumps(res)[:600], flush=True)
        with open(args.out, 'a') as f:
            f.write(json.dumps(res) + '\n')
