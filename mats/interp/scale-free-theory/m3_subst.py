"""M3 (scale-free-theory addendum, PREREG-SF.md): attention substitutions that separate routing from weighting.

For a clean 2-layer net and a layer l in {0, 1} (block 1, block 2), replace that layer's attention (all rows, all heads)
by one of:
  fit        softmax over allowed keys of R_h(pair class) + w_j: R_h fitted per head on fit episodes (label routing),
             w_j the episode's own per-key effect (two-way additive fit of the episode's log-attention around R);
  aligned    the attention of a partner episode (fixed derangement of the test set); with the fixed token layout this
             keeps every row and column on the same (source, question) labels: routing kept, weighting from elsewhere;
  misaligned the partner's attention with its 15 context positions randomly permuted (rows and columns): routing broken.
Score: mean over test episodes of KL(net || variant) and KL(Bayes || variant) on a y-grid.
"""
import json, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/scale-free-theory')
from sf_net import load_clean, forward, allow_mask, episodes, head_logpdf, bayes_logpdf, kl_grid

NETS = [f'w32_h2_L2_s{s}' for s in range(5)] + [f'w32_h2_L2_mix4_s{s}' for s in range(5)]
FIT_SEED, TEST_SEED, PART_SEED, PERM_SEED = 501, 502, 503, 504
N_FIT, N_TEST = 2048, 4000


def pair_codes(T=16):
    src = np.array([t // 5 for t in range(15)] + [-1]); q = np.array([t % 5 for t in range(15)] + [-1])
    isq = np.arange(T) == T - 1; isqq = (q == 4)
    f = [(src[:, None] == src[None]) & (src[:, None] >= 0), (q[:, None] == q[None]) & (q[:, None] >= 0),
         np.repeat(isqq[:, None], T, 1), np.repeat(isqq[None], T, 0), np.repeat(isq[:, None], T, 1),
         np.repeat(isq[None], T, 0), np.eye(T, dtype=bool)]
    code = sum(fi.astype(int) * (2 ** i) for i, fi in enumerate(f))
    uniq, inv = np.unique(code, return_inverse=True)
    return inv.reshape(T, T), len(uniq)


CODES, NC = pair_codes()


def two_way(L, M, R, iters=8):
    """L, M: (n, T, T) log-attention and validity; R: (T, T) class effect. Fit u_i + w_j to L - R on valid entries."""
    resid = np.where(M, L - R[None], 0.0)
    u = np.zeros(L.shape[:2]); w = np.zeros((L.shape[0], L.shape[2]))
    cnt_r = M.sum(2).clip(1); cnt_c = M.sum(1).clip(1)
    for _ in range(iters):
        u = (np.where(M, resid - w[:, None, :], 0)).sum(2) / cnt_r
        w = (np.where(M, resid - u[:, :, None], 0)).sum(1) / cnt_c
    return u, w


def fit_R(L, M, iters=6):
    """Per-head class table R (NC,) fitted with per-episode row and column effects."""
    R = np.zeros(NC)
    for _ in range(iters):
        u, w = two_way(L, M, R[CODES])
        res = np.where(M, L - u[:, :, None] - w[:, None, :], 0.0)
        num = np.zeros(NC); den = np.zeros(NC)
        np.add.at(num, CODES, res.sum(0)); np.add.at(den, CODES, M.sum(0))
        R = num / den.clip(1)
    return R


def main():
    t0 = time.time()
    res = {}
    for name in (sys.argv[1:] or NETS):
        model, meta = load_clean(name)
        dF, XF, mF = episodes(N_FIT, FIT_SEED, meta['ds'])
        dT, XT, mT = episodes(N_TEST, TEST_SEED, meta['ds'])
        allowT = allow_mask(mT).numpy()[:, None]                       # (n, 1, T, T)
        allowF = allow_mask(mF).numpy()[:, None]
        with torch.no_grad():
            headT, attT = forward(model, XT, mT)
            _, attF = forward(model, XF, mF)
        lp_net = head_logpdf(headT); lp_bayes = bayes_logpdf(dT)
        kl_bn = float(kl_grid(lp_bayes, lp_net).mean())
        rng = np.random.default_rng(PART_SEED)
        partner = (np.arange(N_TEST) + rng.integers(1, N_TEST)) % N_TEST    # a fixed derangement
        rngp = np.random.default_rng(PERM_SEED)
        perms = np.stack([np.concatenate([rngp.permutation(15), [15]]) for _ in range(N_TEST)])
        rec = {'kl_bayes_net': kl_bn}
        for li in range(len(model.blocks)):
            AF = attF[li].numpy(); AT = attT[li].numpy()
            H = AF.shape[1]
            # fit R per head on fit episodes; build the 'fit' attention for test episodes
            fitA = np.zeros_like(AT)
            for h in range(H):
                MF = np.broadcast_to(allowF[:, 0], AF[:, h].shape)
                LF = np.log(np.maximum(AF[:, h], 1e-30))
                R = fit_R(LF, MF)
                MT_ = np.broadcast_to(allowT[:, 0], AT[:, h].shape)
                LT = np.log(np.maximum(AT[:, h], 1e-30))
                u, w = two_way(LT, MT_, R[CODES])
                logits = np.where(MT_, R[CODES][None] + w[:, None, :], -np.inf)
                logits = logits - logits.max(2, keepdims=True)
                e = np.exp(logits); fitA[:, h] = e / e.sum(2, keepdims=True)
            aligned = AT[partner]
            mis = aligned.copy()
            for i in range(N_TEST):
                p = perms[i]
                mis[i] = aligned[i][:, p][:, :, p]
            # the query row must still respect the mask (it does: permutation keeps context->context, query fixed)
            out = {}
            for vname, A in [('fit', fitA), ('aligned', aligned), ('misaligned', mis)]:
                with torch.no_grad():
                    hv, _ = forward(model, XT, mT, overrides={li: torch.from_numpy(A.astype(np.float32))})
                lp_v = head_logpdf(hv)
                out[vname] = {'kl_net_variant': float(kl_grid(lp_net, lp_v).mean()),
                              'kl_bayes_variant': float(kl_grid(lp_bayes, lp_v).mean())}
            rec[f'layer{li + 1}'] = out
        res[name] = rec
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else {a: {b: round(c, 4) for b, c in bb.items()} for a, bb in v.items()}) for k, v in rec.items()}),
              round(time.time() - t0), 's', flush=True)
        json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/m3_subst.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
