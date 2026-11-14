"""N7 (scale-free-theory, 2026-10-29): prototype of measurement M2 (routing classes and label fit) on clean tiny-PFN nets.

Tokens: t < 15 is (source r = t // 5, question q = t % 5; q = 4 is the query question's reading), t = 15 the query.
Mask (pfn.py): context rows attend to the 15 context tokens; the query row attends to context + itself.
Per layer and head:
  C(tau): number of distinct context rows per episode (greedy clustering, total variation <= tau), out of 15.
  Label fit on the 15 x 15 context block of log-attention: remove row means (softmax shift) and column means (a per-key
    weight w_j, per episode); regress what is left (the pair-specific part) on pair-class one-hots
    (same source?, same question?, i is a query reading?, j is a query reading?, i == j?), pooled over episodes.
    Reported: share of row-centred variance in column means (weighting), and R^2 of the pair part on labels (routing).
  Query row of the last layer: attention mass on each source's query reading vs that source's Bayes reliability
    log E[1/v_s | labelled residuals]; Pearson r over sources x episodes.
"""
import json, math, sys, os
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import torch
import world as W
import pfn as P

RUNS = '/home/agent/workspaces/mats/work/tiny-pfn/runs'
TAUS = [0.05, 0.1, 0.2]


def greedy_clusters(rows, tau):
    """rows (n, k) distributions; number of clusters: each row joins the first centre within TV <= tau."""
    centres = []
    for r in rows:
        for c in centres:
            if 0.5 * np.abs(r - c).sum() <= tau:
                break
        else:
            centres.append(r)
    return len(centres)


def pair_classes():
    src = np.array([t // 5 for t in range(15)]); q = np.array([t % 5 for t in range(15)])
    feats = np.stack([(src[:, None] == src[None]), (q[:, None] == q[None]), np.repeat((q == 4)[:, None], 15, 1),
                      np.repeat((q == 4)[None], 15, 0), np.eye(15, dtype=bool)], -1).astype(int)   # (15, 15, 5)
    code = (feats * (2 ** np.arange(5))).sum(-1)
    uniq = np.unique(code)
    onehot = (code[..., None] == uniq[None, None]).astype(float)                                  # (15, 15, nc)
    return onehot


def main():
    cfgs = sys.argv[1:] if len(sys.argv) > 1 else ['w32_h2_L2_mix4', 'w32_h2_L2', 'w32_h2_L1']
    n = 1500
    rng = np.random.default_rng(5)
    d = W.sample_np(n, rng)
    rngid = np.random.default_rng(99)
    sp = W.source_posterior(d)                                         # (n, S, NV) from labelled residuals
    logprec = np.log((sp / W.VLEV[None, None]).sum(2))                 # log E[1/v | residuals] (n, S)
    OH = pair_classes(); nc = OH.shape[-1]
    res = {}
    for cfg in cfgs:
        for s in range(5):
            rd = os.path.join(RUNS, f'{cfg}_s{s}')
            if not os.path.exists(os.path.join(rd, 'model.pt')):
                continue
            meta = json.load(open(os.path.join(rd, 'meta.json')))
            model = P.build(meta); model.load_state_dict(torch.load(os.path.join(rd, 'model.pt'))); model.eval()
            semb = rngid.standard_normal((n, W.S, meta['ds'])) / 4
            qemb = rngid.standard_normal((n, W.Q, meta['ds'])) / 4
            X, m = W.assemble(semb, qemb, d['o'], d['y'])
            with torch.no_grad():
                out = model(torch.from_numpy(X), torch.from_numpy(m), want=True)
            attns = [a.numpy() for a in out[-1]]                       # per layer (n, H, T, T)
            key = f'{cfg}_s{s}'; res[key] = {}
            for l, A in enumerate(attns):
                H = A.shape[1]
                for h in range(H):
                    a = A[:, h]                                       # (n, 16, 16)
                    ctx = a[:, :15, :15]
                    Cs = {tau: float(np.mean([greedy_clusters(ctx[e], tau) for e in range(min(n, 400))])) for tau in TAUS}
                    la = np.log(np.maximum(ctx, 1e-12))
                    rc = la - la.mean(2, keepdims=True)                # remove row constants
                    colm = rc.mean(1, keepdims=True)                  # per-key weight w_j (per episode)
                    pairp = rc - colm                                 # pair-specific part
                    tot = rc.var(axis=(1, 2)).mean()
                    col_share = float(colm.var(axis=(1, 2)).mean() / max(tot, 1e-12))
                    Y = pairp.reshape(n, -1)
                    Xd = OH.reshape(-1, nc)
                    beta, *_ = np.linalg.lstsq(np.tile(Xd, (n, 1)), Y.ravel(), rcond=None)
                    fit = (Xd @ beta)[None]
                    r2 = float(1 - ((Y - fit) ** 2).sum() / max(((Y - Y.mean()) ** 2).sum(), 1e-12))
                    rec = {'C_mean_by_tau': Cs, 'col_share': col_share, 'pair_R2_labels': r2}
                    if l == len(attns) - 1:
                        qrow = a[:, 15, :]                             # query row over 16 tokens
                        mass = np.stack([qrow[:, r * 5 + 4] for r in range(W.S)], 1)    # (n, S)
                        x1 = np.log(np.maximum(mass, 1e-12)); x1 = x1 - x1.mean(1, keepdims=True)
                        y1 = logprec - logprec.mean(1, keepdims=True)
                        rec['query_mass_vs_logprec_r'] = float(np.corrcoef(x1.ravel(), y1.ravel())[0, 1])
                        rec['query_self_mass_mean'] = float(qrow[:, 15].mean())
                    res[key][f'L{l + 1}H{h}'] = rec
            print(key, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else {a_: round(b_, 2) for a_, b_ in vv.items()})
                                        for kk, vv in v.items()} for k, v in res[key].items()}), flush=True)
    json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n7_routing_classes.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
