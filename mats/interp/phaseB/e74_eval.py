"""E74 per-net evaluation (PREREG-B amendment 82, planned) for world 1 with S sources (nets of train_n.py, or S = 3 nets of E37 / E60 read
with pfn2 / pfn_own builders), on results/e74_data_S<S>.npz:
  gap         test KL: mean test NLL minus exact Bayes NLL (16,000 episodes);
  rom_large, rom_middle   big-miss contrast on the matched pairs: sum (w_bigmiss - w_spread)[src] / sum W_Bayes[src] per range, with
              w = d mean / d o*_src by autograd on the source's summary token's o column; type A if rom_middle < 0;
  head measures (block 1, summary rows of all S sources, first 2,000 test episodes, the net's own forward and masks): per head P_h (mass
              on the source's own four labelled tokens), C_h (mass on the S summary tokens, self included), leak_h (mass on other
              sources' labelled tokens); pooling heads P_h >= 0.5, consensus heads C_h >= 0.5; for every head the mean norm-scaled own
              logit L_own and signed coefficient K over own keys and the norm term at |r| = 3 (E63's exact pre-LN decomposition, with
              the attention's own 1 / sqrt(dh)); an opposite-sign pair among the pooling heads.
usage: python e74_eval.py S OUTDIR RUN [RUN ...] -> OUTDIR/<tag>.json
"""
import json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_n as PN  # noqa: E402

torch.set_num_threads(1)
Q, QL, DS, OC, YC = 5, 4, 16, 32, 33


def load(run):
    meta = json.load(open(os.path.join(run, 'meta.json')))
    if meta.get('recipe') == 'train_n':
        model = PN.build_any(meta, P2, 2 * DS + 4)
    else:                                                                      # S = 3 reference nets (E37 ctrl, E60 own)
        import pfn_own as PO
        model = PO.build_any(meta, P2, 2 * DS + 4)
    model.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu')); model.eval(); model.requires_grad_(False)
    return model, meta


def mean_of(head):
    mu, logs, lg = head
    return (torch.softmax(lg, -1) * mu).sum(1)


def gap(model, z):
    nll = []
    with torch.no_grad():
        for i in range(0, len(z['test_X']), 2000):
            hd = model(torch.from_numpy(z['test_X'][i:i + 2000]), torch.from_numpy(z['test_valid'][i:i + 2000]))
            nll.append(P2.head_nll_np(tuple(t.numpy() for t in hd), z['test_ystar'][i:i + 2000]))
    return float((np.concatenate(nll) - z['test_nll_bayes']).mean())


def rom(model, z, S):
    src, rg = z['src'], z['range']; rows = np.arange(len(src)); summ = src * Q + QL; ws = []
    for k in (0, 1):
        w = []
        for i in range(0, len(src), 1000):
            Xt = torch.from_numpy(z['X%d' % k][i:i + 1000]).clone().requires_grad_(True)
            m = mean_of(model(Xt, torch.from_numpy(z['valid%d' % k][i:i + 1000])))
            g, = torch.autograd.grad(m.sum(), Xt)
            w.append(g[torch.arange(len(Xt)), torch.from_numpy(summ[i:i + 1000]), OC].numpy().astype(np.float64))
        ws.append(np.concatenate(w))
    Wb = z['WB0'][rows, src]
    return {rr: float((ws[1] - ws[0])[rg == rr].sum() / Wb[rg == rr].sum()) for rr in ('large', 'middle')}


def heads(model, z, S, nb=2000):
    X = z['test_X'][:nb]; v = z['test_valid'][:nb]; T = S * Q + 1
    summ = [s * Q + QL for s in range(S)]; labs = [[s * Q + q for q in range(QL)] for s in range(S)]
    with torch.no_grad():
        A = model(torch.from_numpy(X), torch.from_numpy(v), want=True)[-1][0].numpy()           # (B, H, T, T)
        blk = model.blocks[0]; Xt = torch.from_numpy(X)
        q = blk.qkv(blk.ln1(model.inp(Xt))).split(blk.h * blk.dh, -1)[0].view(nb, T, blk.h, blk.dh).numpy()
    H = A.shape[1]
    P = np.stack([A[:, :, summ[s]][:, :, labs[s]].sum(-1) for s in range(S)], 1)                # (B, S, H)
    C = np.stack([A[:, :, summ[s]][:, :, summ].sum(-1) for s in range(S)], 1)
    Lk = np.stack([A[:, :, summ[s]][:, :, [j for t in range(S) if t != s for j in labs[t]]].sum(-1) for s in range(S)], 1)
    W = model.inp.weight.detach().numpy().astype(np.float64); b = model.inp.bias.detach().numpy().astype(np.float64)
    gam = blk.ln1.weight.detach().numpy().astype(np.float64); d = W.shape[0]
    Wk = blk.qkv.weight.detach().numpy().astype(np.float64)[blk.h * blk.dh:2 * blk.h * blk.dh].reshape(blk.h, blk.dh, d)
    u = W[:, OC] - W[:, OC].mean(); U = np.linalg.norm(u); ku = np.einsum('hed,d->he', Wk, gam * u)
    Lown = np.zeros((nb, S, H)); Kown = np.zeros((nb, S, H)); dl3 = np.zeros((nb, S, H))
    for s in range(S):
        Xl = X[:, labs[s], :].astype(np.float64); X0 = Xl.copy(); X0[..., OC] = X0[..., YC]
        a = X0 @ W.T + b; a = a - a.mean(-1, keepdims=True); Aj = np.linalg.norm(a, axis=-1); rho = (a @ u) / (Aj * U)
        ka = np.einsum('hed,bjd->bjhe', Wk, gam[None, None] * a)                                  # (B, 4, H, dh)
        qs = q[:, summ[s]].astype(np.float64)                                                     # (B, H, dh)
        L = math.sqrt(d / blk.dh) * np.einsum('bhe,bjhe->bjh', qs, ka) / Aj[:, :, None]
        K = math.sqrt(d / blk.dh) * np.einsum('bhe,he->bh', qs, ku)[:, None, :] / Aj[:, :, None]
        r = 3.0
        ratio = 0.5 * (Aj / np.sqrt(Aj ** 2 + 2 * r * Aj * U * rho + r ** 2 * U ** 2) + Aj / np.sqrt(Aj ** 2 - 2 * r * Aj * U * rho + r ** 2 * U ** 2))
        Lown[:, s] = L.mean(1); Kown[:, s] = K.mean(1); dl3[:, s] = (L * (ratio[:, :, None] - 1.0)).mean(1)
    o = dict(P_h=[float(P[:, :, h].mean()) for h in range(H)], C_h=[float(C[:, :, h].mean()) for h in range(H)],
             leak_h=[float(Lk[:, :, h].mean()) for h in range(H)], L_own_h=[float(Lown[:, :, h].mean()) for h in range(H)],
             K_h=[float(Kown[:, :, h].mean()) for h in range(H)], dlogit_r3_h=[float(dl3[:, :, h].mean()) for h in range(H)])
    o['pool_heads'] = [h for h in range(H) if o['P_h'][h] >= 0.5]; o['n_pool'] = len(o['pool_heads'])
    o['n_consensus'] = int(sum(c >= 0.5 for c in o['C_h']))
    ks = [o['K_h'][h] for h in o['pool_heads']]
    o['opposite_pair'] = bool(any(x * y < 0 for i, x in enumerate(ks) for y in ks[i + 1:]))
    o['leak_pool'] = float(np.mean([o['leak_h'][h] for h in o['pool_heads']])) if o['pool_heads'] else None
    return o


def main():
    S = int(sys.argv[1]); outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
    z = np.load('results/e74_data_S%d.npz' % S)
    for run in sys.argv[3:]:
        tag = os.path.basename(run.rstrip('/')); p = os.path.join(outdir, tag + '.json')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, meta = load(run)
        assert meta.get('S', 3) == S
        o = dict(tag=tag, S=S, variant=meta.get('variant'), seed=meta['seed'], gap=gap(model, z), **rom(model, z, S))
        o = {('rom_' + k if k in ('large', 'middle') else k): v for k, v in o.items()}
        o['type'] = 'A' if o['rom_middle'] < 0 else 'B'
        if not o['variant']:
            o.update(heads(model, z, S))
        o['seconds'] = time.time() - t0
        json.dump(o, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p)
        print(tag, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in o.items() if not k.endswith('_h')}), flush=True)


if __name__ == '__main__':
    main()
