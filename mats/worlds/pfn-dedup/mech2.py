"""Eval-only follow-up (coordinator request): exact-duplicate sweep, count decodability, patching interventions,
untrained-network and shuffled baselines. Runs on saved models, CPU or GPU.

usage: python mech2.py RUNS_DIR OUT_JSON [--n 600]
"""
import glob, json, math, os, sys
import numpy as np
import torch
import world as W
from pfn import PFN, DEV, fixed_set

MS = list(range(1, 9))
SWEEP_SEED = 40_000


def masks(valid):
    B, T = valid.shape
    qi = T - 1
    ctx = valid.clone(); ctx[:, qi] = False
    allow = ctx[:, None, :].expand(B, T, T).clone()
    allow[:, qi, qi] = True
    eye = torch.eye(T, dtype=torch.bool, device=valid.device)[None]
    return allow | (eye & ~valid[:, :, None])


def run(model, X, valid, patch=None):
    """Forward replicating PFN.forward; patch(li, x) may replace the residual after block li."""
    allow = masks(valid)
    x = model.inp(X); hs, attns = [x], []
    for li, blk in enumerate(model.blocks):
        x, a = blk(x, allow, True)
        if patch is not None:
            x = patch(li, x)
        hs.append(x); attns.append(a)
    y = model.out(model.ln(x[:, -1]))
    return y[:, 0], y[:, 1].clamp(-6, 3), hs, attns


def base_episodes(n, rng):
    d = W.sample_np(n * 2, rng)
    keep = np.where(d['k'] == 0)[0][:n]
    d = {k: v[keep] for k, v in d.items()}
    semb = rng.standard_normal((n, W.S, W.DS)) / 4
    qemb = rng.standard_normal((n, W.Q, W.DS)) / 4
    gemb = rng.standard_normal((n, W.S, W.DS)) / 4
    s0 = rng.integers(0, W.S, n)
    return d, semb, qemb, gemb, s0


def tokens_with_copies(base, m, keys):
    d, semb, qemb, gemb, s0 = base
    n = len(s0)
    src = np.concatenate([np.tile(np.arange(W.S), (n, 1)), np.repeat(s0[:, None], m - 1, 1)], 1)
    valid = np.ones_like(src, dtype=bool)
    X, vm = W.assemble(semb, qemb, gemb, src, valid, d['o'], d['y'], d['lab'], keys)
    return X, vm, src


def indep_copy_reference(d, s0, m):
    """GLS with the true partition plus (m-1) extra INDEPENDENT readings equal to source s0's query reading."""
    n = len(s0); means = np.zeros(n); vars_ = np.zeros(n)
    for i in range(n):
        Sg = W.SIG[d['pidx'][i]]
        k = m - 1
        Sa = np.zeros((W.S + k, W.S + k)); Sa[:W.S, :W.S] = Sg
        Sa[W.S:, W.S:] = (W.SG2 + W.SE2) * np.eye(k)
        oa = np.concatenate([d['o'][i, :, W.QL], np.full(k, d['o'][i, s0[i], W.QL])])
        Si = np.linalg.inv(Sa); one = np.ones(W.S + k)
        prec = 1 + one @ Si @ one
        means[i] = (one @ Si @ oa) / prec; vars_[i] = 1 / prec
    return means, vars_


def slope(dmu, dx):
    sel = np.abs(dx) > 0.3
    return float((dmu[sel] * dx[sel]).sum() / (dx[sel] ** 2).sum())


def _fit(Ha, t, lam):
    return np.linalg.solve(Ha.T @ Ha + lam * np.eye(Ha.shape[1]), Ha.T @ t)


def ridge_r2(H, t, groups, lams=(1e-2, 1e-1, 1, 10, 100, 1000)):
    """Retrained probe: episode-level split (first half of episodes fit, second half test); features
    standardised on the fit split; ridge lambda chosen by an inner episode-level split of the fit half."""
    ug = np.unique(groups); tr = np.isin(groups, ug[: len(ug) // 2])
    mu, sd = H[tr].mean(0), H[tr].std(0)
    sd = np.maximum(sd, 1e-2 * sd.max() + 1e-12)
    Hs = np.concatenate([(H - mu) / sd, np.ones((len(H), 1))], 1)
    gtr = ug[: len(ug) // 2]; inner = np.isin(groups, gtr[: len(gtr) // 2]) & tr; val = tr & ~inner
    best = min(lams, key=lambda l: ((t[val] - Hs[val] @ _fit(Hs[inner], t[inner], l)) ** 2).mean())
    w = _fit(Hs[tr], t[tr], best)
    p = Hs[~tr] @ w
    return float(1 - ((t[~tr] - p) ** 2).mean() / t[~tr].var())


def analyse(model, keys, base):
    d, semb, qemb, gemb, s0 = base
    n = len(s0)
    out = {'pull': {}, 'logprec_shift': {}, 'attn_total': {}, 'attn_per_copy': {}}
    mus, logs, Hdup, Mlab, Gid, Hfirst = {}, {}, {}, [], [], {}
    with torch.no_grad():
        for m in MS:
            X, vm, src = tokens_with_copies(base, m, keys)
            xb, mb = torch.tensor(X, device=DEV), torch.tensor(vm, device=DEV)
            mu, ls, hs, attns = run(model, xb, mb)
            mus[m], logs[m] = mu.cpu().numpy(), ls.cpu().numpy()
            # query attention on the copies' query-question tokens (rows of source s0)
            rows = (src == s0[:, None])                                       # (n, R)
            qtok = np.arange(src.shape[1]) * W.Q + W.QL
            per_layer = []
            for a in attns:
                A = a[:, :, -1, :].cpu().numpy()                              # (n, heads, T)
                tot = (A[:, :, qtok] * rows[:, None, :]).sum(-1)              # (n, heads)
                per_layer.append(tot.mean(0).tolist())
            out['attn_total'][m] = per_layer
            out['attn_per_copy'][m] = [[v / m for v in hh] for hh in per_layer]
            # hidden states of duplicate tokens at the query question, per layer (one row per copy)
            for li, h in enumerate(hs):
                H = h.cpu().numpy()[:, qtok]                                  # (n, R, d)
                sel = H[rows]                                                 # (n*m, d)
                Hdup.setdefault(li, []).append(sel)
                Hfirst.setdefault(li, {})[m] = H[np.arange(n), s0]            # original source token (n, d)
            Mlab.append(np.repeat(np.full(n, m), m)); Gid.append(np.repeat(np.arange(n) + 10_000 * m, m))
    odup = d['o'][np.arange(n), s0, W.QL]
    for m in MS:
        out['pull'][m] = slope(mus[m] - mus[1], odup - mus[1])
        out['logprec_shift'][m] = float((logs[1] - logs[m]).mean())
    M = np.concatenate(Mlab).astype(float); G = np.concatenate(Gid)
    # count decodability (log m) from duplicate tokens per layer; groups = (episode, m) so copies of one
    # episode-m never split across fit/test
    epi = np.concatenate([np.repeat(np.arange(n), m) for m in MS])
    out['count_r2'] = [ridge_r2(np.concatenate(Hdup[li]), np.log(M), epi) for li in sorted(Hdup)]
    # within-episode version: log m from (h_m - h_1) of the original source token, same episode
    D = {li: np.concatenate([Hfirst[li][m] - Hfirst[li][1] for m in MS[1:]]) for li in Hfirst}
    tm = np.concatenate([np.full(n, np.log(m)) for m in MS[1:]]); ge = np.concatenate([np.arange(n)] * (len(MS) - 1))
    out['count_r2_within'] = [ridge_r2(D[li], tm, ge) if np.abs(D[li]).max() > 0 else 0.0 for li in sorted(D)]
    rng = np.random.default_rng(0)
    out['count_r2_shuffled'] = [ridge_r2(np.concatenate(Hdup[li]), rng.permutation(np.log(M)), epi)
                                for li in sorted(Hdup)]
    # count may enter multiplicatively with episode-specific content: mean ||h_m - h_1|| per m, per layer
    out['delta_norm'] = {li: {m: float(np.linalg.norm(Hfirst[li][m] - Hfirst[li][1], axis=1).mean()) for m in MS}
                         for li in sorted(Hfirst)}
    return out, mus, odup


def patching(model, keys, base, m=4):
    """Replace the m=4 duplicate tokens' residual after layer li with the m=1 run's residual of the original
    source token (same question), all questions; measure the pull toward the duplicated value."""
    d, semb, qemb, gemb, s0 = base
    n = len(s0)
    X1, v1, src1 = tokens_with_copies(base, 1, keys)
    Xm, vmm, srcm = tokens_with_copies(base, m, keys)
    with torch.no_grad():
        mu1, _, hs1, _ = run(model, torch.tensor(X1, device=DEV), torch.tensor(v1, device=DEV))
        mum, _, _, _ = run(model, torch.tensor(Xm, device=DEV), torch.tensor(vmm, device=DEV))
    odup = d['o'][np.arange(n), s0, W.QL]
    mu1 = mu1.cpu().numpy(); mum = mum.cpu().numpy()
    res = {'pull_m': slope(mum - mu1, odup - mu1), 'patched': []}
    R1 = src1.shape[1]; Rm = srcm.shape[1]
    for li in range(len(model.blocks)):
        bi, rr = np.nonzero(srcm == s0[:, None])
        bi = np.repeat(bi, W.Q); qq = np.tile(np.arange(W.Q), len(rr))
        dst = np.repeat(rr, W.Q) * W.Q + qq
        srct = np.repeat(s0[np.nonzero(srcm == s0[:, None])[0]], W.Q) * W.Q + qq
        bi_t, dst_t, src_t = (torch.tensor(v, device=DEV) for v in (bi, dst, srct))
        def patch(l, x, li=li):
            if l != li: return x
            x = x.clone()
            x[bi_t, dst_t] = hs1[li + 1][bi_t, src_t]
            return x
        with torch.no_grad():
            mup, _, _, _ = run(model, torch.tensor(Xm, device=DEV), torch.tensor(vmm, device=DEV), patch)
        res['patched'].append({'layer': li, 'pull': slope(mup.cpu().numpy() - mu1, odup - mu1)})
    return res


def main(runs, outp, n=600):
    rng = np.random.default_rng(SWEEP_SEED)
    base = base_episodes(n, rng)
    d, semb, qemb, gemb, s0 = base
    # references
    _, rn, (md, vd), w = W.references(d)
    ref = {'true_duplicate_pull': {m: 0.0 for m in MS}, 'indep_pull': {}, 'indep_logprec_shift': {}}
    c1m, c1v = indep_copy_reference(d, s0, 1)
    odup = d['o'][np.arange(n), s0, W.QL]
    for m in MS:
        cm, cv = indep_copy_reference(d, s0, m)
        ref['indep_pull'][m] = slope(cm - c1m, odup - c1m)
        ref['indep_logprec_shift'][m] = float((0.5 * np.log(c1v / cv)).mean())
    results = {'reference': ref, 'models': []}
    for f in sorted(glob.glob(os.path.join(runs, 'L*_k*_s*/meta.json'))):
        meta = json.load(open(f))
        L, k, s = meta['depth'], meta['keys'], meta['seed']
        for trained in (True, False):
            torch.manual_seed(s)
            model = PFN(meta['fin'], L).to(DEV)
            if trained:
                model.load_state_dict(torch.load(os.path.join(os.path.dirname(f), 'model.pt'), map_location=DEV))
            model.eval()
            a, mus, _ = analyse(model, k, base)
            p = patching(model, k, base) if trained else None
            results['models'].append({'depth': L, 'keys': k, 'seed': s, 'trained': trained, **a, 'patching': p})
            print(f"L{L} k{k} s{s} {'trained' if trained else 'untrained'}: pull {[round(a['pull'][m], 3) for m in MS]} "
                  f"count_r2 {[round(v, 2) for v in a['count_r2']]} within {[round(v, 2) for v in a['count_r2_within']]}", flush=True)
    json.dump(results, open(outp, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600)
