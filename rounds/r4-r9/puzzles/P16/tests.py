"""Stage-0 unit tests (python tests.py [--quick]). Prints one line per test and writes results/tests.json."""
import json, sys, time, os
import numpy as np
import torch
import gen
from gen import (Episode, Col, xfun_col, bits_table, make_episode, sample_rows, ybits, H, NAMESPACES, ALL_TYPES)
from oracle import cmi_batch, joint_BbY, pack, best_device

QUICK = '--quick' in sys.argv
DEV = best_device()
R = {}


def rec(name, ok, **kw):
    R[name] = dict(ok=bool(ok), **kw)
    print(('PASS ' if ok else 'FAIL ') + name, json.dumps(kw, default=float)[:300], flush=True)


def hand_episode(cand_fn, eps_flip=0.0, w=4, par=()):
    """x in {0,1}^10, y_j = x_j (eta 0, per-bit), B = two Bern(.5) noise columns, candidate built by cand_fn."""
    nx = 10; X = 2 ** nx
    XB = bits_table(nx)
    gtab = XB[:, :w].T.copy()
    YB = bits_table(w)
    py = np.ones((X, 2 ** w))
    for j in range(w):
        py *= (YB[None, :, j] == gtab[j][:, None])
    noise = Col('T0', 'noise', (), np.full((X, 8), .5))
    cols = [noise, Col('T0', 'noise', (), np.full((X, 8), .5)), cand_fn(XB)]
    return Episode('unit', 0, nx, w, 2, np.arange(w), gtab, np.zeros(X, np.int64), 'bit', 0.0, 'pal', py, cols, 'hand')


def H2(e):
    return 0.0 if e in (0, 1) else -(e * np.log2(e) + (1 - e) * np.log2(1 - e))


def main():
    t0 = time.time()
    # namespaces
    ids = list(NAMESPACES.values())
    s = {ns: gen.rng_for(ns, 0, 0).integers(0, 2 ** 62, 4).tolist() for ns in NAMESPACES}
    rec('namespaces_disjoint', len(set(ids)) == len(ids) and len({tuple(v) for v in s.values()}) == len(s))
    # T2 analytic: b = x0 xor Bern(eps) -> CMI = 1 - H(eps)
    errs = []
    for e in (0, .1, .25, .4):
        ep = hand_episode(lambda XB, e=e: xfun_col('T2', 'x0', XB[:, 0], e))
        errs.append(abs(float(cmi_batch([ep], DEV)[0]) - (1 - H2(e))))
    rec('T2_equals_1_minus_H', max(errs) < 1e-9, max_abs_err=max(errs))
    # T4: b = x0 xor B_0 -> I(b; Y | B) = I(x0; Y | B) = 1 bit (y_0 = x0)
    X = 1024
    def t4(XB):
        P1 = np.zeros((X, 8))
        for cfg in range(8):
            P1[:, cfg] = XB[:, 0] ^ (cfg & 1)
        return Col('T4', 'x0^B0', (0,), P1)
    ep = hand_episode(t4)
    ref = hand_episode(lambda XB: xfun_col('T2', 'x0', XB[:, 0], 0))
    a, b = float(cmi_batch([ep], DEV)[0]), float(cmi_batch([ref], DEV)[0])
    rec('T4_equals_I_g_Y_given_B', abs(a - b) < 1e-9 and abs(a - 1) < 1e-9, cmi_T4=a, cmi_g=b)
    # negatives: CMI < 1e-12 (generator episodes, all strata)
    n_per = 3 if QUICK else 8
    worst = {}
    for t in ['T0', 'T3a', 'T3b', 'HN1', 'HN2', 'HN3', 'HN4']:
        vals = []
        for m in (2, 6, 12):
            for r in range(n_per):
                ep = make_episode('unit', ALL_TYPES.index(t) * 1000 + m * 50 + r, t, m, 6)
                vals.append(ep.cmi)
        worst[t] = max(vals)
    rec('negatives_cmi_below_1e-12', all(v < 1e-12 for v in worst.values()), worst=worst)
    # Monte Carlo: E[log p(b,y|B) - log p(b|B) - log p(y|B)] over sampler rows = oracle CMI (within 3 SE)
    nmc = 200_000 if QUICK else 1_000_000
    per = 2 if QUICK else 5
    bad, zs = [], []
    for t in ALL_TYPES:
        for r in range(per):
            m = (2, 6, 12)[r % 3]
            ep = make_episode('unit', 500_000 + ALL_TYPES.index(t) * 1000 + r, t, m, 6)
            p = joint_BbY(pack([ep], DEV, torch.float64))[0].cpu().numpy()      # (2^m, 2, Y)
            rng = gen.rng_for('unit', 900_000 + ALL_TYPES.index(t) * 1000 + r)
            if ep.cand.markov:
                continue   # Markov noise: rows are dependent; its marginal is covered by the negatives test
            _, y, V = sample_rows(ep, nmc, rng, birth=0)
            bi = (V[:, :m].astype(np.int64) << np.arange(m)).sum(1)
            bv = V[:, m].astype(np.int64)
            pB = p.sum((1, 2)); pBb = p.sum(2); pBy = p.sum(1)
            with np.errstate(divide='ignore'):
                ll = np.log2(p[bi, bv, y]) + np.log2(pB[bi]) - np.log2(pBb[bi, bv]) - np.log2(pBy[bi, y])
            est, se = ll.mean(), ll.std() / np.sqrt(nmc)
            z = (est - ep.cmi) / max(se, 1e-12)
            zs.append(float(z))
            if not np.isfinite(est) or (abs(z) > 3 and abs(est - ep.cmi) > 1e-9):
                bad.append((t, m, float(est), float(ep.cmi), float(se)))
    # 3 SE at ~100 tests: a few exceed by chance; fail only if > 5% exceed or any is infinite (a zero-prob sample)
    rec('oracle_matches_monte_carlo', len(bad) <= max(1, int(.05 * len(zs))), n=len(zs), n_beyond_3se=len(bad), bad=bad[:5],
        max_abs_z=max(abs(z) for z in zs))
    # FR masks and pre-birth identity (random init model, fp32)
    from reader import FR
    torch.manual_seed(0)
    for mode in ('joint', 'fact'):
        model = FR(mode=mode).to(DEV).eval()
        ep = make_episode('unit', 777, 'T2', 6, 6)
        _, y, V = sample_rows(ep, H + 32, gen.rng_for('unit', 778))
        Vt = torch.as_tensor(V, dtype=torch.long, device=DEV)[None]
        Yt = torch.as_tensor(ybits(y, 6), dtype=torch.long, device=DEV)[None]
        order = torch.randperm(6, generator=torch.Generator().manual_seed(1))[None].to(DEV)
        with torch.no_grad():
            model.dec.L if mode == 'joint' else None
            base = per_output_logp(model, Vt, Yt, order)
            t = H + 5
            ok = True
            # perturb later rows (inputs and labels)
            V2, Y2 = Vt.clone(), Yt.clone(); V2[:, t + 1:] = 1 - V2[:, t + 1:].clamp(min=0); Y2[:, t + 1:] = 1 - Y2[:, t + 1:]
            o2 = per_output_logp(model, V2, Y2, order)
            ok &= torch.equal(o2[:, :t + 1], base[:, :t + 1])
            # perturb y_t at the output ranked j: outputs ranked < j... their probs must not change; for 'fact' all
            j_rank = 3
            jcol = int((order[0] == j_rank).nonzero()[0, 0])
            Y3 = Yt.clone(); Y3[:, t, jcol] = 1 - Y3[:, t, jcol]
            o3 = per_output_logp(model, Vt, Y3, order, probs_only=True)
            b3 = per_output_logp(model, Vt, Yt, order, probs_only=True)
            earlier_or_same = (order[0] <= j_rank) if mode == 'joint' else torch.ones(6, dtype=torch.bool, device=DEV)
            ok &= torch.equal(o3[:, t][:, earlier_or_same], b3[:, t][:, earlier_or_same])
            ok &= torch.equal(o3[:, :t], b3[:, :t])
            rec(f'FR_{mode}_causal_masks_bitwise', bool(ok))
            # with vs without candidate: identical on rows <= H
            Vn = Vt.clone(); Vn[:, :, -1] = -1
            lw = model.row_logp(Vt, Yt, order); lo = model.row_logp(Vn, Yt, order)
            rec(f'FR_{mode}_prebirth_identity', torch.equal(lw[:, :H + 1], lo[:, :H + 1]) or float((lw[:, :H] - lo[:, :H]).abs().max()) == 0,
                max_abs_diff_rows_lt_H=float((lw[:, :H] - lo[:, :H]).abs().max()),
                note='row H is the first with b present (0-based)')
    # settlement mode: rows >= ctx do not change predictions for rows < ctx, and are invisible to each other
    model = FR(mode='joint').to(DEV).eval()
    with torch.no_grad():
        ctx = H + 8
        a = per_output_logp(model, Vt, Yt, order, ctx=ctx)
        V4, Y4 = Vt.clone(), Yt.clone(); V4[:, ctx + 1:] = 1 - V4[:, ctx + 1:].clamp(min=0); Y4[:, ctx + 1:] = 1 - Y4[:, ctx + 1:]
        b = per_output_logp(model, V4, Y4, order, ctx=ctx)
        rec('FR_settlement_rows_independent', torch.equal(a[:, :ctx + 1], b[:, :ctx + 1]))
    # tautology (not a finding): C1 on parity(y) is at chance
    from baselines import c1_counts
    vals = {}
    for t in ('Lpar', 'T0'):
        eps = [make_episode('unit', 600_000 + ALL_TYPES.index(t) * 1000 + r, t, 6, 6) for r in range(40)]
        Vs, Ys = [], []
        for r, ep in enumerate(eps):
            _, y, V = sample_rows(ep, H + 16, gen.rng_for('unit', 700_000 + r))
            Vs.append(V); Ys.append(ybits(y, 6))
        g = c1_counts(torch.as_tensor(np.stack(Vs), dtype=torch.long, device=DEV),
                      torch.as_tensor(np.stack(Ys), dtype=torch.long, device=DEV), H)
        vals[t] = g.sum(1).cpu().numpy()
    from scipy.stats import mannwhitneyu
    au = mannwhitneyu(vals['Lpar'], vals['T0']).statistic / (40 * 40)
    rec('tautology_C1_chance_on_parity_y', abs(au - .5) < .2, auroc_Lpar_vs_T0=float(au))
    R['_wall_s'] = time.time() - t0
    os.makedirs('results', exist_ok=True)
    json.dump(R, open('results/tests.json', 'w'), indent=1, default=float)
    print('ALL PASS' if all(v['ok'] for k, v in R.items() if not k.startswith('_')) else 'SOME FAILED', f'{time.time()-t0:.0f}s')


def per_output_logp(model, V, Y, order, ctx=None, probs_only=False):
    """(B,T,w) per-output log-prob (or logits when probs_only) from the query stream."""
    B, T, cin = V.shape
    w = Y.shape[2]
    x = model.backbone(V, Y, ctx)
    q = x[:, 1]
    if model.mode == 'fact':
        lg = model.head(q[:, :, cin:])[..., 0]
    else:
        qs = q.reshape(-1, cin + w, q.shape[-1])
        lg = model.dec(qs[:, :cin], qs[:, cin:], Y.reshape(-1, w), order[0]).view(B, T, w)
    if probs_only:
        return lg
    return -torch.nn.functional.binary_cross_entropy_with_logits(lg, Y.float(), reduction='none')


if __name__ == '__main__':
    main()
