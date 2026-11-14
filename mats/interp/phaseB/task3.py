"""Task 3: fit q families per site on a noisy PFN run; coverage eps, wasted bits, bits, efficiency, chain a_l -> r_l.

usage: python task3.py RUN_DIR [--f3]     writes RUN_DIR/fam.json
"""
import argparse, json, math, os, time
import numpy as np
import torch
from sklearn.linear_model import OrthogonalMatchingPursuit
import nnoise as N
import mi, fam
from mirun import trace
import world as W

torch.set_num_threads(1)
TOK = {'QT': 15, 'QQ0': 4}


def named(d):
    _, _, mom, _ = W.references(d)
    Q = d['o'][:, :, W.QL]
    SS = ((d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]) ** 2).sum(2)
    sp = W.source_posterior(d)
    zQT = np.c_[mom['bayes'][0], np.log(mom['bayes'][1]), Q, np.log(SS)]
    zQQ0 = np.c_[Q[:, 0], np.log(SS[:, 0]), sp[:, 0]]
    return {'QT': zQT, 'QQ0': zQQ0}


def episodes(n, seed, ds):
    rng = np.random.default_rng(seed)
    d = W.sample_np(n, rng); X, v = W.tokens_np(d, rng, ds)
    return d, X, v


def mlp_pattern(model, l, xa):
    b = model.blocks[l - 1]
    with torch.no_grad():
        return (b.mlp[0](b.ln2(torch.as_tensor(xa, dtype=torch.float32))) > 0).numpy()


def push_ar(model, l, Ha, sigma, rng):
    """Token-local MLP sublayer of block l with its noise: a_l state -> centres of r_l (and the noisy m)."""
    b = model.blocks[l - 1]
    with torch.no_grad():
        x = torch.as_tensor(Ha, dtype=torch.float32)
        mm = N.sphere(b.hidden_pre(x)); m = mm + sigma * torch.as_tensor(rng.standard_normal(mm.shape), dtype=torch.float32)
        c = N.sphere(b.write_mlp(x, m))
    return c.numpy().astype(np.float64)


def graph_indegree(Aa, Pr, minfrac=0.01, maxk=24):
    """For each r-feature j active in >= minfrac of samples: dense linear R^2 of its pre-activation from the a-code,
    and the OMP in-degree needed to reach 90% of it."""
    act = (Aa > 0)
    use = np.flatnonzero(act.any(0))
    Xa = Aa[:, use]
    ntr = int(0.8 * len(Xa))
    out = []
    for j in range(Pr.shape[1]):
        y = Pr[:, j]
        if (y > 0).mean() < minfrac: continue
        A = fam.ridge(Xa[:ntr], y[:ntr, None], 1e-4)
        p = np.c_[Xa[ntr:], np.ones(len(Xa) - ntr)] @ A
        vt = ((y[ntr:] - y[ntr:].mean()) ** 2).sum()
        r2d = 1 - ((y[ntr:] - p[:, 0]) ** 2).sum() / vt
        deg = maxk + 1
        for kk in (1, 2, 3, 4, 6, 8, 12, 16, 24):
            if kk > Xa.shape[1]: break
            o = OrthogonalMatchingPursuit(n_nonzero_coefs=kk).fit(Xa[:ntr], y[:ntr])
            r2 = 1 - ((y[ntr:] - o.predict(Xa[ntr:])) ** 2).sum() / vt
            if r2 >= 0.9 * r2d:
                deg = kk; break
        out.append((float(r2d), deg))
    return out, len(use)


def run(a):
    model, meta = N.load_run(a.run); sg = meta['sigma']; s2 = sg ** 2
    mij = {(r['site'], r['group']): r for r in json.load(open(os.path.join(a.run, 'mi.json')))} if os.path.exists(os.path.join(a.run, 'mi.json')) else {}
    dF, XF, vF = episodes(a.nfit, 70_000, meta['ds']); zF = named(dF)
    dB, XB, vB = episodes(a.M, 71_000, meta['ds'])
    zbank = named(W.sample_np(a.M, np.random.default_rng(72_000)))
    gen = torch.Generator().manual_seed(73_000)
    muF, hF = trace(model, XF, vF, sg, gen)
    muB, hB = {}, {}
    for i in range(0, a.M, 8192):                           # keep only the analysed tokens of the big bank
        m_, h_ = trace(model, XB[i:i + 8192], vB[i:i + 8192], sg, gen)
        for s_ in m_:
            muB.setdefault(s_, []).append(m_[s_][:, [TOK['QT'], TOK['QQ0']]]); hB.setdefault(s_, []).append(h_[s_][:, [TOK['QT'], TOK['QQ0']]])
    muB = {k: np.concatenate(v) for k, v in muB.items()}; hB = {k: np.concatenate(v) for k, v in hB.items()}
    BT = {'QT': 0, 'QQ0': 1}
    rng = np.random.default_rng(74_000)
    sites = model.site_names()
    rows, qs = [], {}
    t0 = time.time()
    fams = ['F1', 'F2', 'F4k2', 'F4k4'] + (['F3'] if a.f3 else [])
    for s in sites:
        for g, t in TOK.items():
            Hfit = hF[s][:, t].astype(np.float64); Ht = hB[s][:a.neval, BT[g]].astype(np.float64)
            bank = muB[s][:, BT[g]].astype(np.float64); D = Hfit.shape[1]
            hbr = mi.entropy_bracket(Ht, bank, sg, np.arange(a.neval))
            Ir = mij.get((s, g), {})
            for f in fams:
                tq = time.time()
                if f == 'F1':
                    q = fam.fit_F1(Hfit, zF[g], zbank[g], s2)
                elif f == 'F2':
                    q = fam.fit_F2(Hfit, zF[g], s2)
                elif f.startswith('F4'):
                    k = int(f[-1]); q = fam.fit_F4(Hfit, s2, 4 * D, k, seed=1)
                else:
                    l = int(s[1:]) if s != 'emb' else None
                    if l is None: continue
                    xa = hF[f'a{l}'][:, t]
                    q = fam.fit_F3(Hfit, mlp_pattern(model, l, xa), s2)
                ev = fam.evaluate(q, Ht, hbr, bank, sg, rng)
                row = dict(site=s, group=g, fam=f, D=D, sigma=sg, depth=meta['depth'], h_in=hbr['h_in'], h_out=hbr['h_out'], hZ=mi.hZ_bits(D, sg),
                           I_lo=Ir.get('I_lo'), I_hi=Ir.get('I_hi'), **ev)
                for attr in ('top', 'top_r2', 'ncell', 'npat', 'ncomp', 'frac_rare', 'Hpat_mm', 'singletons', 'nsup', 'fve', 'Hsup_mm', 'r2', 'se2'):
                    if hasattr(q, attr): row[attr] = getattr(q, attr)
                if row['I_lo'] is not None:
                    row['eff_lo'] = row['I_lo'] / row['bits_desc']; row['eff_hi'] = row['I_hi'] / row['bits_desc']
                    row['eff_rate_lo'] = row['I_lo'] / row['bits_rate']
                rows.append(row); qs[(s, g, f)] = q
                print(f"{s:4s} {g:4s} {f:5s} eps [{ev['eps_lo']:6.2f},{ev['eps_hi']:6.2f}] wasted<={ev['wasted_ub']:7.2f} rate {ev['bits_rate']:6.2f} "
                      f"desc {ev['bits_desc']:6.2f} I [{row['I_lo'] if row['I_lo'] is None else round(row['I_lo'], 2)}, "
                      f"{row['I_hi'] if row['I_hi'] is None else round(row['I_hi'], 2)}] ({time.time() - tq:.0f}s, {time.time() - t0:.0f}s)", flush=True)
                json.dump(dict(rows=rows), open(os.path.join(a.run, 'fam.json'), 'w'), indent=1)
    # chain a_l -> r_l (token local)
    chain = []
    for l in range(1, meta['depth'] + 1):
        for g, t in TOK.items():
            for f in fams:
                if (f'a{l}', g, f) not in qs or (f'r{l}', g, f) not in qs: continue
                qa, qr = qs[(f'a{l}', g, f)], qs[(f'r{l}', g, f)]
                Ha = qa.sample(a.npush, rng)
                c = push_ar(model, l, Ha, sg, rng)
                tt = c[:a.neval] + sg * rng.standard_normal((a.neval, c.shape[1]))
                br = mi.entropy_bracket(tt, c, sg, np.arange(a.neval))
                CE = float(-qr.logpdf(tt).mean() / mi.LOG2)
                epsr = next(r for r in rows if r['site'] == f'r{l}' and r['group'] == g and r['fam'] == f)
                row = dict(l=l, group=g, fam=f, sigma=sg, depth=meta['depth'], chain_lo=CE - br['h_out'], chain_hi=CE - br['h_in'],
                           eps_r_lo=epsr['eps_lo'], eps_r_hi=epsr['eps_hi'])
                if f.startswith('F4'):
                    with torch.no_grad():
                        Aa = qa.sae.code(torch.as_tensor(hF[f'a{l}'][:, t], dtype=torch.float32)).numpy()
                        Pr = qr.sae.pre(torch.as_tensor(hF[f'r{l}'][:, t], dtype=torch.float32)).numpy()
                    gi, nuse = graph_indegree(Aa.astype(np.float64), Pr.astype(np.float64))
                    if gi:
                        row.update(graph_nfeat=len(gi), graph_used_a=nuse, graph_r2d_med=float(np.median([x[0] for x in gi])),
                                   graph_deg_med=float(np.median([x[1] for x in gi])), graph_deg_mean=float(np.mean([x[1] for x in gi])))
                chain.append(row)
                print('chain', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
                json.dump(dict(rows=rows, chain=chain), open(os.path.join(a.run, 'fam.json'), 'w'), indent=1)
    json.dump(dict(rows=rows, chain=chain), open(os.path.join(a.run, 'fam.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--f3', action='store_true')
    ap.add_argument('--nfit', type=int, default=20_000); ap.add_argument('--M', type=int, default=2 ** 17)
    ap.add_argument('--neval', type=int, default=4096); ap.add_argument('--npush', type=int, default=2 ** 16)
    run(ap.parse_args())
