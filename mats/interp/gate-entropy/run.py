"""Driver: python run.py SPEC OUT.json [--smoke]
SPEC examples:
  rnn:PATH[:name]                         AlgZoo noisy RNN (.npz from work/algzoo-noisy)
  arc:n:L:sigma:seed[:tol]                ARC He-ReLU MLP
  tanh:g:d:L:sigma:seed[:tol]             phase-R tanh LN MLP (bias 0.5)
  pfn:RUNDIR[:tolfrac]                    phase-B PFN (tol = tolfrac x RMS deviation from Bayes)
"""
import sys, os, json, math, time
import numpy as np
import torch
os.environ.setdefault('OMP_NUM_THREADS', '1')
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ent, nets, analyze as AN
from analyze import log

TAUS = (0.01, 0.001)


def build(spec):
    p = spec.split(':')
    if p[0] == 'rnn':
        return nets.RNNNet(p[1], name=(p[2] if len(p) > 2 else None))
    if p[0] == 'arc':
        n, L, sg, seed = int(p[1]), int(p[2]), float(p[3]), int(p[4])
        tol = float(p[5]) if len(p) > 5 else 0.1
        layers = sorted(set([1, 2, 4, 8, 16, 24, L]) if n <= 64 else set([1, 4, 8, 16, 24, L]))
        layers = [l for l in layers if l <= L]
        return nets.ArcNet(n, L, sg, seed, tol=tol, layers=layers)
    if p[0] == 'tanh':
        g, d, L, sg, seed = float(p[1]), int(p[2]), int(p[3]), float(p[4]), int(p[5])
        tol = float(p[6]) if len(p) > 6 else 0.1
        layers = sorted(set([1, 2, 4, 8, 16, 24, L]) if d <= 64 else set([1, 4, 8, 16, 24, L]))
        layers = [l for l in layers if l <= L]
        return nets.TanhNet(d, L, g, sg, seed, bias=0.5, tol=tol, layers=layers)
    if p[0] == 'pfn':
        net = nets.PFNNet(p[1])
        net.tolfrac = float(p[2]) if len(p) > 2 else 0.1
        return net
    raise ValueError(spec)


def sizes(net, smoke):
    if smoke:
        return dict(N_ent=1 << 13, N_flip=256, N_ho=2048, made_hidden=True, max_splits=64, n_cond_in=32, n_cond_rep=64)
    if net.kind == 'rnn':
        return dict(N_ent=1 << 21, N_flip=8192, N_ho=1 << 16, made_hidden=True, max_splits=4096, n_cond_in=512, n_cond_rep=256)
    if net.kind == 'pfn':
        return dict(N_ent=1 << 15, N_flip=1024, N_ho=8192, made_hidden=False, max_splits=512, n_cond_in=0, n_cond_rep=0)
    big = max(net.m.values()) >= 256
    return dict(N_ent=(1 << 17) if big else (1 << 19), N_flip=1024 if big else 4096, N_ho=8192 if big else 1 << 15,
                made_hidden=not big, max_splits=1024 if big else 2048, n_cond_in=128, n_cond_rep=256)


def entropy_panel(Gs, m, sz, groups=None, seed=0, allow_made=False):
    """Gs: unpacked bool (N, m)."""
    big = Gs.shape[1] > 300
    out = ent.all_estimates(Gs[:, :], logK_bits=m, made=False, groups=groups)
    nm = min(len(Gs), (1 << 15) if big else (1 << 17))
    sub = Gs[:nm]
    if Gs.shape[1] > 0:
        out['fvsbn'] = ent.made_upper(sub, hidden=0, seed=seed, epochs=(25 if big else 40), max_train=(1 << 14) if big else (1 << 16))
        ups = [out['indep_upper'], out['fvsbn']['H_up'] + 2 * out['fvsbn']['se']]
        if allow_made and sz['made_hidden'] and Gs.shape[1] <= 64:
            out['made'] = ent.made_upper(sub, hidden=min(max(2 * Gs.shape[1], 64), 256), seed=seed)
            ups.append(out['made']['H_up'] + 2 * out['made']['se'])
        out['upper'] = float(min(ups))
    else:
        out['upper'] = 0.0
    return out


def main():
    spec, outp = sys.argv[1], sys.argv[2]
    if spec == 'exact':                      # queue convenience: the estimator check (exact_check.py)
        import runpy
        runpy.run_path(os.path.join(HERE, 'exact_check.py'), run_name='__main__')
        return
    if spec.startswith('script|'):          # queue convenience: run a helper script with OUT as its argument
        import runpy
        sys.argv = [spec.split('|')[1], outp]
        runpy.run_path(os.path.join(HERE, spec.split('|')[1]), run_name='__main__')
        return
    if spec.startswith('fresh|'):           # queue convenience: fresh-data validation of E3 (e3_fresh.py)
        import runpy
        tag = os.path.basename(outp)[:-len('.json')].replace('FRESH_', '')
        sys.argv = ['e3_fresh.py', spec[len('fresh|'):], tag]
        runpy.run_path(os.path.join(HERE, 'e3_fresh.py'), run_name='__main__')
        return
    if spec == 'l1bound':                    # queue convenience: layer-1 Gaussian chain-rule lower bound
        import runpy
        sys.argv = ['layer1_bound.py', os.path.join(HERE, 'results', 'layer1_bound.json')]
        runpy.run_path(os.path.join(HERE, 'layer1_bound.py'), run_name='__main__')
        return
    smoke = '--smoke' in sys.argv
    t00 = time.time()
    net = build(spec)
    sz = sizes(net, smoke)
    log('net', net.name, 'kind', net.kind, 'layers', net.layers, sz)
    res = dict(spec=spec, name=net.name, kind=net.kind, sizes=sz, layers={}, capacity_bits=getattr(net, 'capacity_bits', None))
    # ---- flip and held-out sets
    st_f = net.sample(sz['N_flip'], seed=1)
    if net.kind == 'pfn':
        a = st_f['ans']; dev = a - st_f['bayes_mean']
        net.tol_abs = float(net.tolfrac * dev.pow(2).mean().sqrt())
    st_h = net.sample(sz['N_ho'], seed=2)
    res['answer_flip_set'] = net.answer_summary(st_f); res['answer_ho_set'] = net.answer_summary(st_h)
    log('answers', res['answer_ho_set'])
    # ---- entropy sample, streamed
    chunk = 1 << 14 if net.kind != 'pfn' else 1 << 12
    packed = {l: [] for l in net.layers}
    capb = {}
    for i in range(0, sz['N_ent'], chunk):
        st = net.sample(min(chunk, sz['N_ent'] - i), seed=1000 + i // chunk)
        for l in net.layers:
            packed[l].append(np.packbits(net.gates(st, l), axis=1))
        if i == 0 and net.kind != 'pfn':
            for l in net.layers:
                if net.kind == 'rnn':
                    if l >= 2:
                        sig = st['A'][l - 2]; nsd = net.sigma * net.r[l - 2]
                        capb[l] = AN.cov_capacity(sig, nsd)
                else:
                    if l >= 2:
                        sig = torch.relu(st['Z'][l - 2]) if net.kind == 'arc' else net.ln(torch.tanh(st['Z'][l - 2]))
                        capb[l] = AN.cov_capacity(sig, net.noise_sd_into(l)) if net.noise_sd_into(l) > 0 else None
        del st
    log('entropy sample done', sz['N_ent'])
    cond_all = cond_entropy_all(net, sz['n_cond_in'], sz['n_cond_rep']) if (sz['n_cond_in'] and net.kind != 'pfn') else {}
    log('conditional entropies done')
    for l in net.layers:
        t0 = time.time()
        m = net.m[l]
        P = np.concatenate(packed[l]); packed[l] = None
        Gs = np.unpackbits(P, axis=1)[:, :m].astype(bool); del P
        L = dict(m=m)
        L['cap_cov_bits'] = capb.get(l)
        groups = None
        if net.kind == 'tanh':
            d = net.d; groups = [np.array([i, i + d]) for i in range(d)]
        # E1
        L['E1'] = entropy_panel(Gs, m, sz, groups=groups, allow_made=True)
        log(f'l={l} E1 plugin={L["E1"]["plugin"]:.2f} mm={L["E1"]["mm"]:.2f} nsb={L["E1"]["nsb"].get("H", float("nan")):.2f} H2={L["E1"]["H2"]:.2f}{"(0coll)" if L["E1"]["H2_zero"] else ""} up={L["E1"]["upper"]:.2f} indep={L["E1"]["indep_upper"]:.2f}')
        # flips
        tf = time.time()
        if net.kind == 'pfn' and l == 2:
            # only the query token's 64 units can reach the output; flips of other tokens' units are exactly inert
            qcols = np.arange((net.T - 1) * net.dm, net.T * net.dm)
            F = np.zeros((sz['N_flip'], m), dtype=bool)
            Fq, base = flip_matrix_cols(net, st_f, l, np.arange(sz['N_flip']), qcols)
            F[:, qcols] = Fq
            # check inertness on a few other-token gates
            chk = np.arange(0, 64)
            Fc, _ = flip_matrix_cols(net, st_f, l, np.arange(min(64, sz['N_flip'])), chk)
            L['pfn_l2_nonquery_check_changes'] = int(Fc.sum())
        else:
            F, base = AN.flip_matrix(net, st_f, l, np.arange(sz['N_flip']))
        f = F.mean(0)
        L['flip_seconds'] = time.time() - tf
        L['flip_rate'] = f.tolist()
        L['base_change_check'] = None
        # E2
        Gf = net.gates(st_f, l)
        L['E2'] = {}
        prevR, preve2 = None, None
        for tau in TAUS:
            R = np.nonzero(f >= tau)[0]
            e2 = dict(D=int(len(R)))
            if prevR is not None and len(R) == len(prevR) and (R == prevR).all():
                e2 = dict(preve2); e2['same_as_previous_tau'] = True
            elif len(R) == m:
                e2['panel'] = L['E1']; e2['all_gates'] = True
            if len(R) and 'panel' not in e2:
                gr = None
                if net.kind == 'tanh':
                    d = net.d; neur = np.unique(R % d)
                    # group the up/dn gates of each relevant neuron that are both in R
                    gr = []
                    Rl = list(R)
                    for nn_ in neur:
                        cols = [j for j, c in enumerate(Rl) if c % d == nn_]
                        gr.append(np.array(cols))
                e2['panel'] = entropy_panel(Gs[:, R], len(R), sz, groups=gr)
            if len(R) and 'subspace' not in e2:
                # subspace of the relevant gates' pre-activations (flip set)
                Z = net.preact(st_f, l)
                if net.kind == 'tanh':
                    neur = np.unique(R % net.d); thr = lambda X: np.concatenate([X > 1, X < -1], 1)
                else:
                    neur = R; thr = lambda X: X > 0
                sub, (mu, Vt) = AN.subspace(Z, Gf, neur)
                sub.update(AN.gate_agreement_k(Z, neur, thr, mu, Vt))
                for key in ('k_var99', 'k_agree0.99', 'k_agree0.999'):
                    if sub.get(key) is not None:
                        sub['log2cells_' + key] = AN.log2_cells(len(R), sub[key])
                e2['subspace'] = sub
            L['E2'][f'tau{tau:g}'] = e2
            prevR, preve2 = R, e2
            log(f'l={l} tau={tau} D={len(R)} ' + (f'H(G_R) plugin={e2["panel"]["plugin"]:.2f} up={e2["panel"]["upper"]:.2f} H2={e2["panel"]["H2"]:.2f} k99={e2["subspace"]["k_var99"]} kag99={e2["subspace"].get("k_agree0.99")}' if len(R) else ''))
        # E3
        eps_list = (0.01, 0.002) if net.kind == 'rnn' else (0.01,)
        L['E3'] = AN.merged_entropy(net, st_f, st_h, l, F, eps_list, max_splits=sz['max_splits'], log_every=False)
        log(f'l={l} E3 ' + ' '.join(f'{k}={v}' for k, v in L['E3'].items() if k.startswith('H_merged') or k.startswith('leaves') or k.startswith('final')))
        # noise-only entropy H(G_l | X) on replicated inputs
        if l in cond_all:
            L['cond'] = cond_all[l]
            log(f'l={l} H(G|X) plugin={L["cond"]["plugin_mean"]:.3f} indep={L["cond"]["indep_mean"]:.3f}')
        L['seconds'] = time.time() - t0
        res['layers'][str(l)] = L
        json.dump(res, open(outp, 'w'))
    res['seconds'] = time.time() - t00
    json.dump(res, open(outp, 'w'))
    log('done', outp, f'{res["seconds"]:.0f}s')


def flip_matrix_cols(net, st, l, idx, cols):
    G = torch.from_numpy(net.gates(st, l))
    idx = np.asarray(idx); n = len(idx); m = G.shape[1]
    base = net.sub(st, l, idx, G[idx])
    F = np.zeros((n, len(cols)), dtype=bool)
    per = max(1, (1 << 14) // len(cols))
    for i in range(0, n, per):
        ii = idx[i:i + per]; k = len(ii)
        Gr = G[ii][:, None, :].repeat(1, len(cols), 1)
        Gr[:, torch.arange(len(cols)), torch.as_tensor(cols)] ^= True
        a1 = net.sub(st, l, np.repeat(ii, len(cols)), Gr.reshape(k * len(cols), m))
        F[i:i + k] = net.changed(base[i:i + k].repeat_interleave(len(cols)), a1).reshape(k, len(cols)).numpy()
    return F, base


def cond_entropy_all(net, n_in, n_rep):
    """H(G_l | X) for every analysed layer: n_in inputs x n_rep noise draws; mean per-input plug-in (biased low,
    capped at log2 n_rep) and mean per-input sum of per-unit entropies."""
    g = torch.Generator().manual_seed(99)
    xin = torch.randn(n_in, net.n if net.kind in ('rnn', 'arc') else net.d, generator=g)
    pl = {l: [] for l in net.layers}; ind = {l: [] for l in net.layers}
    for j in range(n_in):
        st = net.sample(n_rep, seed=5000 + j, x=xin[j:j + 1].repeat(n_rep, 1))
        for l in net.layers:
            G = net.gates(st, l)
            pl[l].append(ent.plugin(ent.pattern_ids(G)[0]))
            if net.kind == 'tanh':
                d = net.d
                ind[l].append(sum(ent.plugin(ent.pattern_ids(G[:, [i, i + d]])[0]) for i in range(d)))
            else:
                ind[l].append(float(ent.h2(G.mean(0)).sum()))
    out = {}
    for l in net.layers:
        p, q = np.array(pl[l]), np.array(ind[l])
        out[l] = dict(n_in=n_in, n_rep=n_rep, plugin_mean=float(p.mean()), plugin_se=float(p.std() / math.sqrt(n_in)),
                      indep_mean=float(q.mean()), indep_se=float(q.std() / math.sqrt(n_in)), frac_inputs_any_noise_flip=float((p > 0).mean()))
    return out


def cond_entropy(net, l, n_in, n_rep):
    """H(G_l | X): for n_in inputs, n_rep noise draws each; mean per-input plug-in (biased low, capped at log2 n_rep)
    and mean per-input independent-gates entropy (sum of per-gate binary entropies)."""
    pl, ind = [], []
    g = torch.Generator().manual_seed(99)
    xin = torch.randn(n_in, net.n if net.kind in ('rnn', 'arc') else net.d, generator=g)
    for j in range(n_in):
        xr = xin[j:j + 1].repeat(n_rep, 1)
        st = net.sample(n_rep, seed=5000 + j, x=xr)
        G = net.gates(st, l)
        c, _ = ent.pattern_ids(G)
        pl.append(ent.plugin(c))
        if net.kind == 'tanh':
            d = net.d
            ind.append(sum(ent.plugin(ent.pattern_ids(G[:, [i, i + d]])[0]) for i in range(d)))
        else:
            ind.append(float(ent.h2(G.mean(0)).sum()))
    return dict(n_in=n_in, n_rep=n_rep, plugin_mean=float(np.mean(pl)), plugin_se=float(np.std(pl) / math.sqrt(n_in)),
                indep_mean=float(np.mean(ind)), indep_se=float(np.std(ind) / math.sqrt(n_in)),
                frac_inputs_any_noise_flip=float(np.mean(np.array(pl) > 0)))


if __name__ == '__main__':
    main()
