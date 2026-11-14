"""E63 (PREREG-B amendment 71; CPU, existing nets): how do the standard nets allocate block 1's summary-token heads, how much does
their routing leak, how do the random source codes set the LayerNorm normaliser, and what does idealising the code geometry gain
at evaluation?  World 1, E27's first NB = 2,000 bases (variant 0), each net in its own format.
Per net (best-validation model.pt, or a checkpoint for the 'traj' cohort):
  M1 head allocation of the summary rows (tokens 4, 9, 14) in block 1, per head h (means over episodes and sources):
     P_h own-labelled mass (the source's four labelled tokens), self mass, osum_h (mass on the other two sources' summary tokens),
     olab_h (other sources' labelled tokens); C_h = self + osum_h (mass on the three query-question tokens).
     consensus head: max_h C_h >= 0.5; pooling head: P_h >= 0.5.
  M2 (random-code nets) cross-source leakage of the pooling heads: per (episode, source), mean over pooling heads of the mass on other
     sources' tokens (labelled and summary); Spearman over (episode, source) with the source code's largest cosine with another
     source's code; leakage mean, 90% and 99% quantiles.
  M3 the per-source LayerNorm normaliser: A_j = |P x_j| at r_j = 0 for labelled tokens (lngeom.py's A); per source the mean over its
     four labelled tokens; its CV across (episode, source), with the real codes and with both codes zeroed.
  M4 (cohorts ctrl and own) code idealisation at evaluation (Bayes-invariant; the source codes only): KL on the bases (+-6 grid,
     e26_run.set_stats) with base codes, eqn (each source code rescaled to norm 1), orth (Loewdin-orthogonalised, own norms kept),
     orthn (orthonormalised, norm 1).
  H3 (descriptive) the norm-scaled logits of the pooling heads: L_j = sqrt(d / dh) q.W_K(gamma * a_j) / A_j, the logit of key j at
     r_j = 0 relative to the zero point of the 1/n_j scaling (exact for pre-LN attention: logit_j = (alpha_j + beta r_j) / n_j +
     const); means over own labelled keys (L_own) and other sources' labelled keys (L_oth); and the implied logit change of an own
     key with |r| = 3 against r = 0 (L_own (A / n - 1), with each token's A, U, rho, averaged over both signs); the odd coefficient
     K_odd = sqrt(d / dh) q.W_K(gamma * u~) / A_j (the logit's slope in r at r = 0 from the signed feature), its RMS and mean over own
     keys. For own nets the pooling heads are both heads (their summary rows see only the source's four labelled tokens).
usage: python e63_run.py COHORT  (ctrl | fix | own | ext | traj; plumb = the explored nets, plumbing only) -> results/e63/<cohort>.json
Overrides (plumbing only): E63_NB, E63_OUT, E63_LIMIT (first N nets of the cohort).
"""
import glob, json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_own as PO  # noqa: E402

P2.build = lambda meta: PO.build_any(meta, P2, P2.fin_of(meta['ds']))
import e26_run as RU  # noqa: E402
import e30_eval as E30  # noqa: E402
import e48_fmt as F  # noqa: E402

E30.F = F
torch.set_num_threads(1)
S, Q, QL, T, DS = 3, 5, 4, 16, 16
SUMM = [s * Q + QL for s in range(S)]
LABS = [[s * Q + q for q in range(QL)] for s in range(S)]
NB = int(os.environ.get('E63_NB', 2000)); OUT = os.environ.get('E63_OUT', 'results/e63')
LIMIT = int(os.environ.get('E63_LIMIT', 0))
EXPLORED = {'e43_newinit_s0', 'e43_newinit_s1', 'e43_newinit_s2', 'e43_newinit_s4', 'e43_newinit_s11'}   # amendment 71's exploration
CKPTS = [0, 2000, 3000, 5000, 10000, 20000, 40000, 60000, 80000, 90000]


def acquired(evalfile):
    return os.path.exists(evalfile) and float(np.load(evalfile, allow_pickle=True)['gap']) < 0.10


def cohort(name):
    """[(tag, run_dir, eval_file, checkpoint or None)]"""
    out = []
    if name in ('ctrl', 'traj'):
        for d in sorted(glob.glob('runs/e37/e37_ctrl_s*')):
            if os.path.isdir(d) and acquired('results/e37/%s.npz' % os.path.basename(d)):
                if name == 'ctrl':
                    out.append((os.path.basename(d), d, 'results/e37/%s.npz' % os.path.basename(d), None))
                else:
                    for st in CKPTS:
                        out.append(('%s@%d' % (os.path.basename(d), st), d, 'results/e37/%s.npz' % os.path.basename(d), st))
    elif name == 'fix':
        for d in sorted(glob.glob('runs/e30/fix_s*')):
            if os.path.isdir(d) and acquired('results/e30/%s.npz' % os.path.basename(d)):
                out.append((os.path.basename(d), d, 'results/e30/%s.npz' % os.path.basename(d), None))
    elif name == 'own':
        for d in sorted(glob.glob('runs/e60/e60_own_s*')):
            if os.path.isdir(d) and acquired('results/e60/%s.npz' % os.path.basename(d)):
                out.append((os.path.basename(d), d, 'results/e60/%s.npz' % os.path.basename(d), None))
    elif name == 'plumb':                                        # plumbing only: the nets explored before amendment 71
        for tag in sorted(EXPLORED):
            out.append((tag, 'runs/e43/' + tag, 'results/e43/%s.npz' % tag, None))
    elif name == 'ext':
        for f in sorted(glob.glob('results/e47/*.npz')):
            tag = os.path.basename(f)[:-4]
            if tag in EXPLORED or tag.startswith('e37_'):
                continue
            sub = tag.split('_')[0]
            d = 'runs/%s/%s' % (sub, tag)
            if os.path.isdir(d) and acquired('results/%s/%s.npz' % (sub, tag)):
                out.append((tag, d, 'results/%s/%s.npz' % (sub, tag), None))
    out.sort(key=lambda t: t[0])
    return out[:LIMIT] if LIMIT else out


def load(run, step):
    model, arm, meta = E30.load(run)
    if step is not None:
        sd = torch.load(os.path.join(run, 'ckpt', 'step_%06d.pt' % step), map_location='cpu'); model.load_state_dict(sd)
    model.eval(); model.requires_grad_(False)
    return model, arm, meta


def transform(se, mode):
    se = se.astype(np.float64).copy()
    nrm = np.linalg.norm(se, axis=-1, keepdims=True)
    if mode == 'base':
        return se
    if mode == 'eqn':
        return se / nrm
    G = np.einsum('nsd,ntd->nst', se, se)
    w, V = np.linalg.eigh(G)
    O = np.einsum('nij,nj,nkj,nkd->nid', V, w ** -0.5, V, se)              # Loewdin: (G^-1/2) se, orthonormal rows
    return O * (nrm if mode == 'orth' else 1.0)


def block1_parts(model, X):
    """Block 1's attention probabilities (B, H, T, T), and per summary row and head the quantities for H3."""
    blk = model.blocks[0]; Xt = torch.from_numpy(X).float(); B = X.shape[0]
    allow = torch.ones(B, T, T, dtype=torch.bool); allow[:, :, T - 1] = False; allow[:, T - 1, T - 1] = True
    with torch.no_grad():
        x = model.inp(Xt); h = blk.ln1(x)
        q, k, v = blk.qkv(h).split(blk.h * blk.dh, -1)
        sp = lambda t: t.view(B, T, blk.h, blk.dh).transpose(1, 2)
        qh, kh = sp(q), sp(k)
        sc = (qh @ kh.transpose(-1, -2) / math.sqrt(blk.dh)).masked_fill(~allow[:, None], float('-inf'))
        A = sc.softmax(-1).numpy()
    return A, qh.numpy(), blk


def h3_logits(model, X, qh, blk):
    """Norm-scaled r=0 logits L_j of labelled keys for every summary query and head: (B, H, S, 12) and own-key logit change at |r|=3."""
    W = model.inp.weight.detach().numpy().astype(np.float64); b = model.inp.bias.detach().numpy().astype(np.float64)
    gam = blk.ln1.weight.detach().numpy().astype(np.float64)
    Wk = blk.qkv.weight.detach().numpy().astype(np.float64)[blk.h * blk.dh:2 * blk.h * blk.dh]          # (H*dh, d)
    d = W.shape[0]; B = X.shape[0]
    lab = [j for s in range(S) for j in LABS[s]]
    Xl = X[:, lab, :].astype(np.float64); X0 = Xl.copy(); X0[..., 32] = X0[..., 33]
    a = X0 @ W.T + b; a = a - a.mean(-1, keepdims=True)                                                  # (B, 12, d)
    u = W[:, 32] - W[:, 32].mean(); Aj = np.linalg.norm(a, axis=-1); U = np.linalg.norm(u); rho = (a @ u) / (Aj * U)
    ka = np.einsum('ed,bjd->bje', Wk, gam[None, None] * a).reshape(B, len(lab), blk.h, blk.dh)              # W_K (gamma a_j), per head
    ku = (Wk @ (gam * u)).reshape(blk.h, blk.dh)                                                              # W_K (gamma u~), per head
    out = np.zeros((B, blk.h, S, len(lab))); odd = np.zeros((B, blk.h, S, len(lab)))
    for s in range(S):
        qs = qh[:, :, SUMM[s], :]                                                                            # (B, H, dh)
        out[:, :, s, :] = math.sqrt(d / blk.dh) * np.einsum('bhe,bjhe->bhj', qs, ka) / Aj[:, None, :]
        odd[:, :, s, :] = math.sqrt(d / blk.dh) * np.einsum('bhe,he->bh', qs, ku)[:, :, None] / Aj[:, None, :]
    r = 3.0
    ratio = 0.5 * (Aj / np.sqrt(Aj ** 2 + 2 * r * Aj * U * rho + r ** 2 * U ** 2) + Aj / np.sqrt(Aj ** 2 - 2 * r * Aj * U * rho + r ** 2 * U ** 2))
    return out, ratio, odd                                                                                   # ratio (B, 12) = mean A / n at |r| = 3


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    name = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    z = np.load('results/e27_data.npz'); sl = slice(0, NB)
    g = z['bmean'][sl, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][sl], z['bvars'], z['bwts'][sl], g)
    se0 = z['semb'][sl]; qe = z['qemb'][sl]
    res = {}
    for tag, run, ev, step in cohort(name):
        model, arm, meta = load(run, step)
        evz = np.load(ev, allow_pickle=True)
        o = dict(arm=arm, variant=meta.get('variant'), gap=float(evz['gap']), rom_large=float(evz['rom_large']),
                 rom_middle=float(evz['rom_middle']), type='A' if float(evz['rom_middle']) < 0 else 'B', step=step)
        X, v = F.make_X(z['o'][sl, 0], z['y'][sl], se0, qe, arm)
        if meta.get('variant'):                                    # own: block 1's summary rows are masked by design; M3, M4 only
            name_m1 = False
        else:
            name_m1 = True
        A, qh, blk = block1_parts(model, X)
        H = A.shape[1]
        P = np.stack([A[:, :, SUMM[s], LABS[s]].sum(-1) for s in range(S)], 1)                                    # (B, S, H)
        slf = np.stack([A[:, :, SUMM[s], SUMM[s]] for s in range(S)], 1)
        osum = np.stack([A[:, :, SUMM[s], [SUMM[t] for t in range(S) if t != s]].sum(-1) for s in range(S)], 1)
        olab = np.stack([A[:, :, SUMM[s], [j for t in range(S) if t != s for j in LABS[t]]].sum(-1) for s in range(S)], 1)
        if not name_m1:
            P = slf = osum = olab = None
        if name_m1:
            o['P_h'] = [float(P[:, :, h].mean()) for h in range(H)]; o['self_h'] = [float(slf[:, :, h].mean()) for h in range(H)]
            o['osum_h'] = [float(osum[:, :, h].mean()) for h in range(H)]; o['olab_h'] = [float(olab[:, :, h].mean()) for h in range(H)]
            o['C_h'] = [o['self_h'][h] + o['osum_h'][h] for h in range(H)]
            o['C_max'] = float(max(o['C_h'])); o['consensus'] = bool(o['C_max'] >= 0.5)
            pool = [h for h in range(H) if o['P_h'][h] >= 0.5]; o['n_pool'] = len(pool)
        if step is None and not name_m1:
            pool = list(range(blk.h))                                  # own: both summary heads pool the source's own four tokens
        if step is None:
            if pool:
                lk = (osum[:, :, pool] + olab[:, :, pool]).mean(-1) if name_m1 else None                            # (B, S)
                if name_m1:
                    o['leak_mean'] = float(lk.mean()); o['leak_q90'] = float(np.quantile(lk, 0.9)); o['leak_q99'] = float(np.quantile(lk, 0.99))
                if arm != 'fix' and name_m1:
                    nr = np.linalg.norm(se0, axis=-1)
                    cs = np.einsum('nsd,ntd->nst', se0, se0) / nr[:, :, None] / nr[:, None, :]
                    for s in range(S):
                        cs[:, s, s] = -np.inf
                    o['leak_spearman_maxcos'] = spearman(lk.ravel(), cs.max(-1).ravel())
                L, ratio, odd = h3_logits(model, X, qh, blk)
                own_i = [[QL * s + q for q in range(QL)] for s in range(S)]                                         # positions in lab
                Lown = np.stack([L[:, :, s, own_i[s]] for s in range(S)], 1)                                         # (B, S, H, 4)
                Loth = np.stack([L[:, :, s, [i for t in range(S) if t != s for i in own_i[t]]] for s in range(S)], 1)
                rat = np.stack([ratio[:, own_i[s]] for s in range(S)], 1)                                            # (B, S, 4)
                o['L_own_h'] = [float(Lown[:, :, h].mean()) for h in range(H)]; o['L_oth_h'] = [float(Loth[:, :, h].mean()) for h in range(H)]
                o['dlogit_r3_h'] = [float((Lown[:, :, h] * (rat - 1.0)).mean()) for h in range(H)]
                Kodd = np.stack([odd[:, :, s, own_i[s]] for s in range(S)], 1)                                       # (B, S, H, 4)
                o['Kodd_rms_h'] = [float(np.sqrt((Kodd[:, :, h] ** 2).mean())) for h in range(H)]
                o['Kodd_mean_h'] = [float(Kodd[:, :, h].mean()) for h in range(H)]
                o['pool_heads'] = pool
        if step is None:
            W = model.inp.weight.detach().numpy().astype(np.float64); b = model.inp.bias.detach().numpy().astype(np.float64)
            lab = [j for s in range(S) for j in LABS[s]]
            X0 = X[:, lab, :].astype(np.float64).copy(); X0[..., 32] = X0[..., 33]
            def nrm(XX):
                e = XX @ W.T + b; e = e - e.mean(-1, keepdims=True); return np.linalg.norm(e, axis=-1).reshape(NB, S, QL).mean(-1)
            As = nrm(X0); Xn = X0.copy(); Xn[..., :2 * DS] = 0; Asn = nrm(Xn)
            o['A_src_cv'] = float(As.std() / As.mean()); o['A_src_cv_nocode'] = float(Asn.std() / Asn.mean())
            if arm != 'fix' and name in ('ctrl', 'own', 'plumb'):
                for mode in ('base', 'eqn', 'orth', 'orthn'):
                    Xm, vm = F.make_X(z['o'][sl, 0], z['y'][sl], transform(se0, mode), qe, arm)
                    hd = RU.heads(model, 'clean', Xm, vm)
                    kl, _, mass = RU.set_stats([hd], lpB, g); o['kl_' + mode] = float(kl.mean()); o['mass_min_' + mode] = float(mass.min())
        res[tag] = o
        print(name, tag, json.dumps({k: (round(x, 5) if isinstance(x, float) else x) for k, x in o.items()}), flush=True)
    json.dump(res, open(os.path.join(OUT, '%s.json' % name), 'w'), indent=1)


if __name__ == '__main__':
    main()
