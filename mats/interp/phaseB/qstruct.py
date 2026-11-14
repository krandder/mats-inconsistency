"""E2 (PREREG-B amendment 9): whole-network structure built from the question (the network's deviation from Bayes).

Stage A: backward build from the output with deviation-sensitivity edge weights (real maps, restricted attention).
Stage B: greedy local moves (prune edge group, drop node group, pool sibling group, named-pattern attention, low-rank MLP
maps, low-rank embedding, skip-merge of an MLP block), scored by proxy deviation R^2 (common random numbers) per cost.
Stage C: Pareto points scored by K = 64 independent-draw deviation R^2 on the 2,000 test episodes and scenario agreement.
usage: python qstruct.py RUN_DIR [--smoke] -> RUN_DIR/qstruct.json
"""
import copy, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import nnoise as N
import world as W
import qw
import dag as DG
import devpred as DP

torch.set_num_threads(1)
T, QT = 16, 15
QQ = [4, 9, 14]
SRC = [list(range(5 * s, 5 * s + 5)) for s in range(3)]
LAB = [list(range(5 * s, 5 * s + 4)) for s in range(3)]
EGROUPS = {'src0': SRC[0], 'src1': SRC[1], 'src2': SRC[2], 'q': [QT]}
NGROUPS = {'lab0': LAB[0], 'lab1': LAB[1], 'lab2': LAB[2], 'qq0': [4], 'qq1': [9], 'qq2': [14], 'q': [QT]}
PGROUPS = {'QQ': QQ, 'lab0': LAB[0], 'lab1': LAB[1], 'lab2': LAB[2]}
SRC_OF = np.array([0] * 5 + [1] * 5 + [2] * 5 + [-1])
Q_OF = np.array(list(range(5)) * 3 + [-1])
DS = 16
CTX = {'fn': None, 'b': None, 'fit': None}      # E3b hook: per-episode free inputs (Bayes quantities); unused in E2


# ------------------------------------------------------------------ named attention pattern
def static_feats():
    F = np.zeros((T, T, 11), np.float32)
    for t in range(T):
        for u in range(T):
            ctx_t, ctx_u = t < QT, u < QT
            F[t, u, 0] = ctx_t and ctx_u and SRC_OF[t] == SRC_OF[u]
            F[t, u, 1] = ctx_t and ctx_u and Q_OF[t] == Q_OF[u]
            F[t, u, 2] = u in QQ
            F[t, u, 3] = ctx_u and Q_OF[u] < 4
            F[t, u, 4] = t == QT and u in QQ
            F[t, u, 5] = t == QT and ctx_u and Q_OF[u] < 4
            F[t, u, 6] = t == u
            F[t, u, 10] = t == QT and u == QT
    return torch.as_tensor(F)


SF = static_feats()


def reliability(X):
    """r_s = -log(mean squared labelled residual of source s + 0.01), from the tokens (value - label on labelled questions)."""
    o, y = X[:, :, 2 * DS], X[:, :, 2 * DS + 1]
    r = np.stack([-np.log(((o[:, LAB[s]] - y[:, LAB[s]]) ** 2).mean(1) + 0.01) for s in range(3)], 1)
    return torch.as_tensor(r, dtype=torch.float32)                   # (n, 3)


def named_feats(rel):
    """(n, T, T, 11): static features plus the reliability features of the key's source."""
    n = rel.shape[0]; F = SF[None].expand(n, T, T, 11).clone()
    rk = torch.zeros(n, T); rk[:, :QT] = rel[:, torch.as_tensor(SRC_OF[:QT])]
    F[..., 7] = rk[:, None, :] * (torch.arange(T) < QT).float()[None, None, :]
    F[..., 8] = F[..., 7] * (torch.arange(T) == QT).float()[None, :, None]
    F[..., 9] = F[..., 7] * SF[None, :, :, 0]
    return F


class AttnNamed(nn.Module):
    """Attention with logits theta . phi(t, t') (named quantities) and the real value/output maps (fixed)."""
    def __init__(self, block):
        super().__init__(); self.ln1 = block.ln1; self.h, self.dh = block.h, block.dh
        d = block.o.weight.shape[0]; hd = self.h * self.dh
        self.Wv = block.qkv.weight[2 * hd:].detach(); self.bv = block.qkv.bias[2 * hd:].detach()
        self.Wo = block.o.weight.detach(); self.bo = block.o.bias.detach()
        self.theta = nn.Parameter(torch.zeros(self.h, 11))
        self.cost = self.theta.numel() + self.Wv.numel() + self.bv.numel() + self.Wo.numel() + self.bo.numel()
    def forward(self, x, mask, feats):
        n = len(x); v = (self.ln1(x) @ self.Wv.T + self.bv).view(n, T, self.h, self.dh).transpose(1, 2)
        lg = torch.einsum('hf,ntuf->nhtu', self.theta, feats).masked_fill(~mask[None, None], float('-inf'))
        return N.sphere(x + ((lg.softmax(-1) @ v).transpose(1, 2).reshape(n, T, -1) @ self.Wo.T + self.bo))


class MergeLR(nn.Module):
    """Skip-merge of MLP block l: r = sphere(a + LN2(a) V U^T + c), rank r."""
    def __init__(self, block, A, r):
        super().__init__(); self.ln2 = block.ln2; d = A.shape[-1]
        with torch.no_grad():
            L = block.ln2(A).reshape(-1, d); Z = (block.mlp[2](N.sphere(block.hidden_pre(A))) ).reshape(-1, d)
        self.U, self.V, self.c = qw.lowrank_init(L.numpy(), Z.numpy(), r); self.cost = r * 2 * d + d
    def forward(self, a): return N.sphere(a + self.ln2(a) @ self.V @ self.U.T + self.c)


# ------------------------------------------------------------------ description
class Desc:
    def __init__(self, depth, D):
        self.depth = depth; self.mask = {l: torch.as_tensor(DG.base_allow()) for l in range(1, depth + 1)}
        self.drop = set(); self.pool = set(); self.sur = {}; self.merge = {}; self.D = D; self.moves = []; self.lastB = None
    def copy(self):
        c = Desc(self.depth, self.D); c.mask = {l: m.clone() for l, m in self.mask.items()}
        c.drop = set(self.drop); c.pool = set(self.pool); c.sur = dict(self.sur); c.merge = dict(self.merge); c.moves = list(self.moves)
        c.lastB = self.lastB
        return c


def needed(desc):
    """Tokens needed (ancestors of the output) at each residual cut r_l (r_0 = emb)."""
    need = {desc.depth: {QT}}
    for l in range(desc.depth, 0, -1):
        s = set(need[l])
        if l == desc.depth and desc.lastB is not None: need[l - 1] = s; continue
        for t in need[l]:
            s |= set(np.flatnonzero(desc.mask[l][t].numpy()).tolist())
        need[l - 1] = s
    return need


def site_tokens(desc, site):
    need = needed(desc)
    if site == 'emb': return need[0]
    l = int(site[1:]); return need[l]


def post(desc, site, v, means):
    for (s, t) in desc.drop:
        if s == site: v = v.clone(); v[:, t] = means[site][t]
    for (s, g) in desc.pool:
        if s == site:
            G = PGROUPS[g]; v = v.clone(); v[:, G] = v[:, G].mean(1, keepdim=True)
    return v


def noise_for(desc, site, z):
    for (s, g) in desc.pool:
        if s == site:
            G = PGROUPS[g]; z = z.clone(); z[:, G] = z[:, G[0]][:, None]
    return z


def forward_desc(model, desc, X, nz, sigma, means, sig_eff, feats=None):
    """One noisy pass of the description with noise dict nz (site -> (n, T, D) standard normal). Returns head."""
    Xt = torch.from_numpy(X); n = len(X)
    if CTX['fn'] is not None: CTX['b'] = CTX['fn'](X)
    if feats is None and any(isinstance(m, AttnNamed) for m in desc.sur.values()): feats = named_feats(reliability(X))
    add = lambda site, v, s_=sigma: post(desc, site, v, means) + s_ * noise_for(desc, site, nz[site])
    x = add('emb', desc.sur['emb'](Xt) if 'emb' in desc.sur else model.f_emb(Xt))
    for l in range(1, model.depth + 1):
        b = model.blocks[l - 1]; a = f'a{l}'
        if l == model.depth and desc.lastB is not None:
            x = add(f'r{l}', desc.lastB(x)); continue
        if a in desc.sur: xa = desc.sur[a](x, desc.mask[l], feats)
        else: xa = N.sphere(x + b.attn_out(x, desc.mask[l][None].expand(n, T, T)))
        xa = add(a, xa)
        if l in desc.merge:
            x = add(f'r{l}', desc.merge[l](xa), sig_eff[l]); continue
        m = add(f'm{l}', desc.sur[f'm{l}'](xa) if f'm{l}' in desc.sur else model.f_m(l, xa))
        x = add(f'r{l}', desc.sur[f'r{l}'](xa, m) if f'r{l}' in desc.sur else model.f_r(l, xa, m))
    return model.head(x)


def noise_iter(seed, n, K, shapes):
    g = torch.Generator().manual_seed(seed)
    for _ in range(K):
        yield {s: torch.randn((n,) + sh, generator=g) for s, sh in shapes.items()}


def marginal_desc(model, desc, X, noises, sigma, means, sig_eff, K=None):
    """noises: a list of noise dicts (common random numbers) or an iterator of K dicts (K given)."""
    feats = named_feats(reliability(X)) if any(isinstance(m, AttnNamed) for m in desc.sur.values()) else None
    mus, lss, lgs = [], [], []; K = K or len(noises)
    with torch.no_grad():
        for nz in noises:
            mu, ls, lg = forward_desc(model, desc, X, nz, sigma, means, sig_eff, feats)
            mus.append(mu); lss.append(ls); lgs.append(torch.log_softmax(lg, -1) - math.log(K))
    return torch.cat(mus, 1).numpy(), torch.cat(lss, 1).numpy(), torch.cat(lgs, 1).numpy()


def cost_of(desc, model, exact):
    need = needed(desc); p = exact['emb'] if 'emb' not in desc.sur else desc.sur['emb'].cost
    for l in range(1, model.depth + 1):
        a = f'a{l}'
        if l == model.depth and desc.lastB is not None: p += desc.lastB.cost; continue
        p += desc.sur[a].cost if a in desc.sur else exact['a']
        if l in desc.merge: p += desc.merge[l].cost
        else:
            p += desc.sur[f'm{l}'].cost if f'm{l}' in desc.sur else exact['m']
            p += desc.sur[f'r{l}'].cost if f'r{l}' in desc.sur else exact['r']
    p += sum(desc.D[s] for (s, t) in desc.drop)
    edges = sum(int(desc.mask[l][sorted(need[l])].sum()) for l in range(1, model.depth + 1) if not (l == model.depth and desc.lastB is not None))
    nodes = 0
    for site in ['emb'] + [f'{k}{l}' for l in range(1, model.depth + 1) for k in 'amr']:
        if site[0] == 'm' and int(site[1:]) in desc.merge: continue
        if site[0] in 'am' and int(site[1:]) == model.depth and desc.lastB is not None: continue
        toks = need[0] if site == 'emb' else need[int(site[1:])]
        toks = set(toks) - {t for (s, t) in desc.drop if s == site}
        for (s, g) in desc.pool:
            if s == site and len(set(PGROUPS[g]) & toks) > 1: toks = (toks - set(PGROUPS[g])) | {PGROUPS[g][0]}
        nodes += len(toks)
    return dict(params=int(p), edges=edges, nodes=nodes, total=int(p) + edges + nodes)


def kl_bits(model, desc, st, nz, X, sigma, means, sig_eff, idx):
    """KL(real path || description path) in bits by the chain rule, on real-path parents (merged blocks: mean term only)."""
    need = needed(desc); n = len(idx); Xi = X[idx.numpy()]; tot = 0.0; per = {}
    if CTX['fn'] is not None: CTX['b'] = CTX['fn'](Xi)
    feats = named_feats(reliability(Xi)) if any(isinstance(m, AttnNamed) for m in desc.sur.values()) else None
    with torch.no_grad():
        for site in ['emb'] + [f'{k}{l}' for l in range(1, model.depth + 1) for k in 'amr']:
            toks = sorted(need[0] if site == 'emb' else need[int(site[1:])])
            f = st[site][idx] - sigma * nz[site][idx]
            if site == 'emb':
                g = desc.sur['emb'](torch.from_numpy(Xi)) if 'emb' in desc.sur else model.f_emb(torch.from_numpy(Xi))
            else:
                l = int(site[1:]); b = model.blocks[l - 1]
                if l == model.depth and desc.lastB is not None:
                    if site[0] in 'am': continue
                    xin = st['emb'][idx] if l == 1 else st[f'r{l - 1}'][idx]
                    g = post(desc, site, desc.lastB(xin), means)
                    e = ((f - g)[:, toks] ** 2).sum((1, 2)).mean() / (2 * sigma ** 2); per[site] = float(e) / math.log(2); tot += per[site]; continue
                if l in desc.merge:
                    if site[0] == 'm': continue
                    if site[0] == 'r':
                        g = desc.merge[l](st[f'a{l}'][idx]); g = post(desc, site, g, means)
                        e = ((f - g)[:, toks] ** 2).sum((1, 2)).mean() / (2 * sig_eff[l] ** 2); per[site] = float(e) / math.log(2); tot += per[site]; continue
                if site[0] == 'a':
                    xin = st['emb'][idx] if l == 1 else st[f'r{l - 1}'][idx]
                    g = desc.sur[site](xin, desc.mask[l], feats) if site in desc.sur else N.sphere(xin + b.attn_out(xin, desc.mask[l][None].expand(n, T, T)))
                elif site[0] == 'm':
                    g = desc.sur[site](st[f'a{l}'][idx]) if site in desc.sur else model.f_m(l, st[f'a{l}'][idx])
                else:
                    g = desc.sur[site](st[f'a{l}'][idx], st[f'm{l}'][idx]) if site in desc.sur else model.f_r(l, st[f'a{l}'][idx], st[f'm{l}'][idx])
            g = post(desc, site, g, means)
            e = ((f - g)[:, toks] ** 2).sum((1, 2)).mean() / (2 * sigma ** 2); per[site] = float(e) / math.log(2); tot += per[site]
    return tot, per


# ------------------------------------------------------------------ surrogate fitting (in the real-net context)
def fit_site(model, mod, site, st, nz, X, sigma, J, ntr, steps_w, steps_e, sds, allow, feats_all=None, merge_l=None, sig_eff=None, last=False):
    """Output-MSE warm start, question-weighted fit with anchor 0.1, then single-site end-to-end tuning (CRN).
    last=True: mod replaces the whole last block for the query token (fit on the query token of r_L only)."""
    if last:
        dq = st[site].shape[-1]; J = J.reshape(len(J), 3, T, dq)[:, :, QT]
    D = (st[site].shape[-1] if last else st[site][0].numel()); eps = 0.1 * float((J[:ntr] ** 2).sum((1, 2)).mean()) / D
    ps = [p for p in mod.parameters() if p.requires_grad]
    if merge_l is not None:
        b = model.blocks[merge_l - 1]
        with torch.no_grad(): tgt_all = N.sphere(b.write_mlp(st[f'a{merge_l}'], N.sphere(b.hidden_pre(st[f'a{merge_l}'])))).reshape(len(X), -1)
    g = torch.Generator().manual_seed(0)
    for phase, steps in (('mse', steps_w // 3), ('w', steps_w)):
        opt = torch.optim.Adam(ps, lr=3e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1))
        for s in range(steps):
            idx = torch.randint(ntr, (256,), generator=g)
            if CTX['fit'] is not None: CTX['b'] = CTX['fit'][idx]
            if last:
                l = model.depth; xin = st['emb'][idx] if l == 1 else st[f'r{l - 1}'][idx]
                e = mod(xin)[:, QT] - (st[site][idx] - sigma * nz[site][idx])[:, QT]
            elif merge_l is not None: e = mod(st[f'a{merge_l}'][idx]).reshape(256, -1) - tgt_all[idx]
            elif isinstance(mod, AttnNamed):
                l = int(site[1:]); xin = st['emb'][idx] if l == 1 else st[f'r{l - 1}'][idx]
                e = mod(xin, torch.as_tensor(DG.base_allow()), feats_all[idx]).reshape(256, -1) - (st[site][idx] - sigma * nz[site][idx]).reshape(256, -1)
            else: e = qw.errors(mod, site, st, nz, X, sigma, idx)
            loss = (e ** 2).sum(1).mean() if phase == 'mse' else (torch.einsum('nkd,nd->nk', J[idx], e) ** 2).sum(1).mean() + eps * (e ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    # single-site end-to-end tuning inside the real net, common random numbers
    desc = Desc(model.depth, {}); means = {}
    if last: desc.lastB = mod
    elif merge_l is not None: desc.merge[merge_l] = mod
    else: desc.sur[site] = mod
    opt = torch.optim.Adam(ps, lr=1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps_e, 1))
    for s in range(steps_e):
        idx = torch.randint(ntr, (256,), generator=g); Xi = X[idx.numpy()]
        nzb = {k: torch.randn((256,) + v.shape[1:], generator=g) for k, v in st.items()}
        with torch.no_grad(): tm, tl = qw.moments_t(*model_forward_nz(model, Xi, nzb, sigma))
        fe = feats_all[idx] if feats_all is not None and isinstance(mod, AttnNamed) else None
        m, ls = qw.moments_t(*forward_desc(model, desc, Xi, nzb, sigma, means, sig_eff or {}, fe))
        loss = (((m - tm) / sds[0]) ** 2 + ((ls - tl) / sds[1]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return mod


def build_cands_E2(c):
    model, st, nz, X, sg, Js, ntr, sw, se, sds, allow = (c[k] for k in ('model', 'st', 'nz', 'X', 'sg', 'Js', 'ntr', 'steps_w', 'steps_e', 'sds', 'allow'))
    cands = {}
    cands['emb:LR8'] = ('sur', 'emb', fit_site(model, qw.EmbLR(model, X, 8), 'emb', st, nz, X, sg, Js['emb'], ntr, sw, se, sds, allow))
    for l in range(1, c['depth'] + 1):
        b = model.blocks[l - 1]
        cands[f'a{l}:named'] = ('sur', f'a{l}', fit_site(model, AttnNamed(b), f'a{l}', st, nz, X, sg, Js[f'a{l}'], ntr, sw, se, sds, allow, c['feats_all']))
        for r in (8, 4):
            cands[f'm{l}:LR{r}'] = ('sur', f'm{l}', fit_site(model, qw.HidLR(b, st[f'a{l}'], r), f'm{l}', st, nz, X, sg, Js[f'm{l}'], ntr, sw, se, sds, allow))
            cands[f'r{l}:LR{r}'] = ('sur', f'r{l}', fit_site(model, qw.WriteLR(b, st[f'm{l}'], r), f'r{l}', st, nz, X, sg, Js[f'r{l}'], ntr, sw, se, sds, allow))
        cands[f'merge{l}:LR8'] = ('merge', l, fit_site(model, MergeLR(b, st[f'a{l}'], 8), f'r{l}', st, nz, X, sg, Js[f'r{l}'], ntr, sw, se, sds, allow, merge_l=l, sig_eff=c['sig_eff']))
    return cands


BUILD_CANDS = [build_cands_E2]
OUT_TAG = ['qstruct']


def model_forward_nz(model, X, nz, sigma):
    desc = Desc(model.depth, {}); return forward_desc(model, desc, X, nz, sigma, {}, {})


# ------------------------------------------------------------------ main
def run(run_dir, smoke=False):
    t0 = time.time()
    model, meta = N.load_run(run_dir); model.requires_grad_(False); sg = meta['sigma']; depth = meta['depth']
    n_fit, ntr, n_px, Kpx = (512, 448, 256, 4) if smoke else (4096, 3584, 1024, 8)
    steps_w, steps_e, rounds = (30, 10, 2) if smoke else (600, 300, 12)
    U9 = json.load(open(os.path.join(run_dir, 'devpred.json'))); sds = [U9['dev_sd'][k] for k in ('mean', 'logsd', 'kl')]
    out = dict(run=run_dir, sigma=sg, depth=depth, stageA=[], stageB=[], pareto=[])
    sites = qw.order(depth)
    rng = np.random.default_rng(92_000); d = W.sample_np(n_fit, rng); X, valid = W.tokens_np(d, rng, meta['ds'])
    st, nz, allow = qw.trace_nz(model, X, valid, sg, torch.Generator().manual_seed(92_001)); B = qw.bayes_t(d)
    Dd = {s: st[s].shape[-1] for s in sites}
    means = {s: (st[s] - sg * nz[s]).mean(0) for s in sites}
    b0 = model.blocks[0]
    exact = {'emb': model.inp.weight.numel() + model.inp.bias.numel(),
             'a': sum(p.numel() for p in list(b0.qkv.parameters()) + list(b0.o.parameters())),
             'm': sum(p.numel() for p in b0.mlp[0].parameters()), 'r': sum(p.numel() for p in b0.mlp[2].parameters())}
    # ---- Jacobians, node weights, edge weights
    Js, node_w = {}, {}
    for s in sites:
        J = qw.jacobian(model, s, st, nz, allow, sg, B, sds); Js[s] = J
        node_w[s] = J[:ntr].reshape(ntr, 3, T, -1).pow(2).sum((1, 3)).mean(0).numpy().tolist()
    edge_w = {}
    with torch.no_grad():
        for l in range(1, depth + 1):
            b = model.blocks[l - 1]; xin = st['emb'] if l == 1 else st[f'r{l - 1}']; hd = b.h * b.dh
            q, k, v = b.qkv(b.ln1(xin[:ntr])).split(hd, -1); spl = lambda t_: t_.view(ntr, T, b.h, b.dh).transpose(1, 2)
            A = ((spl(q) @ spl(k).transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~allow[:ntr, None], float('-inf')).softmax(-1)  # (n,h,T,T)
            vo = torch.stack([spl(v)[:, h_] @ b.o.weight[:, h_ * b.dh:(h_ + 1) * b.dh].T for h_ in range(b.h)], 1)  # (n,h,T,d)
            pre = xin[:ntr] + b.attn_out(xin[:ntr], allow[:ntr]); nrm = pre.norm(dim=-1, keepdim=True); u = pre / nrm
            Jt = Js[f'a{l}'][:ntr].reshape(ntr, 3, T, -1)
            Jp = (Jt - (Jt * u[:, None]).sum(-1, keepdim=True) * u[:, None]) * (math.sqrt(pre.shape[-1]) / nrm[:, None])   # (n,3,T,d)
            E = torch.zeros(T, T)
            for i in range(0, ntr, 256):
                msg = torch.einsum('nhtu,nhud->ntud', A[i:i + 256], vo[i:i + 256])          # (n, t, u, d)
                E += (torch.einsum('nktd,ntud->nktu', Jp[i:i + 256], msg) ** 2).sum((0, 1))
            edge_w[l] = (E / ntr).numpy()
    out['node_w'] = node_w; out['edge_w'] = {l: e.tolist() for l, e in edge_w.items()}
    print('weights done %.0fs' % (time.time() - t0), flush=True)
    # ---- proxy set (common random numbers)
    rngp = np.random.default_rng(93_000); dp = W.sample_np(n_px, rngp); Xp, vp = W.tokens_np(dp, rngp, meta['ds'])
    ref, _, mom, wb = W.references(dp)
    Tp = dict(comp_means=ref['bayes'][0], comp_vars=ref['bayes'][1], w_bayes=wb, mean_bayes=mom['bayes'][0], var_bayes=mom['bayes'][1])
    gpx = torch.Generator().manual_seed(93_001)
    noises = [{s: torch.randn((n_px,) + st[s].shape[1:], generator=gpx) for s in sites} for _ in range(Kpx)]
    sig_eff = {}
    devR = qw.deviations_t(marginal_desc(model, Desc(depth, Dd), Xp, noises, sg, means, sig_eff), Tp)
    feats_all = named_feats(reliability(X))
    idx_kl = torch.arange(ntr, n_fit)
    def proxy(desc):
        dv = qw.deviations_t(marginal_desc(model, desc, Xp, noises, sg, means, sig_eff), Tp)
        return {k: DP.r2(devR[k], dv[k]) for k in dv}
    def record(desc, tag):
        r2 = proxy(desc); c = cost_of(desc, model, exact); kb, per = kl_bits(model, desc, st, nz, X, sg, means, sig_eff, idx_kl)
        need = needed(desc); M1 = desc.mask[1].numpy()
        rows_ctx = [t for t in need[1] if t < QT]
        kept = [(t, u) for t in rows_ctx for u in np.flatnonzero(M1[t]) if u < QT]
        within = float(np.mean([SRC_OF[t] == SRC_OF[u] for t, u in kept])) if kept else float('nan')
        return dict(tag=tag, moves=list(desc.moves), proxy=r2, cost=c, KL_bits=kb, KL_sites=per, within_source_L1=within,
                    n_needed={l: len(v) for l, v in need.items()})
    # ---- stage A: backward build
    descA = {}
    for eps in ([0.3, 0.0] if smoke else [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0]):
        desc = Desc(depth, Dd); need_r = {depth: {QT}}
        for l in range(depth, 0, -1):
            M = torch.as_tensor(DG.base_allow()).clone(); nxt = set(need_r[l])
            for t in need_r[l]:
                e = edge_w[l][t].copy(); e[~DG.base_allow()[t]] = 0
                order = np.argsort(-e); cs = np.cumsum(e[order]); tot = cs[-1]
                kk = int(np.searchsorted(cs, (1 - eps) * tot - 1e-12) + 1) if tot > 0 else 1
                keep = [int(u) for u in order[:max(kk, 1)] if DG.base_allow()[t, u]]
                row = torch.zeros(T, dtype=torch.bool); row[keep] = True; M[t] = row; nxt |= set(keep)
            desc.mask[l] = M; need_r[l - 1] = nxt
        desc.moves.append(f'backward eps={eps}')
        rec = record(desc, f'A eps={eps}'); out['stageA'].append(rec); descA[eps] = desc
        print('A', eps, {k: round(v, 3) for k, v in rec['proxy'].items()}, rec['cost'], 'KL bits %.0f' % rec['KL_bits'], 'within %.2f' % rec['within_source_L1'], flush=True)
        json.dump(out, open(os.path.join(run_dir, OUT_TAG[0] + ('_smoke.json' if smoke else '.json')), 'w'), indent=1)
    base = out['stageA'][-1]['proxy']['mean']
    ok = [r for r in out['stageA'] if r['proxy']['mean'] >= 0.95 * base]
    eps0 = max(float(r['tag'].split('=')[1]) for r in ok); cur = descA[eps0].copy()
    out['stageB_start_eps'] = eps0
    # ---- surrogate candidates (fitted once in the real-net context)
    ts_ = time.time()
    for l in range(1, depth + 1):
        # merged noise: sigma^2 (1 + E ||d f_r / d m||_F^2 / d) per coordinate (moment-matched isotropic)
        b = model.blocks[l - 1]; a_ = st[f'a{l}'][:64].reshape(-1, model.d)[:512]
        with torch.no_grad(): m0 = N.sphere(b.hidden_pre(a_))
        Jf = torch.stack([torch.autograd.functional.jacobian(lambda mv: N.sphere(b.write_mlp(a_[i], mv)), m0[i]) for i in range(0, 512, 4)])
        sig_eff[l] = sg * math.sqrt(1 + float((Jf ** 2).sum((1, 2)).mean()) / model.d)
    cands = BUILD_CANDS[0](dict(model=model, st=st, nz=nz, X=X, sg=sg, Js=Js, ntr=ntr, steps_w=steps_w, steps_e=steps_e, sds=sds,
                                allow=allow, feats_all=feats_all, sig_eff=sig_eff, depth=depth))
    out['sig_eff'] = sig_eff; out['fit_seconds'] = time.time() - ts_
    print('surrogates fitted %.0fs' % (time.time() - ts_), flush=True)
    # single-candidate proxy scores in the full network (P48)
    out['single'] = {}
    for nm, (kind, key, mod) in cands.items():
        dsc = Desc(depth, Dd)
        if kind == 'sur': dsc.sur[key] = mod
        elif kind == 'last': dsc.lastB = mod
        else: dsc.merge[key] = mod
        out['single'][nm] = proxy(dsc)
        print('single', nm, {k: round(v, 3) for k, v in out['single'][nm].items()}, flush=True)
    # ---- stage B: greedy local moves
    recB = record(cur, 'B0'); out['stageB'].append(recB); cur_r2, cur_cost = recB['proxy']['mean'], recB['cost']['total']
    for rnd in range(rounds):
        need = needed(cur); trials = []
        for l in range(1, depth + 1):
            if l == depth and cur.lastB is not None: continue
            for gq, Gq in EGROUPS.items():
                rows = [t for t in Gq if t in need[l]]
                if not rows: continue
                for gk, Gk in EGROUPS.items():
                    M = cur.mask[l].clone(); sub = M[rows][:, Gk]
                    if not sub.any(): continue
                    for t in rows: M[t, Gk] = False
                    if not all(M[t].any() for t in rows): continue
                    c = cur.copy(); c.mask[l] = M; c.moves.append(f'prune L{l} {gq}<-{gk}'); trials.append(c)
        for site in sites:
            toks = site_tokens(cur, site)
            if site[0] == 'm' and int(site[1:]) in cur.merge: continue
            if site != 'emb' and site[0] in 'am' and int(site[1:]) == depth and cur.lastB is not None: continue
            for gn, G in NGROUPS.items():
                tt = [t for t in G if t in toks and (site, t) not in cur.drop]
                if not tt: continue
                c = cur.copy(); c.drop |= {(site, t) for t in tt}; c.moves.append(f'drop {site} {gn}'); trials.append(c)
            for gn, G in PGROUPS.items():
                if (site, gn) in cur.pool or len([t for t in G if t in toks]) < 2: continue
                c = cur.copy(); c.pool.add((site, gn)); c.moves.append(f'pool {site} {gn}'); trials.append(c)
        for nm, (kind, key, mod) in cands.items():
            if kind == 'last':
                if cur.lastB is not None: continue
                c = cur.copy(); c.lastB = mod; c.moves.append(f'last {nm}'); trials.append(c); continue
            l = key if kind == 'merge' else (None if key == 'emb' else int(key[1:]))
            if l == depth and cur.lastB is not None: continue
            if kind == 'sur':
                if key in cur.sur and cur.sur[key].cost <= mod.cost: continue
                if l is not None and key[0] in 'mr' and l in cur.merge: continue
                c = cur.copy(); c.sur[key] = mod; c.moves.append(f'sur {nm}'); trials.append(c)
            else:
                if key in cur.merge or f'm{key}' in cur.sur or f'r{key}' in cur.sur: continue
                c = cur.copy(); c.merge[key] = mod; c.moves.append(f'merge {nm}'); trials.append(c)
        scored = []
        for c in trials:
            cc = cost_of(c, model, exact)['total']
            if cc >= cur_cost: continue
            r2 = proxy(c)['mean']; scored.append((r2 - cur_r2, cur_cost - cc, c))
        if not scored: break
        free = [s_ for s_ in scored if s_[0] >= 0]
        best = max(free, key=lambda s_: s_[1]) if free else max(scored, key=lambda s_: s_[0] / s_[1])
        cur = best[2]; rec = record(cur, f'B{rnd + 1}'); out['stageB'].append(rec); cur_r2, cur_cost = rec['proxy']['mean'], rec['cost']['total']
        print('B', rnd + 1, cur.moves[-1], {k: round(v, 3) for k, v in rec['proxy'].items()}, rec['cost'], 'KL bits %.0f' % rec['KL_bits'], '(%d trials)' % len(scored), flush=True)
        json.dump(out, open(os.path.join(run_dir, OUT_TAG[0] + ('_smoke.json' if smoke else '.json')), 'w'), indent=1)
    # ---- stage C: Pareto points, K = 64 independent draws on the test set, scenario agreement
    # rebuild the stage-B descriptions by replaying the accepted moves (stored in order)
    Bd = [descA[eps0].copy()]
    cur2 = descA[eps0].copy()
    for rec in out['stageB'][1:]:
        mv = rec['moves'][-1]; cur2 = apply_move(cur2, mv, cands); Bd.append(cur2)
    allp = [(r['cost']['total'], r['proxy']['mean'], ('A', i)) for i, r in enumerate(out['stageA'])] + \
           [(r['cost']['total'], r['proxy']['mean'], ('B', i)) for i, r in enumerate(out['stageB'])]
    par = [p for p in allp if not any((q[0] <= p[0] and q[1] > p[1]) or (q[0] < p[0] and q[1] >= p[1]) for q in allp)]
    par = sorted(set(par), key=lambda p: p[0])
    n_test, K = (300, 8) if smoke else (2000, 64)
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz'))
    Tt = {k: Tz[k][:n_test] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    Xt = Tz['X'][:n_test]
    shapes = {s: tuple(st[s].shape[1:]) for s in sites}
    ind_noises = lambda seed, n, K_: noise_iter(seed, n, K_, shapes)
    devT = qw.deviations_t(marginal_desc(model, Desc(depth, Dd), Xt, ind_noises(1, n_test, K), sg, means, sig_eff, K), Tt)
    devT2 = qw.deviations_t(marginal_desc(model, Desc(depth, Dd), Xt, ind_noises(7, n_test, K), sg, means, sig_eff, K), Tt)
    out['floor'] = {k: DP.r2(devT[k], devT2[k]) for k in devT}
    rs = np.random.default_rng(95_000); dsn = W.sample_np(4096 if smoke else 2 ** 15, rs); Xs, vs = W.tokens_np(dsn, rs, meta['ds'])
    ref, _, mom, wb = W.references(dsn)
    Ts = dict(comp_means=ref['bayes'][0], comp_vars=ref['bayes'][1], w_bayes=wb, mean_bayes=mom['bayes'][0], var_bayes=mom['bayes'][1])
    Fsc = DP.scenario_features(dsn, Xs, Ts)
    def scen(desc):
        hs = []
        for i in range(0, len(Xs), 4096):
            hh = marginal_desc(model, desc, Xs[i:i + 4096], ind_noises(21 + i, len(Xs[i:i + 4096]), 8), sg, means, sig_eff, 8); hs.append(hh)
        h = tuple(np.concatenate([x[j] for x in hs]) for j in range(3))
        return DP.score_scenarios(Fsc, qw.deviations_t(h, Ts))
    net_sc = scen(Desc(depth, Dd)); out['network_scenarios'] = net_sc
    for (cst, pr, (stg, i)) in par:
        desc = descA[float(out['stageA'][i]['tag'].split('=')[1])] if stg == 'A' else Bd[i]
        dv = qw.deviations_t(marginal_desc(model, desc, Xt, ind_noises(11, n_test, K), sg, means, sig_eff, K), Tt)
        sc = scen(desc)
        row = dict(stage=stg, index=i, cost=(out['stageA'] if stg == 'A' else out['stageB'])[i]['cost'], KL_bits=(out['stageA'] if stg == 'A' else out['stageB'])[i]['KL_bits'],
                   moves=desc.moves, proxy_mean=pr, R2={k: DP.r2(devT[k], dv[k]) for k in dv},
                   scenario_agree=int(sum(sc[k]['hit'] == net_sc[k]['hit'] for k in sc)), scenarios={k: sc[k]['hit'] for k in sc})
        out['pareto'].append(row)
        print('P', stg, i, row['cost'], {k: round(v, 3) for k, v in row['R2'].items()}, 'scen', row['scenario_agree'], flush=True)
        json.dump(out, open(os.path.join(run_dir, OUT_TAG[0] + ('_smoke.json' if smoke else '.json')), 'w'), indent=1)
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(run_dir, OUT_TAG[0] + ('_smoke.json' if smoke else '.json')), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


def apply_move(desc, mv, cands):
    c = desc.copy(); p = mv.split()
    if p[0] == 'prune':
        l = int(p[1][1:]); gq, gk = p[2].split('<-'); need = needed(c)
        rows = [t for t in EGROUPS[gq] if t in need[l]]
        for t in rows: c.mask[l][t, EGROUPS[gk]] = False
    elif p[0] == 'drop':
        site, gn = p[1], p[2]; toks = site_tokens(c, site)
        c.drop |= {(site, t) for t in NGROUPS[gn] if t in toks}
    elif p[0] == 'pool':
        c.pool.add((p[1], p[2]))
    elif p[0] == 'sur':
        kind, key, mod = cands[p[1]]; c.sur[key] = mod
    elif p[0] == 'merge':
        kind, key, mod = cands[p[1]]; c.merge[key] = mod
    elif p[0] == 'last':
        kind, key, mod = cands[p[1]]; c.lastB = mod
    c.moves.append(mv); return c


if __name__ == '__main__':
    run(sys.argv[1], smoke='--smoke' in sys.argv)
