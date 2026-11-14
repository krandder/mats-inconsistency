"""PFN experiment harness (world-agnostic): set-style transformer, training on fresh episodes, evaluation against
exact Bayes and named references at N far beyond training, per-episode predictions saved as npz.

=====================================================================================================================
WORLD INTERFACE (duck-typed; reference implementation: junta.py)
  A world module exposes  make_world(cfg: dict | None) -> world  (cfg merged over the module's defaults). The object:
    name: str; cfg: dict (full config, saved with every run)
    target_kind: 'real' (Gaussian-mixture predictive) or 'cat' (categorical; then also n_classes: int)
    feat_dim: int                       token feature size F
    train_policies, heldout, test_policies: lists of policy names (test = train + held-out)
    sample(rng, E, N, policy, n=None) -> ep: dict of numpy arrays with at least
        'X'      (E, T, F) float32 tokens, the QUERY token LAST (index T-1); no positional information is added
        'mask'   (E, T) bool, valid tokens (query always valid; context i valid iff i < n[i] when n is given)
        'target' (E,) float (real) or int (cat)
      plus any world-specific arrays its exact_bayes / references / meta need. Must be deterministic given rng.
    sample_train(rng, E, nmin, nmax) -> ep    training batch (policy mixture, per-episode N in [nmin, nmax])
    exact_bayes(ep) -> pred                   real: (w, m, v) arrays (E, K), weights sum to 1 per row, padded
                                              components have w = 0; cat: probs (E, C)
    references(ep) -> dict name -> pred      MUST include 'bayes'; add every reference that separates an ability
    meta(ep) -> dict name -> (E,) array      per-episode columns (saved; used by splits)
  Optional:
    splits: dict name -> fn(meta) -> bool mask        (metrics are reported per split; default {'all'})
    abilities: list of (label, weaker_ref, stronger_ref, split): reports the fraction of the weaker->stronger NLL
               gain that the model achieves, (nll_weak - nll_model) / (nll_weak - nll_strong)
    fit_fixed(rng) -> json-able dict, set_fixed(d)    prior-level references fitted on training episodes
                                                      (fitted once in `evalset`, stored, re-applied at eval)

ENTRY POINTS (python harness.py <cmd> ...; world module must be importable from this directory)
  train   --world junta --world-cfg '{"tau2":0}' --depth 2 --width 64 --heads 4 --mix 4 --seed 0
          --steps 40000 --bs 256 --nmin 8 --nmax 64 --lr 5e-4 [--sink 1] --out runs/X
          (N per training episode is log-uniform in [nmin, nmax]; --sink 1 adds a learned key every token sees;
           --sum-head 16 adds an unnormalised precision head per layer; --nmode batch draws one N per batch and
           sizes the batch as min(bs, tok_budget / N))
          -> runs/X/{model.pt (final), best.pt, meta.json (args, world cfg, curve, timing)}
  evalset --world junta --world-cfg ... --Ns 16,64,...,4096 --eps 1000 --seed 7 --out runs/evalset_X
          -> per (N, policy) cell: cell_N{N}_{policy}.npz (target, meta, per-ref NLL/mean/var, Bayes mixture);
             index.json (world cfg, fixed-ref fit, seeds, ref timing). Episodes are regenerated from seeds at eval.
  eval    --run runs/X [--ckpt best.pt] --evalset runs/evalset_X [--out runs/X/eval]
          -> pred_N{N}_{policy}.npz per cell (model mixture, NLL, KL(Bayes||model) by quadrature, moments)
             summary.json: per cell and split: NLL of model and every ref, gap to Bayes (mean, se), KL,
             decomposition (fraction of model's gain over the weakest ref reproduced by each ref), abilities.
  Library use: load_run(dir) -> (model, world, meta); predict(model, X, mask) -> pred;
               query_attention(model, X, mask) -> list per layer of (E, heads, T) query-row attention;
               hidden_states(model, X, mask) -> residual stream per layer. With --sink, position 0 is the sink
               and token i of X sits at position i+1 in these arrays.
  Metrics helpers: nll(pred, y, kind), kl_bayes_model(pb, pm, kind), moments(pred).
=====================================================================================================================
"""
import argparse, importlib, json, math, os, sys, time, zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================== model
class Block(nn.Module):
    def __init__(self, d, h, mlp_ratio=4, sum_head=0):
        super().__init__()
        self.h = h; self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))
        # optional unnormalised 'sum' head (precision channel): S_i = sum_j sigmoid(q_i.k_j/sqrt(ds)) v_j over valid
        # context keys, passed through sign(S) log(1+|S|) so log-counts / log-precisions are linear in its output
        self.ds = sum_head
        if sum_head:
            self.sqkv = nn.Linear(d, 3 * sum_head); self.so = nn.Linear(sum_head, d)

    def _sum_head(self, h, keymask, chunk=512):
        qs, ks, vs = self.sqkv(h).split(self.ds, -1)
        km = keymask[:, 0, 0, :].to(h.dtype)[:, None, :]                              # (B, 1, T)
        outs = []
        for i in range(0, h.shape[1], chunk):
            g = torch.sigmoid(qs[:, i:i + chunk] @ ks.transpose(-1, -2) / math.sqrt(self.ds)) * km
            outs.append(g @ vs)
        S = torch.cat(outs, 1)
        return self.so(torch.sign(S) * torch.log1p(S.abs()))

    def forward(self, x, keymask, want_q=False):
        B, T, d = x.shape; dh = d // self.h
        q, k, v = self.qkv(self.ln1(x)).split(d, -1)
        q, k, v = (t.view(B, T, self.h, dh).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=keymask)
        qa = None
        if want_q:
            s = (q[:, :, -1:] @ k.transpose(-1, -2))[:, :, 0] / math.sqrt(dh)        # (B, h, T)
            qa = s.masked_fill(~keymask[:, :, 0], float('-inf')).softmax(-1)
        x = x + self.o(o.transpose(1, 2).reshape(B, T, d))
        if self.ds:
            x = x + self._sum_head(self.ln1(x), keymask)
        x = x + self.mlp(self.ln2(x))
        return x, qa


class SetPFN(nn.Module):
    """Pre-norm transformer without positional encoding. Every token attends to the valid CONTEXT tokens only
    (keys = context mask), so context representations do not depend on the query; the prediction is read from the
    query token (last position). Head: K-component Gaussian mixture (K=1: Gaussian) or categorical."""

    def __init__(self, fin, depth, d, h, kind='real', mix=1, n_classes=2, mlp_ratio=4, sink=False, sum_head=0):
        super().__init__()
        self.kind, self.mix, self.use_sink = kind, mix, sink
        self.sink = nn.Parameter(torch.randn(1, 1, d) * 0.02) if sink else None   # learned always-visible key
        self.inp = nn.Linear(fin, d)
        self.blocks = nn.ModuleList([Block(d, h, mlp_ratio, sum_head) for _ in range(depth)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, 3 * mix if kind == 'real' else n_classes)

    def forward(self, X, mask, want_q=False, want_h=False):
        ctx = mask.clone(); ctx[:, -1] = False
        x = self.inp(X)
        if self.use_sink:                                     # sink token at position 0 (attention-sink / BOS ablation)
            x = torch.cat([self.sink.expand(len(x), 1, -1), x], 1)
            ctx = torch.cat([torch.ones_like(ctx[:, :1]), ctx], 1)
        empty = ~ctx.any(1)
        ctx[empty, -1] = True                                 # empty context: query attends to itself (no NaN)
        km = ctx[:, None, None, :]
        qas, hs = [], [x]
        for blk in self.blocks:
            x, qa = blk(x, km, want_q); qas.append(qa); hs.append(x)
        if want_h:
            return hs
        o = self.out(self.ln(x[:, -1]))
        if self.kind == 'real':
            lg, mu, ls = o.split(self.mix, -1)
            o = (lg.log_softmax(-1), mu, ls.clamp(-7, 4))
        return (o, qas) if want_q else o


def loss_fn(out, y, kind):
    if kind == 'real':
        lw, mu, ls = out
        lp = lw - ls - 0.5 * math.log(2 * math.pi) - 0.5 * ((y[:, None] - mu) / ls.exp()) ** 2
        return -torch.logsumexp(lp, -1)
    return F.cross_entropy(out, y.long(), reduction='none')


# ============================================================== predictive utilities (numpy)
def nll(pred, y, kind='real'):
    if kind == 'cat':
        return -np.log(np.maximum(pred[np.arange(len(y)), y.astype(int)], 1e-300))
    w, m, v = pred
    lp = np.log(np.maximum(w, 1e-300)) - 0.5 * np.log(2 * math.pi * v) - 0.5 * (y[:, None] - m) ** 2 / v
    lp[w <= 0] = -np.inf
    mx = lp.max(1, keepdims=True)
    return -(mx[:, 0] + np.log(np.exp(lp - mx).sum(1)))


def moments(pred, kind='real'):
    if kind == 'cat':
        return pred[:, 1] if pred.shape[1] == 2 else pred.argmax(1), pred.max(1)
    w, m, v = pred
    mean = (w * m).sum(1)
    return mean, (w * (v + m ** 2)).sum(1) - mean ** 2


def _logmix(grid, w, m, v):
    """log density of mixtures on per-episode grids: grid (E, G); w, m, v (E, K) -> (E, G)."""
    lp = (np.log(np.maximum(w, 1e-300))[:, None, :] - 0.5 * np.log(2 * math.pi * v)[:, None, :]
          - 0.5 * (grid[:, :, None] - m[:, None, :]) ** 2 / v[:, None, :])
    lp = np.where((w > 0)[:, None, :], lp, -np.inf)
    mx = lp.max(2, keepdims=True)
    return mx[..., 0] + np.log(np.exp(lp - mx).sum(2))


def kl_bayes_model(pb, pm, kind='real', G=4001, chunk=32):
    """KL(p_Bayes || p_model) per episode. Real: trapezoid quadrature on a grid covering every Bayes component
    with weight > 1e-9 to +-10 sd (narrowest components are resolved: grid spacing is also capped at 0.2 sd_min
    by a second local grid around each heavy component)."""
    if kind == 'cat':
        return (pb * (np.log(np.maximum(pb, 1e-300)) - np.log(np.maximum(pm, 1e-300)))).sum(1)
    wb, mb, vb = pb; wm, mm, vm = pm
    E = len(wb); out = np.zeros(E)
    for i0 in range(0, E, chunk):
        sl = slice(i0, i0 + chunk)
        w, m, v = wb[sl], mb[sl], vb[sl]
        live = w > 1e-9
        sd = np.sqrt(v)
        lo = np.where(live, m - 10 * sd, np.inf).min(1); hi = np.where(live, m + 10 * sd, -np.inf).max(1)
        # base grid plus dense points around the heaviest component (guards against very narrow components)
        j = w.argmax(1); mj, sj = m[np.arange(len(j)), j], sd[np.arange(len(j)), j]
        base = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, G)[None, :]
        loc = mj[:, None] + sj[:, None] * np.linspace(-10, 10, 801)[None, :]
        grid = np.sort(np.concatenate([base, loc], 1), 1)
        lb = _logmix(grid, w, m, v); lm = _logmix(grid, wm[sl], mm[sl], vm[sl])
        pbg = np.exp(lb)
        f = pbg * (lb - lm)
        dx = np.diff(grid, axis=1)
        z = (0.5 * (pbg[:, 1:] + pbg[:, :-1]) * dx).sum(1)
        out[sl] = (0.5 * (f[:, 1:] + f[:, :-1]) * dx).sum(1) / z
    return out


# ============================================================== world / model loading
def load_world(name, cfg):
    return importlib.import_module(name).make_world(cfg)


def build_model(world, a):
    return SetPFN(world.feat_dim, a['depth'], a['width'], a['heads'], world.target_kind, a.get('mix', 1),
                  getattr(world, 'n_classes', 2), a.get('mlp_ratio', 4), bool(a.get('sink', 0)), a.get('sum_head', 0))


def load_run(run, ckpt='best.pt'):
    meta = json.load(open(os.path.join(run, 'meta.json')))
    world = load_world(meta['args']['world'], meta['world_cfg'])
    model = build_model(world, meta['args']).to(DEV)
    model.load_state_dict(torch.load(os.path.join(run, ckpt), map_location=DEV))
    model.eval()
    return model, world, meta


def _bs(T, budget=None):
    budget = budget or int(os.environ.get('PFN_EVAL_TOKENS', 1 << 17))
    return max(1, budget // T)


def predict(model, X, mask, bs=None):
    bs = bs or _bs(X.shape[1])
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = model(torch.as_tensor(X[i:i + bs], device=DEV), torch.as_tensor(mask[i:i + bs], device=DEV))
            outs.append([t.float().cpu().numpy() for t in o] if model.kind == 'real' else [o.softmax(-1).cpu().numpy()])
    if model.kind == 'real':
        lw, mu, ls = (np.concatenate([o[k] for o in outs]) for k in range(3))
        return np.exp(lw).astype(np.float64), mu.astype(np.float64), np.exp(2 * ls.astype(np.float64))
    return np.concatenate([o[0] for o in outs]).astype(np.float64)


def query_attention(model, X, mask, bs=None):
    bs = bs or _bs(X.shape[1], 1 << 15)
    per = None
    with torch.no_grad():
        for i in range(0, len(X), bs):
            _, qas = model(torch.as_tensor(X[i:i + bs], device=DEV), torch.as_tensor(mask[i:i + bs], device=DEV), True)
            qas = [a.cpu().numpy() for a in qas]
            per = qas if per is None else [np.concatenate([p, a]) for p, a in zip(per, qas)]
    return per


def hidden_states(model, X, mask, bs=None):
    """Residual stream after the embedding and after each block: list of (E, T(+1 if sink), d) arrays."""
    bs = bs or _bs(X.shape[1], 1 << 15)
    per = None
    with torch.no_grad():
        for i in range(0, len(X), bs):
            hs = model(torch.as_tensor(X[i:i + bs], device=DEV), torch.as_tensor(mask[i:i + bs], device=DEV), want_h=True)
            hs = [h.cpu().numpy() for h in hs]
            per = hs if per is None else [np.concatenate([p, h]) for p, h in zip(per, hs)]
    return per


def _tens(ep, kind):
    return (torch.as_tensor(ep['X'], device=DEV), torch.as_tensor(ep['mask'], device=DEV),
            torch.as_tensor(ep['target'], dtype=torch.float32, device=DEV))


# ============================================================== train
VAL_SEED = 10_000


def train(a):
    cfg = json.loads(a.world_cfg)
    world = load_world(a.world, cfg)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng([a.seed, zlib.crc32(a.out.encode())])
    args = vars(a)
    model = build_model(world, args).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    vep = world.sample_train(np.random.default_rng(VAL_SEED), a.nval, a.nmin, a.nmax)
    vb = nll(world.exact_bayes(vep), vep['target'], world.target_kind).mean()
    os.makedirs(a.out, exist_ok=True)
    curve, best, best_step, off = [], 1e9, -1, 0
    if a.init:                   # continue a previous run: its final weights, curve and best checkpoint
        import shutil
        pm = json.load(open(os.path.join(a.init, 'meta.json')))
        model.load_state_dict(torch.load(os.path.join(a.init, 'model.pt'), map_location=DEV))
        shutil.copy(os.path.join(a.init, 'best.pt'), os.path.join(a.out, 'best.pt'))
        curve, best, best_step, off = [tuple(c) for c in pm['curve']], pm['best_val'], pm['best_step'], pm['args']['steps'] + pm.get('step_offset', 0)
    t_data = t_comp = 0.0; t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups: g['lr'] = lr
        ta = time.time()
        if a.nmode == 'batch':       # one N per batch (log-uniform), batch size from a token budget: no padding waste
            Nb = int(np.exp(rng.uniform(math.log(a.nmin), math.log(a.nmax + 1)))); Nb = min(max(Nb, a.nmin), a.nmax)
            ep = world.sample_train(rng, int(min(a.bs, max(16, a.tok_budget // Nb))), Nb, Nb)
        else:
            ep = world.sample_train(rng, a.bs, a.nmin, a.nmax)
        X, M, Y = _tens(ep, world.target_kind)
        tb = time.time()
        loss = loss_fn(model(X, M), Y, world.target_kind).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if DEV == 'cuda': torch.cuda.synchronize()
        tc = time.time(); t_data += tb - ta; t_comp += tc - tb
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            v = float(nll(predict(model, vep['X'], vep['mask']), vep['target'], world.target_kind).mean())
            model.train()
            curve.append((off + step, v, v - vb))
            if v < best:
                best, best_step = v, off + step
                torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
            print(f'[{os.path.basename(a.out)}] step {step} train {loss.item():.4f} val {v:.4f} gap {v - vb:.4f} '
                  f'best {best:.4f}@{best_step} ({time.time() - t0:.0f}s; data {t_data / step * 1e3:.1f}ms '
                  f'comp {t_comp / step * 1e3:.1f}ms per step)', flush=True)
    torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
    tot = off + a.steps
    tail = [c[1] for c in curve if c[0] >= 0.75 * tot]; head = [c[1] for c in curve if c[0] < 0.75 * tot]
    meta = dict(args=args, world_cfg=world.cfg, n_params=nparam, val_bayes_nll=float(vb), best_val=best,
                best_step=best_step, last_quarter_improvement=(min(head) - min(tail)) if head and tail else None,
                converged=bool(head and tail and (min(head) - min(tail)) < 0.003), step_offset=off, init=a.init,
                curve=curve, seconds=time.time() - t0, ms_per_step=(t_data + t_comp) / a.steps * 1e3,
                ms_data_per_step=t_data / a.steps * 1e3, ms_compute_per_step=t_comp / a.steps * 1e3, device=DEV)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)


# ============================================================== eval set (references computed once)
def _cell_seed(seed, N, pol):
    return [seed, N, zlib.crc32(pol.encode())]


def eps_for(N, base):
    return int(max(100, min(base, base * 512 // max(N, 512))))


def make_evalset(a):
    cfg = json.loads(a.world_cfg)
    world = load_world(a.world, cfg)
    os.makedirs(a.out, exist_ok=True)
    fixed = world.fit_fixed(np.random.default_rng([a.seed, 1])) if hasattr(world, 'fit_fixed') else None
    Ns = [int(x) for x in a.Ns.split(',')]
    pols = a.policies.split(',') if a.policies else world.test_policies
    cells, timing = [], {}
    for N in Ns:
        for pol in pols:
            E = a.eps_fixed or eps_for(N, a.eps)
            t0 = time.time()
            ep = world.sample(np.random.default_rng(_cell_seed(a.seed, N, pol)), E, N, pol)
            t1 = time.time()
            refs = world.references(ep)
            t2 = time.time()
            d = dict(target=ep['target'])
            for k, v in world.meta(ep).items(): d['meta_' + k] = v
            for r, p in refs.items():
                d['nll_' + r] = nll(p, ep['target'], world.target_kind)
                mu, var = moments(p, world.target_kind); d['mean_' + r] = mu; d['var_' + r] = var
            if world.target_kind == 'real':
                w, m, v = refs['bayes']; keep = w > 1e-12
                d['bayes_k'] = keep.sum(1); d['bayes_w'] = w[keep]; d['bayes_m'] = m[keep]; d['bayes_v'] = v[keep]
            else:
                d['bayes_p'] = refs['bayes']
            name = f'cell_N{N}_{pol}.npz'
            np.savez_compressed(os.path.join(a.out, name), **d)
            cells.append(dict(N=N, policy=pol, E=E, seed=_cell_seed(a.seed, N, pol), file=name, refs=list(refs)))
            timing[f'{N}_{pol}'] = dict(sample_s=t1 - t0, refs_s=t2 - t1, per_episode_ms=(t2 - t1) / E * 1e3)
            print(f'evalset N={N} {pol} E={E}: sample {t1 - t0:.1f}s refs {t2 - t1:.1f}s; ' +
                  ' '.join(f'{r}={d["nll_" + r].mean():.3f}' for r in refs), flush=True)
    json.dump(dict(world=a.world, world_cfg=world.cfg, fixed=fixed, cells=cells, timing=timing, seed=a.seed),
              open(os.path.join(a.out, 'index.json'), 'w'), indent=1)


def _unragged(d):
    k = d['bayes_k']; E = len(k); K = int(k.max())
    w = np.zeros((E, K)); m = np.zeros((E, K)); v = np.ones((E, K))
    off = np.concatenate([[0], np.cumsum(k)])
    for i in range(E):
        w[i, :k[i]] = d['bayes_w'][off[i]:off[i + 1]]; m[i, :k[i]] = d['bayes_m'][off[i]:off[i + 1]]
        v[i, :k[i]] = d['bayes_v'][off[i]:off[i + 1]]
    return w, m, v


def _mse(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def summarize_cell(d, nm, kl, world, refs, mom=None):
    meta = {k[5:]: d[k] for k in d if k.startswith('meta_')}
    splits = getattr(world, 'splits', None) or {'all': lambda m: np.ones(len(d['target']), bool)}
    res = {}
    base = 'r0' if 'r0' in refs else ('prior' if 'prior' in refs else None)
    for sname, fn in splits.items():
        sel = fn(meta)
        if sel.sum() < 2: continue
        nb = d['nll_bayes'][sel]
        r = dict(n=int(sel.sum()), nll_model=float(nm[sel].mean()), kl_model=_mse(kl[sel]),
                 gap_model=_mse(nm[sel] - nb))
        if mom is not None:          # bias and variance of the model predictive relative to Bayes
            dm = mom[0][sel] - d['mean_bayes'][sel]
            r['bias_mean'] = float(dm.mean()); r['bias_rms'] = float(np.sqrt((dm ** 2).mean()))
            r['bias_rms_over_sd'] = float(np.sqrt((dm ** 2 / d['var_bayes'][sel]).mean()))
            r['logvar_ratio'] = _mse(np.log(mom[1][sel] / d['var_bayes'][sel]))
            # moment-matched Gaussian split of the KL: variance part 0.5(log r + 1/r - 1), r = v_model / v_Bayes;
            # mean part 0.5 (m_model - m_Bayes)^2 / v_model; the rest of KL(Bayes||model) is shape (non-Gaussian)
            rr = mom[1][sel] / d['var_bayes'][sel]
            r['kl_var_part'] = float((0.5 * (np.log(rr) + 1 / rr - 1)).mean())
            r['kl_mean_part'] = float((0.5 * dm ** 2 / mom[1][sel]).mean())
        for ref in refs:
            r['nll_' + ref] = float(d['nll_' + ref][sel].mean())
            r['gap_' + ref] = _mse(d['nll_' + ref][sel] - nb)
        if base:
            gain = d['nll_' + base][sel].mean() - nm[sel].mean()
            r['decomp_base'] = base; r['model_gain_over_base'] = float(gain)
            r['decomp'] = {ref: float((d['nll_' + base][sel].mean() - d['nll_' + ref][sel].mean()) / gain)
                           if abs(gain) > 1e-9 else None for ref in refs}
        res[sname] = r
    ab = {}
    for label, weak, strong, sname in getattr(world, 'abilities', []):
        if weak not in refs or strong not in refs or sname not in res: continue
        sel = splits[sname](meta)
        nw, ns, nmm = d['nll_' + weak][sel].mean(), d['nll_' + strong][sel].mean(), nm[sel].mean()
        ab[label] = dict(split=sname, nll_weak=float(nw), nll_strong=float(ns), nll_model=float(nmm),
                         ref_gain=float(nw - ns), frac=float((nw - nmm) / (nw - ns)) if abs(nw - ns) > 1e-6 else None)
    return res, ab


def evaluate(a):
    model, world, meta = load_run(a.run, a.ckpt)
    idx = json.load(open(os.path.join(a.evalset, 'index.json')))
    assert idx['world_cfg'] == world.cfg, 'eval set built for a different world config'
    if idx.get('fixed') and hasattr(world, 'set_fixed'): world.set_fixed(idx['fixed'])
    out = a.out or os.path.join(a.run, 'eval')
    os.makedirs(out, exist_ok=True)
    summary = dict(run=a.run, ckpt=a.ckpt, evalset=a.evalset, cells=[])
    kind = world.target_kind
    for c in idx['cells']:
        if a.Ns and c['N'] not in [int(x) for x in a.Ns.split(',')]: continue
        d = dict(np.load(os.path.join(a.evalset, c['file'])))
        t0 = time.time()
        ep = world.sample(np.random.default_rng(c['seed']), c['E'], c['N'], c['policy'])
        t1 = time.time()
        assert np.allclose(ep['target'], d['target']), 'regenerated episodes differ from the eval set'
        if DEV == 'cuda': torch.cuda.synchronize()
        tf = time.time()
        pm = predict(model, ep['X'], ep['mask'])
        if DEV == 'cuda': torch.cuda.synchronize()
        tf2 = time.time()
        nm = nll(pm, ep['target'], kind)
        pb = _unragged(d) if kind == 'real' else d['bayes_p']
        kl = kl_bayes_model(pb, pm, kind)
        mu, var = moments(pm, kind)
        save = dict(nll_model=nm, kl_model=kl, mean_model=mu, var_model=var)
        if kind == 'real':
            save.update(model_w=pm[0].astype(np.float32), model_m=pm[1].astype(np.float32), model_v=pm[2].astype(np.float32))
        else:
            save['model_p'] = pm.astype(np.float32)
        np.savez_compressed(os.path.join(out, f'pred_N{c["N"]}_{c["policy"]}.npz'), **save)
        res, ab = summarize_cell(d, nm, kl, world, c['refs'], (mu, var) if kind == 'real' else None)
        summary['cells'].append(dict(N=c['N'], policy=c['policy'], E=c['E'], forward_s=tf2 - tf,
                                     forward_ms_per_episode=(tf2 - tf) / c['E'] * 1e3, sample_s=t1 - t0,
                                     splits=res, abilities=ab))
        r = res['all']
        print(f'N={c["N"]:5d} {c["policy"]:8s} gap {r["gap_model"][0]:.4f}+-{r["gap_model"][1]:.4f} '
              f'KL {r["kl_model"][0]:.4f} | ' + ' '.join(f'{k}={r["gap_" + k][0]:.3f}' for k in c['refs'] if k != 'bayes')
              + f' | fwd {(tf2 - tf) / c["E"] * 1e3:.2f}ms/ep', flush=True)
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)


# ============================================================== CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['train', 'evalset', 'eval'])
    ap.add_argument('--world', default='junta'); ap.add_argument('--world-cfg', dest='world_cfg', default='{}')
    ap.add_argument('--depth', type=int, default=2); ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--heads', type=int, default=4); ap.add_argument('--mix', type=int, default=4)
    ap.add_argument('--mlp-ratio', dest='mlp_ratio', type=int, default=4)
    ap.add_argument('--sink', type=int, default=0, help='1 = learned attention-sink token visible to every query')
    ap.add_argument('--sum-head', dest='sum_head', type=int, default=0,
                    help='width of one unnormalised sigmoid-gated, log-compressed sum head per layer (0 = off)')
    ap.add_argument('--init', default='', help='continue from this run dir (final weights, curve, best.pt)')
    ap.add_argument('--nmode', default='episode', choices=['episode', 'batch'],
                    help="episode: N per episode, padded batch (default); batch: one log-uniform N per batch")
    ap.add_argument('--tok-budget', dest='tok_budget', type=int, default=16384, help='context tokens per batch (nmode batch)')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup', type=int, default=1000); ap.add_argument('--eval-every', dest='eval_every', type=int, default=2000)
    ap.add_argument('--nval', type=int, default=2048)
    ap.add_argument('--nmin', type=int, default=8); ap.add_argument('--nmax', type=int, default=64)
    ap.add_argument('--out', default=''); ap.add_argument('--run', default=''); ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--evalset', default='')
    ap.add_argument('--Ns', default='16,32,64,128,256,512,1024,2048,4096')
    ap.add_argument('--eps', type=int, default=1000); ap.add_argument('--eps-fixed', dest='eps_fixed', type=int, default=0)
    ap.add_argument('--policies', default='')
    a = ap.parse_args()
    if a.cmd == 'eval' and a.Ns == ap.get_default('Ns'): a.Ns = ''
    {'train': train, 'evalset': make_evalset, 'eval': evaluate}[a.cmd](a)


if __name__ == '__main__':
    main()
