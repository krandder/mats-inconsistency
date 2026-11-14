"""E-SUD (design v2): one any-order network over Sudoku cells plus a homogeneous Solvable token; search chains;
computation-as-evidence for Solvable.

python sud.py train --N 6 --seed 0 --data DIR --out DIR [--layers 4]
python sud.py eval  --N 6 --seed 0 --data DIR --out DIR

Model: transformer over C cell tokens (digit 1..N or MASK) + one Solvable token (0, 1 or MASK) with learned position
ids. It is told nothing about rows, columns, boxes or the rule. Outputs: q(x_i | visible) for every cell and
q(Solvable = 1 | visible) at the Solvable token. Training targets are outcomes only: a sampled completion of each
solvable board, and the solvable bit. Exact marginals are read only by the scorer. No hand-coded checker at test time:
a "witness" is a completed board that the model's own Solvable token accepts.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MS = [0, 1, 2, 4, 8, 16]
RULES = ['rand', 'voi', 'conf']          # P1 random order, P2 stiff-gated VoI, P3 confidence-first (stiff-first)
ARM = {'rand': 'P1', 'voi': 'P2', 'conf': 'P3'}


class Net(nn.Module):
    def __init__(self, N, d=128, L=4, H=4):
        super().__init__()
        self.N, self.C = N, N * N
        self.tok = nn.Embedding(N + 4, d)  # 0 MASK, 1..N digits, N+1 S-MASK, N+2 S=0, N+3 S=1
        self.pos = nn.Parameter(torch.randn(self.C + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.0, batch_first=True, norm_first=True,
                                           activation='gelu')
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, N)
        self.sol = nn.Linear(d, 1)

    def forward(self, x, s):
        """x (B,C) long cells (0 = MASK); s (B,) long Solvable token in {-1 MASK, 0, 1}.
        Returns cell logits (B,C,N), Solvable logit (B,)."""
        st = torch.where(s < 0, self.N + 1, self.N + 2 + s.clamp_min(0))
        h = self.tok(torch.cat([x, st[:, None]], 1)) + self.pos[None]
        h = self.ln(self.enc(h))
        return self.out(h[:, :-1]), self.sol(h[:, -1]).squeeze(-1)


def load(data, N, split):
    d = np.load(os.path.join(data, f's{N}_{split}.npz') if not os.environ.get('ESUD_SPLIT_PREFIX') else
                os.path.join(data, f"s{N}_{os.environ['ESUD_SPLIT_PREFIX']}{split}.npz"))  # R3: s6_hard_{val,test}.npz
    return {k: d[k] for k in d.files}


def t(a, dt=torch.long):
    return torch.as_tensor(np.asarray(a), dtype=dt, device=DEV)


@torch.no_grad()
def fwd(model, x, s, bs=16384):
    lo, so = [], []
    for i in range(0, x.shape[0], bs):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'):
            l, q = model(x[i:i + bs], s[i:i + bs])
        lo.append(l.float()); so.append(q.float())
    return torch.cat(lo), torch.cat(so)


def make_batch(D, idx, gen):
    """Any-order training states. Reveal k ~ Uniform{0..#empty} extra cells: from the sampled completion (solvable
    boards) or from the pre-corruption grid (unsolvable boards; the state stays unsolvable). Solvable boards: the
    Solvable token is revealed (=1) half the time, else masked with target 1; cell loss on all masked cells.
    Unsolvable boards: Solvable masked with target 0; no cell targets."""
    b = t(D['board'][idx]); y = t(D['outcome'][idx]); sv = t(D['solvable'][idx])
    B, C = b.shape
    empty = b == 0
    nemp = empty.sum(1)
    k = (torch.rand(B, device=DEV, generator=gen) * (nemp + 1).float()).floor().long()
    sc = torch.rand(B, C, device=DEV, generator=gen) + (~empty).float() * 2
    rank = sc.argsort(1).argsort(1)
    rev = (rank < k[:, None]) & empty
    x = torch.where(rev, y, b)
    srev = (torch.rand(B, device=DEV, generator=gen) < 0.5) & (sv == 1)
    s_in = torch.where(srev, torch.ones_like(sv), -torch.ones_like(sv))
    cell_mask = (x == 0) & (sv[:, None] == 1)
    return x, y, sv.float(), s_in, cell_mask, ~srev


def losses(model, x, y, sv, s_in, cell_mask, s_mask):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'):
        lg, sl = model(x, s_in)
    lg, sl = lg.float(), sl.float()
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), (y - 1).clamp_min(0).reshape(-1), reduction='none').view_as(cell_mask)
    cl = (ce * cell_mask).sum() / cell_mask.sum().clamp_min(1)
    sm = s_mask.float()
    bl = (F.binary_cross_entropy_with_logits(sl, sv, reduction='none') * sm).sum() / sm.sum().clamp_min(1)
    return cl, bl


@torch.no_grad()
def val_outcome_loss(model, V):
    """Early-stopping loss on OUTCOMES only, at the board state: cell CE vs the sampled completion (Solvable
    revealed = 1, solvable boards) + Solvable BCE (token masked, all boards)."""
    x = t(V['board']); y = t(V['outcome']); sv = t(V['solvable'])
    ones = torch.ones_like(sv)
    cl, _ = losses(model, x, y, sv.float(), ones, (x == 0) & (sv[:, None] == 1), torch.zeros_like(sv, dtype=torch.bool))
    _, bl = losses(model, x, y, sv.float(), -ones, torch.zeros_like(x, dtype=torch.bool), torch.ones_like(sv, dtype=torch.bool))
    return float(cl), float(bl)


def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    D, V = load(a.data, a.N, 'train'), load(a.data, a.N, 'val')
    model = Net(a.N, L=a.layers).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    gen = torch.Generator(device=DEV); gen.manual_seed(a.seed)
    n = len(D['board']); best = (1e9, 0); log = []
    tag = f's{a.N}_seed{a.seed}'
    ck = os.path.join(a.out, f'ck_{tag}.pt'); bestp = os.path.join(a.out, f'model_{tag}.pt')
    step0 = 0
    if os.path.exists(ck):  # resume after preemption
        st = torch.load(ck, map_location=DEV)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); step0 = st['step']; best = tuple(st['best']); log = st['log']
        np.random.seed(a.seed + step0)
    t0 = time.time()
    for step in range(step0, a.steps):
        lr = a.lr * min(1, (step + 1) / 500) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g['lr'] = lr
        idx = np.random.randint(0, n, a.bs)
        x, y, sv, s_in, cm, smask = make_batch(D, idx, gen)
        cl, bl = losses(model, x, y, sv, s_in, cm, smask)
        loss = cl + bl
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            model.eval(); vc, vb = val_outcome_loss(model, V); model.train()
            log.append(dict(step=step + 1, train_cell=round(float(cl), 4), train_sol=round(float(bl), 4), val_cell=round(vc, 4),
                            val_sol=round(vb, 4), sec=round(time.time() - t0, 1)))
            print(json.dumps(log[-1]), flush=True)
            if vc + vb < best[0]:
                best = (vc + vb, step + 1); torch.save(model.state_dict(), bestp)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1, best=best, log=log), ck)
    json.dump(dict(log=log, best_val=best[0], best_step=best[1], layers=a.layers, steps=a.steps),
              open(os.path.join(a.out, f'train_{tag}.json'), 'w'), indent=1)
    if os.path.exists(ck):
        os.remove(ck)


# ------------------------------------------------------------------ search chains
def ent(p):
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


@torch.no_grad()
def chains(model, boards, K, rule, gen, ntop=6, tau=1.0):
    """K materialisation chains per board, run under the hypothesis Solvable = 1 (token revealed), i.e. the
    proposal q(z | board, X = 1). Each step one forward pass on S_{t-1} gives q(z | S_{t-1}) for every masked cell;
    the rule picks z_t, its value is sampled from q(z_t | S_{t-1}), and S_t = S_{t-1} + {z_t}. Rules:
      rand: uniform random masked cell (P1).
      conf: lowest-entropy masked cell = stiffest arc (P3, = P2-Stiff).
      voi : among the 6 lowest-entropy masked cells, the one maximising sum_d q(z=d|S) * sum_Y ||q(Y|S,z=d) - q(Y|S)||^2
            over the other masked cells (P2, stiff-gated VoI; 6*N extra passes per step).
    Returns Z (B,K,C) completions, logq (B,K) = log q_sigma(z), rb (B,K,C,N) the conditional each cell was sampled
    from (Rao-Blackwell claims), chk (B,K) = the model's own q(Solvable=1 | board + z) with the token masked."""
    B, C = boards.shape; N = model.N
    x = boards.repeat_interleave(K, 0).clone(); BK = B * K
    one = torch.ones(BK, dtype=torch.long, device=DEV)
    logq = torch.zeros(BK, device=DEV)
    rb = torch.zeros(BK, C, N, device=DEV)
    logp_bel = torch.zeros(BK, device=DEV)  # R3: untempered believed log-mass of the sampled values
    ar = torch.arange(BK, device=DEV)
    nemp = int((x == 0).sum(1).max())
    for _ in range(nemp):
        m = x == 0
        act = m.any(1)
        if not act.any():
            break
        lg, _ = fwd(model, x, one)
        p = F.softmax(lg / tau, -1)  # R2: proposal temperature (1.0 = round 1)
        if rule == 'rand':
            sc = torch.rand(BK, C, device=DEV, generator=gen)
        else:
            sc = -ent(p) + 1e-6 * torch.rand(BK, C, device=DEV, generator=gen)
        sc = torch.where(m, sc, torch.full_like(sc, -1e9))
        if rule == 'voi':
            k = min(ntop, C)
            cand = sc.topk(k, 1).indices                            # (BK,k)
            okc = m.gather(1, cand)
            xb = x[:, None, None, :].expand(BK, k, N, C).clone()
            dd = torch.arange(1, N + 1, device=DEV)[None, None, :, None].expand(BK, k, N, 1)
            xb.scatter_(3, cand[:, :, None, None].expand(BK, k, N, 1), dd)
            lgb, _ = fwd(model, xb.reshape(-1, C), torch.ones(BK * k * N, dtype=torch.long, device=DEV))
            qb = F.softmax(lgb, -1).view(BK, k, N, C, N)
            other = m[:, None, None, :].expand(BK, k, N, C).clone()
            other.scatter_(3, cand[:, :, None, None].expand(BK, k, N, 1), False)
            diff = (((qb - p[:, None, None]) ** 2).sum(-1) * other).sum(-1)    # (BK,k,N)
            wz = p[ar[:, None], cand]                                          # (BK,k,N)
            voi = (wz * diff).sum(-1)
            voi = torch.where(okc, voi, torch.full_like(voi, -1e9))
            z = cand[ar, voi.argmax(1)]
        else:
            z = sc.argmax(1)
        pz = p[ar, z]
        d = torch.multinomial(pz.clamp_min(1e-12), 1, generator=gen).squeeze(1)
        lp = torch.log(pz.gather(1, d[:, None]).squeeze(1).clamp_min(1e-30))
        logq = torch.where(act, logq + lp, logq)
        pb = F.softmax(lg, -1)[ar, z]
        lpb = torch.log(pb.gather(1, d[:, None]).squeeze(1).clamp_min(1e-30))
        logp_bel = torch.where(act, logp_bel + lpb, logp_bel)
        rb[ar[act], z[act]] = pz[act]
        x[ar[act], z[act]] = d[act] + 1
    _, sl = fwd(model, x, -one)
    return x.view(B, K, C), logq.view(B, K), rb.view(B, K, C, N), torch.sigmoid(sl).view(B, K), logp_bel.view(B, K)


# ------------------------------------------------------------------ estimators and scores
def cell_est(p0, rb, chk, m, lam, weighted):
    """(lam * p0 + sum_{r<=m} w_r rb_r) / (lam + sum w_r), w_r = check (weighted) or 1. lam fitted on validation."""
    if m == 0:
        return p0
    w = chk[:, :m] if weighted else torch.ones_like(chk[:, :m])
    num = lam * p0 + (w[..., None, None] * rb[:, :m]).sum(1)
    return num / (lam + w.sum(1))[:, None, None]


def cell_scores(est, D, msk):
    """Per solvable board: mean KL(exact || est) and mean outcome CE over the board's empty cells."""
    ex = t(D['marg'], torch.float); y = t(D['outcome'])
    e = est.clamp_min(1e-6); e = e / e.sum(-1, keepdim=True)
    kl = (ex * (ex.clamp_min(1e-12).log() - e.log())).sum(-1)
    ce = -e.log().gather(2, (y - 1).clamp_min(0)[..., None]).squeeze(-1)
    n = msk.sum(1).clamp_min(1)
    return ((kl * msk).sum(1) / n).cpu().numpy(), ((ce * msk).sum(1) / n).cpu().numpy(), kl


LAMS = [0.03, 0.1, 0.3, 0.5, 1, 2, 4, 8, 16, 1e6]


def fit_lam(p0, rb, chk, m, weighted, D, msk):
    pos = D['solvable'] == 1
    return min(LAMS, key=lambda l: cell_scores(cell_est(p0, rb, chk, m, l, weighted), D, msk)[1][pos].mean())


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(p, y, nb=10):
    bins = np.minimum((p * nb).astype(int), nb - 1); e = 0.0
    for k in range(nb):
        s = bins == k
        if s.any():
            e += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(e)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sol_terms(Z, logq, chk, m):
    """First m chains: found = some completion the model's own check accepts (chk >= 0.5); best = max check;
    C_m = summed proposal mass q_sigma(z) of DISTINCT refuted completions (capped below 1)."""
    B = len(Z)
    found = (chk[:, :m] >= 0.5).any(1) if m else np.zeros(B, bool)
    best = chk[:, :m].max(1) if m else np.zeros(B)
    Cm = np.zeros(B)
    for b in range(B):
        seen = {}
        for r in range(m):
            if chk[b, r] < 0.5:
                key = Z[b, r].tobytes(); seen[key] = max(seen.get(key, -1e9), logq[b, r])
        Cm[b] = min(sum(math.exp(v) for v in seen.values()), 1 - 1e-9)
    return found, best, Cm


def sol_arms(p0s, found, best, Cm, m, theta, kappa):
    wit = np.maximum(p0s, best)
    return dict(P0=p0s,
                witness=np.where(found, wit, p0s),                                                   # ignore null search
                null=np.where(found, wit, 1 / (1 + np.exp(-(logit(p0s) + theta * np.log1p(-Cm))))),  # eta = 1-(1-C)^theta
                count=np.where(found, wit, 1 / (1 + np.exp(-(logit(p0s) - kappa * m)))))             # control: count only


def one_step_cells(boards, N):
    """Scorer-only: empty cells whose peers among the givens already hold N-1 distinct digits (naked singles)."""
    from boards import geom
    br, bc, box = geom(N); C = N * N
    out = np.zeros(boards.shape, bool)
    for bi, b in enumerate(boards):
        for i in range(C):
            if b[i]:
                continue
            ds = {b[j] for j in range(C) if j != i and b[j] and (j // N == i // N or j % N == i % N or box[j] == box[i])}
            out[bi, i] = len(ds) == N - 1
    return out


def run_eval(a):
    model = Net(a.N, L=a.layers).to(DEV)
    tag = f's{a.N}_seed{a.seed}'
    model.load_state_dict(torch.load(os.path.join(a.out, f'model_{tag}.pt'), map_location=DEV))
    model.eval()
    gen = torch.Generator(device=DEV); gen.manual_seed(1000 + a.seed)
    res = dict(N=a.N, seed=a.seed, M=MS, layers=a.layers)
    K = max(MS)
    rules = a.rules.split(',')
    S = {}
    for split in ['val', 'test']:
        D = load(a.data, a.N, split)
        if a.limit:
            D = {k: v[: a.limit] for k, v in D.items()}
        bd = t(D['board']); B, C = bd.shape
        one = torch.ones(B, dtype=torch.long, device=DEV)
        msk = (bd == 0) & t(D['solvable'], torch.bool)[:, None]
        lg0, _ = fwd(model, bd, one); p0 = F.softmax(lg0, -1)
        _, sl0 = fwd(model, bd, -one); p0s = torch.sigmoid(sl0).cpu().numpy()
        ch = {}
        for r in rules:
            t0 = time.time(); parts = []
            for i in range(0, B, a.chunk):
                parts.append(chains(model, bd[i:i + a.chunk], K, r, gen))
            Z, lq, rb, ck = [torch.cat([q[j] for q in parts]) for j in range(4)]
            ch[r] = dict(Z=Z.cpu().numpy(), lq=lq.cpu().numpy(), rb=rb, chk=ck, chk_np=ck.cpu().numpy())
            res.setdefault('sec', {})[f'{split}_{r}'] = round(time.time() - t0, 1)
            print(split, r, 'chains done', res['sec'][f'{split}_{r}'], 's', flush=True)
        S[split] = dict(D=D, msk=msk, p0=p0, p0s=p0s, ch=ch, bd=bd)

    # ---- fit one-parameter terms on VALIDATION outcomes only
    V = S['val']; yv = V['D']['solvable'].astype(float)
    fit = dict(lam={}, theta={}, kappa={})
    for r in rules:
        c = V['ch'][r]
        for wtd in [0, 1]:
            fit['lam'][f'{r}_{wtd}'] = {m: fit_lam(V['p0'], c['rb'], c['chk'], m, wtd, V['D'], V['msk']) for m in MS if m}
        vt = {m: sol_terms(c['Z'], c['lq'], c['chk_np'], m) for m in MS if m}
        grid = np.exp(np.linspace(np.log(0.01), np.log(1000), 101))
        fit['theta'][r] = float(min(grid, key=lambda th: np.mean([ll(sol_arms(V['p0s'], *vt[m], m, th, 0)['null'], yv).mean() for m in vt])))
        kgrid = np.concatenate([[0], np.exp(np.linspace(np.log(1e-3), np.log(10), 61))])
        fit['kappa'][r] = float(min(kgrid, key=lambda kp: np.mean([ll(sol_arms(V['p0s'], *vt[m], m, 1, kp)['count'], yv).mean() for m in vt])))
    res['fit'] = fit

    # ---- score on TEST
    T = S['test']; D = T['D']; pos = D['solvable'] == 1; ys = D['solvable'].astype(float)
    os1 = t(one_step_cells(D['board'], a.N), torch.bool) & T['msk']
    kl0, ce0, klc = cell_scores(T['p0'], D, T['msk'])
    multi = T['msk'] & ~os1
    res['D0'] = dict(kl_P0_all=float(kl0[pos].mean()),
                     kl_P0_onestep=float((klc * os1).sum() / os1.sum().clamp_min(1)), n_onestep=int(os1.sum()),
                     kl_P0_multistep=float((klc * multi).sum() / multi.sum().clamp_min(1)), n_multistep=int(multi.sum()))
    cells = {0: {'P0': dict(kl=float(kl0[pos].mean()), ce=float(ce0[pos].mean()))}}
    per = {'kl_P0_M0': kl0[pos].astype(np.float32)}
    sol = {}
    for r in rules:
        c = T['ch'][r]; nm = ARM[r]
        for m in MS:
            if m == 0:
                continue
            cells.setdefault(m, {})
            for wtd, suf in [(1, ''), (0, '_nochk')]:
                e = cell_est(T['p0'], c['rb'], c['chk'], m, fit['lam'][f'{r}_{wtd}'][m], wtd)
                kl, ce, _ = cell_scores(e, D, T['msk'])
                cells[m][nm + suf] = dict(kl=float(kl[pos].mean()), ce=float(ce[pos].mean()))
                per[f'kl_{nm}{suf}_M{m}'] = kl[pos].astype(np.float32)
        for m in MS:
            found, best, Cm = sol_terms(c['Z'], c['lq'], c['chk_np'], m)
            arms = sol_arms(T['p0s'], found, best, Cm, m, fit['theta'][r], fit['kappa'][r])
            row = dict(found_pos=float(found[pos].mean()), found_neg=float(found[~pos].mean()),
                       Cm_neg=float(Cm[~pos].mean()), Cm_pos=float(Cm[pos].mean()))
            for k, p in arms.items():
                l = ll(p, ys)
                row[k] = dict(ll_all=float(l.mean()), ll_neg=float(l[~pos].mean()), ll_pos=float(l[pos].mean()), ece=ece(p, ys))
                per[f'sll_{nm}_{k}_M{m}'] = l.astype(np.float32)
            sol.setdefault(nm, {})[m] = row
        # chain diagnostics: share of completions that are truly valid (scorer-only check) vs the model's check
        from boards import valid_full
        Zt = c['Z']; v = np.array([[valid_full(list(Zt[b, k]), a.N) for k in range(K)] for b in range(len(Zt))])
        chk = c['chk_np']
        res.setdefault('chain_diag', {})[nm] = dict(
            true_valid_pos=float(v[pos].mean()), true_valid_neg=float(v[~pos].mean()),
            check_acc=float(((chk >= 0.5) == v).mean()), check_fp=float(((chk >= 0.5) & ~v).sum() / max(1, (~v).sum())),
            check_fn=float(((chk < 0.5) & v).sum() / max(1, v.sum())))
    res['cells'] = cells; res['solvable'] = sol
    res['n_test'] = dict(pos=int(pos.sum()), neg=int((~pos).sum()))
    json.dump(res, open(os.path.join(a.out, f'eval_{tag}.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(a.out, f'perboard_{tag}.npz'), solvable=D['solvable'], **per)
    print(json.dumps(res['D0']))
    print(json.dumps({m: {k: round(v['kl'], 4) for k, v in cells[m].items()} for m in cells}))
    print(json.dumps({nm: {m: {k: round(v['ll_neg'], 3) for k, v in sol[nm][m].items() if isinstance(v, dict)} for m in MS} for nm in sol}))


def dump(a):
    """R2: run K=16 chains per board on val and test for the given rules at proposal temperature tau and save the raw
    chain outputs (completions, log q_sigma, the model's own check) plus p0(Solvable). No scoring here."""
    model = Net(a.N, L=a.layers).to(DEV)
    tag = f's{a.N}_seed{a.seed}'
    model.load_state_dict(torch.load(os.path.join(a.model_dir or a.out, f'model_{tag}.pt'), map_location=DEV)); model.eval()
    gen = torch.Generator(device=DEV); gen.manual_seed(1000 + a.seed)
    out = {}
    for split in ['val', 'test']:
        D = load(a.data, a.N, split)
        bd = t(D['board']); B = bd.shape[0]
        one = torch.ones(B, dtype=torch.long, device=DEV)
        _, sl0 = fwd(model, bd, -one); out[f'{split}_p0s'] = torch.sigmoid(sl0).cpu().numpy()
        out[f'{split}_solvable'] = D['solvable']
        for r in a.rules.split(','):
            t0 = time.time(); parts = []
            for i in range(0, B, a.chunk):
                Z, lq, _, ck, lb = chains(model, bd[i:i + a.chunk], 16, r, gen, tau=a.tau)
                parts.append((Z.cpu().numpy().astype(np.int8), lq.cpu().numpy(), ck.cpu().numpy(), lb.cpu().numpy()))
            out[f'{split}_{r}_Z'] = np.concatenate([q[0] for q in parts])
            out[f'{split}_{r}_lq'] = np.concatenate([q[1] for q in parts])
            out[f'{split}_{r}_chk'] = np.concatenate([q[2] for q in parts])
            out[f'{split}_{r}_lb'] = np.concatenate([q[3] for q in parts])
            print(split, r, 'tau', a.tau, round(time.time() - t0, 1), 's', flush=True)
    np.savez_compressed(os.path.join(a.out, f'chains_tau{a.tau:g}_{a.rules.replace(",", "-")}_{tag}.npz'), **out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode'); p.add_argument('--N', type=int, default=6); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data', default='data'); p.add_argument('--out', default='out')
    p.add_argument('--steps', type=int, default=30000); p.add_argument('--bs', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--eval_every', type=int, default=1000)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0); p.add_argument('--chunk', type=int, default=256)
    p.add_argument('--rules', default='rand,voi,conf')
    p.add_argument('--tau', type=float, default=1.0); p.add_argument('--model_dir', default='')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode == 'train':
        train(a)
    elif a.mode == 'dump':
        dump(a)
    else:
        run_eval(a)
