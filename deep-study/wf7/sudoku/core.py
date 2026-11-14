"""core.py -- everything the arms use: the any-order network, its training loss, order policies, routes in the
state graph (partial fillings, one cell per edge), route densities, the paired-route trust learner and the searchers.

No Sudoku rule appears in this file and it never imports oracle.py. The network is told only WHICH cells share a
row, a column or a box (three attention heads are restricted to those groups; a fourth is global). It is never told
what sharing implies. That is the relational prior of every network here, and it is disclosed in NOTE.md.
"""
import math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

BOXSHAPE = {4: (2, 2), 6: (2, 3), 9: (3, 3)}


def relmask(K):
    """[4, N, N] bool: head 0 may attend within a row, head 1 within a column, head 2 within a box, head 3 anywhere."""
    br, bc = BOXSHAPE[K]; N = K * K
    r = torch.arange(N) // K; c = torch.arange(N) % K; b = (r // br) * (K // bc) + c // bc
    return torch.stack([r[:, None] == r[None, :], c[:, None] == c[None, :], b[:, None] == b[None, :],
                        torch.ones(N, N, dtype=torch.bool)])


class Block(nn.Module):
    def __init__(self, d, H=4):
        super().__init__()
        self.H = H
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h, mask):
        B, T, d = h.shape
        q, k, v = self.qkv(self.n1(h)).view(B, T, 3, self.H, d // self.H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        h = h + self.o(a.transpose(1, 2).reshape(B, T, d))
        return h + self.m(self.n2(h))


class RelFormer(nn.Module):
    """Masked any-order transformer over the cells of a board. No positional embedding: position enters only
    through the four attention masks. Token = value embedding (index kmax = empty) + clue-flag embedding
    (+ optional per-cell real features, used by the trust nets to read M's atoms)."""

    def __init__(self, kmax, d=64, L=4, n_out=None, extra_in=0):
        super().__init__()
        self.kmax = kmax
        self.val = nn.Embedding(kmax + 1, d); self.flag = nn.Embedding(2, d)
        self.extra = nn.Linear(extra_in, d) if extra_in else None
        self.blocks = nn.ModuleList([Block(d) for _ in range(L)])
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, n_out or kmax)
        self.cfg = dict(kmax=kmax, d=d, L=L, n_out=n_out, extra_in=extra_in)

    def forward(self, x, flag, mask, extra=None):
        h = self.val(x) + self.flag(flag)
        if self.extra is not None:
            h = h + self.extra(extra)
        for b in self.blocks:
            h = b(h, mask)
        return self.head(self.norm(h))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def save_model(model, path, **meta):
    torch.save(dict(cfg=model.cfg, state=model.state_dict(), meta=meta), path)


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = RelFormer(**ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
    return m, ck.get("meta", {})


class Net:
    """A frozen any-order network with a call counter. One call = one forward pass on one board."""

    def __init__(self, model, K, name="M"):
        self.m = model.eval(); self.K = K; self.N = K * K; self.mask = relmask(K); self.calls = 0
        self.name = name; self.params = n_params(model)
        self.flops_per_call = 2 * self.params * self.N          # the accounting rule of the portfolio, section 1.2

    @torch.no_grad()
    def logp(self, s, flag=None, alpha=None, chunk=4096):
        """s: [B, N] long, -1 = empty. Returns log q(X_i = v | s) for every cell, [B, N, kmax]. `alpha` ([B, kmax]
        bool) restricts the alphabet (used only by the size-generic model)."""
        self.calls += len(s)
        out = []
        for a in range(0, len(s), chunk):
            x = s[a:a + chunk]; x = torch.where(x < 0, torch.full_like(x, self.m.kmax), x)
            f = torch.zeros_like(x) if flag is None else flag[a:a + chunk].long()
            lg = self.m(x, f, self.mask)
            if alpha is not None:
                lg = lg.masked_fill(~alpha[a:a + chunk][:, None, :], -1e9)
            out.append(F.log_softmax(lg, -1))
        return torch.cat(out)


# ----------------------------------------------------------------------------- training of an any-order model
def anyorder_batch(x, clue, gen):
    """Given complete boards x [B, N] and clue masks [B, N] bool (all False for the grids-only variant), draw a
    uniformly random number r in {0..n_free-1} of extra revealed cells, a uniformly random subset of that size, and
    return (input with -1 at masked cells, target mask). This is the any-order (masked) objective."""
    B, N = x.shape
    nfree = (~clue).sum(1)
    r = torch.floor(torch.rand(B, generator=gen) * nfree).long()
    u = torch.rand(B, N, generator=gen); u[clue] = 2.0
    rank = u.argsort(1).argsort(1)
    revealed = (rank < r[:, None]) & ~clue
    masked = ~(clue | revealed)
    return torch.where(masked, torch.full_like(x, -1), x), masked


def train_anyorder(model, K, sampler, steps, bs=256, lr=2e-3, wd=0.01, seed=0, ckpt=None, log=print, every=250,
                   alpha_fn=None, mask=None):
    """sampler(bs) -> (x [bs, N] long, clue [bs, N] bool[, alpha]). Loss: per-board mean cross-entropy on the
    masked cells. Reality only: no consistency term of any kind."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    start = 0
    if ckpt and os.path.exists(ckpt + ".part"):
        ck = torch.load(ckpt + ".part", map_location="cpu")
        model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
        log(f"  resumed from step {start}")
    masks = {}
    t0 = time.time(); run = 0.0
    model.train()
    for step in range(start, steps):
        warm = min(1.0, (step + 1) / 100); cos = 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr * warm * (0.05 + 0.95 * cos)
        batch = sampler(bs)
        x, clue = batch[0], batch[1]
        Kb = int(round(math.sqrt(x.shape[1])))
        if Kb not in masks:
            masks[Kb] = relmask(Kb)
        s, masked = anyorder_batch(x, clue, gen)
        xin = torch.where(s < 0, torch.full_like(s, model.kmax), s)
        lg = model(xin, clue.long(), masks[Kb])
        if len(batch) > 2:
            lg = lg.masked_fill(~batch[2][:, None, :], -1e9)
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x.reshape(-1), reduction="none").view(x.shape)
        loss = ((ce * masked).sum(1) / masked.sum(1)).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        run = loss.item() if step == start else 0.98 * run + 0.02 * loss.item()
        if (step + 1) % every == 0:
            log(f"  step {step+1}/{steps}: masked-cell NLL (running) {run:.4f}  ({time.time()-t0:.0f}s)")
            if ckpt:
                torch.save(dict(state=model.state_dict(), opt=opt.state_dict(), step=step + 1), ckpt + ".part")
    model.eval()
    return model


# ----------------------------------------------------------------------------- order policies (adapted)
# A policy maps (state s [B, N], M's log-probs lp [B, N, K], empty mask [B, N]) to a score per cell; the cell with
# the LOWEST score among the empty ones is filled next. It reads revealed cells only.
def pol_maxprob(s, lp, emp):            # M's own confidence: highest top probability first (Kim et al. "top-prob")
    return -lp.max(-1).values


def pol_margin(s, lp, emp):             # M's own confidence: largest top-1 minus top-2 probability ("top-prob-margin")
    t = lp.exp().topk(2, -1).values
    return -(t[..., 0] - t[..., 1])


def pol_entropy(s, lp, emp):            # M's own confidence: lowest entropy first
    p = lp.exp()
    return -(p * lp.clamp(min=-50)).sum(-1)


def pol_anti(s, lp, emp):               # deliberately bad: the cell M is least sure about first
    return lp.max(-1).values


def pol_raster(s, lp, emp):
    return torch.arange(s.shape[1], dtype=torch.float32).expand(s.shape[0], -1)


class RandomOrder:
    def __init__(self, seed=0):
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, s, lp, emp):
        return torch.rand(s.shape, generator=self.gen)


class TrustPolicy:
    """Order policy given by a small network trained on real boards only (see `fit_trust`). It reads the revealed
    cells and M's atoms at the current state, and scores every empty cell; lowest score is revealed next."""

    def __init__(self, model, K, name="pi_trust"):
        self.m = model.eval(); self.K = K; self.mask = relmask(K); self.calls = 0; self.name = name
        self.params = n_params(model); self.flops_per_call = 2 * self.params * K * K

    @torch.no_grad()
    def __call__(self, s, lp, emp):
        self.calls += len(s)
        return trust_scores(self.m, s, lp, emp, self.mask)


def trust_features(lp, emp):
    """Per-cell features for the trust net: M's probabilities at the cell (zero at filled cells) and its entropy."""
    p = lp.exp() * emp[..., None]
    ent = -(p * lp.clamp(min=-50)).sum(-1, keepdim=True)
    return torch.cat([p, ent], -1)


def trust_scores(model, s, lp, emp, mask):
    x = torch.where(s < 0, torch.full_like(s, model.kmax), s)
    return model(x, torch.zeros_like(x), mask, extra=trust_features(lp, emp)).squeeze(-1)


# ----------------------------------------------------------------------------- routes in the state graph
@torch.no_grad()
def rollout(M, policy, start, X=None, flag=None, greedy=False, floor=0.0, gen=None, record=False, abort_below=None,
            alpha=None):
    """Walk the state graph from `start` ([B, N], -1 = empty) to complete boards, one cell per edge.
    The cell is chosen by `policy`; the value is
      - X's value at that cell if X is given (a route to a known board: returns its density under M's atoms),
      - else M's most probable value if greedy, else a draw from M's atom mixed with a uniform floor.
    Returns dict: s (final boards), logp (sum of log M-atoms of the realised values = log route density),
    logprop (log-density of the draws, floor included), used (number of steps = calls of M per board), and
    if record: cells [B, T], lps [B, T], ent [B, T] (M's entropy at the chosen cell), pmax [B, T];
    if record == "full" also states [B, T, N] and atoms [B, T, N, K] (M's log-probs at each visited state).
    `abort_below`: stop a board once logp < this value (used by the check: all terms are <= 0)."""
    s = start.clone(); B, N = s.shape
    logp = torch.zeros(B); logprop = torch.zeros(B); alive = torch.ones(B, dtype=torch.bool)
    used = torch.zeros(B, dtype=torch.long)
    T = int((s < 0).sum(1).max()) if B else 0
    if record:
        cells = torch.full((B, T), -1, dtype=torch.long); lps = torch.zeros(B, T)
        ent = torch.zeros(B, T); pmax = torch.zeros(B, T)
        states = torch.full((B, T, N), -2, dtype=torch.long) if record == "full" else None
        atoms = None
    for t in range(T):
        emp = s < 0
        act = emp.any(1) & alive
        if not act.any():
            break
        ia = torch.where(act)[0]
        sa = s[ia]; ea = emp[ia]; ar = torch.arange(len(ia))
        lp = M.logp(sa, None if flag is None else flag[ia], None if alpha is None else alpha[ia])
        used[ia] += 1
        score = policy(sa, lp, ea).masked_fill(~ea, float("inf"))
        cell = score.argmin(1)
        lpc = lp[ar, cell]                                              # [b, K]
        if X is not None:
            v = X[ia, cell]
        elif greedy:
            v = lpc.argmax(1)
        else:
            p = (1 - floor) * lpc.exp() + floor / lpc.shape[1]
            v = torch.multinomial(p, 1, generator=gen).squeeze(1)
        lv = lpc[ar, v]
        logp[ia] += lv
        logprop[ia] += torch.log((1 - floor) * lv.exp() + floor / lpc.shape[1]) if floor > 0 else lv
        if record:
            cells[ia, t] = cell; lps[ia, t] = lv
            ent[ia, t] = -(lpc.exp() * lpc.clamp(min=-50)).sum(-1); pmax[ia, t] = lpc.max(-1).values.exp()
            if record == "full":
                states[ia, t] = sa
                if atoms is None:
                    atoms = torch.zeros(B, T, N, lp.shape[-1])
                atoms[ia, t] = lp
        s[ia, cell] = v
        if abort_below is not None:
            alive = alive & (logp >= abort_below)
    out = dict(s=s, logp=logp, logprop=logprop, alive=alive, used=used)
    if record:
        out.update(cells=cells, lps=lps, ent=ent, pmax=pmax, states=states, atoms=atoms)
    return out


def route_logT(M, policy, X, start=None, flag=None, abort_below=None, alpha=None):
    """log T(x): density of board x under 'choose the cell by `policy`, draw its value from M's atom', starting from
    the empty board (or from `start`). One route per board, so T is normalised and this costs n calls of M."""
    if start is None:
        start = torch.full_like(X, -1)
    return rollout(M, policy, start, X=X, flag=flag, abort_below=abort_below, alpha=alpha)


# ----------------------------------------------------------------------------- the trust learner (reality only)
def collect_paired(M, policy, X, start, flag, n_dev, gen, top=4):
    """Paired routes to the same real board. For each real board x: one route under `policy` (the base route), and
    n_dev deviations: at a uniformly random step of the base route reveal a different empty cell, then follow
    `policy` to the end. delta = (M's total surprise on the deviated continuation) - (on the base continuation).
    Both continuations end at the same real board, so the entropy of reality cancels exactly and delta is a sample of
    the difference in excess error. Returns training tuples (state, atoms, action, delta), the base action included
    with delta = 0, and the base route's total surprise per board."""
    B, N = X.shape
    base = rollout(M, policy, start, X=X, flag=flag, record="full")
    valid = base["cells"] >= 0                                           # steps that exist for each board
    togo = torch.flip(torch.cumsum(torch.flip(-base["lps"] * valid, [1]), 1), [1])     # surprise from step k on
    nsteps = valid.sum(1)
    S, A, At, D = [], [], [], []
    bidx = torch.arange(B); ok = nsteps >= 2
    for d in range(n_dev):
        k = torch.floor(torch.rand(B, generator=gen) * (nsteps - 1).clamp(min=1)).long()     # a step with >= 2 empty cells
        sk = base["states"][bidx, k]; lpk = base["atoms"][bidx, k]; emp = sk < 0
        a0 = base["cells"][bidx, k]
        score = policy(sk, lpk, emp).masked_fill(~emp, float("inf"))
        score[bidx, a0] = float("inf")
        ntop = min(top, N)
        cand_top = score.topk(ntop, 1, largest=False).indices                             # best alternatives
        pick_top = cand_top[bidx, torch.randint(0, ntop, (B,), generator=gen)]
        w = (emp & (torch.arange(N)[None, :] != a0[:, None])).float()
        pick_uni = torch.multinomial(w + 1e-12, 1, generator=gen).squeeze(1)
        use_top = torch.rand(B, generator=gen) < 0.5
        a = torch.where(use_top & torch.isfinite(score[bidx, pick_top]), pick_top, pick_uni)
        ca = -lpk[bidx, a, X[bidx, a]]                                    # M's surprise at the deviating cell
        s2 = sk.clone(); s2[bidx, a] = X[bidx, a]
        rest = rollout(M, policy, s2[ok], X=X[ok], flag=None if flag is None else flag[ok])
        delta = ca[ok] - rest["logp"] - togo[bidx, k][ok]
        S.append(sk[ok]); At.append(lpk[ok]); A.append(a[ok]); D.append(delta)
        if d == 0:                                                        # the base action, delta = 0 by definition
            S.append(sk[ok]); At.append(lpk[ok]); A.append(a0[ok]); D.append(torch.zeros(int(ok.sum())))
    return torch.cat(S), torch.cat(At), torch.cat(A), torch.cat(D), -base["logp"]


def fit_trust(M, K, sample_real, policy0, iters=3, boards_per_iter=1536, n_dev=6, bs_roll=256, d=64, L=2,
              epochs=6, lr=2e-3, seed=0, log=print):
    """Paired-route policy iteration for the trust MDP (state = partial filling of a REAL board; action = which cell
    reality reveals next; cost = M's surprise). `sample_real(n)` returns (X, start, flag) drawn from reality.
    Each iteration: collect paired routes under the current policy, regress a per-cell advantage net on delta, and
    take the new policy 'reveal the cell with the lowest predicted advantage'. No searcher and no solver is involved;
    the only supervision is M's surprise on real boards. Returns the list of policies (one per iteration) and the
    training log; the caller selects among them on real validation boards."""
    import copy
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    policy = policy0; mask = relmask(K); hist = []; policies = []
    net = RelFormer(M.m.kmax, d=d, L=L, n_out=1, extra_in=M.m.kmax + 1)
    for it in range(iters):
        t0 = time.time()
        S, At, A, D, base = [], [], [], [], []
        for a in range(0, boards_per_iter, bs_roll):
            X, start, flag = sample_real(min(bs_roll, boards_per_iter - a))
            s_, at_, a_, d_, b_ = collect_paired(M, policy, X, start, flag, n_dev, gen)
            S.append(s_); At.append(at_); A.append(a_); D.append(d_); base.append(b_)
        S, At, A, D, base = [torch.cat(z) for z in (S, At, A, D, base)]
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
        n = len(S); nb = max(1, n // 512); tot_steps = epochs * nb; stp = 0
        for ep in range(epochs):
            perm = torch.randperm(n, generator=gen); el = 0.0
            for j in range(nb):
                for g in opt.param_groups:
                    g["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * stp / tot_steps)))
                stp += 1
                idx = perm[j * 512:(j + 1) * 512]
                emp = S[idx] < 0
                pred = trust_scores(net, S[idx], At[idx], emp, mask)[torch.arange(len(idx)), A[idx]]
                loss = F.smooth_l1_loss(pred, D[idx])
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                el += loss.item()
        net.eval()
        nz = D != 0
        hist.append(dict(it=it, base_surprise=base.mean().item(), frac_dev_better=(D[nz] < -1e-4).float().mean().item(),
                         mean_delta=D[nz].mean().item(), fit_loss=el / nb, n=n))
        log(f"  trust iter {it}: base route surprise {base.mean():.4f} (n={len(base)}); deviations better than base "
            f"{hist[-1]['frac_dev_better']:.3f}; mean delta {hist[-1]['mean_delta']:+.4f}; fit loss {el/nb:.4f}; "
            f"{n} tuples ({time.time()-t0:.0f}s)")
        policy = TrustPolicy(copy.deepcopy(net), K, name=f"pi_trust_it{it+1}")
        policies.append(policy)
    return policies, hist


# ----------------------------------------------------------------------------- search, corrected against T
def search_first_accept(prop, prop_policy, puz, flag_prop, checker, attempts, gen, floor=0.0, greedy_first=True,
                        rounds=(1, 7, 24, 96, 128), log=None):
    """Search for a batch of puzzles. A candidate is a complete board that agrees with the clues (the proposal never
    changes a clue). Candidate 0 is the proposal's greedy route (if greedy_first), the others are draws.
    `checker(X)` returns (accept [m] bool, logT [m], T-route steps used per candidate [m]).
    Sequential semantics: candidates are tried in order; the first accepted one is returned and never revoked.
    Returns per puzzle: answer (-1 rows if none), index of the accepted candidate (-1 if none within `attempts`),
    proposal steps and check steps used up to and including the accepted candidate, the first candidate (the unchecked
    one-pass answer), and every candidate with its logT, proposal log-density and accept flag."""
    P, N = puz.shape
    nfree = (puz < 0).sum(1)
    answer = torch.full((P, N), -1, dtype=torch.long); acc_idx = torch.full((P,), -1, dtype=torch.long)
    prop_steps = torch.zeros(P, dtype=torch.long); chk_steps = torch.zeros(P, dtype=torch.long)
    first = torch.full((P, N), -1, dtype=torch.long); done = torch.zeros(P, dtype=torch.bool); tried = 0
    cands = [[] for _ in range(P)]
    for r in rounds:
        r = min(r, attempts - tried)
        if r <= 0 or done.all():
            break
        todo = torch.where(~done)[0]; nt = len(todo)
        fl = None if flag_prop is None else flag_prop[todo]
        parts_X, parts_q = [], []
        nd = r
        if greedy_first and tried == 0:
            out0 = rollout(prop, prop_policy, puz[todo], flag=fl, greedy=True)
            first[todo] = out0["s"]; parts_X.append(out0["s"][:, None, :]); parts_q.append(out0["logprop"][:, None]); nd = r - 1
        if nd > 0:
            rep = torch.arange(nt).repeat_interleave(nd)
            outs = rollout(prop, prop_policy, puz[todo][rep], flag=None if fl is None else fl[rep], floor=floor, gen=gen)
            parts_X.append(outs["s"].view(nt, nd, N)); parts_q.append(outs["logprop"].view(nt, nd))
            if tried == 0 and not greedy_first:
                first[todo] = outs["s"].view(nt, nd, N)[:, 0]
        X = torch.cat(parts_X, 1); lq = torch.cat(parts_q, 1)                       # [nt, r, N], [nt, r]
        acc, logT, cc = checker(X.reshape(nt * r, N))
        acc = acc.view(nt, r); logT = logT.view(nt, r); cc = cc.view(nt, r)
        anyhit = acc.any(1); j = acc.float().argmax(1)                              # first accepted candidate
        ncount = torch.where(anyhit, j + 1, torch.full_like(j, r))                  # candidates actually consumed
        csum = torch.cumsum(cc, 1)
        prop_steps[todo] += ncount * nfree[todo]
        chk_steps[todo] += csum[torch.arange(nt), ncount - 1]
        hit = todo[anyhit]
        answer[hit] = X[anyhit, j[anyhit]]; acc_idx[hit] = tried + j[anyhit]; done[hit] = True
        for jj, p in enumerate(todo.tolist()):
            cands[p].append((X[jj], logT[jj], lq[jj], acc[jj]))
        tried += r
        if log:
            log(f"      after {tried} candidates: {int(done.sum())}/{P} puzzles have an accepted answer")
    return dict(answer=answer, acc_idx=acc_idx, prop_steps=prop_steps, chk_steps=chk_steps, first=first, cands=cands,
                nfree=nfree)


def make_checker(M, policy, tau):
    """Accept a complete board iff log T(x) >= tau, where T = 'policy picks the cell from the empty board, M's atom
    gives the value'. The route is abandoned as soon as the running sum falls below tau (all terms are <= 0), so a
    rejected board usually costs only the steps up to its first rule violation."""
    def checker(X):
        out = rollout(M, policy, torch.full_like(X, -1), X=X, abort_below=tau)
        return out["logp"] >= tau, out["logp"], out["used"]
    return checker


# ----------------------------------------------------------------------------- added 03:20Z (addendum A8): pair check
@torch.no_grad()
def pair_min(M, X, abort_below=None):
    """Smallest log q(X_j | only cell i revealed, with value X_i) over all ordered pairs of distinct cells.
    Reads M only at one-cell contexts (every one of them occurs in real boards). N calls of M per board; with
    `abort_below`, a board stops being evaluated once its running minimum is below that value.
    Returns (score [B], calls used [B])."""
    B, N = X.shape
    score = torch.full((B,), float("inf")); used = torch.zeros(B, dtype=torch.long)
    alive = torch.ones(B, dtype=torch.bool)
    for i in range(N):
        ia = torch.where(alive)[0]
        if len(ia) == 0:
            break
        s = torch.full((len(ia), N), -1, dtype=torch.long); s[:, i] = X[ia, i]
        lp = M.logp(s); used[ia] += 1
        v = lp.gather(2, X[ia][:, :, None]).squeeze(2); v[:, i] = float("inf")
        score[ia] = torch.minimum(score[ia], v.min(1).values)
        if abort_below is not None:
            alive = alive & (score >= abort_below)
    return score, used


def make_checker2(M, policy, tau, theta):
    """Accept a complete board iff (i) every one-cell-context atom of M is at least e^theta on it (pair check) and
    (ii) its route density log T(x) >= tau. The pair check runs first; boards it rejects never reach the route."""
    def checker(X):
        sc, used = pair_min(M, X, abort_below=theta)
        ok = sc >= theta
        logT = torch.full((len(X),), float("-inf")); steps = used.clone()
        if ok.any():
            out = rollout(M, policy, torch.full_like(X[ok], -1), X=X[ok], abort_below=tau)
            logT[ok] = out["logp"]; steps[ok] += out["used"]
        return ok & (logT >= tau), logT, steps
    return checker
