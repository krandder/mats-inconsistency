"""P12 nets and loops: DT-recall on the cell grid, its feedback / arbitrageur variants, CostNet+BB.

Arms (cfg dict keys: fb, extra, arb, pen, td):
  DT          fb=0 extra=0 arb=none
  DT+fb       fb=1 extra=8 arb=none   feedback from sigma(u_t), no BP (auto-context control)
  DT+pen      pen=1.0                 local semantic loss on sigma(u_t) at every t
  DT-TD       td=1                    chained TD settlement (bookkeeping)
  DTxARB      fb=1 arb=sg             one flow-BP step per iteration, stop-gradient at the arbitrageur
  DTxARB-noFB fb=0 arb=sg
  DTxARB-bp   fb=1 arb=bp             backprop through BP (joint, CRF-as-RNN)
Loop iteration t: u_t = head(h_t) (edge log-odds); arb: theta_t = unary(u_t) (u_t itself for 'logodds',
log sigma(u_t) - c for 'logprob'), M_t = BP_step(M_{t-1}, sg(theta_t)), b_t = u_t + sg(delta(M_t))
(the spec's b_t = u_t + sg(M_t); 'bp' drops the sg); else b_t = u_t. theta_t is only BP's unary. The first
version used b_t = theta_t + sg(delta): with 'logprob' theta <= -c the on-path target is out of reach, u runs
to +inf, sigma(-u) underflows and the gradient is exactly 0 (DTxARB s0 collapsed at step ~1300, 2026-09-29). Feedback channels = sg([sigma(b_t)_E, sigma(b_t)_S,
log1p(semantic_loss(sigma(b_t)))]) enter the next step's input. delta enters the prices through a position limit,
kappa * tanh(delta / kappa), kappa = 5: with the old clamp at +-30 the proposer had to post |u| > 30 to overrule a
wrong arbitrageur, and DTxARB s0 diverged (loss 1.5e6 at step 500, NaN at 1600, 2026-09-29). Loss: progressive BCE on sigma(b_t).
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
import flow

IN_CH = 15


def make_input(open_, S, G, code):
    """open_ (B,n,n,4) bool, S/G (B,2), code (B,n,n) int -> x (B,15,n,n)."""
    B, n = open_.shape[:2]
    endS = torch.zeros(B, n, n, device=open_.device); endG = torch.zeros_like(endS)
    ar = torch.arange(B, device=open_.device)
    endS[ar, S[:, 0].long(), S[:, 1].long()] = 1; endG[ar, G[:, 0].long(), G[:, 1].long()] = 1
    bits = ((code.long().unsqueeze(-1) >> torch.arange(9, device=open_.device)) & 1).float()
    x = torch.cat([open_.float(), endS[..., None], endG[..., None], bits], -1)
    return x.permute(0, 3, 1, 2).contiguous()


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, 1, 1); self.c2 = nn.Conv2d(w, w, 3, 1, 1)

    def forward(self, h):
        return F.relu(h + self.c2(F.relu(self.c1(h))))


class DTNet(nn.Module):
    """3x3 embed (15 [+3 feedback] -> w), concat with state, project, 2 residual blocks of two 3x3 convs,
    head -> E and S edge logits. State width w + extra."""

    def __init__(self, w=32, fb=False, extra=0):
        super().__init__()
        self.w, self.fb, self.extra = w, fb, extra
        hw = w + extra
        self.embed = nn.Conv2d(IN_CH + (3 if fb else 0), w, 3, 1, 1)
        self.proj = nn.Conv2d(hw + w, hw, 3, 1, 1)
        self.blocks = nn.Sequential(Block(hw), Block(hw))
        self.head = nn.Sequential(nn.Conv2d(hw, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 2, 3, 1, 1))

    def h0(self, x):
        return x.new_zeros(x.shape[0], self.w + self.extra, x.shape[2], x.shape[3])

    def step(self, h, xin):
        e = F.relu(self.embed(xin))
        return self.blocks(self.proj(torch.cat([h, e], 1)))

    def logits(self, h):
        u = self.head(h)
        return flow.es_to_slots(u[:, 0], u[:, 1])


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class Loop:
    """Holds the per-batch constants and runs iterations for one arm."""

    def __init__(self, model, cfg, open_, S, G, code, bp_tau=0.0, bp_damp=0.5, bp_theta="logodds", bp_c=0.0, bp_kappa=5.0):
        self.m, self.cfg = model, cfg
        self.kappa = bp_kappa    # arbitrageur position limit: its price term is kappa * tanh(delta / kappa)
        self.theta_mode, self.theta_c = bp_theta, bp_c
        self.open = open_.bool(); self.S, self.G = S, G
        B, n = open_.shape[:2]
        self.x = make_input(open_, S, G, code)
        self.endm = flow.endpoints(S, G, B, n, open_.device)
        self.arb = cfg.get("arb", "none")
        self.bp = flow.FlowBP(self.open, S, G, tau=bp_tau, damp=bp_damp) if self.arb != "none" else None

    def init(self):
        B, _, n, _ = self.x.shape
        st = {"h": self.m.h0(self.x)}
        if self.cfg.get("fb"):
            p0 = torch.full((B, n, n, 4), 0.5, device=self.x.device)
            st["fb"] = self._fb(p0)
        if self.bp is not None:
            st["M"] = self.bp.init_state()
        return st

    def _limit(self, d):
        return self.kappa * torch.tanh(d.clamp(-1e4, 1e4) / self.kappa)

    def _fb(self, p):
        sem = flow.semantic_loss(p, self.open, self.endm)
        return torch.stack([p[..., 0], p[..., 1], torch.log1p(sem)], 1).detach()

    def detach(self, st):
        out = {"h": st["h"].detach()}
        if "fb" in st: out["fb"] = st["fb"].detach()
        if "M" in st: out["M"] = (st["M"][0].detach(), st["M"][1].detach())
        return out

    def step(self, st):
        """Returns new state, proposer logits u (B,n,n,4), prices b (B,n,n,4)."""
        xin = torch.cat([self.x, st["fb"]], 1) if "fb" in st else self.x
        h = self.m.step(st["h"], xin)
        u = self.m.logits(h)
        new = {"h": h}
        if self.bp is not None:
            th = flow.unary(u, self.theta_mode, self.theta_c)     # arbitrageur's unary (= u for logodds, c = 0)
            if self.arb == "sg":
                M = self.bp.step((st["M"][0].detach(), st["M"][1].detach()), th.detach())
                d = self._limit(self.bp.delta(M)).detach()
            else:  # 'bp': differentiate through the arbitrageur
                M = self.bp.step(st["M"], th)
                d = self._limit(self.bp.delta(M))
            new["M"] = M
            b = u + d          # prices: proposer logit + the arbitrageur's message (theta_t is BP's unary only)
        else:
            b = u
        if "fb" in st:
            new["fb"] = self._fb(torch.sigmoid(b.detach()))
        return new, u, b


def masked_bce(b, path, open_):
    """BCE on sigma(b) over open E and S slots (each edge once), mean over edges."""
    m = open_[..., :2].float()
    l = F.binary_cross_entropy_with_logits(b[..., :2], path[..., :2].float(), reduction="none")
    return (l * m).sum() / m.sum().clamp_min(1)


def train_loss(loop, path, T, alpha, n_nograd, k_grad):
    """Bansal et al. progressive loss: (1-alpha) * loss at T from scratch + alpha * loss after a random
    no-grad prefix of n_nograd iterations followed by k_grad iterations with grad. DT+pen adds
    lambda * mean_t mean_cell semantic_loss(sigma(u_t)); DT-TD replaces the per-t terms by TD."""
    cfg = loop.cfg
    st = loop.init(); pen = 0.0; tds = []; prev_b = None
    for t in range(T):
        st, u, b = loop.step(st)
        if cfg.get("pen"):
            pen = pen + flow.semantic_loss(torch.sigmoid(u), loop.open, loop.endm).mean()
        if cfg.get("td"):
            tds.append(b)
    loss_max = masked_bce(b, path, loop.open)
    extra = {"bce_T": float(loss_max.detach()), "u_absmax": float(u.detach().abs().max())}
    if cfg.get("pen"):
        pen = pen / T; loss_max = loss_max + cfg["pen"] * pen; extra["pen"] = float(pen)
    if cfg.get("td"):
        td = 0.0
        for t in range(T - 1):
            tgt = torch.sigmoid(tds[t + 1]).detach()
            m = loop.open[..., :2].float()
            l = F.binary_cross_entropy_with_logits(tds[t][..., :2], tgt[..., :2], reduction="none")
            td = td + (l * m).sum() / m.sum().clamp_min(1)
        td = td / (T - 1); loss_max = loss_max + td; extra["td"] = float(td)
    loss = (1 - alpha) * loss_max
    if alpha > 0:
        with torch.no_grad():
            st = loop.init()
            for _ in range(n_nograd):
                st, _, _ = loop.step(st)
        st = loop.detach(st)
        for _ in range(k_grad):
            st, u, b = loop.step(st)
        loss = loss + alpha * masked_bce(b, path, loop.open)
    return loss, extra


# ------------------------------------------------------------------ shortest path on GPU (Bellman-Ford) + BB
def shortest_path(ecost, open_, S, G, max_iter=None):
    """Batched min-cost S-G path on a grid. ecost (B,n,n,4) >= 0 symmetric on open slots.
    Returns on (B,n,n,4) bool symmetric path indicator, and ok (B,) (G reached, walk terminated)."""
    B, n = open_.shape[:2]; dev = ecost.device
    INF = 1e12
    w = torch.where(open_, ecost.double(), torch.full_like(ecost, INF, dtype=torch.float64))
    dist = torch.full((B, n, n), INF, dtype=torch.float64, device=dev)
    ar = torch.arange(B, device=dev)
    dist[ar, S[:, 0].long(), S[:, 1].long()] = 0.0
    max_iter = max_iter or n * n + 5
    for _ in range(max_iter):
        nd = flow.nb(dist.unsqueeze(-1).expand(B, n, n, 4).contiguous())   # dist at neighbour across slot d
        # nb gives x[nb(c,d), opp d]; with x constant over slots this is dist[nb(c,d)]
        nd = torch.where(open_, nd, torch.full_like(nd, INF))
        cand = (nd + w).min(-1).values
        new = torch.minimum(dist, cand)
        if torch.equal(new, dist):
            break
        dist = new
    # walk back from G
    on = torch.zeros(B, n, n, 4, dtype=torch.bool, device=dev)
    ci, cj = G[:, 0].long().clone(), G[:, 1].long().clone()
    si, sj = S[:, 0].long(), S[:, 1].long()
    done = (ci == si) & (cj == sj)
    di = torch.tensor([0, 1, 0, -1], device=dev); dj = torch.tensor([1, 0, -1, 0], device=dev)
    nd = flow.nb(dist.unsqueeze(-1).expand(B, n, n, 4).contiguous())
    nd = torch.where(open_, nd, torch.full_like(nd, INF))
    score = nd + w                                           # (B,n,n,4)
    reach = dist[ar, ci, cj] < INF / 2
    for _ in range(n * n + 5):
        if bool(done.all()):
            break
        d = score[ar, ci, cj].argmin(-1)                     # predecessor direction
        act = ~done
        on[ar[act], ci[act], cj[act], d[act]] = True
        ni, nj = ci + di[d], cj + dj[d]
        on[ar[act], ni[act], nj[act], (d[act] + 2) % 4] = True
        ci = torch.where(act, ni, ci); cj = torch.where(act, nj, cj)
        done = done | ((ci == si) & (cj == sj))
    return on, done & reach


class BBShortestPath(torch.autograd.Function):
    """Blackbox differentiation of a combinatorial solver (Vlastelica/Pogancic et al. 2020):
    forward y = solver(w); backward grad_w = -(y - solver(w + lambda * dL/dy)) / lambda."""

    @staticmethod
    def forward(ctx, ecost, open_, S, G, lam):
        y, _ = shortest_path(ecost.detach(), open_, S, G)
        y = y.float()
        ctx.save_for_backward(ecost, open_, S, G, y); ctx.lam = lam
        return y

    @staticmethod
    def backward(ctx, gy):
        ecost, open_, S, G, y = ctx.saved_tensors; lam = ctx.lam
        gsym = gy + flow.nb(gy)                              # an edge appears in two slots
        wp = (ecost.detach() + lam * gsym).clamp_min(0.0)
        yp, _ = shortest_path(wp, open_, S, G)
        g = -(y - yp.float()) / lam
        return g, None, None, None, None


class CostNet(nn.Module):
    """Local 3x3 CNN (2 conv layers, w=32) -> positive cell costs."""

    def __init__(self, w=32):
        super().__init__()
        self.c1 = nn.Conv2d(IN_CH, w, 3, 1, 1); self.c2 = nn.Conv2d(w, 1, 3, 1, 1)

    def forward(self, x):
        return F.softplus(self.c2(F.relu(self.c1(x))))[:, 0] + 0.01


def cell_to_edge_cost(c, open_):
    return flow.edge_cost_tensor(c, open_)


def costnet_loss(net, open_, S, G, code, path, lam=20.0):
    x = make_input(open_, S, G, code)
    c = net(x)
    ec = cell_to_edge_cost(c, open_)
    y = BBShortestPath.apply(ec, open_, S, G, lam)
    # Hamming loss over each edge once (E,S slots)
    tgt = path.float()
    ham = (y * (1 - tgt) + (1 - y) * tgt)[..., :2]
    return ham.sum() / open_.shape[0], y
