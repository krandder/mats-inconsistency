"""P6: trader entry under a growing-depth curriculum (prefix-sum parity).

Arms (one module application per recurrent step, width 32):
  R0   DT-Recall (Schwarzschild/Bansal), progressive loss alpha=1, full BPTT
  DTL  DT-L (Bear et al. 2024): spectral-norm convs, ELU, learned interpolated skips, IPT alpha=0.5
  R1   one trader module, chunks of k=4 with detached carry (TBPTT) + per-chunk LMSR pay (deep supervision)
  F    8 traders: auction + PASS, second-price ledger, tax, regulator top-up at the floor, no exit
  EC   E-clone: a mature trader below W0/10 exits; slot gets a mutated clone of a wealth-sampled parent
  PBT  PBT-uniform: EC ledger + bailouts; replays EC's exit schedule with uniform victim and parent
  EMA  EMA-clone: no ledger/price/sizing (s=1); replays EC's schedule, victim = lowest EMA of raw gain

Economy streams per example: x (one-hot bits, re-read every step), b (one binary LMSR logit per
position), h (16-channel scratch, read through a channel layer norm). Trader weights are stacked
(N, ...) and gathered per row, so one call applies the auction winner of every row.
Reuses market-resnet: mresnet.prefix_batch (data), econ.soft_score, econ.worst_loss_per_bit.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, math, time, hashlib, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "..", "market-resnet")):
    if p not in sys.path:
        sys.path.append(p)
from mresnet import prefix_batch            # noqa: E402  (market-resnet lane owns these)
from econ import soft_score, worst_loss_per_bit  # noqa: E402


def set_numerics():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


set_numerics()

W, H, KCH = 32, 16, 4            # module width, scratch channels, chunk length k
CIN = 3 + 1 + H                  # one-hot x (0/1/PAD), tanh(b/4), LN(h)
FB = 2 * (2 + H)                 # bid features: pooled mean+max of [tanh(b/4), |p-.5|, LN(h)]
def make_shapes(w):
    shapes = {
        "w_in": (w, CIN, 3), "b_in": (w,),
        "w_r1": (w, w, 3), "b_r1": (w,),
        "w_r2": (w, w, 3), "b_r2": (w,),
        "w_out": (1 + H, w, 1), "b_out": (1 + H,),
        "q_w1": (FB, 32), "q_b1": (32,), "q_w2": (32,), "q_b2": (1,),
    }
    fan = {"w_in": CIN * 3, "b_in": CIN * 3, "w_r1": w * 3, "b_r1": w * 3, "w_r2": w * 3, "b_r2": w * 3,
           "w_out": w, "b_out": w, "q_w1": FB, "q_b1": FB, "q_w2": 32, "q_b2": 32}
    return shapes, fan


SHAPES, FAN = make_shapes(W)
STEP_KEYS = [k for k in SHAPES if not k.startswith("q_")]
ECON_ARMS = ("R1", "F", "EC", "PBT", "EMA")
LEDGER_ARMS = ("F", "EC", "PBT")
CAP = 256


# ----------------------------------------------------------------------------- helpers
def xonehot(x):
    return F.one_hot(x, 3).float().transpose(1, 2)


def chan_ln(h):
    return (h - h.mean(1, keepdim=True)) / torch.sqrt(h.var(1, unbiased=False, keepdim=True) + 1e-5)


def score_bits(b, y, mask):
    """Mean log score per valid bit, per example (nats)."""
    return (soft_score(b, y) * mask).sum(1) / mask.sum(1)


CONV_MODE = os.environ.get("P6_CONV", "bmm")
GRAPH = os.environ.get("P6_GRAPH", "1") == "1"      # CUDA-graph the training chunk (econ arms)


def rconv(x, w, bias, shared):
    """Conv1d with per-row weights (unfold + batched matmul, or grouped conv) or one shared weight."""
    pad = w.shape[-1] // 2
    if shared:
        return F.conv1d(x, w, bias, padding=pad)
    B, Ci, L = x.shape
    Co, K = w.shape[1], w.shape[-1]
    if CONV_MODE == "group":
        out = F.conv1d(x.reshape(1, B * Ci, L), w.reshape(B * Co, Ci, K), bias.reshape(B * Co), padding=pad, groups=B)
        return out.reshape(B, Co, L)
    if K == 1:
        return torch.baddbmm(bias.unsqueeze(2), w.reshape(B, Co, Ci), x)
    xp = F.pad(x, (pad, pad))
    unf = torch.stack([xp[:, :, k:k + L] for k in range(K)], 2).reshape(B, Ci * K, L)
    return torch.baddbmm(bias.unsqueeze(2), w.reshape(B, Co, Ci * K), unf)


def module_step(Wr, xoh, b, h, mask, shared):
    z = torch.cat([xoh, torch.tanh(b / 4).unsqueeze(1), chan_ln(h)], 1)
    z = F.relu(rconv(z, Wr["w_in"], Wr["b_in"], shared))
    r = rconv(F.relu(rconv(z, Wr["w_r1"], Wr["b_r1"], shared)), Wr["w_r2"], Wr["b_r2"], shared)
    z = F.relu(z + r)
    o = rconv(z, Wr["w_out"], Wr["b_out"], shared) * mask.unsqueeze(1)
    return b + o[:, 0], h + o[:, 1:]


def bid_feats(b, h, mask):
    f = torch.cat([torch.tanh(b / 4).unsqueeze(1), (torch.sigmoid(b) - 0.5).abs().unsqueeze(1), chan_ln(h)], 1)
    m = mask.unsqueeze(1)
    mean = (f * m).sum(2) / m.sum(2).clamp(min=1)
    mx = (f + (m - 1) * 1e4).amax(2)
    return torch.cat([mean, mx], 1)


def bids_all(P, feats):
    z = F.relu(torch.einsum("bf,nfh->bnh", feats, P["q_w1"]) + P["q_b1"].unsqueeze(0))
    return torch.einsum("bnh,nh->bn", z, P["q_w2"]) + P["q_b2"].view(1, -1)


def gather_rows(P, oh):
    """Per-row weights as a one-hot matmul (deterministic backward on GPU)."""
    B, N = oh.shape
    return {k: (oh @ P[k].reshape(N, -1)).reshape(B, *P[k].shape[1:]) for k in STEP_KEYS}


def slot_weights(P, n):
    return {k: P[k][n] for k in STEP_KEYS}


def phash(P):
    h = hashlib.sha256()
    for k in sorted(P):
        h.update(P[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def gini(ws):
    ws = sorted(max(w, 0.0) for w in ws)
    n, s = len(ws), sum(ws)
    if s <= 0:
        return 0.0
    return sum((2 * (i + 1) - n - 1) * w for i, w in enumerate(ws)) / (n * s)


def spearman(a, b):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(o):
            r[i] = float(pos)
        return r
    ra, rb = rk(a), rk(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra)); vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else float("nan")


# ----------------------------------------------------------------------------- plain baselines
class DTR(nn.Module):
    """dt_net_recall_1d, width w, one residual block per step (Bansal et al. 2022)."""
    def __init__(self, w=W):
        super().__init__()
        self.proj = nn.Conv1d(3, w, 3, padding=1, bias=False)
        self.recall = nn.Conv1d(w + 3, w, 3, padding=1, bias=False)
        self.c1 = nn.Conv1d(w, w, 3, padding=1, bias=False)
        self.c2 = nn.Conv1d(w, w, 3, padding=1, bias=False)
        self.head = nn.Sequential(nn.Conv1d(w, w, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv1d(w, w // 2, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv1d(w // 2, 1, 3, padding=1, bias=False))

    def init_state(self, xoh):
        return F.relu(self.proj(xoh))

    def step(self, h, xoh):
        z = F.relu(self.recall(torch.cat([h, xoh], 1)))
        return F.relu(z + self.c2(F.relu(self.c1(z))))

    def out(self, h):
        return self.head(h)[:, 0]


class DTL(nn.Module):
    """DT-L (Bear, Prugel-Bennett & Hare 2024): recall = SN conv on h + plain biased conv on x,
    ELU, two learned per-channel interpolated skips (1-g) id + g f, spectral norm on every
    recurrent conv acting on h (so each step is <= 1-Lipschitz in h)."""
    def __init__(self, w=W):
        super().__init__()
        SN = torch.nn.utils.parametrizations.spectral_norm
        self.proj = nn.Conv1d(3, w, 3, padding=1)
        self.rh = SN(nn.Conv1d(w, w, 3, padding=1, bias=False))
        self.rx = nn.Conv1d(3, w, 3, padding=1)
        self.c1 = SN(nn.Conv1d(w, w, 3, padding=1))
        self.c2 = SN(nn.Conv1d(w, w, 3, padding=1))
        self.g1 = nn.Parameter(torch.zeros(w))
        self.g2 = nn.Parameter(torch.zeros(w))
        self.head = nn.Sequential(nn.Conv1d(w, w, 3, padding=1), nn.ELU(),
                                  nn.Conv1d(w, w // 2, 3, padding=1), nn.ELU(),
                                  nn.Conv1d(w // 2, 1, 3, padding=1))

    def init_state(self, xoh):
        return F.elu(self.proj(xoh))

    def step(self, h, xoh):
        a1 = torch.sigmoid(self.g1).view(1, -1, 1)
        a2 = torch.sigmoid(self.g2).view(1, -1, 1)
        h1 = (1 - a1) * h + a1 * F.elu(self.rh(h) + self.rx(xoh))
        return (1 - a2) * h1 + a2 * F.elu(self.c2(F.elu(self.c1(h1))))

    def out(self, h):
        return self.head(h)[:, 0]


def bce_bits(logit, y, mask):
    return (-(score_bits(logit, y, mask))).mean()


class Plain:
    def __init__(self, arm, seed, dev):
        torch.manual_seed(10_000 + seed)
        self.arm, self.dev = arm, dev
        self.net = (DTR() if arm == "R0" else DTL()).to(dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def lr_at(self, it):
        return 1e-3 * min(1.0, (it + 1) / 200)

    def train_iter(self, x, y, mask, D, it, gp):
        for g in self.opt.param_groups:
            g["lr"] = self.lr_at(it)
        xoh = xonehot(x)
        n = int(torch.randint(0, D, (1,), generator=gp))
        k = int(torch.randint(1, D - n + 1, (1,), generator=gp))
        self.net.train()
        with torch.nn.utils.parametrize.cached():
            h0 = self.net.init_state(xoh)
            with torch.no_grad():
                h = h0
                for _ in range(n):
                    h = self.net.step(h, xoh)
            h = h.detach() if n else h0
            for _ in range(k):
                h = self.net.step(h, xoh)
            lp = bce_bits(self.net.out(h), y, mask)
            loss = lp
            if self.arm == "DTL":         # IPT alpha = 0.5: half full-depth loss, half progressive
                h = h0
                for _ in range(D):
                    h = self.net.step(h, xoh)
                loss = 0.5 * bce_bits(self.net.out(h), y, mask) + 0.5 * lp
            self.opt.zero_grad()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def rollout(self, x, mask, y, depths, h0=None, want_h=False):
        """Run to max(depths); return {D: (logit, h?)} at the requested depths."""
        self.net.eval()
        xoh = xonehot(x)
        h = self.net.init_state(xoh) if h0 is None else h0
        want = sorted(set(depths))
        out, t, fixed = {}, 0, None
        with torch.nn.utils.parametrize.cached():
            for D in want:
                while t < D and fixed is None:
                    hn = self.net.step(h, xoh)
                    t += 1
                    if t % 64 == 0 and torch.equal(hn, h):    # exact fixed point: nothing changes again
                        fixed = t
                    h = hn
                out[D] = (self.net.out(h), h if want_h else None)
            r = (self.net.step(h, xoh) - h).flatten(1).norm(dim=1) / (h.flatten(1).norm(dim=1) + 1e-12)
        self.last_info = dict(fixed_at=fixed, resid=r)
        return out

    def state(self):
        return {"net": self.net.state_dict(), "opt": self.opt.state_dict()}

    def load(self, s):
        self.net.load_state_dict(s["net"]); self.opt.load_state_dict(s["opt"])


# ----------------------------------------------------------------------------- economy
class Econ:
    def __init__(self, arm, seed, dev, a):
        self.arm, self.dev, self.a = arm, dev, a
        self.N = 1 if arm == "R1" else a.N
        self.ledger = arm in LEDGER_ARMS
        self.sized = arm in LEDGER_ARMS
        gi = torch.Generator().manual_seed(10_000 + seed)
        self.ge = torch.Generator().manual_seed(40_000 + seed)       # entry: parents, victims, mutation
        self.P = {}
        shapes, fan = make_shapes(getattr(a, "width", W))
        for k, s in shapes.items():
            bound = 1.0 / math.sqrt(fan[k])
            self.P[k] = ((torch.rand((self.N,) + s, generator=gi) * 2 - 1) * bound).to(dev).requires_grad_(True)
        self.m = {k: torch.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: torch.zeros_like(v) for k, v in self.P.items()}
        N, W0 = self.N, a.W0
        self.t = [0] * N                # Adam step per slot
        self.warm = [200] * N
        self.age = [0] * N
        self.founder = [True] * N
        self.born = [0] * N
        self.lineage = list(range(N)); self.next_id = N
        self.wealth = [W0] * N
        self.treasury = 0.0; self.sink = 0.0; self.mm = 0.0; self.mint = -N * W0
        self.injected = 0.0
        self.ema = [0.0] * N
        self.comp = [dict(w0=W0, gross=0.0, cost=0.0, price=0.0, tax=0.0) for _ in range(N)]
        self.events = []                # exits / entries / bailouts / replacements
        self.lives = []                 # (id, born, died or None, founder)
        self.max_ledger_err = 0.0
        self.win = self.new_window()
        self.chunk_gain = {}            # chunk index -> summed gross gain
        self.row_gain = {"finished": 0.0, "active": 0.0}
        self._graphs = {}               # (B, L, chunks) -> chunk function (CUDA-graphed on GPU)

    def new_window(self):
        return dict(trades=0, passes=0, rowchunks=0, s_small=0, gross=0.0, exits=0, entries=0, bailouts=0)

    # ---- ledger
    def ledger_total(self):
        terms = self.wealth + [self.treasury, self.sink, self.mm, self.mint]
        return sum(terms), sum(abs(t) for t in terms)

    def check_ledger(self):
        tot, scale = self.ledger_total()
        err = abs(tot) / max(1.0, scale)
        self.max_ledger_err = max(self.max_ledger_err, err)
        assert err < 1e-6, f"ledger broken: total {tot} scale {scale}"

    def eligible(self):
        return [self.founder[n] or self.age[n] >= self.a.m for n in range(self.N)]

    # ---- one training iteration
    def chunk_fn(self, B, nch):
        """One chunk (bid, auction, k=4 steps, sizing, pay, losses) as a pure tensor function, so it can be
        CUDA-graphed. Args: *P values, b, h, xoh, mask, y, ex, up, wealth_t, c_row.
        Returns loss (grad to P), stats (4,N) f64 [gross, cost, price, trades], scal (5,) f64
        [passes, sized-down trades, chunk gross, gross on finished rows, gross on active rows], b_end, h_end."""
        N, a, keys = self.N, self.a, list(self.P.keys())
        shared, sized, ledger = N == 1, self.sized, self.ledger
        norm = float(B * nch)
        ar = torch.arange(N, device=self.dev)

        def fn(*args):
            Pl = dict(zip(keys, args[:len(keys)]))
            b, h, xoh, mask, y, ex, up, wealth_t, c_row = args[len(keys):]
            bids = bids_all(Pl, bid_feats(b, h, mask))           # (B,N), grad to bid heads only
            if N == 1:
                sel = torch.zeros_like(up)
                trade = torch.ones_like(ex)
                price = torch.zeros_like(c_row)
            else:
                bd = bids.detach()
                top = bd.topk(2, dim=1)
                sel = torch.where(ex, up, top.indices[:, 0])
                trade = ex | (top.values[:, 0] > 0)
                bc = bd.clamp(min=0)      # exploration price: mean of the other traders' bids, each clipped at 0
                others = (bc.sum(1) - bc.gather(1, sel[:, None])[:, 0]) / (N - 1)
                price = torch.where(ex, others, top.values[:, 1].clamp(min=0))
                if not ledger:
                    price = torch.zeros_like(price)
            tf = trade.float()
            oh = (sel[:, None] == ar[None, :]).float() * tf[:, None]
            Wr = slot_weights(Pl, 0) if shared else gather_rows(Pl, oh)
            bk, hk = b, h
            for _ in range(KCH):
                bk, hk = module_step(Wr, xoh, bk, hk, mask, shared)
            db = bk - b
            if sized:
                wl = worst_loss_per_bit(b, db.detach(), mask)
                s = torch.clamp(a.frac * wealth_t[sel].clamp(min=0) / (wl + 1e-8), max=1.0)
            else:
                s = torch.ones_like(c_row)
            s = (s * tf).detach()
            b_end = b + s[:, None] * db
            h_end = h + s[:, None, None] * (hk - h)
            gain = score_bits(b_end, y, mask) - score_bits(b, y, mask)          # nats/bit per row
            tgt = gain.detach() - c_row
            ohs = (sel[:, None] == ar[None, :]).float()
            bsel = (bids * ohs).sum(1)           # = bids[row, sel]; avoids gather's scatter backward
            lb = F.smooth_l1_loss(bsel, tgt, reduction="none")
            loss = ((-gain + lb) * tf).sum() / norm
            tfd = tf.double()
            gd = gain.detach().double() * tfd
            ohd = oh.double()
            stats = torch.stack([ohd.t() @ gd, ohd.t() @ (c_row.double() * tfd), ohd.t() @ (price.double() * tfd),
                                 ohd.sum(0)])
            finished = ((((b > 0).float() == y) | (mask == 0)).all(1)).double()
            scal = torch.stack([(1 - tfd).sum(), ((s < 1).double() * tfd).sum(), gd.sum(), (gd * finished).sum(),
                                (gd * (1 - finished)).sum()])
            return loss, stats, scal, b_end.detach(), h_end.detach()
        return fn

    def capture(self, B, L, nch, xoh, mask, y, c_row):
        """CUDA-graph one whole chunk (forward, autograd.grad, in-graph gradient and stats accumulation).
        Static inputs live in S; replay per chunk after copying b, h, ex, up in. Bitwise equal to eager."""
        dev, N = self.dev, self.N
        fn = self.chunk_fn(B, nch)
        Pv = tuple(self.P.values())
        S = dict(b=torch.zeros(B, L, device=dev), h=torch.zeros(B, H, L, device=dev), xoh=xoh.clone(),
                 mask=mask.clone(), y=y.clone(), ex=torch.zeros(B, dtype=torch.bool, device=dev),
                 up=torch.zeros(B, dtype=torch.long, device=dev), w=torch.zeros(N, device=dev), c=c_row.clone())
        gacc = [torch.zeros_like(p) for p in Pv]
        acc = torch.zeros(4, N, dtype=torch.float64, device=dev)

        def body():
            loss, st, scal, bo, ho = fn(*Pv, S["b"], S["h"], S["xoh"], S["mask"], S["y"], S["ex"], S["up"], S["w"],
                                        S["c"])
            torch._foreach_add_(gacc, torch.autograd.grad(loss, Pv))
            acc.add_(st)
            return bo, ho, scal
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(s)
        torch.autograd.graph.set_override_stale_capture_stream(True)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = body()
        torch.cuda.synchronize()
        # keep fn alive: its closure owns tensors (ar) that the graph reads; freeing them let the allocator
        # hand their memory to other tensors, and replays read garbage
        return dict(g=g, S=S, gacc=gacc, acc=acc, out=out, fn=fn)

    def train_iter(self, x, y, mask, D, it, gp):
        a, N, dev = self.a, self.N, self.dev
        B, L = x.shape
        xoh = xonehot(x)
        c_row = 0.1 / mask.sum(1)                          # 0.1 nats per example per chunk, per bit
        wealth_t = torch.tensor(self.wealth, dtype=torch.float32, device=dev)
        b = torch.zeros(B, L, device=dev); h = torch.zeros(B, H, L, device=dev)
        for k in self.P:
            self.P[k].grad = None
        nch = D // KCH
        ex_all = (torch.rand(nch, B, generator=gp) < 0.5).to(dev)       # exploration flags, one draw per iteration
        up_all = torch.randint(N, (nch, B), generator=gp).to(dev)       # exploration picks
        Pv = tuple(self.P.values())
        sc = torch.zeros(nch, 5, dtype=torch.float64, device=dev)
        if dev != "cpu" and GRAPH:
            key = (B, L, nch)
            if key not in self._graphs:
                self._graphs.clear()                  # L_max only grows: free the old graph's pool
                self._graphs[key] = self.capture(B, L, nch, xoh, mask, y, c_row)
            G = self._graphs[key]
            S, gacc, acc, (bo, ho, scal) = G["S"], G["gacc"], G["acc"], G["out"]
            for k_, v_ in (("xoh", xoh), ("mask", mask), ("y", y), ("c", c_row), ("w", wealth_t)):
                S[k_].copy_(v_)
            S["b"].zero_(); S["h"].zero_(); acc.zero_()
            for t_ in gacc:
                t_.zero_()
            for j in range(nch):
                S["ex"].copy_(ex_all[j]); S["up"].copy_(up_all[j])
                G["g"].replay()
                sc[j].copy_(scal); S["b"].copy_(bo); S["h"].copy_(ho)
        else:
            fn = self.chunk_fn(B, nch)
            gacc = [torch.zeros_like(p) for p in Pv]
            acc = torch.zeros(4, N, dtype=torch.float64, device=dev)
            for j in range(nch):
                loss, st, scal, b, h = fn(*Pv, b, h, xoh, mask, y, ex_all[j], up_all[j], wealth_t, c_row)
                torch._foreach_add_(gacc, torch.autograd.grad(loss, Pv))
                acc += st
                sc[j] = scal
        for p, g_ in zip(Pv, gacc):
            p.grad = g_
        acc_l = acc.tolist()
        ntr = acc_l[3]
        self.adam(ntr)
        # ---- ledger (per-bit nats, averaged over the batch)
        ac = {k: [v / B for v in acc_l[i]] for i, k in enumerate(("gross", "cost", "price"))}
        for n in range(N):
            g, c, p = ac["gross"][n], ac["cost"][n], ac["price"][n]
            self.ema[n] = (1 - a.tau) * self.ema[n] + a.tau * g
            if self.ledger:
                self.wealth[n] += g - c - p
                self.mm -= g; self.sink += c; self.treasury += p
                cp = self.comp[n]; cp["gross"] += g; cp["cost"] += c; cp["price"] += p
        if self.ledger:
            for n in range(N):
                tx = a.tau * max(self.wealth[n], 0.0)
                self.wealth[n] -= tx; self.treasury += tx; self.comp[n]["tax"] += tx
            self.check_ledger()
        scl = sc.tolist()
        for j in range(nch):
            self.chunk_gain[j] = self.chunk_gain.get(j, 0.0) + scl[j][2] / B
        self.row_gain["finished"] += sum(r[3] for r in scl) / B
        self.row_gain["active"] += sum(r[4] for r in scl) / B
        w = self.win
        w["trades"] += int(round(sum(ntr))); w["passes"] += int(round(sum(r[0] for r in scl)))
        w["rowchunks"] += B * nch
        w["s_small"] += int(round(sum(r[1] for r in scl))); w["gross"] += sum(ac["gross"])
        for n in range(N):
            self.age[n] += 1
        return {"gross": sum(ac["gross"])}

    def adam(self, ntr):
        """Per-slot Adam (own optimizer per trader), per-slot grad clip 1, per-slot warm-up."""
        a, N = self.a, self.N
        with torch.no_grad():
            sq = torch.zeros(N, device=self.dev)
            for k, p in self.P.items():
                if p.grad is not None:
                    sq += p.grad.pow(2).reshape(N, -1).sum(1)
            clip = torch.clamp(1.0 / (sq.sqrt() + 1e-6), max=1.0)
            active = torch.tensor([1.0 if v > 0 else 0.0 for v in ntr], device=self.dev)
            for n in range(N):
                if ntr[n] > 0:
                    self.t[n] += 1
            t = torch.tensor([max(v, 1) for v in self.t], dtype=torch.float32, device=self.dev)
            lr = torch.tensor([a.lr * min(1.0, (self.age[n] + 1) / self.warm[n]) for n in range(N)],
                              dtype=torch.float32, device=self.dev)
            bc1 = 1 - 0.9 ** t; bc2 = 1 - 0.999 ** t
            for k, p in self.P.items():
                if p.grad is None:
                    continue
                shp = (N,) + (1,) * (p.dim() - 1)
                g = p.grad * clip.view(shp)
                act = active.view(shp)
                self.m[k].mul_(1 - 0.1 * act).add_(0.1 * act * g)
                self.v[k].mul_(1 - 0.001 * act).add_(0.001 * act * g * g)
                upd = (self.m[k] / bc1.view(shp)) / ((self.v[k] / bc2.view(shp)).sqrt() + 1e-8)
                p.sub_(act * lr.view(shp) * upd)

    # ---- population events (end of iteration it)
    def replace(self, n, parent, it, why):
        a = self.a
        rec = dict(it=it, slot=n, old_id=self.lineage[n], parent_slot=parent, parent_id=self.lineage[parent],
                   why=why, age=self.age[n], wealth=self.wealth[n], ema=self.ema[n], comp=dict(self.comp[n]))
        self.lives.append((self.lineage[n], self.born[n], it, self.founder[n]))
        with torch.no_grad():
            for k in self.P:
                src = self.P[k][parent].clone()
                sd = float(src.std()) if src.numel() > 1 else 0.0
                noise = torch.randn(src.shape, generator=self.ge).to(self.dev)
                self.P[k][n] = src + a.mut * sd * noise
                self.m[k][n] = 0; self.v[k][n] = 0
        self.t[n] = 0; self.warm[n] = 50; self.age[n] = 0; self.founder[n] = False; self.born[n] = it + 1
        self.lineage[n] = self.next_id; self.next_id += 1
        self.ema[n] = 0.0
        if self.ledger:
            self.treasury += self.wealth[n]
            if self.wealth[parent] >= 2 * a.W0:
                self.wealth[parent] -= a.W0; rec["funded_by"] = "parent"
            else:
                if self.treasury < a.W0:
                    short = a.W0 - self.treasury
                    self.mint -= short; self.treasury += short; self.injected += short
                    rec["minted"] = short
                self.treasury -= a.W0; rec["funded_by"] = "treasury"
            self.wealth[n] = a.W0
        self.comp[n] = dict(w0=a.W0, gross=0.0, cost=0.0, price=0.0, tax=0.0)
        self.events.append(dict(kind="replace", **rec))
        self.win["exits"] += 1; self.win["entries"] += 1

    def end_events(self, it, replay):
        a, N = self.a, self.N
        if N == 1:
            return 0
        floor = a.W0 / 10
        victims = []
        mature = [n for n in range(N) if self.age[n] >= a.m]
        if self.arm == "EC" and it < a.entry_stop:
            victims = [n for n in mature if self.wealth[n] < floor]
        elif self.arm in ("PBT", "EMA") and it in replay:
            cnt = min(replay[it], len(mature))
            if cnt < replay[it]:
                self.events.append(dict(kind="replay_short", it=it, want=replay[it], got=cnt))
            if self.arm == "PBT":
                pool = list(mature)
                for _ in range(cnt):
                    victims.append(pool.pop(int(torch.randint(len(pool), (1,), generator=self.ge))))
            else:
                victims = sorted(mature, key=lambda n: (self.ema[n], n))[:cnt]
        for n in victims:
            others = [q for q in range(N) if q not in victims]
            if self.arm == "EC":
                wts = [max(self.wealth[q], 0.0) for q in others]
            elif self.arm == "EMA":
                wts = [max(self.ema[q], 0.0) for q in others]
            else:
                wts = [1.0] * len(others)
            if sum(wts) <= 0:
                wts = [1.0] * len(others)
            pi = int(torch.multinomial(torch.tensor(wts, dtype=torch.float64), 1, generator=self.ge))
            self.replace(n, others[pi], it, "floor" if self.arm == "EC" else "replay")
        if self.ledger:
            for n in range(N):
                if self.wealth[n] < floor:
                    inj = a.W0 - self.wealth[n]
                    self.events.append(dict(kind="bailout", it=it, slot=n, id=self.lineage[n], wealth=self.wealth[n],
                                            age=self.age[n], comp=dict(self.comp[n])))
                    self.mint -= inj; self.injected += inj; self.wealth[n] = a.W0
                    self.comp[n] = dict(w0=a.W0, gross=0.0, cost=0.0, price=0.0, tax=0.0)
                    self.win["bailouts"] += 1
            self.check_ledger()
        return len(victims)

    # ---- test-time rollout
    @torch.no_grad()
    def rollout(self, x, mask, y, depths, variant="auction", h0=None, b0=None, want_h=False, wealth=None,
                slot=None):
        """variant auction: per row per chunk the best eligible bid trades, PASS if <= 0 (PASS is
        permanent: the state and so the bids do not change). variant single: `slot` trades every chunk.
        depths must be multiples of KCH. Returns ({D: (b, h?)}, info)."""
        N, dev = self.N, self.dev
        B, L = x.shape
        xoh = xonehot(x)
        b = torch.zeros(B, L, device=dev) if b0 is None else b0.clone()
        h = torch.zeros(B, H, L, device=dev) if h0 is None else h0.clone()
        wl_ = self.wealth if wealth is None else wealth
        wealth_t = torch.tensor(wl_, dtype=torch.float32, device=dev)
        elig = torch.tensor(self.eligible(), device=dev)
        halted = torch.zeros(B, dtype=torch.bool, device=dev)
        halt_step = torch.full((B,), -1, dtype=torch.long, device=dev)
        applied = torch.zeros(B, device=dev)
        s_small = 0; trades = 0
        wins = torch.zeros(3, N, dtype=torch.long, device=dev)     # auction wins per depth band 1-256 / -2048 / beyond
        resid = torch.zeros(B, device=dev)
        fixed = None
        want = sorted(set(depths))
        out, t = {}, 0
        for D in want:
            assert D % KCH == 0
            while t < D:
                idx = (~halted).nonzero()[:, 0]
                if idx.numel() == 0 or fixed is not None:
                    t = D
                    break
                full = idx.numel() == B
                xs, ms = (xoh, mask) if full else (xoh[idx], mask[idx])
                bs, hs = (b, h) if full else (b[idx], h[idx])
                if variant == "single" or N == 1:
                    sl = 0 if N == 1 else slot
                    Wr = slot_weights(self.P, sl); shared = True
                    sel = torch.full((idx.numel(),), sl, dtype=torch.long, device=dev)
                    go = torch.ones(idx.numel(), dtype=torch.bool, device=dev)
                else:
                    bd = bids_all(self.P, bid_feats(bs, hs, ms))
                    bd = bd.masked_fill(~elig.view(1, -1), -1e9)
                    bv, sel = bd.max(1)
                    go = bv > 0
                    if not go.all():
                        newly = idx[~go]
                        halted[newly] = True
                        halt_step[newly] = t
                        idx, sel, xs, ms, bs, hs = idx[go], sel[go], xs[go], ms[go], bs[go], hs[go]
                        full = False
                        if idx.numel() == 0:
                            t += KCH
                            continue
                    Wr = gather_rows(self.P, F.one_hot(sel, N).float()); shared = False
                    wins[0 if t < 256 else (1 if t < 2048 else 2)] += torch.bincount(sel, minlength=N)
                bk, hk = bs, hs
                for _ in range(KCH):
                    bk, hk = module_step(Wr, xs, bk, hk, ms, shared)
                db = bk - bs
                if self.sized:
                    wl = worst_loss_per_bit(bs, db, ms)
                    s = torch.clamp(self.a.frac * wealth_t[sel].clamp(min=0) / (wl + 1e-8), max=1.0)
                    s_small += int((s < 1).sum()); trades += s.numel()
                    bn = bs + s[:, None] * db; hn = hs + s[:, None, None] * (hk - hs)
                else:
                    trades += bk.shape[0]
                    bn, hn = bk, hk
                dd = torch.cat([bn - bs, (hn - hs).flatten(1)], 1).norm(dim=1)
                r = dd / (torch.cat([bs, hs.flatten(1)], 1).norm(dim=1) + 1e-12) / KCH
                if t % 64 == 0 and torch.equal(bn, bs) and torch.equal(hn, hs):
                    fixed = t          # every active row is at an exact fixed point: nothing changes again
                if full:
                    b, h = bn, hn
                    resid = r; applied += KCH
                else:
                    b[idx] = bn; h[idx] = hn
                    resid = torch.zeros(B, device=dev); resid[idx] = r; applied[idx] += KCH
                t += KCH
            out[D] = (b.clone(), h.clone() if want_h else None)
        info = dict(halt_step=halt_step, applied=applied, s_small=s_small, trades=trades, resid=resid, fixed_at=fixed,
                    wins=wins.tolist())
        return out, info

    # ---- persistence
    def state(self):
        return {"P": {k: v.detach().cpu() for k, v in self.P.items()},
                "m": {k: v.cpu() for k, v in self.m.items()}, "v": {k: v.cpu() for k, v in self.v.items()},
                "ge": self.ge.get_state(),
                "py": {k: getattr(self, k) for k in ("t", "warm", "age", "founder", "born", "lineage", "next_id",
                                                     "wealth", "treasury", "sink", "mm", "mint", "injected", "ema",
                                                     "comp", "events", "lives", "max_ledger_err", "win",
                                                     "chunk_gain", "row_gain")}}

    def load(self, s):
        for k in self.P:
            with torch.no_grad():
                self.P[k].copy_(s["P"][k].to(self.dev))
            self.m[k] = s["m"][k].to(self.dev); self.v[k] = s["v"][k].to(self.dev)
        self.ge.set_state(s["ge"])
        for k, v in s["py"].items():
            setattr(self, k, v)
        self.chunk_gain = {int(k): v for k, v in self.chunk_gain.items()}


# ----------------------------------------------------------------------------- shared eval helpers
def batch_size(Lmax):
    return min(128, 8192 // Lmax)


@torch.no_grad()
def exact_rate(M, arm, x, y, mask, D):
    if arm in ("R0", "DTL"):
        logit = M.rollout(x, mask, y, [D])[D][0]
    else:
        logit = M.rollout(x, mask, y, [D])[0][D][0]
    ok = (((logit > 0).float() == y) | (mask == 0)).all(1).float()
    return ok.mean().item()


def curriculum_set(Lmax, dev):
    g = torch.Generator().manual_seed(4242 + Lmax)
    return prefix_batch(256, Lmax, Lmax, g, dev)


def make(arm, seed, dev, a):
    return Plain(arm, seed, dev) if arm in ("R0", "DTL") else Econ(arm, seed, dev, a)


def load_replay(a):
    if a.arm not in ("PBT", "EMA"):
        return {}
    p = os.path.join(a.outdir, "train", f"EC_s{a.seed}.json")
    ev = json.load(open(p))["exit_schedule"]
    return {int(k): int(v) for k, v in ev.items()}


def run(a):
    dev = a.device
    name = f"{a.arm}_s{a.seed}"
    os.makedirs(os.path.join(a.outdir, "train"), exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "models"), exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "ckpt"), exist_ok=True)
    final = os.path.join(a.outdir, "train", name + ".json")
    if os.path.exists(final) and not a.force:
        print("done already", final); return
    M = make(a.arm, a.seed, dev, a)
    gd = torch.Generator().manual_seed(20_000 + a.seed)     # data
    gp = torch.Generator().manual_seed(30_000 + a.seed)     # picks (+ progressive-loss n, k)
    replay = load_replay(a)
    st = dict(it=0, Lmax=a.L0, doublings={str(a.L0): 0}, log=[], hashes=[], exit_schedule={}, cap_it=None,
              pass256_it=None, sec=0.0)
    ck = os.path.join(a.outdir, "ckpt", name + ".pt")
    if os.path.exists(ck) and not a.force:
        s = torch.load(ck, map_location="cpu", weights_only=False)
        M.load(s["model"]); gd.set_state(s["gd"]); gp.set_state(s["gp"]); st = s["st"]
        print("resumed at", st["it"], flush=True)
    t0 = time.time() - st["sec"]; last_ck = time.time()
    for it in range(st["it"], a.iters):
        Lmax = st["Lmax"]
        D = Lmax
        x, y, m = prefix_batch(batch_size(Lmax), min(a.lmin, Lmax), Lmax, gd, dev)
        lg = M.train_iter(x, y, m, D, it, gp)
        if a.arm in ECON_ARMS:
            nex = M.end_events(it, replay)
            if a.arm == "EC" and nex:
                st["exit_schedule"][str(it)] = nex
        if a.hash_every and (it % a.hash_every == 0):
            st["hashes"].append([it, phash(M.P if a.arm in ECON_ARMS else dict(M.net.named_parameters()))])
        if (it + 1) % 100 == 0:
            cx, cy, cm = curriculum_set(Lmax, dev)
            em = exact_rate(M, a.arm, cx, cy, cm, D)
            rec = dict(it=it + 1, Lmax=Lmax, exact=em, sec=round(time.time() - t0, 1), **lg)
            if a.arm in ECON_ARMS and M.N > 1:
                pos = [max(w, 0.0) for w in M.wealth]
                tot = sum(pos)
                rec.update(wealth=[round(w, 3) for w in M.wealth], ema=[round(e, 5) for e in M.ema],
                           age=list(M.age), eff_n=(1 / sum((w / tot) ** 2 for w in pos)) if tot > 0 else 0.0,
                           gini=gini(M.wealth), treasury=round(M.treasury, 3), injected=round(M.injected, 3),
                           spear_w_ema=spearman(M.wealth, M.ema), win=M.win)
                M.win = M.new_window()
            st["log"].append(rec)
            print(json.dumps(rec), flush=True)
            if em >= 0.95:
                if Lmax < CAP:
                    st["Lmax"] = Lmax * 2
                    st["doublings"][str(Lmax * 2)] = it + 1
                    if Lmax * 2 == CAP:
                        st["cap_it"] = it + 1
                elif st["pass256_it"] is None:
                    st["pass256_it"] = it + 1
        st["it"] = it + 1
        if time.time() - last_ck > a.ck_sec or it + 1 == a.iters:
            st["sec"] = time.time() - t0
            torch.save({"model": M.state(), "gd": gd.get_state(), "gp": gp.get_state(), "st": st}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            last_ck = time.time()
        if a.stop_at and it + 1 == a.stop_at and it + 1 < a.iters:     # simulated preemption (tests)
            st["sec"] = time.time() - t0
            torch.save({"model": M.state(), "gd": gd.get_state(), "gp": gp.get_state(), "st": st}, ck)
            print("STOPPED", it + 1, flush=True)
            return
    st["sec"] = time.time() - t0
    res = dict(arm=a.arm, seed=a.seed, args=vars(a), L_reached=st["Lmax"], doublings=st["doublings"],
               cap_it=st["cap_it"], pass256_it=st["pass256_it"], train_sec=st["sec"],
               sec_per_iter=st["sec"] / max(1, a.iters), log=st["log"], hashes=st["hashes"],
               exit_schedule=st["exit_schedule"])
    if a.arm in ECON_ARMS:
        nparam = sum(v[0].numel() for v in M.P.values())
        res.update(n_params_per_trader=nparam, N=M.N)
        if M.N > 1:
            res.update(events=M.events, lives=M.lives + [(M.lineage[n], M.born[n], None, M.founder[n]) for n in range(M.N)],
                       wealth=M.wealth, ema=M.ema, age=M.age, founder=M.founder, treasury=M.treasury,
                       injected=M.injected, max_ledger_err=M.max_ledger_err, chunk_gain=M.chunk_gain,
                       row_gain=M.row_gain, spear_w_ema_final=spearman(M.wealth, M.ema))
        torch.save({"arm": a.arm, "P": {k: v.detach().cpu() for k, v in M.P.items()}, "wealth": M.wealth,
                    "ema": M.ema, "age": M.age, "founder": M.founder, "args": vars(a)},
                   os.path.join(a.outdir, "models", name + ".pt"))
    else:
        res.update(n_params=sum(p.numel() for p in M.net.parameters()))
        torch.save({"arm": a.arm, "net": M.net.state_dict(), "args": vars(a)}, os.path.join(a.outdir, "models", name + ".pt"))
    json.dump(res, open(final + ".tmp", "w"))
    os.replace(final + ".tmp", final)
    if os.path.exists(ck):
        os.remove(ck)
    print("DONE", name, round(st["sec"], 1), "s L_reached", st["Lmax"], flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["R0", "DTL", "R1", "F", "EC", "PBT", "EMA"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--W0", type=float, default=20.0)
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--m", type=int, default=200)
    p.add_argument("--mut", type=float, default=0.02)
    p.add_argument("--entry_stop", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hash_every", type=int, default=10)
    p.add_argument("--ck_sec", type=float, default=180)
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--stop_at", type=int, default=0)
    p.add_argument("--width", type=int, default=32, help="trader module width (economy arms and R1)")
    p.add_argument("--lmin", type=int, default=8)
    p.add_argument("--L0", type=int, default=32)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    torch.set_num_threads(a.threads)
    run(a)
