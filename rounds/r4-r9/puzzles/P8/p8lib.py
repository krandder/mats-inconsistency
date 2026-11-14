"""P8 core: data, grouped nets (many independent small nets trained in one pass), prequential codelengths.

Every codelength in P8 is a member of a group: G independent nets with the same architecture, trained on the
same minibatch indices (paired runs: shared data order, identical init for members with the same init seed).
Early stopping is per member (best-val snapshot, patience), so each member is exactly what it would be alone.
"""
import hashlib
import json
import math
import os
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from maze_gen import bfs, gen_mazes

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LN2 = math.log(2.0)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("P8_RES", os.path.join(HERE, "results"))
SMOKE = os.environ.get("P8_SMOKE") == "1"

CFG = dict(lr=3e-3, batch=32, max_steps=2000, patience=100, eval_every=20, hid=16)
if SMOKE:
    CFG.update(max_steps=40, patience=20, eval_every=10)


def seed_int(*parts):
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8], 16)


def cpu_secs():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


# ============================================================ data
N_MAZE, H_Y, DENS = 11, 8, 0.3
X_CH = ["wall", "start", "inside", "decoy"]
POOL_MAZE = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "m2", "m4", "m6", "m8", "nz05", "nz20", "d2", "d4", "d6", "reach"]
SANITY_MAZE = ["copyY", "random", "null", "open", "startZ"]


def maze_data(num, n, seed):
    """Channels (float32, (num,n,n)): inputs wall/start/inside/decoy, target Y = 1[dist<=8], the ranked pool and
    sanity items. inside = 1 on the grid (0 in the conv zero padding), so out-of-bounds is visible (P9b convention)."""
    rng = np.random.default_rng(seed)
    w, s, d, _ = gen_mazes(num, n, DENS, rng)
    r2 = np.random.default_rng(seed_int("decoy", seed))
    dec = np.zeros_like(w)
    dD = np.zeros_like(d)
    man = np.zeros_like(d)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for t in range(num):
        opn = np.argwhere((w[t] == 0) & (s[t] == 0))
        c = tuple(opn[r2.integers(len(opn))])
        dec[t][c] = 1
        dD[t] = bfs(w[t], c)
        si, sj = np.argwhere(s[t] == 1)[0]
        man[t] = np.abs(ii - si) + np.abs(jj - sj)
    Y = d <= H_Y
    ch = {"wall": w, "start": s, "inside": np.ones_like(w), "decoy": dec, "Y": Y}
    for k in range(1, 8):
        ch[f"g{k}"] = d <= k
    for k in (2, 4, 6, 8):
        ch[f"m{k}"] = man <= k
    for k in (2, 4, 6):
        ch[f"d{k}"] = dD <= k
    ch["reach"] = d < 10 ** 6
    ch["nz05"] = Y ^ (r2.random(Y.shape) < 0.05)
    ch["nz20"] = Y ^ (r2.random(Y.shape) < 0.2)
    ch["copyY"] = Y.copy()
    ch["random"] = r2.random(Y.shape) < 0.5
    ch["null"] = np.zeros_like(w)
    ch["open"] = 1 - w
    ch["startZ"] = s.copy()
    return {k: np.asarray(v, np.float32) for k, v in ch.items()}


def _nbits():
    if os.environ.get("P8_NBITS"):
        return int(os.environ["P8_NBITS"])
    try:
        return int(open(os.path.join(RES, "par_nbits.txt")).read().strip())
    except (OSError, ValueError):
        return 32


N_BITS = _nbits()
POOL_PAR = ["p1", "p2", "p3", "p4", "q1", "q2", "dp1", "dp2", "dp3", "wp1", "wp2", "maj", "and", "bit"]
SANITY_PAR = ["copyY", "random"]


def parity_data(num, seed, key="stream", n_bits=None):
    """32 uniform bits; hidden 8-bit set b0..b7 fixed per seed (not per split). p_i = XOR of a pair, q1 = p1^p2,
    q2 = p3^p4, Y = q1^q2 (8-parity). Decoy pairs use irrelevant bits; wrong pairs cross the true pairs."""
    nb = n_bits or N_BITS
    hs = np.random.default_rng(seed_int("parity-set", seed)).permutation(nb)
    b = hs[:8]
    irr = hs[8:]
    rng = np.random.default_rng(seed_int("parity", key, seed))
    X = (rng.random((num, nb)) < 0.5)
    xb = X[:, b]
    p = [xb[:, 2 * i] ^ xb[:, 2 * i + 1] for i in range(4)]
    q1, q2 = p[0] ^ p[1], p[2] ^ p[3]
    Y = q1 ^ q2
    ch = {f"x{i}": X[:, i] for i in range(nb)}
    ch.update({"Y": Y, "p1": p[0], "p2": p[1], "p3": p[2], "p4": p[3], "q1": q1, "q2": q2})
    for i in range(3):
        ch[f"dp{i + 1}"] = X[:, irr[2 * i]] ^ X[:, irr[2 * i + 1]]
    ch["wp1"] = xb[:, 0] ^ xb[:, 2]
    ch["wp2"] = xb[:, 4] ^ xb[:, 6]
    ch["maj"] = xb.sum(1) >= 5
    ch["and"] = xb[:, 0] & xb[:, 1]
    ch["bit"] = xb[:, 0]
    ch["copyY"] = Y.copy()
    ch["random"] = rng.random(num) < 0.5
    ch["k3"] = xb[:, 0] ^ xb[:, 1] ^ xb[:, 2]
    ch["k6"] = np.bitwise_xor.reduce(xb[:, :6], 1)
    out = {k: np.asarray(v, np.float32) for k, v in ch.items()}
    out["_hidden"] = b
    return out


X_PAR = [f"x{i}" for i in range(N_BITS)]


class Bank:
    """Channel store on DEV. t = values in [0,1] (targets); inp = network input (maze: same as t; parity: 2t-1,
    so a missing/padded channel is 0 = 'no information'). '_zero' is the padding channel."""

    def __init__(self, chans, kind, hats=()):
        self.kind = kind
        names = [k for k in chans if not k.startswith("_")]
        self.names = names + ["_zero"] + list(hats)
        self.idx = {k: i for i, k in enumerate(self.names)}
        a0 = chans[names[0]]
        z = np.zeros_like(a0)
        h = np.full_like(a0, 0.5)
        arr = np.stack([chans[k] for k in names] + [z] + [h for _ in hats], 1)
        self.t = torch.tensor(arr, device=DEV)
        self.N = arr.shape[0]
        self.sp = tuple(arr.shape[2:])
        self.cells = int(np.prod(self.sp)) if self.sp else 1
        self._sync_inp()

    def _sync_inp(self):
        if self.kind == "par":
            self.inp = self.t * 2 - 1
            self.inp[:, self.idx["_zero"]] = 0
        else:
            self.inp = self.t

    def set(self, name, rows, vals):
        j = self.idx[name]
        self.t[rows, j] = vals
        if self.kind == "par":
            self.inp[rows, j] = vals * 2 - 1

    def add(self, name, vals):
        """Append a channel (vals tensor (N,*sp))."""
        self.names.append(name)
        self.idx[name] = len(self.names) - 1
        self.t = torch.cat([self.t, vals.to(DEV).float().unsqueeze(1)], 1)
        self._sync_inp()


# ============================================================ grouped models
def _unif(gen, shape, bound):
    return (torch.rand(shape, generator=gen) * 2 - 1) * bound


class GConv:
    """G independent convnets: `layers` 3x3 convs (ReLU) + 1x1 readout to `out` channels. Receptive radius = layers.
    Members with the same init seed get identical initial weights (paired runs)."""

    def __init__(self, G, cin, hid, layers, out=1, init_seeds=None):
        self.G, self.cin, self.hid, self.layers, self.out = G, cin, hid, layers, out
        seeds = init_seeds if init_seeds is not None else [0] * G
        cache = {}
        Ws = [[] for _ in range(layers + 1)]
        bs = [[] for _ in range(layers + 1)]
        for s in seeds:
            if s not in cache:
                g = torch.Generator().manual_seed(seed_int("gconv", s, cin, hid, layers, out))
                ws, bb = [], []
                c = cin
                for _ in range(layers):
                    bd = 1 / math.sqrt(c * 9)
                    ws.append(_unif(g, (hid, c, 3, 3), bd))
                    bb.append(_unif(g, (hid,), bd))
                    c = hid
                bd = 1 / math.sqrt(c)
                ws.append(_unif(g, (out, c, 1, 1), bd))
                bb.append(_unif(g, (out,), bd))
                cache[s] = (ws, bb)
            ws, bb = cache[s]
            for l in range(layers + 1):
                Ws[l].append(ws[l])
                bs[l].append(bb[l])
        self.W = [torch.stack(w).to(DEV).requires_grad_() for w in Ws]
        self.b = [torch.stack(b).to(DEV).requires_grad_() for b in bs]
        self.params = self.W + self.b

    def n_params(self):
        return sum(p[0].numel() for p in self.params)

    def macs(self):
        c, m = self.cin, 0
        for _ in range(self.layers):
            m += c * self.hid * 9
            c = self.hid
        return m + c * self.out  # per cell

    def features(self, x, upto):
        G = self.G
        h = x
        for l in range(upto):
            W = self.W[l]
            h = F.relu(F.conv2d(h, W.reshape(-1, *W.shape[2:]), self.b[l].reshape(-1), padding=1, groups=G))
        return h

    def logits(self, x):  # x (B, G*cin, n, n) -> (B, G, out, n, n)
        B = x.shape[0]
        h = self.features(x, self.layers)
        W = self.W[-1]
        o = F.conv2d(h, W.reshape(-1, *W.shape[2:]), self.b[-1].reshape(-1), groups=self.G)
        return o.view(B, self.G, self.out, *o.shape[2:])

    def train_logits(self, x):
        return [(1.0, self.logits(x))]


class GMLP:
    """G independent MLPs cin -> hid (x layers, ReLU) -> out."""

    def __init__(self, G, cin, hid, layers, out=1, init_seeds=None):
        self.G, self.cin, self.hid, self.layers, self.out = G, cin, hid, layers, out
        seeds = init_seeds if init_seeds is not None else [0] * G
        cache = {}
        Ws = [[] for _ in range(layers + 1)]
        bs = [[] for _ in range(layers + 1)]
        for s in seeds:
            if s not in cache:
                g = torch.Generator().manual_seed(seed_int("gmlp", s, cin, hid, layers, out))
                ws, bb, c = [], [], cin
                for l in range(layers + 1):
                    o = out if l == layers else hid
                    bd = 1 / math.sqrt(c)
                    ws.append(_unif(g, (c, o), bd))
                    bb.append(_unif(g, (1, o), bd))
                    c = o
                cache[s] = (ws, bb)
            ws, bb = cache[s]
            for l in range(layers + 1):
                Ws[l].append(ws[l])
                bs[l].append(bb[l])
        self.W = [torch.stack(w).to(DEV).requires_grad_() for w in Ws]
        self.b = [torch.stack(b).to(DEV).requires_grad_() for b in bs]
        self.params = self.W + self.b

    def n_params(self):
        return sum(p[0].numel() for p in self.params)

    def macs(self):
        c, m = self.cin, 0
        for _ in range(self.layers):
            m += c * self.hid
            c = self.hid
        return m + c * self.out

    def logits(self, x):  # x (B, G*cin) -> (B, G, out)
        B = x.shape[0]
        h = x.view(B, self.G, self.cin).transpose(0, 1)
        for l in range(self.layers + 1):
            h = torch.baddbmm(self.b[l], h, self.W[l])
            if l < self.layers:
                h = F.relu(h)
        return h.transpose(0, 1)

    def train_logits(self, x):
        return [(1.0, self.logits(x))]


class GStack:
    """Free-latent tower (B5 and the deployed-truth control): K rungs, each a 2-layer trader emitting one sigmoid
    channel from base + previous latents; then a 2-layer Y trader on base + all latents. Trained end to end on Y.
    kind 'conv' (GConv hid 16) or 'mlp' (GMLP hid 128)."""

    def __init__(self, G, cbase, K, kind="conv", hid=None, init_seeds=None):
        self.G, self.cbase, self.K, self.kind = G, cbase, K, kind
        Mk = GConv if kind == "conv" else GMLP
        hid = hid or (CFG["hid"] if kind == "conv" else 128)
        self.rungs = [Mk(G, cbase + j, hid, 2, 1, init_seeds) for j in range(K)]
        self.head = Mk(G, cbase + K, hid, 2, 1, init_seeds)
        self.params = [p for r in self.rungs for p in r.params] + self.head.params
        self.out = 1

    def n_params(self):
        return sum(r.n_params() for r in self.rungs) + self.head.n_params()

    def macs(self):
        return sum(r.macs() for r in self.rungs) + self.head.macs()

    def logits(self, x):
        B = x.shape[0]
        sp = tuple(x.shape[2:])
        cur = x.view(B, self.G, self.cbase, *sp)
        for r in self.rungs:
            z = torch.sigmoid(r.logits(cur.reshape(B, -1, *sp)))  # (B,G,1,*sp)
            if self.kind == "mlp":
                z = z * 2 - 1
            cur = torch.cat([cur, z], 2)
        return self.head.logits(cur.reshape(B, -1, *sp))

    def train_logits(self, x):
        return [(1.0, self.logits(x))]


class DTR:
    """DT-recall (Bansal et al. 2022; vendored dtnet_vendored.py = P0/dtnet.py) as a 1-member group, progressive loss."""

    def __init__(self, cin, width, iters, alpha=0.5, seed=0):
        from dtnet_vendored import DTRecall
        torch.manual_seed(seed_int("dtr", seed, width, iters))
        self.net = DTRecall(width, dim=2, in_channels=cin).to(DEV)
        self.iters, self.alpha, self.G, self.out, self.cin, self.width = iters, alpha, 1, 1, cin, width
        self.params = list(self.net.parameters())

    def n_params(self):
        return sum(p.numel() for p in self.params)

    def macs(self):
        st, hd = self.net.macs_per_pos()
        return self.cin * self.width * 9 + self.iters * st + hd

    def _lg(self, o):
        return (o[:, 1] - o[:, 0])[:, None, None]

    def logits(self, x):
        h = self.net.init(x)
        for _ in range(self.iters):
            h = self.net.step(h, x)
        return self._lg(self.net.readout(h))

    def train_logits(self, x):
        full = self.logits(x)
        if self.alpha <= 0:
            return [(1.0, full)]
        n = int(torch.randint(0, self.iters, (1,)))
        k = int(torch.randint(1, self.iters - n + 1, (1,)))
        with torch.no_grad():
            h = self.net.init(x)
            for _ in range(n):
                h = self.net.step(h, x)
        h = h.detach()
        for _ in range(k):
            h = self.net.step(h, x)
        return [(1 - self.alpha, full), (self.alpha, self._lg(self.net.readout(h)))]


# ============================================================ input builders
def member_builder(bank, in_lists, tgts, cin=None, member_w=None, drop=None):
    """Standard members: member g reads channels in_lists[g] (padded with _zero to cin) and predicts tgts[g].
    member_w (G, N) optional per-member example weights (cross-fitting folds). drop: optional (G,) list of
    channel-name lists to zero for that member (unused by default)."""
    cin = cin or max(len(l) for l in in_lists)
    flat = []
    for l in in_lists:
        assert len(l) <= cin, (l, cin)
        flat += [bank.idx[c] for c in l] + [bank.idx["_zero"]] * (cin - len(l))
    flat = torch.tensor(flat, device=DEV)
    tg = torch.tensor([bank.idx[t] for t in tgts], device=DEV)

    def build(bi, bk=None):
        bk = bk or bank
        assert bk.names == bank.names
        xb = bk.inp[bi]
        xin = xb[:, flat]
        if bk.kind == "maze":
            xin = xin.reshape(len(bi), -1, *bk.sp)
        tgt = bk.t[bi][:, tg].unsqueeze(2)
        w = None
        if member_w is not None and bk is bank:
            w = member_w[:, bi].T.unsqueeze(-1)
        return xin, tgt, w

    build.cin = cin
    return build


# ============================================================ training
def per_example_loss(lg, tgt):
    """(B,G,out,*sp) logits/targets -> (B,G,out) nats summed over cells."""
    l = F.binary_cross_entropy_with_logits(lg, tgt.expand_as(lg), reduction="none")
    return l.reshape(*l.shape[:3], -1).sum(-1)


def _wmean(l, w):
    if w is None:
        return l.mean((0, 2))
    return (l * w).sum((0, 2)) / w.sum((0, 2)).clamp_min(1e-9)


@torch.no_grad()
def eval_loss(model, build, idx, chunk=128, bk=None):
    """Per-example, per-member loss in nats (len(idx), G) (summed over out channels, weighted if w)."""
    outs = []
    for i in range(0, len(idx), chunk):
        bi = idx[i:i + chunk]
        xin, tgt, w = build(bi, bk)
        l = per_example_loss(model.logits(xin), tgt)
        if w is not None:
            l = l * w
        outs.append(l.sum(2))
    return torch.cat(outs, 0)


@torch.no_grad()
def val_loss(model, build, idx, chunk=128):
    tot, wt = 0, 0
    for i in range(0, len(idx), chunk):
        bi = idx[i:i + chunk]
        xin, tgt, w = build(bi)
        l = per_example_loss(model.logits(xin), tgt)
        if w is None:
            w = torch.ones_like(l)
        tot = tot + (l * w).sum((0, 2))
        wt = wt + w.sum((0, 2))
    return tot / wt.clamp_min(1e-9)


def fit(model, build, tr_idx, va_idx, gen, cfg=None, fixed_steps=None):
    """Adam, per-member early stopping on va_idx (best snapshot, patience), or a fixed number of steps.
    Returns dict(steps, stop_step per member)."""
    cfg = cfg or CFG
    G = model.G
    params = model.params
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    tr_idx = torch.as_tensor(tr_idx, device=DEV)
    va_idx = torch.as_tensor(va_idx, device=DEV)
    best = [p.detach().clone() for p in params]
    bv = val_loss(model, build, va_idx) if fixed_steps is None else None
    bstep = torch.zeros(G, device=DEV)
    maxs = fixed_steps or cfg["max_steps"]
    E, P = cfg["eval_every"], cfg["patience"]
    step = 0
    while step < maxs:
        step += 1
        j = torch.randint(len(tr_idx), (cfg["batch"],), generator=gen)
        bi = tr_idx[j.to(DEV)]
        build.mode = "train"
        xin, tgt, w = build(bi)
        build.mode = "eval"
        tl = model.train_logits(xin)
        loss = sum(wt * _wmean(per_example_loss(lg, tgt), w) for wt, lg in tl)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        if fixed_steps is None and step % E == 0:
            v = val_loss(model, build, va_idx)
            active = (step - bstep) < P  # a member whose patience ran out is frozen at its snapshot (as if alone)
            imp = (v < bv - 1e-7) & active
            if imp.any():
                bv = torch.where(imp, v, bv)
                bstep = torch.where(imp, torch.full_like(bstep, step), bstep)
                with torch.no_grad():
                    for p, b in zip(params, best):
                        m = imp.view(G, *([1] * (p.dim() - 1))) if (G > 1 and p.shape[0] == G) else imp.any()
                        b.copy_(torch.where(m, p, b))
            if bool(((step - bstep) >= P).all()):
                break
    if fixed_steps is None:
        with torch.no_grad():
            for p, b in zip(params, best):
                p.copy_(b)
        stop = torch.clamp(bstep + P, max=step)
    else:
        stop = torch.full((G,), float(step), device=DEV)
    return {"steps": step, "stop": stop.cpu().numpy()}


def blocks_for(N, first=8):
    b = [first]
    while b[-1] < N:
        b.append(min(N, b[-1] * 2))
    return b


class Preq:
    """Prequential codelength (bits) per member over examples [0,N) in ratio-2 blocks 8,16,...,N.
    Block 0 is coded uniform (1 bit per coded cell and output channel). Stage i: the warm-started model is
    trained on [0,a) (replay of all past examples; 10% val slice for early stopping), then block [a,b) is coded.
    Callers drive the stages so several groups can interleave (hooks between train and code)."""

    def __init__(self, model, build, N, seed, tag, cells, cfg=None, fixed_steps=None, uniform=None):
        self.m, self.build, self.N, self.seed, self.tag = model, build, N, seed, tag
        self.blocks = blocks_for(N)
        self.cfg, self.fixed = cfg, fixed_steps
        G = model.G
        ub = uniform if uniform is not None else self.blocks[0] * cells * model.out
        self.per_block = [np.full(G, float(ub))]
        self.steps, self.stop_tot = [], np.zeros(G)
        self.work_stage = []  # per-stage per-member training MACs (own stop step)
        self.cells = cells
        self.last_ex = None
        self.t0 = time.time()

    @property
    def n_stages(self):
        return len(self.blocks) - 1

    def ab(self, i):
        return self.blocks[i], self.blocks[i + 1]

    def train(self, i):
        a, b = self.ab(i)
        g = torch.Generator().manual_seed(seed_int(self.seed, self.tag, "order", i))
        perm = torch.randperm(a, generator=g)
        nv = max(1, int(round(0.1 * a)))
        va, tr = perm[:nv], perm[nv:]
        if len(tr) == 0:
            tr = va
        info = fit(self.m, self.build, tr, va, g, self.cfg, self.fixed)
        self.steps.append(info["steps"])
        self.stop_tot += info["stop"]
        bs = (self.cfg or CFG)["batch"]
        self.work_stage.append(info["stop"] * bs * self.cells * self.m.macs())

    def code(self, i):
        a, b = self.ab(i)
        ex = eval_loss(self.m, self.build, torch.arange(a, b, device=DEV)) / LN2
        self.per_block.append(ex.sum(0).cpu().numpy())
        self.last_ex = ex
        return ex

    def run(self, hook=None):
        for i in range(self.n_stages):
            self.train(i)
            if hook is not None:
                hook(i)
            self.code(i)
        return self.result()

    def result(self):
        pb = np.stack(self.per_block, 1)
        return {"bits": pb.sum(1), "per_block": pb, "blocks": self.blocks, "steps": self.steps,
                "stop_steps": self.stop_tot, "work": np.sum(self.work_stage, 0),
                "work_cum": np.cumsum(self.work_stage, 0),
                "last_block_mean": self.last_ex.mean(0).cpu().numpy(), "secs": time.time() - self.t0}


def prequential(model, build, N, seed, tag, cells, hook=None, cfg=None, fixed_steps=None):
    return Preq(model, build, N, seed, tag, cells, cfg, fixed_steps).run(hook)


@torch.no_grad()
def predict(model, build, idx, bk=None, chunk=128):
    """sigmoid predictions (len(idx), G, out, *sp)."""
    outs = []
    for i in range(0, len(idx), chunk):
        bi = idx[i:i + chunk]
        xin, _, _ = build(bi, bk)
        outs.append(torch.sigmoid(model.logits(xin)))
    return torch.cat(outs, 0)


# ============================================================ statistics
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def ci(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    m = float(np.mean(xs))
    if n == 1:
        return {"mean": m, "ci95": None, "n": 1}
    sd = float(np.std(xs, ddof=1))
    return {"mean": m, "ci95": T975.get(n - 1, 1.96) * sd / math.sqrt(n), "n": n}


def kendall_b(a, b):
    from scipy.stats import kendalltau
    t = kendalltau(a, b, variant="b").statistic
    return float(t) if t == t else 0.0


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return float(r) if r == r else 0.0


def auc(score, label):
    score = np.asarray(score, float)
    label = np.asarray(label).astype(bool)
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(score)
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def exact_acc(prob, Y):
    """prob (B,G,1,n,n) or (B,G,1); Y (B,n,n) or (B,) -> (B,G) bool: every cell correct."""
    pr = prob[:, :, 0] > 0.5
    yy = Y.bool().unsqueeze(1)
    eq = pr == yy
    if eq.dim() > 2:
        eq = eq.flatten(2).all(-1)
    return eq


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, torch.Tensor):
        return jsonable(o.detach().cpu().numpy())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if f != f else round(f, 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump(name, obj):
    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, name + ".json")
    with open(p + ".tmp", "w") as f:
        json.dump(jsonable(obj), f)
    os.replace(p + ".tmp", p)
    return p
