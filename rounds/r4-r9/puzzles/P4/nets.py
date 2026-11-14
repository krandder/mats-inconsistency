"""P4 nets: 4-hidden-layer GELU/LayerNorm MLP aligned to a program by IIT/SIIT
(stage 1, noise check) or trained on plain BCE with a 24-unit sign layer (stage 2).

Layer l (1..4): h_l = GELU(LN(W_l [h_{l-1}; x] + b_l)), with h_0 absent (layer 1
reads x only). x is fed into every hidden layer. Planted nets read each output
logit from its own level-4 group of 4 channels; natural nets use a full readout.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gen

W = 64
CH = 4  # channels per program node


class MLP(nn.Module):
    def __init__(self, nin=12, nout=8, width=W, nlev=4, sign_layer=None, sign_width=24, group_readout=True):
        super().__init__()
        self.nin, self.nout, self.nlev = nin, nout, nlev
        self.sign_layer = sign_layer
        widths = [width if (l != sign_layer) else sign_width for l in range(1, nlev + 1)]
        self.widths = widths
        self.lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        prev = 0
        for l in range(nlev):
            self.lins.append(nn.Linear(prev + nin, widths[l]))
            self.lns.append(nn.LayerNorm(widths[l]))
            prev = widths[l]
        self.readout = nn.Linear(widths[-1], nout)
        self.group_readout = group_readout
        if group_readout:
            m = torch.zeros(nout, widths[-1])
            for j in range(nout):
                m[j, CH * j:CH * (j + 1)] = 1
            self.register_buffer('rmask', m)

    def act(self, l, z):
        z = self.lns[l - 1](z)
        if l == self.sign_layer:
            ht = F.hardtanh(z)
            return ht + (torch.sign(z) + (z == 0).float() - ht).detach()  # straight-through sign
        return F.gelu(z)

    def layer(self, l, hprev, x):
        inp = x if l == 1 else torch.cat([hprev, x], -1)
        return self.act(l, self.lins[l - 1](inp))

    def logits_from(self, h4):
        w = self.readout.weight * self.rmask if self.group_readout else self.readout.weight
        return h4 @ w.t() + self.readout.bias

    def forward(self, x, want=False):
        hs = []
        h = None
        for l in range(1, self.nlev + 1):
            h = self.layer(l, h, x)
            hs.append(h)
        z = self.logits_from(h)
        return (z, hs) if want else z

    def from_level(self, l, hl, x):
        """Logits when post-activation h_l is set to hl; later layers read x (base)."""
        h = hl
        for k in range(l + 1, self.nlev + 1):
            h = self.layer(k, h, x)
        return self.logits_from(h)


def xin(Xbits):
    return torch.tensor(2.0 * Xbits - 1.0, dtype=torch.float32)


def groups_for(prog, l, width=W):
    """Channel groups per candidate node at level l (4 each); the inert node takes all
    leftover channels (the 'inert player')."""
    nodes = prog['levels'][l]
    gs = []
    for k, nd in enumerate(nodes):
        if nd['role'] == 'inert':
            gs.append(np.arange(CH * k, width))
        else:
            gs.append(np.arange(CH * k, CH * (k + 1)))
    assert all(nd['role'] != 'inert' for nd in nodes[:-1]) and nodes[-1]['role'] == 'inert'
    return gs


def chan_mask(groups, sel, width):
    """sel: (B, n) bool of interchanged groups -> (B, width) channel mask."""
    G = torch.zeros(len(groups), width)
    for k, g in enumerate(groups):
        G[k, torch.as_tensor(g)] = 1
    return (sel.float() @ G).clamp(max=1)


def bce_ls(z, y, ls=0.02):
    return F.binary_cross_entropy_with_logits(z, y * (1 - ls) + 0.5 * ls)


def train_siit(prog, seed, steps=6000, bs=512, lr=1e-3, eps=None, log_every=500, min_steps=2000, verbose=True,
               all_levels=True, accept=0.995, node_sup=0.0):
    """IIT/SIIT training (InterpBench recipe). eps: per-output label noise (noise check).
    node_sup > 0 adds a linear-probe loss that makes each gate's 4-channel group decode its program
    value (used only for H1, whose 8-bit parity tree does not leave the chance plateau under IIT alone)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = gen.allx(prog['nin'])
    V = gen.values(prog, X)
    nlev = len(prog['levels']) - 1
    nout = V[-1].shape[0]
    net = MLP(nin=prog['nin'], nout=nout)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = xin(X)
    Y = torch.tensor(V[-1].T, dtype=torch.float32)
    groups = {l: groups_for(prog, l) for l in range(1, nlev)}
    # epiphenomenal probe targets
    epi = {l: prog['plants'][l].get('epi') for l in range(1, nlev) if prog['plants'].get(l, {}).get('epi') is not None}
    probes = nn.ModuleDict({str(l): nn.Linear(CH, 1) for l in epi})
    opt.add_param_group({'params': probes.parameters()})
    nprobe = nn.ModuleDict({f'{l}_{k}': nn.Linear(CH, 1) for l in range(1, nlev) for k, nd in enumerate(prog['levels'][l])
                            if nd['role'] != 'inert'}) if node_sup > 0 else nn.ModuleDict()
    if node_sup > 0:
        opt.add_param_group({'params': nprobe.parameters()})
    epsv = None if eps is None else torch.tensor(eps, dtype=torch.float32)

    def noisy(t):
        if epsv is None:
            return t
        flip = (torch.rand_like(t) < epsv).float()
        return t * (1 - flip) + (1 - t) * flip

    hist = []
    t0 = time.time()
    for it in range(steps):
        b = rng.integers(0, len(X), bs)
        s = rng.integers(0, len(X), bs)
        z, hb = net(xt[b], want=True)
        _, hsrc = net(xt[s], want=True)
        loss = bce_ls(z, noisy(Y[b]))
        # IIT: random multi-group interchange at one level; all_levels does one per level each step
        for l in (range(1, nlev) if all_levels else [int(rng.integers(1, nlev))]):
            n = len(groups[l])
            sel = rng.random((bs, n)) < 0.5
            vb, vs = V[l][:, b], V[l][:, s]
            hl = np.where(sel.T, vs, vb)
            tgt = gen.run_from(prog, l, hl, X[b])
            cm = chan_mask(groups[l], torch.tensor(sel), net.widths[l - 1])
            hp = hb[l - 1] * (1 - cm) + hsrc[l - 1] * cm
            zi = net.from_level(l, hp, xt[b])
            loss = loss + bce_ls(zi, noisy(torch.tensor(tgt.T, dtype=torch.float32)))
        # SIIT: interchange the inert and epiphenomenal groups at a random level; target unchanged
        l2 = int(rng.integers(1, nlev))
        roles = [nd['role'] for nd in prog['levels'][l2]]
        sel2 = np.array([[ro in ('inert', 'epi') for ro in roles]] * bs)
        cm2 = chan_mask(groups[l2], torch.tensor(sel2), net.widths[l2 - 1])
        hp2 = hb[l2 - 1] * (1 - cm2) + hsrc[l2 - 1] * cm2
        zs = net.from_level(l2, hp2, xt[b])
        loss = loss + bce_ls(zs, noisy(Y[b]))
        # linear probe on the epiphenomenal group (keeps it decodable)
        for lk, e in epi.items():
            src = prog['plants'][lk]['epi_src']
            pz = probes[str(lk)](hb[lk - 1][:, CH * e:CH * (e + 1)]).squeeze(1)
            loss = loss + 0.2 * F.binary_cross_entropy_with_logits(pz, torch.tensor(V[lk][src, b], dtype=torch.float32))
        for key, pr in nprobe.items():
            lk, k = map(int, key.split('_'))
            pz = pr(hb[lk - 1][:, CH * k:CH * (k + 1)]).squeeze(1)
            loss = loss + node_sup * F.binary_cross_entropy_with_logits(pz, torch.tensor(V[lk][k, b], dtype=torch.float32))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % log_every == 0 or it == steps - 1:
            ev = eval_iia(net, prog, V, X, groups, seed=10 ** 6 + it)
            ev.update(it=it + 1, loss=loss.item(), sec=round(time.time() - t0, 1))
            hist.append(ev)
            if verbose:
                print(prog['name'], seed, ev, flush=True)
            if it + 1 >= min_steps and ev['iia'] >= accept and ev['strict_iia'] >= accept and ev['acc'] >= 0.999:
                break
    net.eval()
    return net, hist


@torch.no_grad()
def eval_iia(net, prog, V, X, groups, seed, npair=2048):
    """Held-out interchange accuracy (per output bit): IIA over random multi-group
    interchanges at each level; strict IIA over interchanges of the inert and
    epiphenomenal groups (target unchanged)."""
    rng = np.random.default_rng(seed)
    xt = xin(X)
    b = rng.integers(0, len(X), npair)
    s = rng.integers(0, len(X), npair)
    z, hb = net(xt[b], want=True)
    _, hs = net(xt[s], want=True)
    Yb = V[-1][:, b].T
    acc = float(((z > 0).numpy() == Yb).mean())
    iia, strict, iia_ex = [], [], []
    for l in groups:
        n = len(groups[l])
        sel = rng.random((npair, n)) < 0.5
        hl = np.where(sel.T, V[l][:, s], V[l][:, b])
        tgt = gen.run_from(prog, l, hl, X[b]).T
        cm = chan_mask(groups[l], torch.tensor(sel), net.widths[l - 1])
        zi = net.from_level(l, hb[l - 1] * (1 - cm) + hs[l - 1] * cm, xt[b])
        iia.append(((zi > 0).numpy() == tgt).mean())
        iia_ex.append(((zi > 0).numpy() == tgt).all(1).mean())
        roles = [nd['role'] for nd in prog['levels'][l]]
        sel2 = np.array([[ro in ('inert', 'epi') for ro in roles]] * npair)
        cm2 = chan_mask(groups[l], torch.tensor(sel2), net.widths[l - 1])
        zs = net.from_level(l, hb[l - 1] * (1 - cm2) + hs[l - 1] * cm2, xt[b])
        strict.append(((zs > 0).numpy() == Yb).mean())
    return dict(acc=acc, iia=float(np.mean(iia)), iia_min_level=float(np.min(iia)),
                iia_example=float(np.mean(iia_ex)), strict_iia=float(np.mean(strict)))


def train_natural(prog, seed, steps=6000, bs=512, lr=1e-3, verbose=True):
    """Stage 2: plain BCE (label smoothing 0.02), hidden layer 2 = 24 sign units (STE)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = gen.allx(prog['nin'])
    V = gen.values(prog, X)
    net = MLP(nin=prog['nin'], nout=V[-1].shape[0], sign_layer=2, group_readout=False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = xin(X)
    Y = torch.tensor(V[-1].T, dtype=torch.float32)
    hist = []
    for it in range(steps):
        b = rng.integers(0, len(X), bs)
        loss = bce_ls(net(xt[b]), Y[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 1000 == 0:
            with torch.no_grad():
                acc = float(((net(xt) > 0).float() == Y).float().mean())
            hist.append(dict(it=it + 1, loss=float(loss), acc=acc))
            if verbose:
                print(prog['name'], seed, hist[-1], flush=True)
    net.eval()
    return net, hist
