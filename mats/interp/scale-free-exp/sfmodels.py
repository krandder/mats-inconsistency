"""Architectures for the scale-free experiments (2026-10-29).

All nets output a K-component Gaussian mixture for y* (logits, means, log-stds), as in phase B (tiny-pfn/pfn.py).
  tree    TreeNet   per-source encoder -> m-dim message -> SUM over sources -> readout. No cross-source interaction.
  hier    HierNet   known groups: per-source encoder -> m1 -> sum within group (or one attention block inside the
                    group) -> group encoder -> m2 -> sum over groups -> readout.
  set     SetTF     dense transformer, one token per source + a query token (pre-LN, PFN masks); cross=False
                    restricts every source token to itself (a tree with an attention-pooling readout).
  tok     TokTF     dense transformer on phase-B tokens, one token per (source, question) + a query token, with
                    per-layer attention modes and optional bottlenecks on the edges that cross a cut (exp 2, 3).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.GELU()]; d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


def split_head(y, K):
    return y[..., K:2 * K], y[..., 2 * K:].clamp(-6, 3), y[..., :K]            # mu, logstd, logits


def head_nll(head, y):
    mu, ls, lg = head
    lp = torch.log_softmax(lg, -1) - ls - 0.5 * math.log(2 * math.pi) - 0.5 * ((y[:, None] - mu) / ls.exp()) ** 2
    return -torch.logsumexp(lp, -1)


class TreeNet(nn.Module):
    def __init__(self, fin, m, h=64, K=4, enc_layers=2, read_layers=2, pool='sum'):
        super().__init__()
        self.K, self.pool, self.m = K, pool, m
        self.enc = mlp(fin, h, m, enc_layers)
        self.read = mlp(m, h, 3 * K, read_layers)

    def message(self, xs):
        return self.enc(xs)

    def forward(self, xs, **kw):                     # xs (B, N, fin)
        z = self.enc(xs)
        z = z.sum(1) if self.pool == 'sum' else z.mean(1)
        return split_head(self.read(z), self.K)


class TwoPassTree(nn.Module):
    """Two rounds (theory note 5.3): pass 1, per-source encoder -> m1 numbers, summed -> pilot (mean, log-sd of y*),
    trained with an auxiliary NLL; the pilot is broadcast; pass 2, per-source encoder sees its features, the pilot and
    its query reading standardised by the pilot -> m2 numbers, summed -> readout (also sees the pilot)."""
    def __init__(self, fin, m1=3, m2=8, h=64, K=4):
        super().__init__()
        self.K, self.m = K, m2
        self.enc1 = mlp(fin, h, m1); self.pil = mlp(m1, h, 2)
        self.enc2 = mlp(fin + 3, h, m2); self.read = mlp(m2 + 2, h, 3 * K)

    def forward(self, xs, **kw):
        B, N, f = xs.shape
        p = self.pil(self.enc1(xs).sum(1))
        mu0, ls0 = p[:, 0], p[:, 1].clamp(-6, 3)
        self.pilot = (mu0, ls0)
        rel = ((xs[:, :, 8] - mu0[:, None]) / ls0.exp()[:, None]).clamp(-30, 30)
        ctx = torch.stack([mu0, ls0], 1)[:, None, :].expand(B, N, 2)
        z2 = self.enc2(torch.cat([xs, ctx, rel[..., None]], 2)).sum(1)
        return split_head(self.read(torch.cat([z2, mu0[:, None], ls0[:, None]], 1)), self.K)

    def aux_loss(self, y):
        mu0, ls0 = self.pilot
        return (ls0 + 0.5 * math.log(2 * math.pi) + 0.5 * ((y - mu0) / ls0.exp()) ** 2).mean()


class Block(nn.Module):
    """Pre-LN block. attention: 'dense' (scores from content) or 'uniform' (fixed pattern = uniform over allowed
    keys; values still learned) or 'idroute' (scores from the id part of the token only).
    Optional cut bottleneck: on edges flagged by `cut` (B, T, T), keys and values pass through rank-k maps per head."""
    def __init__(self, d, h, mlp_ratio=2, attn='dense', cut_k=None, cut_kv='kv'):
        super().__init__()
        self.h, self.dh, self.attn = h, d // h, attn
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))
        self.cut_k, self.cut_kv = cut_k, cut_kv
        if cut_k is not None and cut_k > 0:
            self.kd = nn.Parameter(torch.randn(h, self.dh, cut_k) / math.sqrt(self.dh))
            self.ku = nn.Parameter(torch.randn(h, cut_k, self.dh) / math.sqrt(cut_k))
            self.vd = nn.Parameter(torch.randn(h, self.dh, cut_k) / math.sqrt(self.dh))
            self.vu = nn.Parameter(torch.randn(h, cut_k, self.dh) / math.sqrt(cut_k))

    def init_cut_pca(self, x, allow):
        """Initialise the rank-k maps to the top-k principal directions of keys and values (per head)."""
        with torch.no_grad():
            B, T, d = x.shape
            _, k, v = self.qkv(self.ln1(x)).split(d, -1)
            for t, (dn, up) in ((k, (self.kd, self.ku)), (v, (self.vd, self.vu))):
                t = t.reshape(-1, self.h, self.dh)
                for hh in range(self.h):
                    U, S, Vt = torch.linalg.svd(t[:, hh] - 0 * t[:, hh].mean(0), full_matrices=False)
                    P = Vt[:self.cut_k].T                                       # (dh, k)
                    dn[hh] = P; up[hh] = P.T

    def forward(self, x, allow, cut=None, idx=None, want=False):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, -1)
        if self.attn == 'idroute':                    # routing (queries, keys) from ids only; values from content
            q, k, _ = self.qkv(self.ln1(idx)).split(d, -1)
        sp = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        neg = torch.finfo(x.dtype).min
        if self.attn == 'uniform':
            s = torch.zeros(B, self.h, T, T, device=x.device, dtype=x.dtype)
        else:
            s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if cut is not None and self.cut_k is not None:
            if self.cut_k == 0:
                allow = allow & ~cut
            else:
                kb = torch.einsum('bhtd,hdk,hke->bhte', k, self.kd, self.ku) if 'k' in self.cut_kv else k
                vb = torch.einsum('bhtd,hdk,hke->bhte', v, self.vd, self.vu) if 'v' in self.cut_kv else v
                if self.attn != 'uniform':
                    sb = q @ kb.transpose(-1, -2) / math.sqrt(self.dh)
                    s = torch.where(cut[:, None], sb, s)
        s = s.masked_fill(~allow[:, None], neg)
        a = s.softmax(-1)
        if cut is not None and self.cut_k is not None and self.cut_k > 0:
            ac = a * cut[:, None]
            out = (a - ac) @ v + ac @ vb
        else:
            out = a @ v
        x = x + self.o(out.transpose(1, 2).reshape(B, T, d))
        x = x + self.mlp(self.ln2(x))
        return (x, a) if want else x


class SetTF(nn.Module):
    """One token per source (+ query token, last). cross=False: source tokens attend only to themselves."""
    def __init__(self, fin, d=32, L=2, H=2, K=4, cross=True):
        super().__init__()
        self.K, self.cross = K, cross
        self.inp = nn.Linear(fin + 1, d)
        self.blocks = nn.ModuleList([Block(d, H) for _ in range(L)])
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, 3 * K)

    def forward(self, xs, **kw):
        B, N, f = xs.shape
        tok = torch.cat([torch.cat([xs, xs.new_zeros(B, N, 1)], 2),
                         torch.cat([xs.new_zeros(B, 1, f), xs.new_ones(B, 1, 1)], 2)], 1)
        T = N + 1
        allow = torch.zeros(T, T, dtype=torch.bool, device=xs.device)
        allow[:, :N] = True
        if not self.cross:
            allow[:N, :N] = torch.eye(N, dtype=torch.bool, device=xs.device)
        allow[N, N] = True
        allow = allow[None].expand(B, T, T)
        x = self.inp(tok)
        for b in self.blocks:
            x = b(x, allow)
        return split_head(self.out(self.ln(x[:, -1])), self.K)


class HierNet(nn.Module):
    """Known contiguous groups of size gsize. inner='sum': sum of per-source messages (m1) inside the group;
    inner='attn': one dense attention block over the group's source tokens, then sum. Group messages m2 summed."""
    def __init__(self, fin, gsize, m1=16, m2=8, h=64, K=4, inner='sum', d_in=32):
        super().__init__()
        self.K, self.g, self.inner = K, gsize, inner
        if inner == 'sum':
            self.enc1 = mlp(fin, h, m1)
        else:
            self.inp = nn.Linear(fin, d_in); self.blk = Block(d_in, 2); self.enc1 = mlp(d_in, h, m1, 1)
        self.enc2 = mlp(m1, h, m2)
        self.read = mlp(m2, h, 3 * K)

    def forward(self, xs, **kw):
        B, N, f = xs.shape
        G = N // self.g
        if self.inner == 'sum':
            z1 = self.enc1(xs).view(B, G, self.g, -1).sum(2)
        else:
            t = self.inp(xs).view(B * G, self.g, -1)
            allow = torch.ones(B * G, self.g, self.g, dtype=torch.bool, device=xs.device)
            z1 = self.enc1(self.blk(t, allow)).view(B, G, self.g, -1).sum(2)
        z2 = self.enc2(z1).sum(1)
        return split_head(self.read(z2), self.K)


class TokTF(nn.Module):
    """Phase-B token transformer (one token per (source, question) + query token, last).
    attn: per-layer specs '<scoring>:<mask>', scoring in {dense, uniform, idroute}, mask in {all, self, src, grp}
      (all = PFN mask; self = each token only itself; src / grp = context tokens see only their own source / group;
      the query token always sees every context token and itself).
    cut: None | 'xsrc' (edges between different sources' tokens) | 'toq' (edges into the query token) |
      'labout' (edges out of a source's labelled-question tokens to any token that is not one of them) | 'xgrp';
    cut_k per layer: None = no bottleneck, 0 = cut edges removed, k = keys and values on cut edges pass rank-k maps.
    The id part of a token (source id + question id [+ group id]) is tok[..., :nid] (used by idroute)."""
    def __init__(self, fin, d=32, L=2, H=2, K=4, attn=None, cut=None, cut_k=None, nid=32, cut_kv="kv"):
        super().__init__()
        self.K, self.L, self.cut, self.nid = K, L, cut, nid
        attn = attn or ['dense:all'] * L
        attn = [a if ':' in a else a + ':all' for a in attn]
        self.scoring = [a.split(':')[0] for a in attn]; self.maskt = [a.split(':')[1] for a in attn]
        cut_k = cut_k if isinstance(cut_k, (list, tuple)) else [cut_k] * L
        self.inp = nn.Linear(fin, d)
        self.inp_id = nn.Linear(nid, d) if 'idroute' in self.scoring else None
        self.blocks = nn.ModuleList([Block(d, H, attn=self.scoring[l], cut_k=cut_k[l], cut_kv=cut_kv) for l in range(L)])
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, 3 * K)

    @staticmethod
    def masks(src, qq, grp, maskt, cut):
        B, T = src.shape
        dev = src.device
        isq = src < 0
        ctx = ~isq
        eye = torch.eye(T, dtype=torch.bool, device=dev)[None].expand(B, T, T)
        qrow = isq[:, :, None] & (ctx[:, None, :] | eye)                  # query sees context + itself
        same_src = src[:, :, None] == src[:, None, :]
        same_grp = grp[:, :, None] == grp[:, None, :]
        both = ctx[:, :, None] & ctx[:, None, :]
        allows = {}
        for m in set(maskt):
            if m == 'all':
                a = both
            elif m == 'self':
                a = eye & both
            elif m == 'src':
                a = both & same_src
            elif m == 'grp':
                a = both & same_grp
            else:
                raise ValueError(m)
            allows[m] = a | qrow
        if cut is None:
            return allows, None
        lab = (qq >= 0) & (qq < 4)
        c = {'xsrc': both & ~same_src, 'both': (both & ~same_src) | (isq[:, :, None] & ctx[:, None, :]),
             'xgrp': both & ~same_grp,
             'toq': isq[:, :, None] & ctx[:, None, :],
             'labout': lab[:, None, :] & ~(same_src & lab[:, :, None] & lab[:, None, :])}[cut]
        return allows, c

    def forward(self, tok, src=None, qq=None, grp=None, want=False, **kw):
        if grp is None:
            grp = src
        allows, cut = self.masks(src, qq, grp, self.maskt, self.cut)
        x = self.inp(tok)
        idx = self.inp_id(tok[..., :self.nid]) if self.inp_id is not None else None
        hs, attns = [x], []
        for l, b in enumerate(self.blocks):
            r = b(x, allows[self.maskt[l]], cut, idx, want)
            if want:
                x, a = r; attns.append(a)
            else:
                x = r
            hs.append(x)
        head = split_head(self.out(self.ln(x[:, -1])), self.K)
        return (head, hs, attns) if want else head


def n_params(m):
    return sum(p.numel() for p in m.parameters())
