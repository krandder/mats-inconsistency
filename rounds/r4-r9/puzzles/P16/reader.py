"""FR: a 2D (row x column) in-context reader, AVICI/TabPFN style, with XLNet-style two streams over rows.

Cells (t, c): c ranges over the m+1 input columns (listed bits and the candidate; value 0/1/absent) and the w output
columns. No column IDs and no row positions: columns are anonymous (the model is permutation-equivariant over columns)
and rows are exchangeable except for causality.

Streams per layer (shared weights):
  F (content): row attention over the full row t (with its labels); column attention over F cells of rows <= t.
  Q (query):   row attention over Q cells of row t only (output values masked); column attention over F cells of rows
               < t plus itself. Q never sees row t's labels, so one pass gives the prequential log p(y_t | rows < t, x_t).
Rows > ctx (query-only rows, used for the settlement block) are invisible as keys to every other row.

Joint readout ('joint'): a 2-layer two-stream autoregressive decoder over the w output cells of a row in a random order
(content h_j = Q_j + E[y_j], query g_j = Q_j; g_j sees the row's input cells and h_i earlier in the order).
'fact': per-output logits straight from Q (factorised ablation). 'direct' (FR-direct): F stream only, candidate flag,
attention-pooled head on the candidate's column -> (CMI regression in bits, useful logit).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class Attn(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh = nh
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, mask=None, kv=None):
        """x (N, L, d) queries; kv (N, Lk, d) keys/values source (default x); mask (L, Lk) bool (True = attend)."""
        N, L, d = x.shape
        h = self.nh
        if kv is None:
            q, k, v = self.qkv(x).view(N, L, 3, h, d // h).permute(2, 0, 3, 1, 4)
        else:
            Wq, Wk, Wv = self.qkv.weight.split(d, 0)
            bq, bk, bv = self.qkv.bias.split(d, 0)
            Lk = kv.shape[1]
            q = Fn.linear(x, Wq, bq).view(N, L, h, d // h).transpose(1, 2)
            k = Fn.linear(kv, Wk, bk).view(N, Lk, h, d // h).transpose(1, 2)
            v = Fn.linear(kv, Wv, bv).view(N, Lk, h, d // h).transpose(1, 2)
        y = Fn.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(N, L, d))


class FF(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        self.a = nn.Linear(d, mult * d)
        self.b = nn.Linear(mult * d, d)

    def forward(self, x):
        return self.b(Fn.gelu(self.a(x)))


class Block(nn.Module):
    def __init__(self, d, nh, mult):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.row = Attn(d, nh)
        self.col = Attn(d, nh)
        self.ff = FF(d, mult)

    def forward(self, x, colmask, two):
        """x (B, S, T, C, d) with S = 2 streams (F, Q) or 1 (F only)."""
        B, S, T, C, d = x.shape
        x = x + self.row(self.n1(x).reshape(B * S * T, C, d)).view(B, S, T, C, d)
        z = self.n2(x).permute(0, 3, 1, 2, 4).reshape(B * C, S * T, d)      # column sequences [F rows..., Q rows...]
        x = x + self.col(z, colmask).view(B, C, S, T, d).permute(0, 2, 3, 1, 4)
        return x + self.ff(self.n3(x))


def column_mask(T, ctx, two, device):
    """(S*T, S*T) bool. F_t -> F_t' iff t' <= t and (t' < ctx or t' == t); Q_t -> F_t' iff t' < t and t' < ctx; Q_t -> Q_t."""
    t = torch.arange(T, device=device)
    vis = t < ctx
    ff = (t[None, :] <= t[:, None]) & (vis[None, :] | (t[None, :] == t[:, None]))
    if not two:
        return ff
    qf = (t[None, :] < t[:, None]) & vis[None, :]
    qq = torch.eye(T, dtype=torch.bool, device=device)
    fq = torch.zeros(T, T, dtype=torch.bool, device=device)
    return torch.cat([torch.cat([ff, fq], 1), torch.cat([qf, qq], 1)], 0)


class Decoder(nn.Module):
    def __init__(self, d, nh, mult, L=2):
        super().__init__()
        self.L = L
        self.att = nn.ModuleList([Attn(d, nh) for _ in range(L)])
        self.ff = nn.ModuleList([FF(d, mult) for _ in range(L)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.ey = nn.Embedding(2, d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, qin, qout, y, order):
        """qin (N, Cin, d), qout (N, w, d) query-stream states of one row, y (N, w) labels, order (w,) rank of each output.
        Returns logits (N, w) of p(y_j = 1 | inputs, y_i with order[i] < order[j])."""
        N, w, d = qout.shape
        Cin = qin.shape[1]
        h = qout + self.ey(y)
        g = qout
        r = order if order.dim() == 2 else order[None]                       # (N or 1, w) ranks
        ones = torch.ones(r.shape[0], w, Cin, dtype=torch.bool, device=qin.device)
        hmask = torch.cat([ones, r[:, None, :] <= r[:, :, None]], 2)
        gmask = torch.cat([ones, r[:, None, :] < r[:, :, None]], 2)
        mask = torch.cat([hmask, gmask], 1)[:, None]                         # (N or 1, 1, 2w, Cin + w)
        for l in range(self.L):
            kv = torch.cat([qin, self.n1[l](h)], 1)
            upd = self.att[l](self.n1[l](torch.cat([h, g], 1)), mask, kv=kv)
            h = h + upd[:, :w]; g = g + upd[:, w:]
            h = h + self.ff[l](self.n2[l](h)); g = g + self.ff[l](self.n2[l](g))
        return self.out(g)[..., 0]


class FR(nn.Module):
    def __init__(self, d=128, L=6, nh=4, mult=2, mode='joint', dec_L=2):
        super().__init__()
        self.mode = mode
        self.two = mode != 'direct'
        self.e_in = nn.Embedding(3, d)       # 0, 1, absent
        self.e_out = nn.Embedding(3, d)      # 0, 1, masked
        self.e_role = nn.Embedding(3, d)     # input, output, candidate flag (direct only)
        self.blocks = nn.ModuleList([Block(d, nh, mult) for _ in range(L)])
        self.norm = nn.LayerNorm(d)
        if mode == 'joint':
            self.dec = Decoder(d, nh, mult, dec_L)
        elif mode == 'fact':
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        else:
            self.pq = nn.Parameter(torch.randn(nh, d) / math.sqrt(d))
            self.pool = Attn(d, nh)
            self.dhead = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def embed(self, V, Y, cand_flag=False):
        """V (B,T,m+1) in {-1,0,1}, Y (B,T,w) in {0,1} -> x (B,S,T,C,d)."""
        vin = torch.where(V < 0, torch.full_like(V, 2), V)
        xin = self.e_in(vin) + self.e_role.weight[0]
        if cand_flag:
            xin[:, :, -1] = xin[:, :, -1] + self.e_role.weight[2]
        xf = torch.cat([xin, self.e_out(Y) + self.e_role.weight[1]], 2)
        if not self.two:
            return xf[:, None]
        xq = torch.cat([xin, self.e_out(torch.full_like(Y, 2)) + self.e_role.weight[1]], 2)
        return torch.stack([xf, xq], 1)

    def backbone(self, V, Y, ctx=None, cand_flag=False):
        B, T, _ = V.shape
        x = self.embed(V, Y, cand_flag)
        mask = column_mask(T, T if ctx is None else ctx, self.two, V.device)
        for blk in self.blocks:
            x = blk(x, mask, self.two)
        return self.norm(x)

    def row_logp(self, V, Y, order, ctx=None):
        """Prequential log p(y_t | rows < t (visible), inputs_t) for every row: (B, T) nats. order (B, w) ranks."""
        B, T, cin = V.shape
        w = Y.shape[2]
        x = self.backbone(V, Y, ctx)
        q = x[:, 1]                                                          # (B,T,C,d)
        if self.mode == 'fact':
            lg = self.head(q[:, :, cin:])[..., 0]
        else:
            qs = q.reshape(B * T, cin + w, q.shape[-1])
            o = order.to(V.device)
            o = o[0] if bool((o == o[:1]).all()) else o[:, None, :].expand(B, T, w).reshape(B * T, w)
            lg = self.dec(qs[:, :cin], qs[:, cin:], Y.reshape(B * T, w), o).view(B, T, w)
        return -Fn.binary_cross_entropy_with_logits(lg.float(), Y.float(), reduction='none').sum(2)

    def direct(self, V, Y, lengths=None):
        """FR-direct: (B,) predicted CMI (bits) and useful logit from the with-b context (rows < lengths)."""
        x = self.backbone(V, Y, cand_flag=True)[:, 0]                        # (B,T,C,d)
        cb = x[:, :, V.shape[2] - 1]                                         # candidate column (B,T,d)
        B, T = cb.shape[:2]
        qry = self.pq[None].expand(B, -1, -1).to(cb.dtype)
        mask = None
        if lengths is not None:
            mask = (torch.arange(T, device=V.device)[None] < lengths[:, None])[:, None, None, :]   # (B,1,1,T)
        pooled = self.pool(qry, mask, kv=cb).mean(1)
        o = self.dhead(pooled)
        return o[:, 0], o[:, 1]


def random_order(B, w, gen, device):
    return torch.stack([torch.randperm(w, generator=gen) for _ in range(B)]).to(device)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
