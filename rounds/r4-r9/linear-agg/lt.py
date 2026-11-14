"""Linear transformers of Vladymyrov et al. 2024 (arXiv 2402.14180), linear-agg 2026-10-16.

Tokens e_i = (x_i, y_i). One layer (paper eq. 2-3, unnormalised sums, no softmax, MLP or LayerNorm):
    e_i <- e_i + sum_h sum_j (e_j' Q_h e_i) P_h e_j          j over the CONTEXT tokens only (query-like tokens are never attended to)
         = e_i + (sum_h P_h C Q_h) e_i,   C = sum_j e_j e_j'
  full : P_h, Q_h arbitrary (D x D), H heads -> 2 H D^2 parameters per layer.
  diag : the paper's eq. 10 in the omega reparameterisation (4 free scalars per layer, multi-head form):
         x_i <- x_i + w_xx Sigma x_i + w_xy y_i alpha ;  y_i <- y_i + w_yx <alpha, x_i> + w_yy y_i lambda
         Sigma = sum_j x_j x_j', alpha = sum_j y_j x_j, lambda = sum_j y_j^2  (current layer's context states)
  gdpp : diag with w_xy = w_yy = 0 (paper eq. 11; 2 scalars per layer).
Readout (paper): prediction = -y of the first query token after the last layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinTF(nn.Module):
    def __init__(self, kind: str, L: int, D: int, heads: int = 1, init: float = 0.02, ydim: int = 1, seed: int = 0):
        super().__init__()
        assert kind in ("full", "diag", "gdpp")
        self.kind, self.L, self.D, self.H, self.ydim = kind, L, D, heads, ydim
        g = torch.Generator().manual_seed(10_000 + seed)
        if kind == "full":
            self.P = nn.Parameter(torch.randn(L, heads, D, D, generator=g) * init)
            self.Q = nn.Parameter(torch.randn(L, heads, D, D, generator=g) * init)
        else:
            assert ydim == 1
            n = 4 if kind == "diag" else 2
            self.w = nn.Parameter(torch.randn(L, n, generator=g) * init)    # diag: xx, xy, yx, yy ; gdpp: xx, yx

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def omegas(self):
        """[L, 4] (w_xx, w_xy, w_yx, w_yy) for diag / gdpp (zeros where absent)."""
        if self.kind == "diag":
            return self.w.detach().clone()
        if self.kind == "gdpp":
            z = torch.zeros(self.L, 4, dtype=self.w.dtype, device=self.w.device)
            z[:, 0], z[:, 2] = self.w.detach()[:, 0], self.w.detach()[:, 1]
            return z
        raise ValueError(self.kind)

    def forward(self, ctx, qry, keep_states=False):
        """ctx [B, n, D], qry [B, m, D] (query-like tokens, y part = 0). Returns final (ctx, qry) and, if keep_states,
        the list of context x-states [B, n, D - ydim] for l = 0..L."""
        states = [ctx[..., : self.D - self.ydim]] if keep_states else None
        for l in range(self.L):
            if self.kind == "full":
                C = torch.einsum("bnd,bne->bde", ctx, ctx)
                M = torch.einsum("hde,bef,hfg->bdg", self.P[l], C, self.Q[l])
                ctx = ctx + torch.einsum("bdg,bng->bnd", M, ctx)
                qry = qry + torch.einsum("bdg,bng->bnd", M, qry)
            else:
                d = self.D - 1
                x, y = ctx[..., :d], ctx[..., d]
                xq, yq = qry[..., :d], qry[..., d]
                Sig = torch.einsum("bnd,bne->bde", x, x)
                alpha = torch.einsum("bn,bnd->bd", y, x)
                lam = (y * y).sum(1)
                if self.kind == "diag":
                    wxx, wxy, wyx, wyy = self.w[l]
                else:
                    wxx, wyx = self.w[l]
                    wxy = wyy = torch.zeros((), device=ctx.device)

                def upd(xx, yy):
                    nx = xx + wxx * torch.einsum("bde,bne->bnd", Sig, xx) + wxy * yy[..., None] * alpha[:, None, :]
                    ny = yy + wyx * torch.einsum("bd,bnd->bn", alpha, xx) + wyy * yy * lam[:, None]
                    return torch.cat([nx, ny[..., None]], -1)
                ctx, qry = upd(x, y), upd(xq, yq)
            if keep_states:
                states.append(ctx[..., : self.D - self.ydim])
        return ctx, qry, states
