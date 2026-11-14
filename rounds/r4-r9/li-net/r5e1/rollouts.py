"""R5-E1 rollout engine, preference-driven search policy and bags.

A rollout r = (patch A_r, temperature tau_r). Starting from the root state (no cell revealed), theta_real unmasks 4 of the 6 cells
of A_r one at a time, confidence-first (the not-yet-revealed patch cell with the lowest binary entropy under the current
prediction), each value sampled from sigmoid(logit / tau_r). After the 4 reveals:
  - final logits at every unsampled cell = theta_real(x_root, z_{A_r});
  - "local consistency logits" at each sampled cell j = theta_real's leave-one-out re-prediction of j from x_root and the other
    3 sampled cells (z_{A_r} minus j); the rollout's belief at j uses this re-prediction, never the sampled value itself;
  - forward-reverse disagreement = mean_j |logit at sampling time - leave-one-out logit|.
Delta l_r = l_r - l0 (36 cells). Cost: 7 forward passes per rollout (3 sampling steps after the root, 1 final, 3 leave-one-out;
the 4th leave-one-out state is the 3rd sampling state).

Search policy pi_pol(A | x_root, U) (Part A): the preference edge U in {U_NW, U_SE} picks the half searched: each rollout is
in the U-preferred half w.p. p_pref = 0.8 (else the other half), and within half h the patch is drawn
w.p. proportional to exp(gamma * <U_h, VOI(A)>), VOI_i(A) = q_i (1 - q_i) at the root for i in A (theta_real's own
uncertainty; no oracle), U_h = indicator of half h's cells. Unskewed = p = 0.5.
"""
from __future__ import annotations

import numpy as np
import torch

import env6
import wm

_T = {}


def geo(dev):
    if dev not in _T:
        _T[dev] = {
            "patches": torch.as_tensor(env6.PATCHES, device=dev),
            "in_nw": torch.as_tensor(env6.IN_NW, device=dev),
            "in_se": torch.as_tensor(env6.IN_SE, device=dev),
            "pmask": torch.as_tensor(env6.PATCH_MASK, device=dev).float(),
            "overlap": torch.as_tensor(env6.OVERLAP, device=dev),
            "sector": torch.as_tensor(env6.SECTOR, device=dev),
            "nwcell": (torch.arange(36, device=dev) < 18).float(),
        }
    return _T[dev]


@torch.no_grad()
def forward_chunks(model, ev, note, rev=None, val=None, chunk=8192, need_h=False):
    out_l, out_h = [], []
    for a in range(0, ev.shape[0], chunk):
        e, n = ev[a:a + chunk], note[a:a + chunk]
        r = torch.zeros_like(e, dtype=torch.bool) if rev is None else rev[a:a + chunk]
        v = torch.zeros_like(e) if val is None else val[a:a + chunk]
        l, h, _ = model(e, n, r, v)
        out_l.append(l.float())
        if need_h:
            out_h.append(h.float())
    if need_h:
        return torch.cat(out_l), torch.cat(out_h)
    return torch.cat(out_l)


def voi_half(l0):
    """<U_h, VOI(A)> for every patch: [B, 40] for the NW half and the SE half."""
    g = geo(l0.device)
    q = torch.sigmoid(l0)
    var = q * (1 - q)
    vnw = (var * g["nwcell"]) @ g["pmask"].T
    vse = (var * (1 - g["nwcell"])) @ g["pmask"].T
    return vnw, vse


def sample_patches(l0, n, p_nw, gen, gamma=4.0, exclude=None):
    """n patches per board. p_nw [B]: probability that a rollout searches the NW half. exclude [B, 40] bool (optional):
    patches not allowed (controlled-bag background). Returns patch [B, n] long, half_nw [B, n] bool."""
    g = geo(l0.device)
    B = l0.shape[0]
    vnw, vse = voi_half(l0)
    ninf = torch.full_like(vnw, -1e9)
    lg_nw = torch.where(g["in_nw"][None], gamma * vnw, ninf)
    lg_se = torch.where(g["in_se"][None], gamma * vse, ninf)
    if exclude is not None:
        lg_nw = torch.where(exclude, ninf, lg_nw)
        lg_se = torch.where(exclude, ninf, lg_se)
    ok_nw = (lg_nw > -1e8).any(1)
    ok_se = (lg_se > -1e8).any(1)
    half = torch.rand(B, n, device=l0.device, generator=gen) < p_nw[:, None]
    half = torch.where(ok_nw[:, None], half, torch.zeros_like(half))
    half = torch.where(ok_se[:, None], half, torch.ones_like(half))
    s_nw = torch.multinomial(torch.softmax(lg_nw, 1), n, replacement=True, generator=gen)
    s_se = torch.multinomial(torch.softmax(lg_se, 1), n, replacement=True, generator=gen)
    return torch.where(half, s_nw, s_se), half


@torch.no_grad()
def rollouts(model, ev, note, l0, patch, tau, gen, chunk=4096):
    """ev [R,36], note [R], l0 [R,36] root logits (repeated per rollout), patch [R] long, tau [R] float.
    Returns dict of tensors (R rows): lr (rollout logits), dl, hbar [R,d], ent, dis, cells [R,4], vals [R,4]."""
    outs = []
    for a in range(0, ev.shape[0], chunk):
        sl = slice(a, a + chunk)
        outs.append(_roll(model, ev[sl], note[sl], l0[sl], patch[sl], tau[sl], gen))
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def _roll(model, ev, note, l0, patch, tau, gen):
    g = geo(ev.device)
    R = ev.shape[0]
    rows = torch.arange(R, device=ev.device)
    P6 = g["patches"][patch]                                            # [R, 6]
    rev = torch.zeros(R, 36, dtype=torch.bool, device=ev.device)
    val = torch.zeros(R, 36, dtype=torch.long, device=ev.device)
    logits = l0.clone()
    cells, vals, fwd = [], [], []
    l_s3 = None
    for t in range(4):
        lp = logits.gather(1, P6)
        Hc = wm.bern_ent(torch.sigmoid(lp))
        Hc = torch.where(rev.gather(1, P6), torch.full_like(Hc, 1e9), Hc)
        c = P6[rows, Hc.argmin(1)]
        lc = logits[rows, c]
        z = torch.rand(R, device=ev.device, generator=gen) < torch.sigmoid(lc / tau)
        cells.append(c)
        vals.append(z)
        fwd.append(lc)
        if t == 3:
            l_s3 = logits
        rev[rows, c] = True
        val[rows, c] = z.long()
        if t < 3:
            logits = model(ev, note, rev, val)[0].float()
    lf, h, att = model(ev, note, rev, val, need_attn=True)
    lf, h, att = lf.float(), h.float(), att.float()
    loo = []
    for t in range(3):
        rv = rev.clone()
        rv[rows, cells[t]] = False
        loo.append(model(ev, note, rv, val)[0].float()[rows, cells[t]])
    loo.append(l_s3[rows, cells[3]])
    C = torch.stack(cells, 1)                                           # [R, 4]
    L = torch.stack(loo, 1)
    Fw = torch.stack(fwd, 1)
    lr = lf.clone()
    lr.scatter_(1, C, L)
    a = att[:, :, :36].gather(2, C[:, None, :].expand(R, att.shape[1], 4)).mean(1)     # [R, 4]
    a = a / a.sum(1, keepdim=True).clamp_min(1e-12)
    hb = (a[..., None] * h[:, :36].gather(1, C[..., None].expand(R, 4, h.shape[-1]))).sum(1)
    return {"lr": lr, "dl": lr - l0, "hbar": hb, "ent": wm.bern_ent(torch.sigmoid(lr)).mean(1),
            "dis": (Fw - L).abs().mean(1), "cells": C, "vals": torch.stack(vals, 1)}


def trust_features(ro, tau):
    return torch.cat([ro["hbar"], ro["ent"][:, None], torch.log(tau)[:, None], ro["dis"][:, None]], 1)


def es_target(ro, patch, occ_rows):
    """Real-world excess surprise of the rollout belief on its own patch cells vs the realised board (calibration only)."""
    g = geo(ro["lr"].device)
    P6 = g["patches"][patch]
    q = torch.sigmoid(ro["lr"].gather(1, P6))
    y = occ_rows.gather(1, P6)
    return wm.excess_surprise(q, y).mean(1)


@torch.no_grad()
def bag_rollouts(model, B, l0, patch, tau, gen, chunk=4096):
    """Rollouts for boards B (dict of [nb, ...] tensors) with patch/tau [nb, n]. Returns dict of [nb, n, ...] tensors."""
    nb, n = patch.shape
    rep = lambda x: x.repeat_interleave(n, 0)
    ro = rollouts(model, rep(B["ev"]), rep(B["note"]), rep(l0), patch.reshape(-1), tau.reshape(-1), gen, chunk)
    return {k: v.view(nb, n, *v.shape[1:]) for k, v in ro.items()}


def draw_tau(shape, gen, dev, p03=0.5):
    """tau_r in {0.3, 1.5}: high-trust deep rollouts vs noisy shallow rollouts, 50/50."""
    return torch.where(torch.rand(shape, device=dev, generator=gen) < p03, torch.full(shape, 0.3, device=dev),
                       torch.full(shape, 1.5, device=dev))
