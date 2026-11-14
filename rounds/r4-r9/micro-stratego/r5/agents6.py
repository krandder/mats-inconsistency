"""R5-E2 agents. Every model-based agent sees only its own observation (canonical encoding of the public state, its own
pieces, the public move history it witnessed) and its legal moves at the real root; all lookahead is inside theta_real
(no simulator, no oracle).
  P0           argmax_a l0(a) (theta_real 0-step action value + theta_amp correction)
  Unified-LIR  stiff_voi cone sampling, theta_trust beta_r, 4 waves of K/4 rollouts allocated by key-space PUCT
               (visit counts replaced by key-deduplicated counts sum 1/c_r), per-action consensus by attn_L3:
               l*_a = l0(a) + sum_r w_r[value] * (logit(e_r) - l0(a)) over the action's rollouts; argmax under the
               utility edge U: U_Std = effective win (l*_a); U_Miner = mean predicted Miner events of the action's
               rollouts (+1 enemy Miner captured, -1 own Miner lost; ties broken by l0)
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

import env6 as E
import model6 as M
import msgame6 as G
import train6 as T

C_PUCT = 1.0
REC_KEYS = ("sq_content", "sq_ownflag", "sq_slot", "slot_loc", "slot_flags", "slot_mcount", "slot_pubrev", "slot_true",
            "turn")


class Ctx:
    def __init__(self, net, amp, strust, rtrust, att, dev, seed):
        self.net, self.amp, self.strust, self.rtrust, self.att = net, amp, strust, rtrust, att
        self.dev = dev
        self.gen = torch.Generator(device=dev).manual_seed(seed)


def live_roots(ctx, states: List[E.State], hists: List[List], dev):
    """Root batch from live game states. hists[i] = list of (state, action) pairs of game i before states[i]."""
    rec = {k: [] for k in REC_KEYS}
    rec0 = {k: [] for k in REC_KEYS}
    wply, nw, legal, nl, env_legal = [], [], [], [], []
    for st, h in zip(states, hists):
        P = st.turn % 2
        t = st.turn
        t0 = max(t - M.WIN, t % 2)
        r = G.encode_root(st)
        st0 = h[t0][0] if t0 < t else st
        r0 = G.encode_root(st0)
        for k in REC_KEYS:
            rec[k].append(r[k])
            rec0[k].append(r0[k])
        w = np.zeros((M.WIN, 6), np.int64)
        for jj, tp_ in enumerate(range(t0, t)):
            w[jj] = G.encode_ply(h[tp_][0], h[tp_][1], P)
        wply.append(w)
        nw.append(t - t0)
        legal.append(G.legal_canon(st))
        acts = E.legal_actions(st)[:G.LMAX]
        nl.append(len(acts))
        env_legal.append(acts)
    R = {k: torch.tensor(np.stack(v).astype(np.int64), device=dev) for k, v in rec.items()}
    R0 = {k: torch.tensor(np.stack(v).astype(np.int64), device=dev) for k, v in rec0.items()}
    for RR in (R, R0):   # the opponent's hidden types never reach an agent
        RR["slot_true"] = torch.where(RR["slot_pubrev"].bool(), RR["slot_true"], torch.zeros_like(RR["slot_true"]))
    ix = torch.arange(len(states), device=dev)
    tp = M.tape_from(R, ix)
    tpw = M.tape_from(R0, ix)
    with T.autocast(dev):
        l0b, _ = M.belief0(ctx.net, tp)
        l0v = M.root_value_logit(ctx.net, tp)
    rt = {"tp": tp, "tpw": tpw, "wply": torch.tensor(np.stack(wply), device=dev), "nw": torch.tensor(nw, device=dev),
          "l0b": l0b.float(), "l0v": l0v.float(), "hidden": (tp["slot_val"] == 0) & (tp["slot_loc"] < M.NSQ),
          "legal": torch.tensor(np.stack(legal).astype(np.int64), device=dev), "n_legal": torch.tensor(nl, device=dev)}
    return rt, env_legal


@torch.no_grad()
def legal_l0(ctx, rt, with_amp=True):
    lg, n = rt["legal"], rt["n_legal"]
    B = lg.shape[0]
    l0 = torch.full((B, G.LMAX), -1e4, device=lg.device)
    for j in range(int(n.max())):
        m = n > j
        if m.any():
            with T.autocast(ctx.dev):
                lj, _ = M.action_l0(ctx.net, ctx.amp if with_amp else None, M.tape_index(rt["tp"], m), lg[m, j, 0], lg[m, j, 1])
            l0[m, j] = lj.float()
    return l0


@torch.no_grad()
def policy_prior(ctx, rt):
    with T.autocast(ctx.dev):
        h, _ = ctx.net(rt["tp"])
        pl = ctx.net.h_policy(h[:, M.GLOB]).float()
    lg = rt["legal"]
    idx = (lg[..., 0] * M.NSQ + lg[..., 1]).clamp_min(0)
    return torch.softmax(pl.gather(1, idx).masked_fill(lg[..., 0] < 0, -1e9), 1)


@torch.no_grad()
def per_action_attn(ctx, l0, A, e, dlb, beta, key, valid):
    """attn_L3 per-action value consensus. A [B,K] action of each rollout. Returns l* [B, LMAX]."""
    B, K = A.shape
    out = l0.clone()
    acts = torch.unique(A)
    for a in acts.tolist():
        m = (A == a) & valid
        rows = torch.nonzero(m.any(1)).squeeze(1)
        if rows.numel() == 0:
            continue
        mm = m[rows]
        bag = {"dlb": dlb[rows] * mm[..., None, None].float(), "dlv": torch.where(mm, M.logit(e[rows]) - l0[rows, a][:, None],
                                                                                  torch.zeros_like(e[rows])),
               "beta": torch.where(mm, beta[rows], torch.ones_like(beta[rows])), "valid": mm,
               "l0b": torch.zeros(rows.shape[0], M.NS, 6, device=A.device), "l0v": l0[rows, a]}
        w = ctx.att.weights(bag, key[rows])
        out[rows, a] = l0[rows, a] + (w[..., M.NS] * bag["dlv"]).sum(1)
    return out


@torch.no_grad()
def dedup_counts(A, key, valid, L, gamma=2.0):
    same = (A[:, :, None] == A[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    c = ((torch.relu(key @ key.transpose(1, 2)) ** gamma) * same.float()).sum(2).clamp_min(1e-6)
    neff = torch.zeros(A.shape[0], L, device=A.device)
    neff.scatter_add_(1, A, (1.0 / c) * valid.float())
    return neff


class Unified:
    name = "Unified-LIR"

    def __init__(self, ctx, K=16, utility="std"):
        self.ctx, self.K, self.utility = ctx, K, utility

    @torch.no_grad()
    def choose(self, rt, return_info=False):
        ctx = self.ctx
        l0 = legal_l0(ctx, rt)
        prior = policy_prior(ctx, rt)
        lg, n = rt["legal"], rt["n_legal"]
        B = lg.shape[0]
        W = 4
        per = self.K // W
        As, Es, DLB, BETA, KEY, MU = [], [], [], [], [], []
        lstar = l0.clone()
        neff = torch.zeros_like(l0)
        mu = torch.zeros_like(l0)
        for w in range(W):
            if w == 0:
                order = prior.argsort(1, descending=True)
            else:
                val = torch.sigmoid(lstar) if self.utility == "std" else mu
                tot = neff.sum(1, keepdim=True)
                score = val + C_PUCT * prior * torch.sqrt(tot + 1) / (1 + neff)
                order = score.masked_fill(lg[..., 0] < 0, -1e9).argsort(1, descending=True)
            act = torch.gather(order, 1, torch.arange(per, device=lg.device)[None] % n[:, None])
            o = T.run_rollouts(ctx.net, ctx.strust, ctx.rtrust, rt, act, 4, ctx.gen, ctx.dev)
            As.append(act)
            Es.append(o["e"])
            DLB.append(o["dlb"])
            BETA.append(o["beta"])
            KEY.append(ctx.amp.key(o["ebar"].reshape(-1, o["ebar"].shape[-1])).view(B, per, -1))
            MU.append(o["miner_u"])
            A = torch.cat(As, 1)
            valid = torch.ones_like(A, dtype=torch.bool)
            key = torch.cat(KEY, 1)
            lstar = per_action_attn(ctx, l0, A, torch.cat(Es, 1), torch.cat(DLB, 1), torch.cat(BETA, 1), key, valid)
            neff = dedup_counts(A, key, valid, l0.shape[1])
            s = torch.zeros_like(l0)
            c = torch.zeros_like(l0)
            s.scatter_add_(1, A, torch.cat(MU, 1))
            c.scatter_add_(1, A, torch.ones_like(torch.cat(MU, 1)))
            mu = s / c.clamp_min(1)
        illegal = lg[..., 0] < 0
        if self.utility == "std":
            choice = lstar.masked_fill(illegal, -1e9).argmax(1)
        else:
            choice = (mu + 1e-3 * torch.sigmoid(l0)).masked_fill(illegal, -1e9).argmax(1)
        if return_info:
            return choice, {"lstar": lstar, "mu": mu, "l0": l0, "A": A}
        return choice


class P0:
    name = "P0"

    def __init__(self, ctx, K=16, utility="std"):
        self.ctx = ctx

    @torch.no_grad()
    def choose(self, rt, return_info=False):
        l0 = legal_l0(self.ctx, rt)
        return l0.masked_fill(rt["legal"][..., 0] < 0, -1e9).argmax(1)


def play_match(A, B_, setups_pairs, dev, max_batch=200):
    """Duplicate-paired games: each (red, blue) setup pair is played twice, agent A as Red then as Blue.
    Returns per game: effective result for A (1 win, 0.5 draw, 0 loss), A's colour, length."""
    games = []
    for (rd, bl) in setups_pairs:
        games.append((rd, bl, 0))
        games.append((rd, bl, 1))
    n = len(games)
    res = np.zeros(n)
    side = np.array([g[2] for g in games])
    length = np.zeros(n, int)
    for g0 in range(0, n, max_batch):
        gs = list(range(g0, min(n, g0 + max_batch)))
        sts = [E.new_game(int(games[i][0]), int(games[i][1])) for i in gs]
        hists = [[] for _ in gs]
        while True:
            live = [k for k, st in enumerate(sts) if st.winner is None]
            if not live:
                break
            for agent, mine in ((A, True), (B_, False)):
                sel = [k for k in live if sts[k].winner is None and ((sts[k].turn % 2) == side[gs[k]]) == mine]
                if not sel:
                    continue
                rt, env_legal = live_roots(agent.ctx, [sts[k] for k in sel], [hists[k] for k in sel], dev)
                ch = agent.choose(rt).tolist()
                for k, c, el in zip(sel, ch, env_legal):
                    a = el[min(int(c), len(el) - 1)]
                    hists[k].append((sts[k], a))
                    sts[k] = E.step(sts[k], a)
        for k, i in enumerate(gs):
            w = sts[k].winner
            res[i] = 0.5 if w == -1 else (1.0 if w == side[i] else 0.0)
            length[i] = sts[k].turn
    return res, side, length
