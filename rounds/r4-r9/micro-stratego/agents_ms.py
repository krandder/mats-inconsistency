"""E-MS game-playing agents. Every model-based agent sees only its own observation (canonical encoding of the public
state + its own pieces) and its legal moves at the real root (the game tells a player its legal moves); all lookahead is
inside theta_real (no simulator, no oracle). The heuristic opponent (behaviour policy, tau = 1) is part of the
environment for D4.2.
  P0            argmax_a l0(a) (theta_amp distilled 0-step action value)
  Ataraxos1     K rollouts round-robin over legal actions, row_major Z, depth 4, l*_a = l0(a) + eta * sum dl_r (beta = 1)
  Unified       stiff_voi Z, 4 waves of K/4 rollouts allocated by key-space PUCT (visit counts replaced by key-deduplicated
                counts), theta_trust beta_r, kc_dedup per-action consensus l*_a = l0(a) + u * sum beta_r / c_r dl_r
  MCTS          determinised PUCT over theta_real (K simulations, fresh Z per simulation via random order, depth 4,
                children = legal moves at the root and the top-6 policy moves below), most-visited root action
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

import model_ms as M
import msgame as G
from env import MicroStrategoEnv as E

C_PUCT = 1.0


def batch_rec(states, mcounts, dev):
    """Build a record dict of tensors (same fields as data/*.npz) for a list of live states (observer = mover)."""
    cols = {k: [] for k in ("sq_content", "sq_ownflag", "sq_slot", "slot_loc", "slot_flags", "slot_mcount", "slot_pubrev",
                            "slot_true", "turn", "legal")}
    env_legal = []
    for st, mc in zip(states, mcounts):
        r = G.encode_root(st, mc)
        for k in cols:
            if k != "legal":
                cols[k].append(r[k])
        P = st.turn % 2
        acts = E.legal_actions(st)
        lg = np.full((14, 2), -1, np.int64)
        for i, (s, nr, nc) in enumerate(acts[:14]):
            lg[i] = (G.canon_sq(int(st.pos[P, s, 0]), int(st.pos[P, s, 1]), P), G.canon_sq(nr, nc, P))
        cols["legal"].append(lg)
        env_legal.append(acts[:14])
    R = {k: torch.tensor(np.stack(v).astype(np.int64), device=dev) for k, v in cols.items()}
    # the opponent's true types must never reach an agent: blank them (only used for revealed slots via pubrev)
    R["slot_true"] = torch.where(R["slot_pubrev"].bool(), R["slot_true"], torch.zeros_like(R["slot_true"]))
    return R, env_legal


class Ctx:
    def __init__(self, net, amp, strust, rtrust, scales, dev, seed):
        self.net, self.amp, self.strust, self.rtrust = net, amp, strust, rtrust
        self.eta = scales["mmd"]["scale"]
        self.u = scales["kc"]["scale"]
        self.dev = dev
        self.gen = torch.Generator(device=dev).manual_seed(seed)


def legal_l0(ctx, R, roots):
    lg = R["legal"][roots]
    n = (lg[..., 0] >= 0).sum(1)
    B = roots.shape[0]
    tp0 = M.root_tape(R, roots)
    l0 = torch.full((B, 14), -1e4, device=roots.device)
    for j in range(14):
        m = n > j
        if m.any():
            lj, _ = M.action_l0(ctx.net, ctx.amp, M.tape_index(tp0, m), lg[m, j, 0], lg[m, j, 1])
            l0[m, j] = lj
    return l0, lg, n, tp0


@torch.no_grad()
def run_rollouts(ctx, tp0, lg, act, order, depth=4, trust=True):
    """act [B,k] legal-slot indices -> per rollout q_eff, l (logit), beta, key, miner_u (flattened [B*k])."""
    B, k = act.shape
    ar = torch.arange(B, device=act.device)[:, None]
    fr = lg[ar, act, 0].reshape(-1)
    to = lg[ar, act, 1].reshape(-1)
    tp = M.tape_repeat(tp0, k)
    tpz, info = M.sample_Z(ctx.net, tp, order, ctx.gen, ctx.strust)
    r = M.rollout(ctx.net, tpz, fr, to, depth, ctx.gen)
    q = r["q_leaf"]
    qe = (q[:, 0] + 0.5 * q[:, 1]).clamp(1e-5, 1 - 1e-5)
    l = torch.log(qe) - torch.log1p(-qe)
    beta = torch.ones_like(qe)
    if trust:
        lr_ = M.path_logp(ctx.net, tpz, info["hidden"], info["step"])
        pdis = (info["logp"] - lr_).abs() / info["hidden"].sum(1).clamp_min(1).float()
        l0r, _ = M.action_l0(ctx.net, None, tp, fr, to)
        from train_ms import trust_beta
        beta = trust_beta(ctx.rtrust, M.trust_features(r, l0r, depth, pdis))
    key = ctx.amp.key(r["ebar"])
    return {"q": qe.view(B, k), "l": l.view(B, k), "beta": beta.view(B, k), "key": key.view(B, k, -1),
            "miner": r["miner_u"].view(B, k)}


def kc_consensus(l0, act, dl, beta, key, valid, u, gamma=2.0):
    """Per-action kc_dedup consensus with crowding among rollouts of the same action."""
    same = (act[:, :, None] == act[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    c = ((torch.relu(key @ key.transpose(1, 2)) ** gamma) * same.float()).sum(2).clamp_min(1e-6)
    w = (beta / c) * valid.float()
    contrib = torch.zeros_like(l0)
    contrib.scatter_add_(1, act, w * dl)
    neff = torch.zeros_like(l0)
    neff.scatter_add_(1, act, (1.0 / c) * valid.float())
    return l0 + u * contrib, neff


class Agent:
    name = "base"

    def __init__(self, ctx, K=16, utility="flag"):
        self.ctx, self.K, self.utility = ctx, K, utility

    def act(self, states, mcounts):
        R, env_legal = batch_rec(states, mcounts, self.ctx.dev)
        roots = torch.arange(len(states), device=self.ctx.dev)
        with torch.no_grad():
            choice = self.choose(R, roots)
        return [env_legal[i][int(choice[i])] for i in range(len(states))]


class P0(Agent):
    name = "P0"

    def choose(self, R, roots):
        l0, lg, n, _ = legal_l0(self.ctx, R, roots)
        return l0.argmax(1)


class Ataraxos1(Agent):
    name = "Ataraxos-1Step"

    def choose(self, R, roots):
        l0, lg, n, tp0 = legal_l0(self.ctx, R, roots)
        act = torch.arange(self.K, device=roots.device)[None] % n[:, None]
        r = run_rollouts(self.ctx, tp0, lg, act, "row_major", trust=False)
        dl = r["l"] - l0.gather(1, act)
        contrib = torch.zeros_like(l0)
        contrib.scatter_add_(1, act, dl)
        return (l0 + self.ctx.eta * contrib).argmax(1)


class Unified(Agent):
    name = "Unified-LIR"

    def choose(self, R, roots):
        ctx = self.ctx
        l0, lg, n, tp0 = legal_l0(ctx, R, roots)
        B = roots.shape[0]
        h, _ = ctx.net(tp0)
        pl = ctx.net.h_policy(h[:, M.GLOB]).float()
        idx = (lg[..., 0] * 16 + lg[..., 1]).clamp_min(0)
        prior = pl.gather(1, idx).masked_fill(lg[..., 0] < 0, -1e9)
        prior = torch.softmax(prior, 1)
        W = 4
        per = self.K // W
        acts, ls, betas, keys, miners = [], [], [], [], []
        lstar = l0.clone()
        neff = torch.zeros_like(l0)
        mu = torch.zeros_like(l0)
        for w in range(W):
            if w == 0:
                order = prior.argsort(1, descending=True)
                act = torch.gather(order, 1, (torch.arange(per, device=roots.device)[None] % n[:, None]))
            else:
                val = torch.sigmoid(lstar) if self.utility == "flag" else mu
                tot = neff.sum(1, keepdim=True)
                score = val + C_PUCT * prior * torch.sqrt(tot + 1) / (1 + neff)
                score = score.masked_fill(lg[..., 0] < 0, -1e9)
                order = score.argsort(1, descending=True)
                act = torch.gather(order, 1, (torch.arange(per, device=roots.device)[None] % n[:, None]))
            r = run_rollouts(ctx, tp0, lg, act, "stiff_voi", trust=True)
            acts.append(act)
            ls.append(r["l"])
            betas.append(r["beta"])
            keys.append(r["key"])
            miners.append(r["miner"])
            A = torch.cat(acts, 1)
            L = torch.cat(ls, 1)
            valid = torch.ones_like(A, dtype=torch.bool)
            lstar, neff = kc_consensus(l0, A, L - l0.gather(1, A), torch.cat(betas, 1), torch.cat(keys, 1), valid, ctx.u)
            if self.utility == "miner":
                Mu = torch.cat(miners, 1)
                s = torch.zeros_like(l0)
                c = torch.zeros_like(l0)
                s.scatter_add_(1, A, Mu)
                c.scatter_add_(1, A, torch.ones_like(Mu))
                mu = s / c.clamp_min(1)
        if self.utility == "flag":
            return lstar.masked_fill(lg[..., 0] < 0, -1e9).argmax(1)
        tie = 1e-3 * torch.sigmoid(l0)
        sc = mu + tie
        sc = sc.masked_fill(lg[..., 0] < 0, -1e9)
        return sc.argmax(1)


class MCTS(Agent):
    name = "Tabular-MCTS"

    def choose(self, R, roots):
        ctx = self.ctx
        net = ctx.net
        dev = roots.device
        B = roots.shape[0]
        l0, lg, n, tp0 = legal_l0(ctx, R, roots)
        h, _ = net(tp0)
        pl = net.h_policy(h[:, M.GLOB]).float()
        idx = (lg[..., 0] * 16 + lg[..., 1]).clamp_min(0)
        pr = torch.softmax(pl.gather(1, idx).masked_fill(lg[..., 0] < 0, -1e9), 1).cpu().numpy()
        lgn = lg.cpu().numpy()
        nn_ = n.cpu().numpy()
        # node: dict(children=[(fr,to)], P=np.array, N=np.array, W=np.array, kids=[node or None])
        trees = [{"children": [tuple(lgn[b, j]) for j in range(nn_[b])], "P": pr[b, :nn_[b]] / pr[b, :nn_[b]].sum(),
                  "N": np.zeros(nn_[b]), "W": np.zeros(nn_[b]), "kids": [None] * nn_[b]} for b in range(B)]
        ar = torch.arange(B, device=dev)
        for sim in range(self.K):
            tp = M.tape_clone(tp0)
            tp, _ = M.sample_Z(net, tp, "random", ctx.gen, None)
            paths = [[] for _ in range(B)]
            node = list(trees)
            leaf_done = np.zeros(B, bool)
            value = np.full(B, np.nan)
            n_ply = np.zeros(B, int)
            for j in range(4):
                fr = torch.zeros(B, dtype=torch.long, device=dev)
                to = torch.zeros(B, dtype=torch.long, device=dev)
                active = np.zeros(B, bool)
                for b in range(B):
                    if leaf_done[b]:
                        continue
                    nd = node[b]
                    if nd is None or len(nd["children"]) == 0:
                        leaf_done[b] = True
                        continue
                    tot = nd["N"].sum()
                    Q = np.where(nd["N"] > 0, nd["W"] / np.maximum(nd["N"], 1), 0.5)
                    if j % 2 == 1:
                        Q = 1.0 - Q
                    u = Q + C_PUCT * nd["P"] * math.sqrt(tot + 1) / (1 + nd["N"])
                    c = int(np.argmax(u))
                    paths[b].append((nd, c))
                    fr[b], to[b] = int(nd["children"][c][0]), int(nd["children"][c][1])
                    active[b] = True
                if not active.any():
                    break
                act_t = torch.tensor(active, device=dev)
                for k_, v_ in (("ply_actor", torch.full_like(fr, j % 2)), ("ply_from", fr), ("ply_to", to)):
                    t_ = tp[k_].clone()
                    t_[act_t, j] = v_[act_t]
                    tp[k_] = t_
                hh, _ = net(tp)
                lo, la, ld = net.trans_logits(hh[:, M.a_idx(j)])
                oc = torch.multinomial(torch.softmax(lo.float(), -1), 1, generator=ctx.gen).squeeze(1)
                at = torch.multinomial(torch.softmax(la.float(), -1), 1, generator=ctx.gen).squeeze(1)
                de = torch.multinomial(torch.softmax(ld.float(), -1), 1, generator=ctx.gen).squeeze(1)
                for k_, v_ in (("ply_out", oc), ("ply_att", at), ("ply_def", de)):
                    t_ = tp[k_].clone()
                    t_[act_t, j] = v_[act_t]
                    tp[k_] = t_
                hh, _ = net(tp)
                vq = torch.softmax(net.h_value(hh[:, M.o_idx(j)]).float(), -1)
                ve = (vq[:, 0] + 0.5 * vq[:, 1]).cpu().numpy()
                pol = net.h_policy(hh[:, M.o_idx(j)]).float()
                top = pol.topk(6, 1)
                tpi = top.indices.cpu().numpy()
                tpp = torch.softmax(top.values, 1).cpu().numpy()
                ocn = oc.cpu().numpy()
                turn = tp["turn"].cpu().numpy()
                for b in range(B):
                    if not active[b]:
                        continue
                    nd, c = paths[b][-1]
                    n_ply[b] = j + 1
                    if ocn[b] == 4:
                        value[b] = 1.0 if j % 2 == 0 else 0.0
                        leaf_done[b] = True
                        continue
                    if turn[b] + j + 1 >= 24:
                        value[b] = 0.5
                        leaf_done[b] = True
                        continue
                    if nd["kids"][c] is None:
                        nd["kids"][c] = {"children": [(int(m) // 16, int(m) % 16) for m in tpi[b]], "P": tpp[b].copy(),
                                         "N": np.zeros(6), "W": np.zeros(6), "kids": [None] * 6}
                        value[b] = ve[b]
                        leaf_done[b] = True
                    else:
                        node[b] = nd["kids"][c]
                        value[b] = ve[b]
            for b in range(B):
                v = value[b] if not np.isnan(value[b]) else 0.5
                for nd, c in paths[b]:
                    nd["N"][c] += 1
                    nd["W"][c] += v
        out = []
        for b in range(B):
            t = trees[b]
            q = np.where(t["N"] > 0, t["W"] / np.maximum(t["N"], 1), 0)
            out.append(int(np.lexsort((q, t["N"]))[-1]))
        return torch.tensor(out, device=dev)


class Heuristic:
    name = "Heuristic-tau1"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, states, mcounts):
        out = []
        for st in states:
            acts, p = G.policy_probs(st, 1.0)
            out.append(acts[int(self.rng.choice(len(acts), p=p))])
        return out


def play_match(A, B_, n_games, setups, seed, max_batch=400):
    """A plays Red in even games, Blue in odd games. Returns per game: effective result for A, A's miner captures (did A
    capture the opponent Miner), A lost own Miner, length."""
    rng = np.random.default_rng(seed)
    res = np.zeros(n_games)
    cap = np.zeros(n_games, bool)
    lost = np.zeros(n_games, bool)
    length = np.zeros(n_games, int)
    for g0 in range(0, n_games, max_batch):
        gs = list(range(g0, min(n_games, g0 + max_batch)))
        sts = [E.new_game(int(rng.choice(setups)), int(rng.choice(setups))) for _ in gs]
        mcs = [np.zeros((2, 6), np.int64) for _ in gs]
        a_side = [0 if g % 2 == 0 else 1 for g in gs]
        while True:
            live = [i for i, st in enumerate(sts) if st.winner is None]
            if not live:
                break
            for agent, mine in ((A, True), (B_, False)):
                sel = [i for i in live if sts[i].winner is None and (sts[i].turn % 2 == a_side[i]) == mine]
                if not sel:
                    continue
                acts = agent.act([sts[i] for i in sel], [mcs[i] for i in sel])
                for i, a in zip(sel, acts):
                    st = sts[i]
                    X = st.turn % 2
                    oc, at, de = G.outcome_code(st, None, a)
                    if X == a_side[i]:
                        if oc in (1, 3, 4) and de == G.MINER:
                            cap[gs[i]] = True
                        if oc in (2, 3) and at == G.MINER:
                            lost[gs[i]] = True
                    else:
                        if oc in (1, 3, 4) and de == G.MINER:
                            lost[gs[i]] = True
                        if oc in (2, 3) and at == G.MINER:
                            cap[gs[i]] = True
                    mcs[i][X, a[0]] += 1
                    sts[i] = E.step(st, a)
        for i, g in enumerate(gs):
            w = sts[i].winner
            res[g] = 0.5 if w == -1 else (1.0 if w == a_side[i] else 0.0)
            length[g] = sts[i].turn
    return res, cap, lost, length
