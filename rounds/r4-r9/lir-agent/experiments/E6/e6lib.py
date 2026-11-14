"""E6 (reshaped, hawks/E6.md): Gao-2022-style gold-vs-proxy on the agent's own adapter features.

gold  = ShipHunt env_reward of an environment-written outcome (grounded utility)
proxy = linear head on mean-pooled final hidden states of the outcome text, read through the
        agent's forecast adapter (roles layout) or its fused adapter (B0), trained prequentially on
        realised env_reward of committed episodes. Arm a: same head on the frozen base (adapters off).
Nothing here modifies harness code; it imports it.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Sequence

import numpy as np
import torch

from env import ShipHunt, PRESETS
from protocol.render import Seq, _context, LETTERS
from protocol.rollout import pad_batch, belief_pass
from protocol.render import belief_seq, proposal_seq
from protocol import losses as LS


# ---------------------------------------------------------------- bank (environment-written outcomes)
def _policies(env: ShipHunt, rng: np.random.Generator, n_per: int):
    """Plans from many policies: uniform, Bayes-greedy with preference noise of several strengths,
    Bayes-sampled, anti-Bayes. Returns list of plans."""
    nc, L = env.cfg.n_cells, env.cfg.plan_len
    p = env.bayes_cell_probs()
    plans = []
    for _ in range(n_per):
        plans.append([int(c) for c in rng.permutation(nc)[:L]])                       # uniform
        s = p + rng.normal(scale=rng.choice([0.05, 0.15, 0.4]), size=nc)
        plans.append([int(c) for c in np.argsort(-s)[:L]])                            # noisy greedy
        w = np.exp(4 * p); w /= w.sum()
        plans.append([int(c) for c in rng.choice(nc, size=L, replace=False, p=w)])    # Bayes-sampled
        s = -p + rng.normal(scale=0.1, size=nc)
        plans.append([int(c) for c in np.argsort(-s)[:L]])                            # anti-Bayes
    return plans


def build_bank(preset: str, board_seeds: Sequence[int], per_board: int, seed: int) -> List[Dict]:
    cfg = PRESETS[preset]
    rng = np.random.default_rng(seed)
    bank = []
    for bs in board_seeds:
        env = ShipHunt(cfg, int(bs))
        outs = {}
        for plan in _policies(env, rng, 30):
            o = env.snapshot().execute(plan)
            key = tuple((c, bool(h)) for c, h in o.shots)
            outs.setdefault(key, o)
        # stratify by gold utility so every level is present, then fill at random
        by_u: Dict[float, list] = {}
        for k, o in outs.items():
            by_u.setdefault(round(o.env_reward, 3), []).append(k)
        keys = []
        levels = sorted(by_u)
        for u in levels:
            ks = by_u[u]
            rng.shuffle(ks)
        while len(keys) < per_board and any(by_u[u] for u in levels):
            for u in levels:
                if by_u[u] and len(keys) < per_board:
                    keys.append(by_u[u].pop())
        for k in keys:
            o = outs[k]
            bank.append(dict(board=int(bs), fired=[[int(c), bool(h)] for c, h in k],
                             u=round(float(o.env_reward), 3), success=bool(o.success)))
    return bank


def bank_hash(bank) -> str:
    return hashlib.sha256(json.dumps(bank, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- outcome text (env-written template)
def outcome_seq(V, env: ShipHunt, fired) -> Seq:
    """Environment-written outcome record: board context + fired shots. Same template for every
    policy, so surface form carries no policy identity. Returns Seq with q['o_start'] = first token
    of the outcome segment."""
    s = Seq(V)
    _context(s, env, fired=False)
    s.q["o_start"] = len(s.ids)
    s.text("\nShots fired:")
    for c, h in fired:
        s.tok(V.cells[c]).text(" hit;" if h else " miss;")
    s.text(" end.")
    return s


@torch.no_grad()
def features_and_ll(model, V, seqs: List[Seq], role: str, base_only: bool = False, mb: int = 16,
                    want_ll: bool = True):
    """Mean-pooled final hidden state over the outcome segment (proxy features) and the agent's
    log-likelihood of the outcome segment tokens (familiarity covariate)."""
    feats, lls = [], []
    for i in range(0, len(seqs), mb):
        ch = seqs[i:i + mb]
        ids, attn = pad_batch([s.ids for s in ch], V.pad)
        if base_only:
            with model.peft.disable_adapter():
                lg = model.logits(role, ids, attn)
        else:
            lg = model.logits(role, ids, attn)
        h = lg.h.float()
        W = lg.W.float()
        for b, s in enumerate(ch):
            a, e = s.q["o_start"], len(s.ids)
            feats.append(h[b, a:e].mean(0).cpu().numpy())
            if want_ll:
                logits = h[b, a - 1:e - 1] @ W.T
                lp = torch.log_softmax(logits, -1)
                tgt = torch.tensor(s.ids[a:e], device=lp.device)
                lls.append(float(lp.gather(1, tgt[:, None]).sum()))
    return np.stack(feats), (np.array(lls) if want_ll else None)


# ---------------------------------------------------------------- proxy head
class ProxyHead:
    """Linear head on standardised features. init: ridge fit on a calibration set of
    environment-written outcomes (disjoint boards). online: SGD on MSE vs realised env_reward."""

    def __init__(self, X0: np.ndarray, y0: np.ndarray, lam: float = 10.0, lr: float = 0.5):
        self.mu = X0.mean(0)
        sd = X0.std(0)
        self.sd = np.maximum(sd, np.median(sd)) + 1e-6      # floor: no near-constant dim blows up later
        Z = (X0 - self.mu) / self.sd
        d = Z.shape[1]
        self.b = float(y0.mean())
        self.w = np.linalg.solve(Z.T @ Z + lam * np.eye(d), Z.T @ (y0 - self.b))
        self.lr = lr

    def score(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.b

    def update(self, X, y):
        """Normalised LMS (stable for 0 < lr < 2 whatever the feature scale)."""
        Z = (X - self.mu) / self.sd
        err = Z @ self.w + self.b - y
        self.w -= self.lr * (Z.T @ err) / (float((Z ** 2).sum(1).mean()) + 1.0) / len(y)
        self.b -= self.lr * float(err.mean())
        return float((err ** 2).mean())

    def state(self):
        return {"w": self.w.tolist(), "b": self.b}


# ---------------------------------------------------------------- policy support of bank outcomes
@torch.no_grad()
def support_logp(model, V, bank, cfg, mb: int = 16):
    """log P(current policy produces this outcome) = log P(plan prefix = fired cells), with the
    situation report at its mode (belief pass), proposal pass under the action adapter."""
    boards = sorted({it["board"] for it in bank})
    envs = {b: ShipHunt(cfg, b) for b in boards}
    nc = cfg.n_cells
    bseqs = [belief_seq(V, envs[b], "deleted", None, self_query=False) for b in boards]
    sit = {}
    for i in range(0, len(boards), mb):
        cl, _ = belief_pass(model, V, bseqs[i:i + mb], nc, None)
        for j, b in enumerate(boards[i:i + mb]):
            sit[b] = cl[j].argmax(-1).tolist()
    p_ids = {b: proposal_seq(V, envs[b], sit[b]).ids for b in boards}
    recs = [{"p_ids": p_ids[it["board"]], "plans": [[c for c, _ in it["fired"]]], "k": 0} for it in bank]
    out = []
    for i in range(0, len(recs), mb):
        out += LS.plan_logp(model, V, recs[i:i + mb], nc=nc).float().cpu().tolist()
    return np.array(out)


def adapter_norms(model, ref: Dict[str, List[torch.Tensor]]):
    out = {}
    for g, old in ref.items():
        new = model.group_params(g)
        out[g] = float(sum((a.float() - b.detach().float().cpu()).pow(2).sum() for a, b in zip(old, new)) ** 0.5)
    return out
