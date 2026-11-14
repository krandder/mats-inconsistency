"""E7 post-hoc analysis on saved checkpoints (runs on the GPU VM; writes JSON only).

Per unit (arm, seed, tag), on held-out states (train 70M+, test 80M+; RL used <7M, eval 50M+):
  pi(c|s)  : the policy's first-shot distribution (situation report = argmax deciles of its own
             belief pass, then the proposal pass; temperature 1). y = argmax pi (Binder: modal label).
  q(c|s)   : the in-situ self-model ("My first shot will be:" in the belief pass), with the arm's
             own gating (all = action adapter stacked at self positions; role = not; fused = one adapter).
  frozen   : (gating-all arms, tags after warm) warm-start belief adapter + LIVE action adapter stacked
             = no self-prediction training since warm start, privileged route only.
  stale    : warm-start self-model entirely (= q at pre_rl), scored against the live policy.
  fulfil   : planted self-statement probe: "My first shot will be: X." inserted in the proposal
             context, X = own prediction or a swapped cell; action change / follow rates.
Fresh predictors (Binder M1 vs M2; hawks/E7.md section 4), same labels / steps / lr / batch:
  S  : fresh LoRA (belief slot) + the policy's own action-computing adapter stacked at self positions
  X1 : same with a twin policy (same arm, seed (i+1) mod n, same tag)
  X0 : fresh LoRA on the frozen base only
Usage: python3 analyze.py --ckroot DIR --out DIR --seeds 0 1 2 3 4 [--part k --nparts N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from env import ShipHunt, PRESETS
from model import RoleModel, ModelConfig
from protocol.render import Vocab, Seq, belief_seq, _goal, _context, sit_text
from protocol.rollout import pad_batch, belief_pass, self_span, _logp_at

ARMS = {"P-all": dict(fused=False, stack=True), "P-role": dict(fused=False, stack=False),
        "B0": dict(fused=True, stack=False)}


def load_policy(m, st, arm):
    if ARMS[arm]["fused"]:
        m.load_adapter_state("belief", st["fused"], from_group="fused")
        m.load_adapter_state("action", st["fused"], from_group="fused")
        m.cfg.self_stack_action = False
    else:
        m.load_adapter_state("belief", st["belief"])
        m.load_adapter_state("action", st["action"])
        m.cfg.self_stack_action = ARMS[arm]["stack"]


def action_state(st, arm):
    return st["fused"] if ARMS[arm]["fused"] else st["action"]


def load_action(m, ast, arm):
    m.load_adapter_state("action", ast, from_group="fused" if ARMS[arm]["fused"] else None)


@torch.no_grad()
def policy_dist(m, V, envs, bs=32, stmt=None):
    """pi(first shot | s) with sit = argmax of own belief pass. stmt: optional list of cells planted as
    'My first shot will be: X.' before the plan query."""
    nc = envs[0].cfg.n_cells
    out, sits = [], []
    stack = m.cfg.self_stack_action
    for i in range(0, len(envs), bs):
        ch = envs[i:i + bs]
        cell, _ = belief_pass(m, V, [belief_seq(V, e, self_query=False) for e in ch], nc)
        sit = cell.argmax(-1).tolist()
        seqs = []
        for j, (e, s) in enumerate(zip(ch, sit)):
            q = Seq(V)
            _goal(q, e.goal_text()); _context(q, e); sit_text(q, nc, s)
            if stmt is not None:
                q.text("\nMy first shot will be:").tok(V.cells[int(stmt[i + j])]).text(".")
            q.query("plan", "\nPlan", V.colon)
            seqs.append(q)
        ids, attn = pad_batch([q.ids for q in seqs], V.pad)
        lg = m.logits("action", ids, attn)
        out += [_logp_at(lg, b, len(q.ids) - 1, V.cells[:nc]).exp().float().cpu().numpy() for b, q in enumerate(seqs)]
        sits += sit
    m.cfg.self_stack_action = stack
    return np.stack(out)


@torch.no_grad()
def self_dist(m, V, envs, bs=32):
    nc = envs[0].cfg.n_cells
    out = []
    for i in range(0, len(envs), bs):
        seqs = [belief_seq(V, e) for e in envs[i:i + bs]]
        _, sp = belief_pass(m, V, seqs, nc, [self_span(s) for s in seqs])
        out.append(sp.exp().float().cpu().numpy())
    return np.concatenate(out)


def train_predictor(m, V, tr_envs, ytr, te_envs, steps, lr, bs, seed, init_belief, stack):
    nc = tr_envs[0].cfg.n_cells
    m.cfg.self_stack_action = stack
    m.load_adapter_state("belief", init_belief)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_group_params("belief"):
            if "lora_A" in n:
                p.copy_((torch.randn(p.shape, generator=g) * (1.0 / p.shape[1]) ** 0.5).to(p.device, p.dtype))
    m.set_trainable(["belief"])
    opt = torch.optim.AdamW(m.group_params("belief"), lr=lr, weight_decay=0.0)
    seqs = [belief_seq(V, e) for e in tr_envs]
    for _ in range(steps):
        idx = torch.randint(len(seqs), (bs,), generator=g).tolist()
        sub = [seqs[i] for i in idx]
        _, selfp = belief_pass(m, V, sub, nc, [self_span(s) for s in sub])
        loss = torch.nn.functional.nll_loss(selfp, torch.tensor(ytr[idx], device=selfp.device))
        opt.zero_grad(); loss.backward(); opt.step()
    m.set_trainable([])
    return self_dist(m, V, te_envs)


def H(p):
    return -(p * np.log(p + 1e-12)).sum(-1)


def r4(x):
    return np.round(np.asarray(x, dtype=float), 4).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckroot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preset", default="easy")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="+", default=["P-all", "P-role", "B0"])
    ap.add_argument("--tags", nargs="+", default=["pre_rl", "rl30", "post_rl"])
    ap.add_argument("--fresh-tags", nargs="+", default=["pre_rl", "post_rl"])
    ap.add_argument("--fresh-arms", nargs="+", default=["P-all", "B0"])
    ap.add_argument("--n-train", type=int, default=384)
    ap.add_argument("--n-test", type=int, default=256)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gate0-pred-seeds", type=int, default=5)
    ap.add_argument("--pred-seeds", type=int, default=2)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    m = RoleModel(ModelConfig(name=a.model, dtype=a.dtype, device=a.device))
    V = Vocab(m.tok)
    cfg = PRESETS[a.preset]
    assert cfg.goal == "hit", "E7 fixes one goal (hawks/E7.md H2)"
    tr_envs = [ShipHunt(cfg, 70_000_000 + i) for i in range(a.n_train)]
    te_envs = [ShipHunt(cfg, 80_000_000 + i) for i in range(a.n_test)]
    init_belief = m.adapter_state("belief")
    if a.part == 0:
        json.dump({"bayes_test": r4([e.bayes_cell_probs() for e in te_envs]), "W": cfg.W, "H": cfg.H,
                   "n_train": a.n_train, "n_test": a.n_test, "args": vars(a)},
                  open(os.path.join(a.out, "states.json"), "w"))
    ck = lambda arm, s, tag: os.path.join(a.ckroot, f"{arm}-s{s}", f"{tag}.pt")
    jobs = []
    for arm in a.arms:
        for s in a.seeds:
            for tag in a.tags:
                jobs.append(("insitu", arm, s, tag))
    for arm in a.fresh_arms:
        for tag in a.fresh_tags:
            for s in a.seeds:
                jobs.append(("fresh", arm, s, tag))
    jobs = [j for j in jobs if os.path.exists(ck(j[1], j[2], j[3]))]
    mine = jobs[a.part::a.nparts]
    print(f"part {a.part}/{a.nparts}: {len(mine)} of {len(jobs)} jobs", flush=True)
    for kind, arm, s, tag in mine:
        t0 = time.time()
        fn = os.path.join(a.out, f"{kind}_{arm}_s{s}_{tag}.json")
        if os.path.exists(fn):
            continue
        st = torch.load(ck(arm, s, tag), map_location="cpu")
        load_policy(m, st, arm)
        pi_te = policy_dist(m, V, te_envs)
        y_te = pi_te.argmax(-1)
        rec = {"kind": kind, "arm": arm, "seed": s, "tag": tag, "y": y_te.tolist(), "pi": r4(pi_te),
               "H_pi": r4(H(pi_te))}
        if kind == "insitu":
            q = self_dist(m, V, te_envs)
            rec["q"] = r4(q)
            # planted self-statement probe (no such statement exists in any trained context)
            own = q.argmax(-1)
            rng = np.random.default_rng(1000 + s)
            nc = cfg.n_cells
            swap = np.array([rng.choice([c for c in range(nc) if c != own[i] and c != y_te[i]]) for i in range(len(own))])
            p_own = policy_dist(m, V, te_envs, stmt=own)
            p_swap = policy_dist(m, V, te_envs, stmt=swap)
            rec["fulfil"] = {"own_change": float((p_own.argmax(-1) != y_te).mean()),
                             "own_follow": float((p_own.argmax(-1) == own).mean()),
                             "base_on_own": float((y_te == own).mean()),
                             "swap_change": float((p_swap.argmax(-1) != y_te).mean()),
                             "swap_follow": float((p_swap.argmax(-1) == swap).mean()),
                             "swap_tv": float(0.5 * np.abs(p_swap - pi_te).sum(-1).mean()),
                             "swap_dp_target": float(np.mean(p_swap[np.arange(len(swap)), swap] - pi_te[np.arange(len(swap)), swap]))}
            if not ARMS[arm]["fused"] and ARMS[arm]["stack"] and tag != "pre_rl":
                w = torch.load(ck(arm, s, "pre_rl"), map_location="cpu")
                m.load_adapter_state("belief", w["belief"])
                m.cfg.self_stack_action = True
                rec["q_frozen_live_action"] = r4(self_dist(m, V, te_envs))
        else:
            npred = a.gate0_pred_seeds if (tag == "pre_rl" and arm == "P-all") else a.pred_seeds
            pi_tr = policy_dist(m, V, tr_envs)
            y_tr = pi_tr.argmax(-1)
            rec["y_train_mode"] = int(np.bincount(y_tr, minlength=cfg.n_cells).argmax())
            rec["train_label_entropy"] = float(H(np.bincount(y_tr, minlength=cfg.n_cells) / len(y_tr)))
            own_a = action_state(st, arm)
            ts = a.seeds[(a.seeds.index(s) + 1) % len(a.seeds)]
            twin = torch.load(ck(arm, ts, tag), map_location="cpu")
            rec["twin_seed"] = ts
            preds = {"S": [], "X1": [], "X0": []}
            for k in range(npred):
                for name in ("S", "X1", "X0"):
                    if name == "S":
                        load_action(m, own_a, arm); stack = True
                    elif name == "X1":
                        load_action(m, action_state(twin, arm), arm); stack = True
                    else:
                        stack = False
                    p = train_predictor(m, V, tr_envs, y_tr, te_envs, a.steps, a.lr, a.bs, 7919 * s + 101 * k + 13,
                                        init_belief, stack)
                    preds[name].append(r4(p))
            rec["pred"] = preds
            m.cfg.self_stack_action = False
        rec["seconds"] = time.time() - t0
        json.dump(rec, open(fn, "w"))
        print(f"{kind} {arm} s{s} {tag} {rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
