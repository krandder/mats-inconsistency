"""E6 driver. Imports the harness unchanged (no code fork); adds stream logging, checkpoints,
the online proxy head and the bank evaluation around Trainer.

  train : arm P (=c, masked LIR) or B0 (=d, fused); saves every RL step's episode stream
  replay: yoked truth-only belief learning b_X: replays X's stream, runs only the harness truth phases
          (belief + forecast, Trainer._truth_phases) from X's warm checkpoint; no reward phase.
Both update the proxy head on realised env_reward of the stream's committed outcomes and evaluate
the frozen bank every --eval-every steps. Arm a (frozen backbone) = the same head trained on the same
stream on base features (adapters disabled), run inside `train`.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol.train as PT
from protocol.train import Trainer, RunConfig
from env import ShipHunt, PRESETS
import e6lib as E

KEEP = ("seed", "goal", "need_hits", "K", "group", "occ", "note_cell", "b_ids", "b_q", "b_span", "sit",
        "p_ids", "plans", "fc", "c_ids", "c_q", "k", "prop", "outcomes", "env_reward", "success", "reward",
        "agree", "conf", "adv")
_CAP = {"on": False, "recs": None}
_orig_rollout = PT.rollout


def _capturing(*a, **kw):
    recs = _orig_rollout(*a, **kw)
    if _CAP["on"]:
        _CAP["recs"] = recs
    return recs


PT.rollout = _capturing


def strip(recs):
    out = []
    for r in recs:
        d = {k: r[k] for k in KEEP if k in r}
        d["fc"] = [{k: f[k] for k in ("ids", "q", "stated_succ", "stated_steps", "stated_hits")} for f in r["fc"]]
        out.append(d)
    return out


class E6:
    def __init__(self, a, tr: Trainer):
        self.a, self.tr = a, tr
        self.cfg = tr.ecfg
        bank = json.load(open(a.bank))
        self.bank, self.calib = bank["bank"], bank["calib"]
        self.role = "forecast"          # fused: every pass uses the fused adapter anyway
        V = tr.V
        self.bank_seqs = [E.outcome_seq(V, ShipHunt(self.cfg, it["board"]), it["fired"]) for it in self.bank]
        self.cal_seqs = [E.outcome_seq(V, ShipHunt(self.cfg, it["board"]), it["fired"]) for it in self.calib]
        self.cal_y = np.array([it["u"] for it in self.calib])
        self.ref = {g: [p.detach().float().cpu().clone() for p in tr.model.group_params(g)] for g in tr.model.groups()}
        Xc, _ = E.features_and_ll(tr.model, V, self.cal_seqs, self.role, want_ll=False)
        self.head = E.ProxyHead(Xc, self.cal_y, lr=a.head_lr)
        self.base_head = None
        if a.mode == "train":
            Xb, _ = E.features_and_ll(tr.model, V, self.cal_seqs, self.role, base_only=True, want_ll=False)
            self.base_head = E.ProxyHead(Xb, self.cal_y, lr=a.head_lr)
            self.bank_base_X, _ = E.features_and_ll(tr.model, V, self.bank_seqs, self.role, base_only=True, want_ll=False)
        self.f = open(os.path.join(a.out, "e6eval.jsonl"), "a")
        self.hf = open(os.path.join(a.out, "e6head.jsonl"), "a")

    def head_step(self, step, recs):
        V = self.tr.V
        seqs, y = [], []
        for r in recs:
            o = r["outcomes"][r["k"]]
            seqs.append(E.outcome_seq(V, ShipHunt(self.cfg, r["seed"]), o["shots"]))
            y.append(o["env_reward"])
        y = np.array(y)
        X, _ = E.features_and_ll(self.tr.model, V, seqs, self.role, want_ll=False)
        row = {"step": step, "preq_mse": float(((self.head.score(X) - y) ** 2).mean())}
        row["train_mse"] = self.head.update(X, y)
        if self.base_head is not None:
            Xb, _ = E.features_and_ll(self.tr.model, V, seqs, self.role, base_only=True, want_ll=False)
            row["a_preq_mse"] = float(((self.base_head.score(Xb) - y) ** 2).mean())
            self.base_head.update(Xb, y)
        self.hf.write(json.dumps(row) + "\n"); self.hf.flush()

    def evaluate(self, step):
        t0 = time.time()
        tr, V = self.tr, self.tr.V
        X, ll = E.features_and_ll(tr.model, V, self.bank_seqs, self.role)
        row = {"step": step, "score": self.head.score(X).tolist(), "ll": ll.tolist(),
               "norms": E.adapter_norms(tr.model, self.ref), "head": {"b": self.head.b}}
        if self.base_head is not None:
            row["a_score"] = self.base_head.score(self.bank_base_X).tolist()
        if self.a.mode == "train":
            row["support_logp"] = E.support_logp(tr.model, V, self.bank, self.cfg).tolist()
        row["t_eval"] = time.time() - t0
        self.f.write(json.dumps(row) + "\n"); self.f.flush()
        tr.save(f"e6_step{step}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "replay"], required=True)
    ap.add_argument("--arm", required=True)            # P or B0
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--preset", default="easy")
    ap.add_argument("--sft-steps", type=int, default=8)
    ap.add_argument("--sft-batch", type=int, default=8)
    ap.add_argument("--rl-steps", type=int, default=32)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="lr for every adapter (harness default 2e-4)")
    ap.add_argument("--head-lr", type=float, default=0.5)
    ap.add_argument("--bank", default=os.path.join(HERE, "bank.json"))
    ap.add_argument("--stream", default=None, help="replay: X's stream dir")
    ap.add_argument("--warm", default=None, help="replay: X's warm checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default="/tmp/lir-agent/e6/ckpt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rc = RunConfig(arm=a.arm, model=a.model, preset=a.preset, seed=a.seed, threads=a.threads,
                   sft_steps=a.sft_steps if a.mode == "train" else 0, sft_batch=a.sft_batch,
                   rl_steps=a.rl_steps, groups=a.groups, group_size=a.group_size, eval_every=0,
                   out=a.out, ckpt_dir=a.ckpt_dir, init_from=a.warm, micro_batch=a.micro_batch,
                   lr_action=a.lr, lr_belief=a.lr, lr_forecast=a.lr)
    tr = Trainer(rc)
    t0 = time.time()
    if a.mode == "train":
        for i in range(rc.sft_steps):
            tr.sft_step(i)
        warm = tr.save("warm")
        torch.save({g: o.state_dict() for g, o in tr.opts.items()}, warm[:-3] + "_opt.pt")
    else:
        tr.model.load(a.warm)
        warm = a.warm
        ost = torch.load(a.warm[:-3] + "_opt.pt")      # same optimiser state as X at the warm point
        for g, o in tr.opts.items():
            o.load_state_dict(ost[g])
    e6 = E6(a, tr)
    e6.evaluate(0)
    sdir = os.path.join(a.out, "stream") if a.mode == "train" else None
    if sdir:
        os.makedirs(sdir, exist_ok=True)
    for s in range(1, a.rl_steps + 1):
        if a.mode == "train":
            _CAP["on"] = True
            tr.rl_step()
            _CAP["on"] = False
            recs = _CAP["recs"]
            with open(os.path.join(sdir, f"step{s:04d}.pkl"), "wb") as f:
                pickle.dump(strip(recs), f)
        else:
            recs = pickle.load(open(os.path.join(a.stream, f"step{s:04d}.pkl"), "rb"))
            tr.step += 1
            st = {"phase": "replay", "step": tr.step}
            st.update(tr._truth_phases(recs, tr.rc.proto))
            tr._log(st)
        e6.head_step(s, recs)
        if s % a.eval_every == 0 or s == a.rl_steps:
            e6.evaluate(s)
    tr.check_base()
    summ = {"mode": a.mode, "arm": a.arm, "seed": a.seed, "mask_violations": tr.mask_violations,
            "base_checksum_unchanged": True, "warm_ckpt": warm, "wall_s": time.time() - t0,
            "rl_steps": a.rl_steps, "bank": a.bank, "stream": a.stream or sdir}
    json.dump(summ, open(os.path.join(a.out, "e6summary.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
