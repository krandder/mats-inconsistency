#!/usr/bin/env python3
"""Swarm predictor runner: chains -> claims -> pool -> settlement -> sampler update (PROPOSAL.md).

Modes
  gate0  S0 chains on Gate-0 boards (n = 16, L in --gate-L, R = 1), then a few timed training steps.
         Writes gate0.json (parse rate, informativeness per claim, pool Brier vs Bayes, decode timing).
  train  trains the sampler: S1 settlement (GRPO), S2 own log-score gain (GRPO), S3 trajectory balance.
         Small evaluations (n = 16, L = 128, R = 1) at step 0, every --small-every steps and at the end.
  eval   the full evaluation grid for one sampler checkpoint (S0 = adapter at zero).
Reuses the harness unchanged: model/roles.py (RoleModel: frozen base + one LoRA adapter "chain"),
env/shiphunt.py (boards, exact Bayes). Chains are sampled from a merged bf16 copy of base + LoRA
(refreshed before each sampling phase) at temperature --temp with no top-k / top-p truncation, so
the sampled distribution is the tempered policy the losses score.
Outputs: --out gets compact JSON/JSONL (claims, payments, timings; synced to /srv); --text-out gets
chain texts (JSONL; kept off /srv). Checkpoints: --ckpt-out (LoRA adapter only).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pool import Board, Claims, fit_stakes, logit, price  # noqa: E402  (puts the harness on sys.path)
from settle import informativeness, settle  # noqa: E402
from refocus import (K_CLAIMS, K_TARGETS, assign, make_board, open_mask, residual_targets,  # noqa: E402
                     round1_targets, roundrobin_targets)
from env.shiphunt import LETTERS, PRESETS  # noqa: E402

SEEDS = {"dev": 700_000, "gate0": 800_000, "small": 850_000, "test": 900_000, "train": 10_000_000}
LINE = re.compile(r"S(\d{1,2})\W{0,6}?p\s*=\s*[A-Za-z*]{0,3}\s*([0-9]*\.?[0-9]+)\W{0,8}?conf(?:idence)?\s*=\s*[A-Za-z*]{0,3}\s*"
                  r"([0-9]*\.?[0-9]+)", re.I)
PROMPT = {"style": "A"}   # A: one numeric example line; B: "Answer:" header + two example lines with varied values


# ---- prompts and parsing ---------------------------------------------------------------------------
def ships_text(cfg) -> str:
    words = [f"one of length {L}" for L in cfg.ships]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


ANSWER_TOKENS = 16      # generation allowance per answer line ("S1: p=0.70 conf=0.8"; 6 lines = 89 Qwen3 tokens)


def words_for(L: int) -> int:
    """Reasoning budget stated in the prompt: about 0.75 words per token."""
    return int(round(0.75 * L))


def chain_prompt(env, board: Board, stmts: List[int], words: Optional[int] = None) -> str:
    cfg = env.cfg
    rows = " / ".join(" ".join(LETTERS[r * cfg.W + c] for c in range(cfg.W)) for r in range(cfg.H))
    probes = "; ".join(f"{LETTERS[c]} {'hit' if h else 'miss'}" for c, h in env.probes)
    shots = "; ".join(f"{LETTERS[c]} {'hit' if h else 'miss'}" for c, h in env.fired) or "none"
    lines = "\n".join(f"S{k + 1}: {board.text(j)}" for k, j in enumerate(stmts))
    return (f"Battleship, {cfg.W} columns x {cfg.H} rows, cells named row by row: {rows}.\n"
            f"Ships: {ships_text(cfg)}, each straight (horizontal or vertical), not overlapping. "
            f"Every layout that follows these rules was equally likely, and {sum(cfg.ships)} of the {cfg.n_cells} cells hold a ship.\n"
            f"Probe reports (each is wrong with probability {cfg.eps}): {probes}.\n"
            f"Shots fired (always correct): {shots}.\n"
            f"Statements:\n{lines}\n" + answer_instructions(words))


def answer_instructions(words: Optional[int]) -> str:
    budget = f"Reason briefly, in at most {words} words" if words else "Reason briefly"
    if PROMPT["style"] == "B":
        return (f"{budget}. Then write the line Answer: and under it exactly one line per statement, "
                f"for example (use your own numbers):\n"
                f"S1: p=0.15 conf=0.6\n"
                f"S2: p=0.80 conf=0.9\n"
                f"p is your probability (0 to 1) that the statement is true; conf (0 to 1) is how sure you are of p.")
    return (f"{budget}, then end with one line per statement, exactly like\n"
            f"S1: p=0.70 conf=0.8\n"
            f"where p is your probability that the statement is true and conf (0 to 1) is how sure you are of p.")


def parse_claims(text: str, k: int):
    """All k lines 'Sj: p=.. conf=..' (last occurrence wins) -> [(p, conf)] * k, else None."""
    found = {}
    for m in LINE.finditer(text):
        i = int(m.group(1))
        if 1 <= i <= k:
            found[i] = (float(m.group(2)), float(m.group(3)))
    if len(found) < k:
        return None
    return [(float(np.clip(found[i][0], 0.01, 0.99)), float(np.clip(found[i][1], 0.05, 1.0))) for i in range(1, k + 1)]


# ---- models -----------------------------------------------------------------------------------------
def prealloc_cache(gen, cap: int):
    """A DynamicCache whose layers write into preallocated buffers and expose views (no torch.cat per step)."""
    from transformers.cache_utils import DynamicCache, DynamicLayer

    class PreallocLayer(DynamicLayer):
        def __init__(self, cap_):
            super().__init__()
            self.cap, self.n = cap_, 0

        def update(self, key_states, value_states, cache_kwargs=None):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
                B, H, _, D = key_states.shape
                self.kb = key_states.new_empty((B, H, self.cap, D))
                self.vb = value_states.new_empty((B, H, self.cap, D))
                self.n = 0
            t = key_states.shape[-2]
            self.kb[:, :, self.n:self.n + t] = key_states
            self.vb[:, :, self.n:self.n + t] = value_states
            self.n += t
            self.keys, self.values = self.kb[:, :, :self.n], self.vb[:, :, :self.n]
            return self.keys, self.values

    c = DynamicCache(config=gen.config)
    c.layers = [PreallocLayer(cap) for _ in range(len(c.layers))]
    return c


# ---- models (continued) ---------------------------------------------------------------------------
class Swarm:
    def __init__(self, args):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from model.roles import ModelConfig, RoleModel
        self.args = args
        self.dev = torch.device(args.device)
        self.dtype = getattr(torch, args.dtype)
        mc = ModelConfig(name=args.model, arm_adapters=("chain",), dtype=args.dtype, device=args.device, seed=args.seed)
        self.rm = RoleModel(mc)
        if args.init:
            self.rm.load(args.init, groups=["chain"])
        self.tok = self.rm.tok
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.eos = sorted({self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id})
        base = self.rm.peft.base_model.model
        cfg = base.config.__class__.from_dict(base.config.to_dict())
        cfg.use_cache = True
        self.gen = AutoModelForCausalLM.from_config(cfg).to(self.dtype)
        sd = {n.replace("base_model.model.", "", 1).replace(".base_layer", ""): p.detach()
              for n, p in self.rm.peft.named_parameters() if "lora_" not in n}
        missing, unexpected = self.gen.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:5]
        assert all("lm_head" in m for m in missing), missing[:5]
        self.gen.tie_weights()
        self.gen.to(self.dev).eval()
        for p in self.gen.parameters():
            p.requires_grad_(False)
        self.sync_gen()
        self.set_gen_impl(getattr(args, "gen_impl", "dynamic"))

    def set_gen_impl(self, impl: str):
        """dynamic: HF DynamicCache; static: preallocated cache; static-compile: static cache + torch.compile
        (CUDA graphs). All three sample the same distribution; only speed differs. 'auto' reads genbench.json."""
        if impl == "auto-bench":
            impl = self.bench_impls(("dynamic", "prealloc"))
        if impl == "auto":
            impl = "dynamic"
            fp = os.path.join(os.path.dirname(self.args.out.rstrip("/")), "genbench", "genbench.json")
            for cand in (fp, "/root/runs/swarm/genbench/genbench.json"):
                if os.path.exists(cand):
                    impl = json.load(open(cand)).get("fastest_impl", "dynamic")
                    break
        self.gen_impl = impl
        self.gen.generation_config.cache_implementation = "static" if impl.startswith("static") else None
        if impl == "static-compile" and not getattr(self, "_compiled", False):
            self.gen.forward = torch.compile(self.gen.forward, mode="reduce-overhead", fullgraph=False)
            self._compiled = True

    @torch.no_grad()
    def sync_gen(self):
        """Sampling copy = base + LoRA delta, computed in fp32 and cast once (no merge/unmerge drift)."""
        for name, mod in self.rm.peft.named_modules():
            la = getattr(mod, "lora_A", None)
            if isinstance(la, nn.ModuleDict) and "chain" in la:
                W = mod.base_layer.weight.float()
                d = (mod.lora_B["chain"].weight.float() @ la["chain"].weight.float()) * mod.scaling["chain"]
                self.gen.get_submodule(name.replace("base_model.model.", "", 1)).weight.copy_((W + d).to(self.dtype))

    def bench_impls(self, impls) -> str:
        """Time each cache implementation on 64 synthetic prompts x 96 new tokens; return the fastest."""
        if self.dev.type != "cuda":
            return "dynamic"
        board = Board(PRESETS[self.args.preset])
        rng = np.random.default_rng(1)
        prompts = []
        for b in range(4):
            env = make_board(board.cfg, SEEDS["dev"] + 900 + b)
            for a in assign(round1_targets(board, env, rng), 16, rng):
                prompts.append(self.prompt_ids(chain_prompt(env, board, a, words_for(64))))
        res = {}
        for impl in impls:
            self.gen_impl = impl
            self.gen.generation_config.cache_implementation = None
            st = {}
            self.generate(prompts[:16], 16, 5, st)                 # warm-up
            st = {}
            self.generate(prompts, 96, 6, st)
            res[impl] = st["gen_s"] / max(st["decode_steps"], 1)
        best = min(res, key=res.get)
        self.bench_result = {"s_per_decode_step": res, "chosen": best}
        print(json.dumps({"gen_impl_bench": self.bench_result}), flush=True)
        return best

    def prompt_ids(self, text: str) -> List[int]:
        chat = self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        return self.tok(chat, add_special_tokens=False)["input_ids"]

    def gen_batch_size(self, plen: int, L: int) -> int:
        if self.dev.type != "cuda":
            return self.args.cpu_batch
        per_tok = 2 * self.gen.config.num_hidden_layers * self.gen.config.num_key_value_heads * \
            getattr(self.gen.config, "head_dim", self.gen.config.hidden_size // self.gen.config.num_attention_heads) * 2
        return int(max(8, min(self.args.gen_batch, self.args.kv_gb * 1e9 / ((plen + L) * per_tok))))

    @torch.no_grad()
    def generate(self, prompts: List[List[int]], L: int, seed: int, stats: Dict) -> List[Dict]:
        """Sample one chain per prompt (left-padded batches). Returns dicts with gen ids, eos flag, text."""
        out = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        plen = max(len(p) for p in prompts)
        B = self.gen_batch_size(plen, L)
        for s in range(0, len(order), B):
            idx = order[s:s + B]
            m = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), m), self.pad, dtype=torch.long)
            att = torch.zeros((len(idx), m), dtype=torch.long)
            for b, i in enumerate(idx):
                ids[b, m - len(prompts[i]):] = torch.tensor(prompts[i])
                att[b, m - len(prompts[i]):] = 1
            torch.manual_seed(seed * 100_003 + s)
            t0 = time.time()
            kw = {"past_key_values": prealloc_cache(self.gen, m + L)} if getattr(self, "gen_impl", "dynamic") == "prealloc" else {}
            res = self.gen.generate(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev), do_sample=True,
                                    temperature=self.args.temp, top_k=0, top_p=1.0, max_new_tokens=L,
                                    eos_token_id=self.eos, pad_token_id=self.pad, use_cache=True, **kw)
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            new = res[:, m:].cpu()
            stats["gen_s"] = stats.get("gen_s", 0.0) + dt
            stats["decode_steps"] = stats.get("decode_steps", 0) + new.shape[1]
            stats["gen_calls"] = stats.get("gen_calls", 0) + 1
            for b, i in enumerate(idx):
                toks = new[b].tolist()
                cut = next((k for k, t in enumerate(toks) if t in self.eos), None)
                g = toks[:cut + 1] if cut is not None else toks
                stats["gen_tokens"] = stats.get("gen_tokens", 0) + len(g)
                out[i] = {"gen": g, "eos": cut is not None,
                          "text": self.tok.decode(g[:-1] if cut is not None else g, skip_special_tokens=True)}
        return out

    # ---- log-probs for the losses -----------------------------------------------------------------
    def token_logprobs(self, seqs: List[List[int]], n_prompt: List[int], adapter: bool, grad: bool):
        """Tempered per-token log-probs of the generated part of each sequence (prompt + gen)."""
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        gm = torch.zeros((len(seqs), T), dtype=torch.bool)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s)
            att[b, :len(s)] = 1
            gm[b, n_prompt[b]:len(s)] = True
        ctx = contextlib.nullcontext() if adapter else self.rm.peft.disable_adapter()
        with ctx, torch.set_grad_enabled(grad):
            lz = self.rm.logits("chain", ids, att)
            tm = gm[:, 1:].to(self.dev)
            hs = lz.h[:, :-1][tm]
            tg = ids[:, 1:].to(self.dev)[tm]
            W = lz.W.float()
            parts = []
            for i in range(0, hs.shape[0], 1024):
                lg = (hs[i:i + 1024].float() @ W.T) / self.args.temp
                parts.append(torch.gather(torch.log_softmax(lg, -1), 1, tg[i:i + 1024, None])[:, 0])
            lp = torch.cat(parts) if parts else torch.zeros(0, device=self.dev)
        seg = torch.nonzero(tm)[:, 0]
        return lp, seg


# ---- episodes ---------------------------------------------------------------------------------------
def claims_of(recs: List[Dict]) -> Claims:
    ch, st, q, w = [], [], [], []
    for r in recs:
        if r["c"] is None:
            continue
        for j, (p, c) in zip(r["t"], r["c"]):
            ch.append(r["cid"]); st.append(j); q.append(p); w.append(c)
    return Claims(np.array(ch, int), np.array(st, int), np.array(q, float), np.array(w, float))


def board_truth(board: Board, env):
    z = board.truth(env)
    return z, board.F[z].copy(), board.bayes(env) @ board.F


def run_round(sw: Swarm, board: Board, items: List[Dict], L: int, seed: int, stats: Dict, texts) -> List[Dict]:
    """items: dicts with env, targets (statement list) and id fields. Generates, parses, returns records."""
    prompts = [sw.prompt_ids(chain_prompt(it["env"], board, it["t"], words_for(L))) for it in items]
    gens = sw.generate(prompts, L + ANSWER_TOKENS * max(len(it["t"]) for it in items), seed, stats)
    recs = []
    for it, pr, g in zip(items, prompts, gens):
        c = parse_claims(g["text"], len(it["t"]))
        r = {k: v for k, v in it.items() if k != "env"}
        r.update(c=c, nt=len(g["gen"]), eos=g["eos"])
        r["_seq"] = (pr, g["gen"])
        recs.append(r)
        if texts is not None:
            texts.write(json.dumps({k: v for k, v in r.items() if k not in ("_seq", "c")} | {"text": g["text"]}) + "\n")
    stats["chains"] = stats.get("chains", 0) + len(recs)
    stats["parsed"] = stats.get("parsed", 0) + sum(r["c"] is not None for r in recs)
    return recs


def strip(r: Dict) -> Dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


# ---- small evaluation (training curves, stop rule) -------------------------------------------------
def small_eval(sw: Swarm, board: Board, step: int, stakes: Dict, args, fh, texts) -> Dict:
    rng = np.random.default_rng([args.seed, step, 5])
    items, envs = [], {}
    for b in range(args.small_boards):
        env = make_board(board.cfg, SEEDS["small"] + b)
        envs[b] = env
        t = round1_targets(board, env, rng)
        for i, a in enumerate(assign(t, args.small_n, rng)):
            items.append({"env": env, "b": b, "seed": SEEDS["small"] + b, "cell": "small", "step": step,
                          "L": args.small_L, "n": args.small_n, "r": 1, "i": i, "cid": i, "t": a})
    stats = {}
    recs = run_round(sw, board, items, args.small_L, args.seed * 7 + step, stats, texts)
    info, se_mean, se_d, nclaims = [], [], [], 0
    for b, env in envs.items():
        rb = [r for r in recs if r["b"] == b]
        cl = claims_of(rb)
        z, y, bay = board_truth(board, env)
        om = open_mask(board, env)
        if len(cl.q):
            p0 = board.p0_marg[cl.stmt]
            info += list(bay[cl.stmt] * np.log(cl.q / p0) + (1 - bay[cl.stmt]) * np.log((1 - cl.q) / (1 - p0)))
            nclaims += len(cl.q)
        pm, _ = price(board, cl, "mean")
        pd, _ = price(board, cl, "D-fit", stakes)
        se_mean.append(float(np.mean((pm[om] - bay[om]) ** 2)))
        se_d.append(float(np.mean((pd[om] - bay[om]) ** 2)))
    for r in recs:
        fh.write(json.dumps(strip(r)) + "\n")
    info = np.array(info)
    rng2 = np.random.default_rng(0)
    ci = [float(np.percentile(info[rng2.integers(len(info), size=(1000, len(info)))].mean(1), q)) for q in (2.5, 97.5)] \
        if len(info) > 1 else [float("nan")] * 2
    out = {"step": step, "parse_rate": stats["parsed"] / max(stats["chains"], 1), "info_per_claim": float(info.mean()) if len(info) else float("nan"),
           "info_ci": ci, "n_claims": nclaims, "brier_vs_bayes_mean": float(np.mean(se_mean)),
           "brier_vs_bayes_Dfit": float(np.mean(se_d)), "gen_s": stats.get("gen_s", 0.0),
           "decode_steps": stats.get("decode_steps", 0), "gen_tokens": stats.get("gen_tokens", 0)}
    return out


# ---- training ----------------------------------------------------------------------------------------
def rewards(arm: str, board: Board, recs: List[Dict], all_cl: Claims, stakes: Dict, y: np.ndarray):
    """Payment per chain of one board-round (S1: settlement at D-fit prices incl. own claims; S2: own
    log-score gain over the prior; S3: log of the claim likelihood q(y) for the TB reward)."""
    vals = [None] * len(recs)
    if arm == "S1":
        pd, _ = price(board, all_cl, "D-fit", stakes)
        for k, r in enumerate(recs):
            if r["c"] is not None:
                cl = claims_of([r])
                vals[k] = float(settle(board, cl, np.ones(len(cl.q)), pd, y, n_chains=r["cid"] + 1)["s"][r["cid"]])
    else:
        for k, r in enumerate(recs):
            if r["c"] is not None:
                q = np.array([p for p, _ in r["c"]])
                t = np.array(r["t"])
                yy = y[t]
                lq = np.where(yy == 1, np.log(q), np.log(1 - q))
                if arm == "S2":
                    p0 = board.p0_marg[t]
                    lq = lq - np.where(yy == 1, np.log(p0), np.log(1 - p0))
                vals[k] = float(lq.sum())
    good = [v for v in vals if v is not None]
    floor = (min(good) - 0.1) if good else 0.0
    return [v if v is not None else floor for v in vals], [v is not None for v in vals]


def train(args):
    sw = Swarm(args)
    board = Board(PRESETS[args.preset])
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.text_out, exist_ok=True)
    params = [p for p in sw.rm.group_params("chain")]
    sw.rm.set_trainable(["chain"])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    stakes = {"kappa": 1.0, "alpha": 1.0}
    hist = deque(maxlen=80)
    fsteps = open(os.path.join(args.out, "steps.jsonl"), "a")
    fsmall = open(os.path.join(args.out, "small_chains.jsonl"), "a")
    fchains = open(os.path.join(args.out, "train_chains.jsonl"), "a")
    ftext = open(os.path.join(args.text_out, "train_texts.jsonl"), "a")
    fsm = open(os.path.join(args.out, "small_eval.jsonl"), "a")
    t_start = time.time()
    Ls = [int(x) for x in args.L_train.split(",")]

    def do_small(step):
        sw.sync_gen()
        s = small_eval(sw, board, step, stakes, args, fsmall, ftext)
        s.update(stakes=dict(stakes), elapsed_min=(time.time() - t_start) / 60)
        fsm.write(json.dumps(s) + "\n"); fsm.flush(); fsmall.flush()
        print(json.dumps(s), flush=True)
        return s

    small_hist = {}
    step0 = 0
    bad_parse = bad_collapse = 0
    state_path = args.ckpt_out + ".state" if args.ckpt_out else ""
    if args.resume and state_path and os.path.exists(state_path):
        stt = torch.load(state_path, map_location="cpu", weights_only=False)
        sw.rm.load(args.ckpt_out, groups=["chain"])
        opt.load_state_dict(stt["opt"])
        step0, stakes, small_hist = stt["step"], stt["stakes"], {int(k): v for k, v in stt["small_hist"].items()}
        bad_parse, bad_collapse = stt.get("bad_parse", 0), stt.get("bad_collapse", 0)
        for e in stt["hist"]:
            hist.append(e)
        print(f"resumed from step {step0}", flush=True)
        fsteps.write(json.dumps({"step": step0, "resumed": True}) + "\n")
    elif args.small_every > 0:
        small_hist[0] = do_small(0)

    def save_state(st_step):
        if not args.ckpt_out:
            return
        sw.rm.save(args.ckpt_out)
        torch.save({"opt": opt.state_dict(), "step": st_step, "stakes": stakes, "small_hist": small_hist,
                    "bad_parse": bad_parse, "bad_collapse": bad_collapse, "hist": list(hist)}, state_path)
    step = step0
    for step in range(step0 + 1, args.steps + 1):
        if (time.time() - t_start) / 60 > args.max_minutes:
            print(f"time limit reached before step {step}", flush=True)
            fsteps.write(json.dumps({"step": step, "time_stop": True}) + "\n")
            step -= 1
            break
        t0 = time.time()
        rng = np.random.default_rng([args.seed, step, 3])
        L = int(rng.choice(Ls))
        sw.sync_gen()
        envs = [make_board(board.cfg, SEEDS["train"] + args.seed * 1_000_000 + step * args.boards + b) for b in range(args.boards)]
        truth = [board_truth(board, e) for e in envs]
        per_board = [[] for _ in envs]
        batch, gstats = [], {}
        for rnd in range(1, args.rounds + 1):
            items = []
            for b, env in enumerate(envs):
                if rnd == 1:
                    t = round1_targets(board, env, rng)
                else:
                    cl = claims_of(per_board[b])
                    pavg, _ = price(board, cl, "D-avg")
                    t = residual_targets(board, env, cl, pavg, rng)
                for i, a in enumerate(assign(t, args.chains, rng)):
                    items.append({"env": env, "b": b, "seed": env.seed, "cell": "train", "step": step, "L": L,
                                  "n": args.chains, "r": rnd, "i": i, "cid": (rnd - 1) * args.chains + i, "t": a})
            recs = run_round(sw, board, items, L, args.seed * 1_000 + step * 10 + rnd, gstats,
                             ftext if step % args.text_every == 0 else None)
            for b in range(len(envs)):
                rb = [r for r in recs if r["b"] == b]
                per_board[b] += rb
                rw, ok = rewards(args.sampler, board, rb, claims_of(per_board[b]), stakes, truth[b][1])
                for r, v, o in zip(rb, rw, ok):
                    r["rw"], r["ok"], r["g"] = v, o, b * args.rounds + rnd - 1
                batch += rb
        t_gen = time.time() - t0
        # stakes: refit every 10 steps on the last 80 boards (outcomes only)
        for b, env in enumerate(envs):
            hist.append({"claims": claims_of(per_board[b]), "y": truth[b][1], "zstar": truth[b][0], "open": open_mask(board, env)})
        if step % 10 == 0 and len(hist) >= 16:
            f = fit_stakes(board, list(hist), "stmt")
            stakes = {"kappa": f["kappa"], "alpha": f["alpha"]}
        # advantages / targets
        groups = {}
        for k, r in enumerate(batch):
            groups.setdefault(r["g"], []).append(k)
        adv = np.zeros(len(batch))
        for g, ks in groups.items():
            x = np.array([batch[k]["rw"] for k in ks])
            adv[ks] = (x - x.mean()) / (x.std(ddof=1) + 1e-4) if len(ks) > 1 else 0.0
        for r, a in zip(batch, adv):
            r["adv"] = float(a)
        # loss
        t1 = time.time()
        opt.zero_grad(set_to_none=True)
        seqs = [r["_seq"][0] + r["_seq"][1] for r in batch]
        npr = [len(r["_seq"][0]) for r in batch]
        logs = {"kl": [], "tb_delta": []}
        order = sorted(range(len(batch)), key=lambda k: len(seqs[k]))
        logR = None
        if args.sampler == "S3":   # reference log-probs first: log R = log p_base(chain) + log q(y), log Z per group
            ref_sum = np.zeros(len(batch))
            for s in range(0, len(order), args.micro):
                ks = order[s:s + args.micro]
                lp, seg = sw.token_logprobs([seqs[k] for k in ks], [npr[k] for k in ks], adapter=False, grad=False)
                for j, k in enumerate(ks):
                    ref_sum[k] = float(lp[seg == j].sum())
            logR = ref_sum + np.array([r["rw"] for r in batch])
            pol_sum = np.zeros(len(batch))
            with torch.no_grad():
                for s in range(0, len(order), args.micro):
                    ks = order[s:s + args.micro]
                    lp, seg = sw.token_logprobs([seqs[k] for k in ks], [npr[k] for k in ks], adapter=True, grad=False)
                    for j, k in enumerate(ks):
                        pol_sum[k] = float(lp[seg == j].sum())
            logZ = np.zeros(len(batch))
            for g, ks in groups.items():
                logZ[ks] = np.mean(logR[ks] - pol_sum[ks])
        for s in range(0, len(order), args.micro):
            ks = order[s:s + args.micro]
            lp, seg = sw.token_logprobs([seqs[k] for k in ks], [npr[k] for k in ks], adapter=True, grad=True)
            if args.sampler == "S3":
                tot = torch.zeros(len(ks), device=lp.device).index_add(0, seg, lp)
                d = torch.tensor(logZ[ks] - logR[ks], device=lp.device, dtype=tot.dtype) + tot
                loss = (d ** 2).sum() / len(batch)
                logs["tb_delta"] += d.detach().abs().tolist()
            else:
                with torch.no_grad():
                    ref, _ = sw.token_logprobs([seqs[k] for k in ks], [npr[k] for k in ks], adapter=False, grad=False)
                cnt = torch.zeros(len(ks), device=lp.device).index_add(0, seg, torch.ones_like(lp))
                a = torch.tensor(adv[ks], device=lp.device, dtype=lp.dtype)
                dlt = ref - lp
                klt = torch.exp(dlt) - dlt - 1
                pg = torch.zeros(len(ks), device=lp.device).index_add(0, seg, lp) / cnt.clamp(min=1)
                kl = torch.zeros(len(ks), device=lp.device).index_add(0, seg, klt) / cnt.clamp(min=1)
                loss = (-(a * pg) + args.kl * kl).sum() / len(batch)
                logs["kl"] += kl.detach().tolist()
            loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        opt.step()
        t_loss = time.time() - t1
        ok = [r["ok"] for r in batch]
        rws = np.array([r["rw"] for r in batch if r["ok"]]) if any(ok) else np.array([np.nan])
        distinct, collapsed = [], []      # per group: distinct rounded p per multi-claimed statement
        for g, ks in groups.items():
            by = {}
            for k in ks:
                if batch[k]["c"] is not None:
                    for j, (pp, _) in zip(batch[k]["t"], batch[k]["c"]):
                        by.setdefault(j, []).append(round(pp, 1))
            multi = [v for v in by.values() if len(v) >= 2]
            if multi:
                distinct.append(float(np.mean([len(set(v)) / len(v) for v in multi])))
                collapsed.append(all(len(set(v)) == 1 for v in multi))
        st = {"step": step, "L": L, "parse_rate": float(np.mean(ok)), "pay_mean": float(np.nanmean(rws)),
              "pay_sd": float(np.nanstd(rws)), "pay_pos_share": float(np.mean(rws > 0)) if any(ok) else float("nan"),
              "kl": float(np.mean(logs["kl"])) if logs["kl"] else None,
              "tb_abs_delta": float(np.mean(logs["tb_delta"])) if logs["tb_delta"] else None,
              "distinct_p_share": float(np.mean(distinct)) if distinct else None,
              "collapsed_group_share": float(np.mean(collapsed)) if collapsed else None,
              "grad_norm": gn, "stakes": dict(stakes), "t_gen": t_gen, "t_loss": t_loss, "t_step": time.time() - t0,
              "decode_steps": gstats.get("decode_steps", 0), "gen_s": gstats.get("gen_s", 0.0),
              "gen_tokens": gstats.get("gen_tokens", 0), "elapsed_min": (time.time() - t_start) / 60}
        fsteps.write(json.dumps(st) + "\n"); fsteps.flush()
        for r in batch:
            fchains.write(json.dumps(strip(r)) + "\n")
        fchains.flush()
        print(json.dumps({k: st[k] for k in ("step", "L", "parse_rate", "pay_mean", "kl", "t_step", "elapsed_min")}), flush=True)
        bad_parse = bad_parse + 1 if st["parse_rate"] < 0.5 else 0
        bad_collapse = bad_collapse + 1 if (st["collapsed_group_share"] or 0.0) >= 1.0 else 0
        if bad_parse >= 20 or bad_collapse >= 20:
            fsteps.write(json.dumps({"step": step, "stop_rule": "parse<0.5 x20" if bad_parse >= 20 else "collapse x20"}) + "\n")
            print("stop rule hit", flush=True)
            break
        if args.small_every > 0 and step % args.small_every == 0 and step < args.steps:
            s_now = do_small(step)
            small_hist[step] = s_now
            if args.stop100 and step == 100 and 0 in small_hist and s_now["info_ci"][0] <= small_hist[0]["info_per_claim"]:
                fsteps.write(json.dumps({"step": step, "stop_rule": "S1 step-100: info per claim not above step 0"}) + "\n")
                print("step-100 stop rule hit", flush=True)
                break
        if step % 25 == 0:
            save_state(step)
    save_state(step)
    if args.small_every > 0 and step > 0 and step not in small_hist:
        do_small(step)
    json.dump({"steps_done": step, "minutes": (time.time() - t_start) / 60, "stakes": stakes},
              open(os.path.join(args.out, "train_done.json"), "w"))
    return sw


# ---- full evaluation grid --------------------------------------------------------------------------
def eval_grid(args, sw: Optional[Swarm] = None):
    sw = sw or Swarm(args)
    board = Board(PRESETS[args.preset])
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.text_out, exist_ok=True)
    fch = open(os.path.join(args.out, "eval_chains.jsonl"), "a")
    ftx = open(os.path.join(args.text_out, "eval_texts.jsonl"), "a")
    meta = {"cells": [], "t_start": time.time()}
    t_start = time.time()
    envs = {b: make_board(board.cfg, SEEDS["test"] + args.seed * 1000 + b) for b in range(args.eval_boards)}
    sw.sync_gen()

    def out_of_time():
        return (time.time() - t_start) / 60 > args.eval_max_minutes

    def write(recs):
        for r in recs:
            fch.write(json.dumps(strip(r)) + "\n")
        fch.flush(); ftx.flush()

    def grid(L: int, ns_r23: List[int], nmax_r1: int, kind: str):
        rng = np.random.default_rng([args.seed, L, 17 if kind == "main" else 19])
        stats = {}
        t0 = time.time()
        # round 1: nmax_r1 chains per board; smaller n are the first n of them
        items = []
        for b, env in envs.items():
            t = round1_targets(board, env, rng) if kind == "main" else roundrobin_targets(board, env, 0)
            for i, a in enumerate(assign(t, nmax_r1, rng)):
                items.append({"env": env, "b": b, "seed": env.seed, "cell": kind, "L": L, "n": nmax_r1, "r": 1, "i": i,
                              "cid": i, "t": a})
        r1 = run_round(sw, board, items, L, args.seed * 31 + L, stats, ftx)
        write(r1)
        hist = {(n, b): [r for r in r1 if r["b"] == b and r["i"] < n] for n in ns_r23 for b in envs}
        for rnd in (2, 3):
            if out_of_time() or not ns_r23:
                break
            items = []
            for n in ns_r23:
                for b, env in envs.items():
                    cl = claims_of(hist[(n, b)])
                    if kind == "main":
                        pavg, _ = price(board, cl, "D-avg")
                        t = residual_targets(board, env, cl, pavg, rng)
                    else:
                        t = roundrobin_targets(board, env, rnd - 1)
                    for i, a in enumerate(assign(t, n, rng)):
                        items.append({"env": env, "b": b, "seed": env.seed, "cell": kind, "L": L, "n": n, "r": rnd,
                                      "i": i, "cid": 1000 * rnd + i, "t": a})
            rr = run_round(sw, board, items, L, args.seed * 31 + L + 7 * rnd, stats, ftx)
            write(rr)
            for r in rr:
                hist[(r["n"], r["b"])].append(r)
        meta["cells"].append({"kind": kind, "L": L, "n_r23": ns_r23, "n_r1": nmax_r1, "minutes": (time.time() - t0) / 60,
                              "chains": stats.get("chains", 0), "parsed": stats.get("parsed", 0),
                              "decode_steps": stats.get("decode_steps", 0), "gen_s": stats.get("gen_s", 0.0),
                              "gen_tokens": stats.get("gen_tokens", 0)})
        json.dump(meta, open(os.path.join(args.out, "eval_meta.json"), "w"), indent=1)

    def long_chains():
        rng = np.random.default_rng([args.seed, 23])
        stats = {}
        t0 = time.time()
        for budget in args.long_budgets:
            if out_of_time():
                break
            items = []
            for b, env in envs.items():
                t = round1_targets(board, env, rng)
                items.append({"env": env, "b": b, "seed": env.seed, "cell": "long", "L": budget, "n": 1, "r": 1, "i": 0,
                              "cid": 0, "t": t})
            write(run_round(sw, board, items, budget, args.seed * 37 + budget, stats, ftx))
        meta["cells"].append({"kind": "long", "budgets": args.long_budgets, "minutes": (time.time() - t0) / 60,
                              "chains": stats.get("chains", 0), "parsed": stats.get("parsed", 0),
                              "decode_steps": stats.get("decode_steps", 0), "gen_s": stats.get("gen_s", 0.0)})
        json.dump(meta, open(os.path.join(args.out, "eval_meta.json"), "w"), indent=1)

    for L in args.eval_L:          # the primary cell (n = 16, L = 256, R = 2) is in the first grid
        if out_of_time():
            break
        ns = args.eval_n if L <= 64 else args.eval_n_long
        grid(L, ns, 64, "main")
    if args.long and not out_of_time():
        long_chains()
    if args.roundrobin:
        for L in (args.rr_L or args.eval_L):
            if not out_of_time():
                grid(L, [16], 16, "roundrobin")
    meta["minutes"] = (time.time() - t_start) / 60
    meta["complete"] = not out_of_time()
    json.dump(meta, open(os.path.join(args.out, "eval_meta.json"), "w"), indent=1)
    return sw


# ---- generation speed benchmark ------------------------------------------------------------------------
def genbench(args):
    sw = Swarm(args)
    board = Board(PRESETS[args.preset])
    rng = np.random.default_rng(0)
    prompts = []
    for b in range(8):
        env = make_board(board.cfg, SEEDS["dev"] + 500 + b)
        for a in assign(round1_targets(board, env, rng), 16, rng):
            prompts.append(sw.prompt_ids(chain_prompt(env, board, a, words_for(256))))
    out = {"model": args.model, "max_new": 256 + 6 * ANSWER_TOKENS, "runs": []}
    os.makedirs(args.out, exist_ok=True)
    for impl, B in (("dynamic", 128), ("dynamic", 32), ("static", 128), ("static-compile", 128)):
        sw.set_gen_impl(impl)
        for rep in range(2 if impl == "static-compile" else 1):     # first compiled call includes compilation
            st = {}
            args.gen_batch = B
            t0 = time.time()
            try:
                g = sw.generate(prompts[:B], out["max_new"], 99 + rep, st)
                ok = True
            except Exception as e:   # record and continue
                ok, g = False, []
                st["error"] = repr(e)[:300]
            row = {"impl": impl, "batch": B, "rep": rep, "ok": ok, "seconds": time.time() - t0,
                   "s_per_decode_step": st.get("gen_s", 0) / max(st.get("decode_steps", 1), 1),
                   "chains_per_s": len(g) / max(time.time() - t0, 1e-9), "error": st.get("error")}
            out["runs"].append(row)
            print(json.dumps(row), flush=True)
            json.dump(out, open(os.path.join(args.out, "genbench.json"), "w"), indent=1)
    best = [r for r in out["runs"] if r["ok"] and r["batch"] == 128 and (r["impl"] != "static-compile" or r["rep"] == 1)]
    out["fastest_impl"] = min(best, key=lambda r: r["s_per_decode_step"])["impl"] if best else "dynamic"
    json.dump(out, open(os.path.join(args.out, "genbench.json"), "w"), indent=1)


# ---- Gate 0 --------------------------------------------------------------------------------------------
def gate0(args):
    sw = Swarm(args)
    board = Board(PRESETS[args.preset])
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.text_out, exist_ok=True)
    fch = open(os.path.join(args.out, "gate0_chains.jsonl"), "a")
    ftx = open(os.path.join(args.text_out, "gate0_texts.jsonl"), "a")
    res = {"model": args.model, "cells": {}, "prompt_dev": {}}
    variants = [v for v in args.prompt_variants.split(",") if v]
    if len(variants) > 1:   # choose the answer format on separate dev boards (never the gate or test boards)
        for v in variants:
            PROMPT["style"] = v
            rng = np.random.default_rng([v.encode()[0], 31])
            items, envs = [], {}
            for b in range(args.dev_boards):
                env = make_board(board.cfg, SEEDS["dev"] + b)
                envs[b] = env
                for i, a in enumerate(assign(round1_targets(board, env, rng), args.gate_n, rng)):
                    items.append({"env": env, "b": b, "seed": env.seed, "cell": "dev-" + v, "L": args.dev_L, "n": args.gate_n,
                                  "r": 1, "i": i, "cid": i, "t": a})
            stats = {}
            recs = run_round(sw, board, items, args.dev_L, 777, stats, ftx)
            for r in recs:
                fch.write(json.dumps(strip(r)) + "\n")
            fch.flush(); ftx.flush()
            info = []
            for b, env in envs.items():
                cl = claims_of([r for r in recs if r["b"] == b])
                z, y, bay = board_truth(board, env)
                if len(cl.q):
                    p0 = board.p0_marg[cl.stmt]
                    info += list(bay[cl.stmt] * np.log(cl.q / p0) + (1 - bay[cl.stmt]) * np.log((1 - cl.q) / (1 - p0)))
            res["prompt_dev"][v] = {"parse_rate": stats["parsed"] / stats["chains"],
                                    "info_per_claim": float(np.mean(info)) if info else float("nan"), "chains": stats["chains"]}
            print(json.dumps({"dev": v} | res["prompt_dev"][v]), flush=True)
        ok = [v for v in variants if res["prompt_dev"][v]["parse_rate"] >= 0.8]
        pool_v = ok or variants
        key = (lambda v: res["prompt_dev"][v]["info_per_claim"]) if ok else (lambda v: res["prompt_dev"][v]["parse_rate"])
        PROMPT["style"] = max(pool_v, key=lambda v: (np.nan_to_num(key(v), nan=-1e9)))
        res["prompt_chosen"] = PROMPT["style"]
        json.dump(res, open(os.path.join(args.out, "gate0.json"), "w"), indent=1)
    for L in args.gate_L:
        rng = np.random.default_rng([L, 29])
        items, envs = [], {}
        for b in range(args.gate_boards):
            env = make_board(board.cfg, SEEDS["gate0"] + b)
            envs[b] = env
            for i, a in enumerate(assign(round1_targets(board, env, rng), args.gate_n, rng)):
                items.append({"env": env, "b": b, "seed": env.seed, "cell": "gate0", "L": L, "n": args.gate_n, "r": 1,
                              "i": i, "cid": i, "t": a})
        stats = {}
        t0 = time.time()
        recs = run_round(sw, board, items, L, 4242 + L, stats, ftx)
        for r in recs:
            fch.write(json.dumps(strip(r)) + "\n")
        fch.flush(); ftx.flush()
        info, se = [], {m: [] for m in ("prior", "mean", "vote", "D-avg")}
        for b, env in envs.items():
            cl = claims_of([r for r in recs if r["b"] == b])
            z, y, bay = board_truth(board, env)
            om = open_mask(board, env)
            if len(cl.q):
                p0 = board.p0_marg[cl.stmt]
                info += list(bay[cl.stmt] * np.log(cl.q / p0) + (1 - bay[cl.stmt]) * np.log((1 - cl.q) / (1 - p0)))
            for m in se:
                p, _ = price(board, cl, m)
                se[m].append(float(np.mean((p[om] - bay[om]) ** 2)))
        info = np.array(info)
        r2 = np.random.default_rng(0)
        ci = [float(np.percentile(info[r2.integers(len(info), size=(2000, len(info)))].mean(1), q)) for q in (2.5, 97.5)] \
            if len(info) > 1 else [float("nan")] * 2
        res["cells"][str(L)] = {
            "chains": stats["chains"], "parse_rate": stats["parsed"] / stats["chains"],
            "eos_rate": float(np.mean([r["eos"] for r in recs])), "mean_gen_tokens": float(np.mean([r["nt"] for r in recs])),
            "info_per_claim": float(info.mean()) if len(info) else float("nan"), "info_ci": ci, "n_claims": int(len(info)),
            "brier_vs_bayes": {m: float(np.mean(v)) for m, v in se.items()},
            "gen_s": stats["gen_s"], "decode_steps": stats["decode_steps"], "gen_calls": stats["gen_calls"],
            "s_per_decode_step": stats["gen_s"] / max(stats["decode_steps"], 1), "gen_tokens": stats["gen_tokens"],
            "minutes": (time.time() - t0) / 60}
        print(json.dumps({"L": L} | {k: v for k, v in res["cells"][str(L)].items() if k != "brier_vs_bayes"}), flush=True)
        json.dump(res, open(os.path.join(args.out, "gate0.json"), "w"), indent=1)
    if args.gate_train_steps > 0:
        a2 = argparse.Namespace(**vars(args))
        a2.steps, a2.small_every, a2.ckpt_out, a2.sampler = args.gate_train_steps, 0, "", "S1"
        a2.out = os.path.join(args.out, "timing_train")
        a2.text_out = os.path.join(args.text_out, "timing_train")
        del sw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        train(a2)
        rows = [json.loads(l) for l in open(os.path.join(a2.out, "steps.jsonl"))]
        res["train_steps"] = [{k: r.get(k) for k in ("step", "L", "t_step", "t_gen", "t_loss", "decode_steps", "gen_s", "parse_rate")} for r in rows]
        if torch.cuda.is_available():
            res["cuda_max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    json.dump(res, open(os.path.join(args.out, "gate0.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate0", "train", "eval", "train+eval", "genbench"], required=True)
    ap.add_argument("--gen-impl", default="dynamic", choices=["dynamic", "static", "static-compile", "prealloc", "auto", "auto-bench"])
    ap.add_argument("--sampler", choices=["S0", "S1", "S2", "S3"], default="S0")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="")
    ap.add_argument("--ckpt-out", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-out", default="")
    ap.add_argument("--eval-out", default="", help="train+eval: where the evaluation writes (default --out)")
    ap.add_argument("--eval-text-out", default="")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--gen-batch", type=int, default=192)
    ap.add_argument("--kv-gb", type=float, default=9.0)
    ap.add_argument("--cpu-batch", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=1e9, help="soft limit for training (checked before each step)")
    ap.add_argument("--eval-max-minutes", type=float, default=1e9, help="soft limit for evaluation (checked between cells)")
    ap.add_argument("--threads", type=int, default=0)
    # training
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--boards", type=int, default=8)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--L-train", default="64,256")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl", type=float, default=0.04)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--text-every", type=int, default=10)
    ap.add_argument("--small-every", type=int, default=50)
    ap.add_argument("--small-boards", type=int, default=100)
    ap.add_argument("--small-n", type=int, default=16)
    ap.add_argument("--small-L", type=int, default=128)
    ap.add_argument("--stop100", action="store_true", help="S1 rule: stop at step 100 if info per claim is not above step 0")
    ap.add_argument("--resume", action="store_true", help="continue from --ckpt-out and its .state file if present")
    # evaluation
    ap.add_argument("--eval-boards", type=int, default=100)
    ap.add_argument("--eval-L", default="256,64")
    ap.add_argument("--eval-n", default="1,4,16,64")
    ap.add_argument("--eval-n-long", default="1,4,16")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--long-budgets", default="256,1024,4096")
    ap.add_argument("--roundrobin", action="store_true")
    ap.add_argument("--rr-L", default="", help="lengths for the round-robin ablation (default: --eval-L)")
    # gate 0
    ap.add_argument("--gate-boards", type=int, default=100)
    ap.add_argument("--gate-n", type=int, default=16)
    ap.add_argument("--gate-L", default="64,256")
    ap.add_argument("--gate-train-steps", type=int, default=5)
    ap.add_argument("--prompt", default="A", choices=["A", "B"])
    ap.add_argument("--prompt-variants", default="", help="gate0: choose among these on --dev-boards first, e.g. A,B")
    ap.add_argument("--dev-boards", type=int, default=30)
    ap.add_argument("--dev-L", type=int, default=256)
    args = ap.parse_args()
    args.text_out = args.text_out or args.out
    for k in ("eval_L", "eval_n", "eval_n_long", "long_budgets", "gate_L", "rr_L"):
        setattr(args, k, [int(x) for x in getattr(args, k).split(",") if x and int(x) > 0])
    PROMPT["style"] = args.prompt
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, f"config-{args.mode}.json"), "w"), indent=1)
    if args.mode == "genbench":
        genbench(args)
    elif args.mode == "gate0":
        gate0(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval_grid(args)
    else:
        sw = train(argparse.Namespace(**vars(args)))
        e = argparse.Namespace(**vars(args))
        e.out, e.text_out = args.eval_out or args.out, args.eval_text_out or args.text_out
        os.makedirs(e.out, exist_ok=True)
        json.dump(vars(e), open(os.path.join(e.out, "config-eval.json"), "w"), indent=1)
        eval_grid(e, sw)


if __name__ == "__main__":
    main()
