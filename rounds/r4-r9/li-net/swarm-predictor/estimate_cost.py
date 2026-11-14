"""L4 cost estimate for the swarm-predictor experiment, anchored on the harness's measured L4 timings.

Measured inputs (read from files, never typed in):
  harness/runs/gpu-probe/l4_estimates.json   per-seed L4 table quoted in harness/README.md
  harness/runs/gpu-probe/q06-P/steps.jsonl   0.6B RL step split: t_rollout (no-grad), t_train (grad)
  harness/runs/gpu-probe/q17-P/steps.jsonl   the same for 1.7B
  token counts of the harness's own sequences (easy preset, Qwen3 tokenizer) and of the draft chain
  prompt below (medium preset, 6 statements, chat template with thinking off).
From these: r_fwd = no-grad tokens/s and r_train = forward+backward tokens/s of the harness code path
on one L4 (these include the harness's Python overheads, so they are conservative).
ASSUMED, not measured anywhere yet: tau = seconds per decoding step (one new token for a whole batch of
up to 128 chains, KV cache on, LoRA merged before sampling). The harness never decodes with a KV
cache. Stage 0 measures tau first; the plan is re-costed if it is more than 1.5x the assumed value.
Run:  cd harness && source scripts/env.sh && python3 ../../li-net/swarm-predictor/estimate_cost.py
Writes cost_estimate.json next to this file.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.environ.get("LIR_HARNESS", "/home/agent/workspaces/eurisko-study/lir-agent/harness")
sys.path.insert(0, H)
RATE_USD_H = json.load(open(f"{H}/runs/gpu-probe/l4_estimates.json"))["reference"]["rate_usd_h_ceiling"]
TAU = {"Qwen/Qwen3-0.6B": 0.035, "Qwen/Qwen3-1.7B": 0.045}       # ASSUMPTION (s per decode step)
TAU_LONG = 1.5                                                    # ASSUMPTION: x tau for 1-4k-token chains
SETUP_H = 0.1                                                     # VM boot + pip (setup-times.txt: pip 162 s)


def harness_tokens():
    from transformers import AutoTokenizer
    from protocol.render import Vocab, belief_seq, proposal_seq, forecast_seq, commit_seq
    from env.shiphunt import ShipHunt, PRESETS
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    V = Vocab(tok)
    b, p, f, c = [], [], [], []
    for s in range(50):
        e = ShipHunt(PRESETS["easy"], 1000 + s)
        sit, plans = [3] * e.cfg.n_cells, [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        b.append(len(belief_seq(V, e).ids)); p.append(len(proposal_seq(V, e, sit).ids))
        f.append(len(forecast_seq(V, e, sit, plans[0]).ids)); c.append(len(commit_seq(V, e, sit, plans, [5, 5, 5]).ids))
    B, P, F, C = (st.mean(x) for x in (b, p, f, c))
    K, L = 3, 3
    no_grad = B + K * sum(P + j for j in range(L)) + K * F + C      # rollout passes per episode (arm P, A4)
    grad = (P + L) + C + B + K * F                                  # plan, commit, belief, 3 forecasts
    return tok, {"belief": B, "proposal": P, "forecast": F, "commit": C, "no_grad_per_episode": no_grad,
                 "grad_per_episode": grad}


def chain_prompt(env, board, stmts):
    from env.shiphunt import LETTERS
    cfg = env.cfg
    rows = " / ".join(" ".join(LETTERS[r * cfg.W + c] for c in range(cfg.W)) for r in range(cfg.H))
    probes = "; ".join(f"{LETTERS[c]} {'hit' if h else 'miss'}" for c, h in env.probes)
    shots = "; ".join(f"{LETTERS[c]} {'hit' if h else 'miss'}" for c, h in env.fired) or "none"
    lines = "\n".join(f"S{k + 1}: {board.text(j)}" for k, j in enumerate(stmts))
    return (f"Battleship, {cfg.W} columns x {cfg.H} rows, cells named row by row: {rows}.\n"
            f"Ships: one of length 3 and one of length 2, each straight (horizontal or vertical), not overlapping. "
            f"Every layout that follows these rules was equally likely.\n"
            f"Probe reports (each is wrong with probability {cfg.eps}): {probes}.\n"
            f"Shots fired (always correct): {shots}.\n"
            f"Statements:\n{lines}\n"
            f"Reason briefly, then end with one line per statement, exactly like\n"
            f"S1: p=0.70 conf=0.8\n"
            f"where p is your probability that the statement is true and conf (0 to 1) is how sure you are of p.")


def prompt_tokens(tok):
    import numpy as np
    from env.shiphunt import ShipHunt, PRESETS
    from pool import Board
    board = Board(PRESETS["medium"])
    n = []
    for s in range(30):
        env = ShipHunt(PRESETS["medium"], 3000 + s)
        rng = np.random.default_rng(s)
        for c in rng.choice([c for c in range(20) if c not in {x for x, _ in env.probes}], 4, replace=False):
            env.fire(int(c))
        text = chain_prompt(env, board, list(rng.choice(board.J, 6, replace=False)))
        chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        n.append(len(tok(chat, add_special_tokens=False)["input_ids"]))
    return st.mean(n), max(n), text


def rates():
    out = {}
    for run, model in (("q06-P", "Qwen/Qwen3-0.6B"), ("q17-P", "Qwen/Qwen3-1.7B")):
        rows = [json.loads(l) for l in open(f"{H}/runs/gpu-probe/{run}/steps.jsonl")]
        rl = [r for r in rows if r.get("phase") == "rl"]
        out[model] = {"t_rollout": st.mean(r["t_rollout"] for r in rl), "t_train": st.mean(r["t_train"] for r in rl),
                      "episodes": 32, "probe_run": run}
    return out


def plan_cost(model, R, P, tau_scale=1.0, stage0_extra_h=0.0, eval_boards=100, train_steps=150, trained_arms=3, seeds=3,
              s0_evals=3, long_chain_evals=6, L_eval=(64, 256), n_full=(1, 4, 16, 64), n_wide_L256=(1, 4, 16)):
    tau = TAU[model] * tau_scale
    r_fwd, r_train = R[model]["r_fwd"], R[model]["r_train"]
    # --- one GRPO step: 8 boards x 2 rounds x 8 chains, L drawn from {64, 256} (mean 160)
    Lm, chains, per_round = 160, 128, 64
    t_dec = 2 * Lm * tau                               # two rounds of batched decoding (64 chains each)
    t_pre = chains * P / r_fwd                         # prompt prefill
    t_grpo = chains * (P + Lm) / r_train               # policy log-probs, forward + backward
    t_ref = chains * (P + Lm) / r_fwd                  # reference (base) log-probs for KL / GFlowNet reward
    step = t_dec + t_pre + t_grpo + t_ref
    mid_eval = 2 * ((100 * 16 / 128) * 128 * tau + 100 * 16 * P / r_fwd)     # n=16, L=128, R=1, 100 boards, 2x
    train_h = (train_steps * step + mid_eval) / 3600
    # --- one full evaluation of a checkpoint: R = 3 rounds (R = 1, 2 are its prefixes); round 1 draws
    # 64 chains (n = 1, 4, 16, 64 are nested subsets); rounds 2-3 run separately per n >= 4
    dec_steps, pre_tok = 0.0, 0.0
    for L in L_eval:
        ns = [n for n in (n_full if L == 64 else n_wide_L256) if n >= 4]
        r1 = eval_boards * 64
        r23 = 2 * eval_boards * sum(ns)
        dec_steps += (r1 / 128 + r23 / 128) * L
        pre_tok += (r1 + r23) * P
    eval_h = (dec_steps * tau + pre_tok / r_fwd) / 3600
    # one long chain of n*L tokens for n*L in {256, 1024, 4096}: batch 128, 128, 32
    long_h = (sum((eval_boards / b) * nl for nl, b in ((256, 128), (1024, 128), (4096, 32))) * tau * TAU_LONG
              + 3 * eval_boards * P / r_fwd) / 3600
    # round-robin Refocus ablation: S1 only, every seed, n = 16, R = 3, L = 64 and 256
    abl_h = seeds * sum((eval_boards * 16 * 3 / 128) * L * tau + eval_boards * 16 * 3 * P / r_fwd for L in L_eval) / 3600
    stage0_h = 2 * ((100 * 16 * 2 / 128) * 160 * tau + 100 * 32 * P / r_fwd) / 3600 + 5 * step / 3600 + SETUP_H
    vms = 1 + trained_arms * seeds // 2 + 2                      # stage 0, training pairs, eval batches
    total = (stage0_h + stage0_extra_h + trained_arms * seeds * train_h + (trained_arms * seeds + s0_evals) * eval_h
             + long_chain_evals * long_h + abl_h + vms * SETUP_H)
    return {"model": model, "tau_s_per_decode_step": tau, "grpo_step_s": round(step, 1),
            "grpo_step_parts_s": {"decode": round(t_dec, 1), "prefill": round(t_pre, 1), "policy_fwd_bwd": round(t_grpo, 1),
                                  "reference_fwd": round(t_ref, 1)},
            "train_L4h_per_arm_seed": round(train_h, 2), "eval_L4h_per_checkpoint": round(eval_h, 2),
            "long_chain_L4h_per_checkpoint": round(long_h, 2), "roundrobin_ablation_L4h": round(abl_h, 2),
            "stage0_L4h": round(stage0_h, 2),
            "settings": {"eval_boards": eval_boards, "train_steps": train_steps, "trained_arms": trained_arms, "seeds": seeds,
                         "s0_evals": s0_evals, "long_chain_evals": long_chain_evals, "L_eval": list(L_eval),
                         "n_full_L64": list(n_full), "n_L256_rounds2_3": list(n_wide_L256)},
            "total_L4h": round(total, 1), "total_usd_at_ceiling": round(total * RATE_USD_H, 1)}


if __name__ == "__main__":
    tok, ht = harness_tokens()
    P_mean, P_max, example = prompt_tokens(tok)
    R = rates()
    for m, r in R.items():
        r["r_fwd_tok_s"] = r["r_fwd"] = 32 * ht["no_grad_per_episode"] / r["t_rollout"]
        r["r_train_tok_s"] = r["r_train"] = 32 * ht["grad_per_episode"] / r["t_train"]
    # Stage 0 gates both model sizes, so every plan carries the other size's Stage 0 as well
    s06 = plan_cost("Qwen/Qwen3-0.6B", R, P_mean)["stage0_L4h"]
    s17 = plan_cost("Qwen/Qwen3-1.7B", R, P_mean)["stage0_L4h"]
    s06x = plan_cost("Qwen/Qwen3-0.6B", R, P_mean, tau_scale=2.0)["stage0_L4h"]
    s17x = plan_cost("Qwen/Qwen3-1.7B", R, P_mean, tau_scale=2.0)["stage0_L4h"]
    plans = {
        "0.6B, full plan": plan_cost("Qwen/Qwen3-0.6B", R, P_mean, stage0_extra_h=s17),
        "0.6B, decoding 2x slower": plan_cost("Qwen/Qwen3-0.6B", R, P_mean, tau_scale=2.0, stage0_extra_h=s17x),
        "0.6B, decoding 2x slower, no S2, long chains for S1 only": plan_cost(
            "Qwen/Qwen3-0.6B", R, P_mean, tau_scale=2.0, stage0_extra_h=s17x, trained_arms=2, long_chain_evals=3),
        "1.7B, no S2, long chains for S1 only": plan_cost(
            "Qwen/Qwen3-1.7B", R, P_mean, stage0_extra_h=s06, trained_arms=2, long_chain_evals=3),
        "1.7B, decoding 2x slower, no S2, long chains for S1 only, 80 boards": plan_cost(
            "Qwen/Qwen3-1.7B", R, P_mean, tau_scale=2.0, stage0_extra_h=s06x, trained_arms=2, long_chain_evals=3,
            eval_boards=80),
    }
    s0 = plan_cost("Qwen/Qwen3-0.6B", R, P_mean)["stage0_L4h"] + plan_cost("Qwen/Qwen3-1.7B", R, P_mean)["stage0_L4h"]
    stage0_both = {"L4h": round(s0, 2), "usd_at_ceiling": round(s0 * RATE_USD_H, 2),
                   "what": "Gate 0 on both model sizes: 100 boards x 16 chains x L in {64, 256}, R = 1, plus 5 timed GRPO steps each"}
    out = {"stage0_both_models": stage0_both, "rate_usd_h_ceiling": RATE_USD_H, "harness_tokens": ht, "chain_prompt_tokens_mean": P_mean,
           "chain_prompt_tokens_max": P_max, "chain_prompt_example": example,
           "rates": {m: {k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()} for m, r in R.items()},
           "assumptions": {"tau_s_per_decode_step": TAU, "tau_long_factor": TAU_LONG, "setup_h_per_vm": SETUP_H},
           "plans": plans}
    with open(os.path.join(HERE, "cost_estimate.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in plans.items():
        print(f"{k:70s} {v['total_L4h']:5.1f} L4-h  ${v['total_usd_at_ceiling']:5.1f}  step {v['grpo_step_s']} s  "
              f"train/arm-seed {v['train_L4h_per_arm_seed']} h  eval/ckpt {v['eval_L4h_per_checkpoint']} h")
    print("stage0 both models", stage0_both)
    print("prompt tokens", P_mean, P_max, "rates", {m: (round(r['r_fwd']), round(r['r_train'])) for m, r in R.items()})
