"""E3 training wrapper around the unmodified harness (harness/protocol/train.py).

Adds, without editing harness code:
  * preset "hardsyco" = "hard" + captain's note (registered at runtime in env.PRESETS), so the
    syco positive control runs at hard scale (hawks/E3.md items 6, 12);
  * --reward-mode forecast: the GRPO reward of each RL episode is the forecaster's own expected
    P(success) for the committed plan (RAP-style internal reward, hawks/E3.md item 5);
  * proposer checkpoints at 0/25/50/100% of RL (hawks/E3.md item 4b);
  * summary.json with mask_violations and the base-checksum assertion.
Usage: python3 e3_train.py --out DIR --ckpt DIR --arm P --preset hard --seed 0 [--init-from warm.pt]
       [--reward-mode env|forecast] [--harness-args "..."]
"""
import argparse, dataclasses, json, os, sys, time
from dataclasses import replace

import numpy as np

H = os.environ.get("LIR_HARNESS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../harness"))
sys.path.insert(0, H)
import env as ENV                       # noqa: E402
from env import shiphunt as SH          # noqa: E402
SH.PRESETS["hardsyco"] = replace(SH.PRESETS["hard"], note=True, note_rel=0.4)
ENV.PRESETS["hardsyco"] = SH.PRESETS["hardsyco"]
import protocol.train as T              # noqa: E402
T.PRESETS["hardsyco"] = SH.PRESETS["hardsyco"]
from protocol.rollout import DEC, ProtoConfig   # noqa: E402

REWARD_MODE = {"mode": "env"}
_orig_rollout = T.rollout


def _rollout(model, V, envs, pc, gen, groups=None, **kw):
    recs = _orig_rollout(model, V, envs, pc, gen, groups=groups, **kw)
    if REWARD_MODE["mode"] == "forecast" and groups is not None:     # RL step only (sft/eval pass no groups)
        for r in recs:
            r["env_reward_true"] = r["reward"]
            r["reward"] = float((r["fc"][r["k"]]["succ_p"] * DEC).sum())
    return recs


T.rollout = _rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", default="P")
    ap.add_argument("--preset", default="hard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--sft-steps", type=int, default=80)
    ap.add_argument("--sft-batch", type=int, default=16)
    ap.add_argument("--rl-steps", type=int, default=80)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--reward-mode", default="env", choices=["env", "forecast"])
    ap.add_argument("--p-reward", default=None)        # harness proto reward (approval for B0 syco)
    ap.add_argument("--p-goal-mode", default=None)     # visible = no content mask (matched comparator)
    ap.add_argument("--warm-only", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rl-deadline", type=float, default=0, help="unix time; if >0, after 3 RL steps shrink rl_steps to finish by it")
    a = ap.parse_args()
    REWARD_MODE["mode"] = a.reward_mode
    pc = ProtoConfig()
    rc = T.RunConfig(arm=a.arm, preset=a.preset, seed=a.seed, model=a.model, device=a.device, dtype=a.dtype,
                     sft_steps=a.sft_steps, sft_batch=a.sft_batch, rl_steps=a.rl_steps, groups=a.groups,
                     group_size=a.group_size, micro_batch=a.micro_batch, eval_every=0, eval_n=0, eval_presets="",
                     out=a.out, ckpt_dir=a.ckpt, init_from=a.init_from, threads=a.threads, proto=pc)
    rc.resolve()
    if a.p_reward:
        rc.proto.reward = a.p_reward
    if a.p_goal_mode:
        rc.proto.goal_mode = a.p_goal_mode
    rc.resolve = lambda: rc
    tr = T.Trainer(rc)
    os.makedirs(a.ckpt, exist_ok=True)
    t0 = time.time()
    if a.init_from:
        tr.model.load(a.init_from)
    else:
        for i in range(a.sft_steps):
            tr.sft_step(i)
    t_sft = time.time() - t0
    tr.model.save(os.path.join(a.ckpt, "c000.pt"))
    mk = lambda n: {max(1, round(n * f)): f"c{int(f * 100):03d}" for f in (0.25, 0.5, 1.0)}
    planned = a.rl_steps
    marks = {} if a.warm_only else mk(a.rl_steps)
    t1 = time.time()
    s = 0
    while not a.warm_only and s < a.rl_steps:
        tr.rl_step()
        s += 1
        if s == 3 and a.rl_deadline > 0:
            sps = (time.time() - t1) / 3
            fit = 3 + int((a.rl_deadline - time.time()) / (sps * 1.05))
            if fit < a.rl_steps:
                a.rl_steps = max(8, fit)
                marks = mk(a.rl_steps)
                print(f"rl budget: {sps:.1f} s/step, rl_steps {planned} -> {a.rl_steps}", flush=True)
        if s in marks:
            tr.model.save(os.path.join(a.ckpt, marks[s] + ".pt"))
    t_rl = time.time() - t1
    tr.check_base()
    summ = dict(arm=a.arm, preset=a.preset, seed=a.seed, reward_mode=a.reward_mode, p_reward=rc.proto.reward,
                goal_mode=rc.proto.goal_mode, fused=rc.fused, alpha=rc.alpha, sft_steps=0 if a.init_from else a.sft_steps,
                init_from=a.init_from, rl_steps=0 if a.warm_only else a.rl_steps, rl_steps_planned=planned, t_sft_s=t_sft, t_rl_s=t_rl,
                sec_per_rl_step=t_rl / max(1, a.rl_steps), mask_violations=tr.mask_violations,
                base_checksum_unchanged=True, ckpts=sorted(os.listdir(a.ckpt)))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    print(json.dumps(summ, default=float))


if __name__ == "__main__":
    main()
