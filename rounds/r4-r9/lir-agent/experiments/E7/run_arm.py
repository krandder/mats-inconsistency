"""E7 runner: one arm x one seed, harness Trainer with two E7 additions (no harness code changed):
  --gating all|role : ModelConfig.self_stack_action True|False (hawks/E7.md H1: "all" = the policy's
                      action adapter is stacked at the self-model query positions = privileged access
                      AND reward leak; "role" = self-model runs base + belief adapter only = Binder M2).
  checkpoints saved at every eval tag (pre_rl, rlN, post_rl) for post-hoc analysis (analyze.py).
Warm start only: --rl-steps 0. RL from a warm start: --init-from WARM.pt --sft-steps 0.
All other flags are the harness's own (protocol.train CLI)."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

import torch

from protocol.train import Trainer, RunConfig, run as _unused  # noqa: F401
from protocol.rollout import ProtoConfig


class E7Trainer(Trainer):
    def __init__(self, rc, gating):
        super().__init__(rc)
        if not rc.fused:
            self.model.cfg.self_stack_action = (gating == "all")
        self.gating = gating

    def evaluate(self, tag):
        out = super().evaluate(tag)
        self.save(tag)
        return out


def parse(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gating", choices=["all", "role"], default="all")
    ap.add_argument("--label", required=True)
    TY = {"int": int, "float": float, "str": str, "Optional[float]": float, "Optional[str]": str}
    rc0 = RunConfig()
    for f in dataclasses.fields(RunConfig):
        if f.name in ("proto", "fused", "truth_losses", "freeze_bf"):
            continue
        if f.type == "bool":
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=lambda x: x.lower() in ("1", "true", "yes"),
                            default=getattr(rc0, f.name))
        else:
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=TY[f.type], default=getattr(rc0, f.name))
    a = vars(ap.parse_args(argv))
    gating, label = a.pop("gating"), a.pop("label")
    rc = RunConfig(**a, proto=ProtoConfig())
    return rc.resolve(), gating, label


def main(argv=None):
    rc, gating, label = parse(argv if argv is not None else sys.argv[1:])
    tr = E7Trainer(rc, gating)
    t0 = time.time()
    if rc.init_from:
        tr.model.load(rc.init_from)
    else:
        for i in range(rc.sft_steps):
            tr.sft_step(i)
    sft_time = time.time() - t0
    tr.evaluate("pre_rl")
    t1 = time.time()
    for s in range(rc.rl_steps):
        tr.rl_step()
        if rc.eval_every and (s + 1) % rc.eval_every == 0 and s + 1 < rc.rl_steps:
            tr.evaluate(f"rl{s + 1}")
    rl_time = time.time() - t1
    if rc.rl_steps:
        tr.evaluate("post_rl")
    tr.check_base()
    summ = {"label": label, "arm": rc.arm, "gating": gating if not rc.fused else "fused (one adapter at every position)",
            "self_stack_action": bool(tr.model.cfg.self_stack_action) if not rc.fused else None,
            "seed": rc.seed, "sft_time_s": sft_time, "rl_time_s": rl_time, "rl_steps": rc.rl_steps,
            "sec_per_rl_step": rl_time / max(rc.rl_steps, 1), "mask_violations": tr.mask_violations,
            "base_checksum_unchanged": True, "init_from": rc.init_from, "goal": tr.ecfg.goal,
            "past_action_typing": "no own past actions in any self-model context: one decision per episode, "
                                  "env.fired is empty at the belief pass; role passes are separate sequences "
                                  "(R2), so there are no token types; the self-model input is grid + evidence only",
            "self_loss": "nll of own committed first shot, phase B, every step, lr_belief (fused: same lr on the fused adapter)",
            "cuda_max_mem_gb": (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else None}
    with open(os.path.join(rc.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    print(json.dumps(summ, default=float))


if __name__ == "__main__":
    main()
