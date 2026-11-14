#!/usr/bin/env python
"""GRPO pilot: training a forecaster FOR consistency (docs/grpo-consistency-pilot-20261008.md).

One prompt = one Boolean form of one pair (v7b_explicit wording, brief reasoning, "Probability: 0.xx").
Each optimizer step takes whole pairs: ``--pairs-per-step`` pairs x 12 forms x G rollouts, so the reward
function sees all 12 forms of every pair in the batch and can score consistency per group.

Rewards (per completion i of form f in pair k; p_i parsed and clipped to [0.01, 0.99]):
  A  (accuracy)     a_i = y_f log p_i + (1 - y_f) log(1 - p_i)                     (log score, nats)
  C  (consistency)  c_i = -min(D(v_i), 6), v_i = p_i in slot f and, in the 11 other slots, the mean of
                    that form's parsed rollouts in the same step; D = sum over forms of Bernoulli
                    KL(v_g || projection_g), the kl coherence projection of scripts/coherence_projection.py
                    (one distribution over the four atoms). Within a group only p_i varies, so the group-mean
                    baseline leaves the extra incoherence that this answer causes.
  A+C               a_i + lam * c_i (lam = 1.0, pre-registered).
An unparseable completion gets the arm's floor minus 1 (A: -5.61, C: -7, A+C: -11.61), below every parsed reward.
Advantage = reward - group mean (no std division), as in training/turtel_grpo.py.

Rollouts: TRL GRPOTrainer, LoRA, vLLM server mode (one GPU serves, one GPU trains). Validation every
``--eval-every`` steps (and at step 0 = base) on all 12 forms of the validation pairs, greedy, through the same
server after a weight sync; it writes validation/metrics.jsonl and applies the stop gates:
  G1 format    validation parse rate < 0.90
  G2 collapse  share of literal answers in [0.45, 0.55] exceeds the base share by > 0.40, or the SD of the
               P and Q answers < 0.05, or their AUC against the labels < 0.55 when the base AUC is >= 0.60
  G3 kill      validation single-form log loss (P and Q) > base + 0.05 nats at two consecutive validations
  G4 train     training parse rate < 0.50 for 5 consecutive steps
A gate writes STOP (reason) in the run dir, saves a checkpoint and ends training; the job chain then stops.
Checkpoints: every ``--save-minutes`` minutes of wall time (and at the end); ``--resume`` continues from the
latest one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from training.verbalized_baseline import parse_probability  # noqa: E402
from training.supervised_pairs import FORM_ORDER  # noqa: E402

LO = 0.01
C_CAP = 6.0
FLOOR = {"acc": math.log(LO) - 1.0, "cons": -C_CAP - 1.0, "acc_cons": math.log(LO) - C_CAP - 1.0}
LITERALS = ("P", "NOT P", "Q", "NOT Q")


# ------------------------------------------------------------------------------------------ coherence

FORM_ATOMS = {
    "P": (1, 1, 0, 0), "NOT P": (0, 0, 1, 1), "Q": (1, 0, 1, 0), "NOT Q": (0, 1, 0, 1),
    "(P) AND (Q)": (1, 0, 0, 0), "(P) OR (Q)": (1, 1, 1, 0),
    "(P) AND (NOT Q)": (0, 1, 0, 0), "(P) OR (NOT Q)": (1, 1, 0, 1),
    "(NOT P) AND (Q)": (0, 0, 1, 0), "(NOT P) OR (Q)": (1, 0, 1, 1),
    "(NOT P) AND (NOT Q)": (0, 0, 0, 1), "(NOT P) OR (NOT Q)": (0, 1, 1, 1),
}


def kl_bern(s: np.ndarray, m: np.ndarray) -> np.ndarray:
    m = np.clip(m, 1e-6, 1 - 1e-6)
    return s * np.log(s / m) + (1 - s) * np.log((1 - s) / (1 - m))


def incoherence(stated: dict[str, float]) -> tuple[float, np.ndarray]:
    """D = min over atom distributions q of sum_f KL(stated_f || A_f q); returns (D, q).

    Same objective, clip, starts and solver as scripts/coherence_projection.fit(method="kl").
    """
    from coherence_projection import fit
    forms = list(stated)
    s = np.clip(np.array([stated[f] for f in forms], dtype=float), LO, 1 - LO)
    q = fit(forms, s, "kl")
    A = np.array([FORM_ATOMS[f] for f in forms], dtype=float)
    return float(np.sum(kl_bern(s, A @ q))), q


# ------------------------------------------------------------------------------------------ reward


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(str(m.get("content", "")) for m in completion if isinstance(m, dict))
    return str(completion)


def score_batch(arm: str, lam: float, pair_ids, forms, labels, texts, group_size: int) -> tuple[list[float], dict]:
    """Rewards for one generation batch (whole pairs, every form, G rollouts each). Pure; unit-tested."""
    n = len(texts)
    parsed = []
    for t in texts:
        p, _ = parse_probability(t)
        parsed.append(None if p is None else min(1 - LO, max(LO, p)))
    by_pair: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i in range(n):
        by_pair[pair_ids[i]][forms[i]].append(i)
    for pid, fm in by_pair.items():
        if set(fm) != set(FORM_ORDER) or any(len(v) != group_size for v in fm.values()):
            raise ValueError(f"batch does not hold whole pairs: {pid} has "
                             f"{ {f: len(v) for f, v in fm.items()} } (expected 12 forms x {group_size})")
    acc = [None] * n
    inc = [None] * n
    for pid, fm in by_pair.items():
        means = {}
        for f, idx in fm.items():
            vals = [parsed[i] for i in idx if parsed[i] is not None]
            if vals:
                means[f] = float(np.mean(vals))
        for f, idx in fm.items():
            others = {g: m for g, m in means.items() if g != f}
            for i in idx:
                p = parsed[i]
                if p is None:
                    continue
                y = labels[i]
                acc[i] = y * math.log(p) + (1 - y) * math.log(1 - p)
                inc[i] = incoherence({**others, f: p})[0]  # every arm: same compute, logged as a diagnostic
    rewards = []
    for i in range(n):
        if parsed[i] is None:
            rewards.append(FLOOR[arm])
            continue
        c = -min(inc[i], C_CAP)
        rewards.append(acc[i] if arm == "acc" else c if arm == "cons" else acc[i] + lam * c)
    ok = [i for i in range(n) if parsed[i] is not None]
    lit = [parsed[i] for i in ok if forms[i] in LITERALS]
    stats = {"parse_rate": len(ok) / max(n, 1),
             "log_score": float(np.mean([acc[i] for i in ok])) if ok else None,
             "incoherence": float(np.mean([inc[i] for i in ok])) if ok else None,
             "mean_p": float(np.mean([parsed[i] for i in ok])) if ok else None,
             "literal_mid_share": float(np.mean([0.45 <= p <= 0.55 for p in lit])) if lit else None,
             "literal_sd": float(np.std(lit)) if lit else None,
             "reward_mean": float(np.mean(rewards))}
    return rewards, {"parsed": parsed, "acc": acc, "inc": inc, "stats": stats}


class Reward:
    def __init__(self, args, run_dir: Path):
        self.args, self.run_dir = args, run_dir
        self.__name__ = f"reward_{args.arm}"
        self.low_parse_streak = 0
        self.stop_reason: str | None = None

    def __call__(self, prompts, completions, completion_ids=None, **kw):
        texts = [completion_text(c) for c in completions]
        started = time.time()
        rewards, extra = score_batch(self.args.arm, self.args.lam, kw["pair_id"], kw["form"], kw["label"], texts,
                                     self.args.group_size)
        stats = extra["stats"]
        stats["reward_seconds"] = round(time.time() - started, 2)
        state = kw.get("trainer_state")
        step = getattr(state, "global_step", None)
        log_metric = kw.get("log_metric")
        if log_metric:
            for k, v in stats.items():
                if v is not None:
                    log_metric(f"pilot/{k}", v)
        with (self.run_dir / "rollouts.jsonl").open("a") as fh:
            for i, t in enumerate(texts):
                fh.write(json.dumps({"step": step, "pair_id": kw["pair_id"][i], "form": kw["form"][i],
                                     "label": kw["label"][i], "p": extra["parsed"][i], "acc": extra["acc"][i],
                                     "inc": extra["inc"][i], "reward": rewards[i],
                                     "tokens": len(completion_ids[i]) if completion_ids is not None else None,
                                     "text": t[-600:]}) + "\n")
        with (self.run_dir / "train-stats.jsonl").open("a") as fh:
            fh.write(json.dumps({"step": step, "time": time.time(), **stats}) + "\n")
        self.low_parse_streak = self.low_parse_streak + 1 if stats["parse_rate"] < 0.5 else 0
        if self.low_parse_streak >= 5:
            self.stop_reason = f"G4 training parse rate < 0.50 for 5 steps (last {stats['parse_rate']:.3f})"
        return rewards


# ------------------------------------------------------------------------------------------ validation


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}]


def summarize(rows: list[dict], ps: list[float | None]) -> dict:
    """Validation metrics on all 12 forms of each pair (missing -> 0.5 for log loss, as in earlier runs)."""
    def ll(p, y):
        p = 0.5 if p is None else min(1 - LO, max(LO, p))
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))
    single = [ll(p, r["label"]) for r, p in zip(rows, ps) if r["form"] in ("P", "Q")]
    all12 = [ll(p, r["label"]) for r, p in zip(rows, ps)]
    lit = [p for r, p in zip(rows, ps) if r["form"] in LITERALS and p is not None]
    pq = [(min(1 - LO, max(LO, p)), r["label"]) for r, p in zip(rows, ps) if r["form"] in ("P", "Q") and p is not None]
    by_pair = defaultdict(dict)
    for r, p in zip(rows, ps):
        if p is not None:
            by_pair[r["pair_id"]][r["form"]] = min(1 - LO, max(LO, p))
    neg_a, neg_b, viol, incs = [], [], [], []
    for fm in by_pair.values():
        for a, na in (("P", "NOT P"), ("Q", "NOT Q")):
            if a in fm and na in fm:
                neg_a.append(fm[a]); neg_b.append(fm[na])
        for conj, l, r_ in (("(P) AND (Q)", "P", "Q"), ("(P) AND (NOT Q)", "P", "NOT Q"),
                            ("(NOT P) AND (Q)", "NOT P", "Q"), ("(NOT P) AND (NOT Q)", "NOT P", "NOT Q")):
            if conj in fm and l in fm and r_ in fm:
                viol.append(fm[conj] > min(fm[l], fm[r_]) + 0.1)
        if len(fm) >= 2:
            incs.append(incoherence(fm)[0])
    corr = float(np.corrcoef(neg_a, neg_b)[0, 1]) if len(neg_a) > 2 and np.std(neg_a) > 0 and np.std(neg_b) > 0 else None
    return {"n": len(rows), "parse_rate": sum(p is not None for p in ps) / max(len(ps), 1),
            "single_log_loss": float(np.mean(single)), "all12_log_loss": float(np.mean(all12)),
            "negation_corr": corr, "and_bound_violation_share": float(np.mean(viol)) if viol else None,
            "mean_incoherence": float(np.mean(incs)) if incs else None,
            "literal_mid_share": float(np.mean([0.45 <= p <= 0.55 for p in lit])) if lit else None,
            "literal_sd": float(np.std(lit)) if lit else None,
            "pq_sd": float(np.std([p for p, _ in pq])) if pq else None, "pq_auc": auc(pq)}


def auc(pairs: list[tuple[float, int]]) -> float | None:
    """Rank AUC of the answers against the labels (ties count one half)."""
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def write_status(run_dir: Path, line: str) -> None:
    with (run_dir / "status.txt").open("a") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + line + "\n")
    print("[status] " + line, flush=True)


def latest_checkpoint(run_dir: Path) -> Path | None:
    cands = [p for p in (run_dir / "checkpoints").glob("checkpoint-*") if (p / "trainer_state.json").exists()]
    return max(cands, key=lambda p: int(p.name.rsplit("-", 1)[1])) if cands else None


# ------------------------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["acc", "cons", "acc_cons"], required=True)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--pairs-per-step", type=int, default=4)
    p.add_argument("--per-device-batch", type=int, default=8)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.005)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-completion-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--vllm-server-url", required=True)
    p.add_argument("--vllm-group-port", type=int, default=51216)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-limit", type=int, default=None, help="validation pairs (smoke only)")
    p.add_argument("--save-minutes", type=float, default=25.0)
    p.add_argument("--kill-margin", type=float, default=0.05)
    p.add_argument("--no-validation", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    run_dir = args.run_dir
    (run_dir / "validation").mkdir(parents=True, exist_ok=True)
    if (run_dir / "STOP").exists() or (run_dir / "RUN_DONE").exists():
        write_status(run_dir, "nothing to do (STOP or RUN_DONE present)")
        return 0

    train_rows = read_jsonl(args.train)
    for a, b in zip(train_rows, train_rows[1:]):
        if b["t"] < a["t"]:
            raise SystemExit("train rows are not in time order")
    for k in range(0, len(train_rows), 12):
        chunk = train_rows[k:k + 12]
        if len({r["pair_id"] for r in chunk}) != 1 or [r["form"] for r in chunk] != list(FORM_ORDER):
            raise SystemExit(f"train rows are not pair-contiguous at row {k}")
    val_rows = read_jsonl(args.validation)
    if args.eval_limit:
        keep = list(dict.fromkeys(r["pair_id"] for r in val_rows))[:args.eval_limit]
        val_rows = [r for r in val_rows if r["pair_id"] in set(keep)]

    prompts_per_step = 12 * args.pairs_per_step
    completions = prompts_per_step * args.group_size
    if completions % args.per_device_batch:
        raise SystemExit("completions per step must be a multiple of --per-device-batch")
    grad_accum = completions // args.per_device_batch
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    template = {"enable_thinking": False}
    dataset = Dataset.from_list([{"prompt": messages(r), "pair_id": r["pair_id"], "form": r["form"],
                                  "label": int(r["label"])} for r in train_rows])
    cfg = GRPOConfig(
        output_dir=str(run_dir / "checkpoints"), seed=args.seed, learning_rate=args.lr,
        lr_scheduler_type="constant", warmup_steps=0, weight_decay=0.0, max_grad_norm=args.max_grad_norm,
        bf16=True, per_device_train_batch_size=args.per_device_batch, gradient_accumulation_steps=grad_accum,
        num_generations=args.group_size, max_completion_length=args.max_completion_tokens,
        temperature=args.temperature, top_p=1.0, beta=args.beta, epsilon=args.clip, num_iterations=1,
        scale_rewards="none", loss_type="grpo", shuffle_dataset=False, max_steps=args.max_steps,
        logging_steps=1, save_strategy="steps", save_steps=10**9, save_total_limit=2, report_to=[],
        remove_unused_columns=False, log_completions=False, mask_truncated_completions=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        chat_template_kwargs=template, use_vllm=True, vllm_mode="server",
        vllm_server_base_url=args.vllm_server_url, vllm_server_timeout=900.0, vllm_group_port=args.vllm_group_port,
    )
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                      target_modules=[t for t in args.lora_targets.split(",") if t], task_type="CAUSAL_LM")
    reward = Reward(args, run_dir)
    trainer = GRPOTrainer(model=model, reward_funcs=[reward], args=cfg, train_dataset=dataset,
                          processing_class=tokenizer, peft_config=lora)

    val_prompt_ids = []
    for r in val_rows:
        ids = tokenizer.apply_chat_template(messages(r), add_generation_prompt=True, tokenize=True,
                                            return_dict=True, **template)["input_ids"]
        val_prompt_ids.append(list(ids))
    base_path = run_dir / "validation" / "base.json"
    gate_state = {"kill_streak": 0}

    def validate(step: int) -> dict:
        vg = trainer.vllm_generation
        vg.sync_weights()
        completion_ids = []
        for k in range(0, len(val_prompt_ids), 512):  # the server accepts at most 1,024 prompts per request
            out = vg.vllm_client.generate(prompts=val_prompt_ids[k:k + 512], n=1, temperature=0.0,
                                          max_tokens=args.max_completion_tokens)
            completion_ids += out["completion_ids"]
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        ps = [parse_probability(t)[0] for t in texts]
        with (run_dir / "validation" / f"step-{step}.jsonl").open("w") as fh:
            for r, t, p in zip(val_rows, texts, ps):
                fh.write(json.dumps({"pair_id": r["pair_id"], "form": r["form"], "label": r["label"], "p": p,
                                     "text": t[-600:]}) + "\n")
        m = {"step": step, **summarize(val_rows, ps)}
        with (run_dir / "validation" / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(m) + "\n")
        print(f"[validation step {step}] " + json.dumps(m), flush=True)
        return m

    def gates(m: dict) -> str | None:
        base = json.loads(base_path.read_text())
        if m["parse_rate"] < 0.90:
            return f"G1 validation parse rate {m['parse_rate']:.3f} < 0.90"
        if m["literal_mid_share"] is not None and base.get("literal_mid_share") is not None and \
                m["literal_mid_share"] > base["literal_mid_share"] + 0.40:
            return f"G2 collapse: literal answers in [0.45,0.55] {m['literal_mid_share']:.3f} vs base {base['literal_mid_share']:.3f}"
        if m["pq_sd"] is not None and m["pq_sd"] < 0.05:
            return f"G2 collapse: SD of the P and Q answers {m['pq_sd']:.3f} < 0.05"
        if m["pq_auc"] is not None and base.get("pq_auc") is not None and base["pq_auc"] >= 0.60 and m["pq_auc"] < 0.55:
            return f"G2 collapse: P/Q discrimination AUC {m['pq_auc']:.3f} < 0.55 (base {base['pq_auc']:.3f})"
        if m["single_log_loss"] > base["single_log_loss"] + args.kill_margin:
            gate_state["kill_streak"] += 1
        else:
            gate_state["kill_streak"] = 0
        if gate_state["kill_streak"] >= 2:
            return (f"G3 kill: validation single log loss {m['single_log_loss']:.4f} > base "
                    f"{base['single_log_loss']:.4f} + {args.kill_margin} at two consecutive validations")
        return None

    class Pilot(TrainerCallback):
        def __init__(self):
            self.last_save = time.time()

        def on_train_begin(self, a, state, control, **kw):
            if not args.no_validation and not base_path.exists():
                m = validate(state.global_step)
                base_path.write_text(json.dumps(m, indent=1))
                write_status(run_dir, f"BASE_VALIDATION single_ll={m['single_log_loss']:.4f} parse={m['parse_rate']:.3f}")
            hist = run_dir / "validation" / "metrics.jsonl"
            if hist.exists() and base_path.exists():  # rebuild the kill streak after a resume
                base = json.loads(base_path.read_text())
                for line in hist.read_text().splitlines()[1:]:
                    v = json.loads(line)
                    gate_state["kill_streak"] = gate_state["kill_streak"] + 1 if \
                        v["single_log_loss"] > base["single_log_loss"] + args.kill_margin else 0

        def on_log(self, a, state, control, logs=None, **kw):
            if logs:
                with (run_dir / "metrics.jsonl").open("a") as fh:
                    fh.write(json.dumps({"step": state.global_step, "time": time.time(), **logs}) + "\n")

        def on_step_end(self, a, state, control, **kw):
            step = state.global_step
            reason = reward.stop_reason
            if not reason and not args.no_validation and args.eval_every > 0 and step % args.eval_every == 0:
                try:
                    reason = gates(validate(step))
                except Exception as error:  # a validation failure must not kill training; it is logged
                    write_status(run_dir, f"VALIDATION_ERROR step={step} {type(error).__name__}: {error}")
                    traceback.print_exc()
            if reason:
                (run_dir / "STOP").write_text(f"step {step}: {reason}\n")
                write_status(run_dir, f"STOP step={step} {reason}")
                control.should_training_stop = True
                control.should_save = True
            if time.time() - self.last_save > args.save_minutes * 60:
                control.should_save = True
            if step % 5 == 0:
                write_status(run_dir, f"STEP {step}/{state.max_steps}")

        def on_save(self, a, state, control, **kw):
            self.last_save = time.time()
            write_status(run_dir, f"CHECKPOINT step={state.global_step}")

    trainer.add_callback(Pilot())
    resume = latest_checkpoint(run_dir) if args.resume else None
    if not (run_dir / "config.json").exists():
        (run_dir / "config.json").write_text(json.dumps({
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "derived": {"completions_per_step": completions, "gradient_accumulation_steps": grad_accum,
                        "train_pairs": len(train_rows) // 12, "validation_pairs": len(val_rows) // 12},
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    write_status(run_dir, f"START arm={args.arm} resume={resume} max_steps={args.max_steps}")
    try:
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    except Exception as error:
        write_status(run_dir, f"RUN_FAILED {type(error).__name__}: {error}")
        raise
    trainer.save_model(str(run_dir / "final"))
    if not (run_dir / "STOP").exists():
        if not args.no_validation and trainer.state.global_step % args.eval_every:
            validate(trainer.state.global_step)
        (run_dir / "RUN_DONE").write_text(f"steps {trainer.state.global_step}\n")
    write_status(run_dir, f"TRAINED steps={trainer.state.global_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
