#!/usr/bin/env python
"""Turtel et al. 2025 (arXiv 2505.17989) replication: modified GRPO with LoRA on DeepSeek-R1-Distill-Qwen-14B.

Recipe (paper v4 sec. 3, as summarised in docs/self-trade-rl-proposal-20260921.md): G = 4 rollouts per
question, advantage = reward - group mean with NO division by the group std (Turtel's "modified GRPO", Dr. GRPO
style), KL beta 0.005 against the initial policy, PPO clip 0.2, AdamW (0.9, 0.999, 1e-8, no weight decay),
LR 1e-6 constant, global grad-norm clip 1.0, entropy bonus 0.001, bf16, one strict online chronological pass,
reasoning budget 4,096 tokens, scorer input truncated at 16,000 characters, reward = -(p_hat - y)^2 with the
strict rule (loss 1.0 when no probability parses) plus guardrail deductions (training/turtel_reward.py).
ReMax (paper LR 2e-6 with a learned per-output value baseline) is not implemented: TRL has no such baseline
and the modified-GRPO arm is the paper's equally good variant (Table 1: 0.192 vs 0.190).

Rollouts: TRL GRPOTrainer with a vLLM server (``--vllm-server-url``; run.sh puts it on GPU 3 and the trainer on
GPUs 0-2 with accelerate/DDP; LoRA weights are merged and pushed to the server before every generation).
Without a server URL the trainer uses HF generate (the CPU dry run).

Outputs under --run-dir: config.json, checkpoints/checkpoint-N (resumable with --resume), metrics.jsonl (every
optimizer step), rollouts.rank{k}.jsonl (every completion: id, p_hat, y, reward, guardrail fields),
validation/step-N.jsonl + validation/metrics.jsonl (held-out chronological slice every --eval-every steps),
final/ (adapter), test/predictions.jsonl + test/metrics.json (Turtel's test set, soft Brier with 0.25 for
unparseable, next to the Table 1 comparators stored in the released file), status.txt (phase lines; RUN_DONE
at the end, RUN_FAILED on an exception).

Dry run: ``--dry-run`` swaps in Qwen3-0.6B on CPU, 8 prompts, 2 optimizer steps, tiny LoRA, HF generate, and
verifies reward parsing, advantage = reward - group mean, a finite loss, one optimizer step, checkpoint
save/load and the test-set evaluation on 20 rows (see docs/turtel-replication-runbook.md).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.turtel_reward import (GuardrailConfig, expected_calibration_error, parse_probability,  # noqa: E402
                                    score_completion, soft_brier)

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DRY_RUN_MODEL = "/home/agent/data/models/Qwen/Qwen3-0.6B/6130ef31402718485ca4d80a6234f70d9a4cf362"
TEST_PARQUET = Path("/home/agent/data/forecasting-benchmarks/LightningRodLabs--outcome-rl-test-dataset/data/"
                    "train-00000-of-00001.parquet")
COMPARATORS = ("prediction_polymarket", "prediction_o1", "prediction_remax")
TABLE1 = {"base_deepseek_r1_14b": 0.215, "remax_ensemble7_100k": 0.190, "modified_grpo_100k": 0.192,
          "remax_10k_only": 0.199, "o1": 0.202, "polymarket": 0.151, "constant_0.5": 0.25}


# --------------------------------------------------------------------------------------------- arguments


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None)
    p.add_argument("--train", type=Path, default=ROOT / "data/turtel-prompts-20260922/train.jsonl")
    p.add_argument("--validation", type=Path, default=ROOT / "data/turtel-prompts-20260922/validation.jsonl")
    p.add_argument("--test-set", type=Path, default=TEST_PARQUET, help="parquet (pyarrow) or JSONL export")
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/turtel-14b-grpo")
    p.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in --run-dir")
    # algorithm
    p.add_argument("--algo", choices=["grpo-nostd", "grpo"], default="grpo-nostd",
                   help="grpo-nostd = Turtel's modified GRPO (no std division); grpo = standard normalisation")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.005, help="KL coefficient against the initial policy")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--entropy-coef", type=float, default=0.001)
    p.add_argument("--max-completion-tokens", type=int, default=4096)
    p.add_argument("--max-score-chars", type=int, default=16_000)
    p.add_argument("--temperature", type=float, default=1.0, help="rollout sampling temperature (on-policy)")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--prompts-per-step", type=int, default=6, help="questions per optimizer step (x G rollouts)")
    p.add_argument("--per-device-batch", type=int, default=2, help="sequences per GPU per micro-step")
    p.add_argument("--num-processes", type=int, default=None, help="default: WORLD_SIZE or 1")
    p.add_argument("--max-steps", type=int, default=None, help="default: one pass over --train")
    p.add_argument("--seed", type=int, default=20260922)
    # guardrails (paper gives no lambda values; these are ours)
    p.add_argument("--lambda-lang", type=float, default=0.5)
    p.add_argument("--lambda-gib", type=float, default=0.5)
    p.add_argument("--lambda-miss", type=float, default=0.25)
    p.add_argument("--min-reasoning-words", type=int, default=30)
    # LoRA
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    # generation backend
    p.add_argument("--vllm-server-url", default=None, help="e.g. http://127.0.0.1:8000; None = HF generate")
    p.add_argument("--attn", default="sdpa", help="attn_implementation for the trainer model")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--chat-template-kwargs", default="{}", help='JSON, e.g. {"enable_thinking": true} for Qwen3')
    # evaluation
    p.add_argument("--eval-every", type=int, default=100, help="optimizer steps between validation passes")
    p.add_argument("--eval-size", type=int, default=200, help="validation questions per pass (chronological head)")
    p.add_argument("--eval-samples", type=int, default=1, help="samples per question; median of parseable ones")
    p.add_argument("--eval-temperature", type=float, default=0.6)
    p.add_argument("--eval-top-p", type=float, default=0.95)
    p.add_argument("--test-samples", type=int, default=1)
    p.add_argument("--test-limit", type=int, default=None)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--eval-batch", type=int, default=32, help="prompts per generation request/batch")
    # checkpoints
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=3)
    # modes
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="evaluate --adapter (or the base model) on the test set")
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter directory for --eval-only")
    p.add_argument("--vllm-inprocess", action="store_true", help="--eval-only: generate with vllm.LLM + LoRARequest")
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    return p


def apply_dry_run_defaults(args: argparse.Namespace) -> None:
    args.model = DRY_RUN_MODEL
    args.chat_template_kwargs = json.dumps({"enable_thinking": True})
    args.prompts_per_step = 4
    args.per_device_batch = 4
    args.num_processes = 1
    args.max_steps = 2
    args.max_completion_tokens = 48
    args.lora_r, args.lora_alpha = 8, 16
    args.lora_targets = "q_proj,v_proj"
    args.eval_every = 2
    args.eval_size = 4
    args.eval_samples = 1
    args.test_limit = 20
    args.save_every = 1
    args.save_total_limit = 2
    args.eval_batch = 4
    args.vllm_server_url = None
    args.attn = "sdpa"
    args.no_gradient_checkpointing = True
    args.min_reasoning_words = 5  # 48-token completions cannot carry 30 words


# --------------------------------------------------------------------------------------------- data


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def assert_chronological(rows: list[dict], key: str = "prediction_date") -> None:
    for previous, current in zip(rows, rows[1:]):
        if current[key] < previous[key]:
            raise ValueError(f"training rows are not chronological by {key}: {previous['id']} -> {current['id']}")


def load_test_rows(path: Path, limit: int | None = None) -> list[dict]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(str(path)).to_pylist()
    else:
        rows = [r for r in read_jsonl(path) if "prompt" in r]
    out = []
    for row in rows:
        record = {"id": row.get("question_id") or row.get("id"), "prompt": row["prompt"],
                  "resolution": int(row["resolution"]), "question_close_date": row.get("question_close_date"),
                  "question_text": row.get("question_text")}
        for key in COMPARATORS:
            value = row.get(key)
            record[key] = float(value) if value is not None and not (isinstance(value, float) and math.isnan(value)) else None
        out.append(record)
        if limit is not None and len(out) >= limit:
            break
    return out


def to_conversational(rows: list[dict]):
    from datasets import Dataset
    return Dataset.from_list([{"prompt": [{"role": "user", "content": r["prompt"]}], "id": r["id"],
                               "resolution": int(r["resolution"]), "prediction_date": r["prediction_date"]}
                              for r in rows])


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(str(m.get("content", "")) for m in completion if isinstance(m, dict))
    return str(completion)


# --------------------------------------------------------------------------------------------- reward


class RolloutLog:
    """Per-rank JSONL of every scored completion; also keeps in-memory rewards for the dry-run checks."""

    def __init__(self, run_dir: Path, rank: int, guardrails: GuardrailConfig):
        self.path = run_dir / f"rollouts.rank{rank}.jsonl"
        self.guardrails = guardrails
        self.rank = rank
        self.calls: list[list[float]] = []
        self.ids_seen: list[str] = []
        self.parsed = 0
        self.total = 0

    def __call__(self, prompts, completions, completion_ids=None, **kwargs):
        resolutions = kwargs.get("resolution")
        ids = kwargs.get("id") or [None] * len(completions)
        dates = kwargs.get("prediction_date") or [None] * len(completions)
        state = kwargs.get("trainer_state")
        step = getattr(state, "global_step", None)
        rewards = []
        with self.path.open("a") as handle:
            for i, completion in enumerate(completions):
                text = completion_text(completion)
                result = score_completion(text, int(resolutions[i]), self.guardrails)
                rewards.append(result.reward)
                self.total += 1
                self.parsed += int(result.parsed)
                record = {"step": step, "rank": self.rank, "index": i, "id": ids[i], "prediction_date": dates[i],
                          "resolution": int(resolutions[i]), "completion_chars": len(text),
                          "completion_tokens": len(completion_ids[i]) if completion_ids is not None else None,
                          **result.to_dict()}
                handle.write(json.dumps(record) + "\n")
                if ids[i] is not None and (not self.ids_seen or self.ids_seen[-1] != ids[i]):
                    self.ids_seen.append(ids[i])
        self.calls.append(rewards)
        return rewards


# --------------------------------------------------------------------------------------------- generation backends


def chat_template_kwargs(args) -> dict:
    return json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else {}


def prompt_token_ids(tokenizer, prompt: str, template_kwargs: dict) -> list[int]:
    """Same call TRL's GRPOTrainer makes to tokenize a conversational prompt."""
    tokenized = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": prompt}],
                                              add_generation_prompt=True, tokenize=True, return_dict=True,
                                              **template_kwargs)
    ids = tokenized["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


class HFGenerator:
    """model.generate on the training model (LoRA enabled); used by the dry run and as a fallback."""

    def __init__(self, model, tokenizer, batch_size: int):
        self.model, self.tokenizer, self.batch_size = model, tokenizer, batch_size

    def generate(self, prompts_ids: list[list[int]], n: int, temperature: float, top_p: float,
                 max_tokens: int) -> list[list[str]]:
        import torch
        self.tokenizer.padding_side = "left"
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        outputs: list[list[str]] = []
        for start in range(0, len(prompts_ids), self.batch_size):
            chunk = prompts_ids[start:start + self.batch_size]
            width = max(len(ids) for ids in chunk)
            input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            attention = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, ids in enumerate(chunk):
                input_ids[i, width - len(ids):] = torch.tensor(ids)
                attention[i, width - len(ids):] = 1
            kwargs = {"max_new_tokens": max_tokens, "num_return_sequences": n, "pad_token_id": pad_id,
                      "do_sample": temperature > 0}
            if temperature > 0:
                kwargs.update(temperature=temperature, top_p=top_p)
            with torch.no_grad():
                generated = self.model.generate(input_ids=input_ids.to(device), attention_mask=attention.to(device),
                                                **kwargs)
            texts = self.tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
            for i in range(len(chunk)):
                outputs.append(texts[i * n:(i + 1) * n])
        if was_training:
            self.model.train()
        return outputs


class ServerGenerator:
    """OpenAI-compatible /v1/completions on the TRL-managed vLLM server (weights = last rollout sync)."""

    def __init__(self, base_url: str, batch_size: int, concurrency: int = 4, timeout: float = 7200.0):
        import requests
        self.session = requests.Session()
        self.base_url = base_url.rstrip("/")
        self.batch_size, self.concurrency, self.timeout = batch_size, concurrency, timeout
        self.model = self.session.get(f"{self.base_url}/v1/models", timeout=60).json()["data"][0]["id"]

    def _request(self, chunk: list[list[int]], n: int, temperature: float, top_p: float, max_tokens: int):
        payload = {"model": self.model, "prompt": chunk, "n": n, "temperature": temperature, "top_p": top_p,
                   "max_tokens": max_tokens}
        response = self.session.post(f"{self.base_url}/v1/completions", json=payload, timeout=self.timeout)
        response.raise_for_status()
        choices = response.json()["choices"]
        texts = [[] for _ in chunk]
        for choice in choices:
            texts[choice["index"] // n].append(choice["text"])
        return texts

    def generate(self, prompts_ids, n, temperature, top_p, max_tokens) -> list[list[str]]:
        chunks = [prompts_ids[i:i + self.batch_size] for i in range(0, len(prompts_ids), self.batch_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(pool.map(lambda c: self._request(c, n, temperature, top_p, max_tokens), chunks))
        return [texts for chunk in results for texts in chunk]


class VLLMInprocessGenerator:
    """vllm.LLM with an optional LoRA adapter (eval-only mode after training; unverified against vllm 0.28)."""

    def __init__(self, model: str, adapter: Path | None, tensor_parallel: int, max_model_len: int, lora_rank: int,
                 revision: str | None = None):
        from vllm import LLM
        kwargs = {"model": model, "tensor_parallel_size": tensor_parallel, "max_model_len": max_model_len,
                  "dtype": "bfloat16", "gpu_memory_utilization": 0.9}
        if revision:
            kwargs["revision"] = revision
        if adapter is not None:
            kwargs.update(enable_lora=True, max_lora_rank=lora_rank)
        self.llm = LLM(**kwargs)
        self.lora_request = None
        if adapter is not None:
            from vllm.lora.request import LoRARequest
            self.lora_request = LoRARequest("turtel-adapter", 1, str(adapter))

    def generate(self, prompts_ids, n, temperature, top_p, max_tokens) -> list[list[str]]:
        from vllm import SamplingParams
        params = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        outputs = self.llm.generate([{"prompt_token_ids": ids} for ids in prompts_ids], params,
                                    lora_request=self.lora_request, use_tqdm=False)
        return [[o.text for o in out.outputs] for out in outputs]


# --------------------------------------------------------------------------------------------- evaluation


def evaluate(rows: list[dict], generator, tokenizer, args, n_samples: int, out_path: Path, tag: str) -> dict:
    template_kwargs = chat_template_kwargs(args)
    started = time.time()
    prompts_ids = [prompt_token_ids(tokenizer, r["prompt"], template_kwargs) for r in rows]
    completions = generator.generate(prompts_ids, n_samples, args.eval_temperature, args.eval_top_p,
                                     args.max_completion_tokens)
    predictions = []
    for row, comps in zip(rows, completions):
        probabilities = [parse_probability(c, args.max_score_chars) for c in comps]
        valid = [p for p in probabilities if p is not None]
        p_hat = statistics.median(valid) if valid else None
        predictions.append({"id": row["id"], "resolution": row["resolution"], "probability": p_hat,
                            "samples": probabilities, "n_parsed": len(valid), "n_samples": len(comps),
                            "completion_chars": [len(c) for c in comps], "completion": comps[0][:20_000],
                            **{k: row.get(k) for k in COMPARATORS if k in row}})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for record in predictions:
            handle.write(json.dumps(record) + "\n")
    pairs = [(p["probability"], p["resolution"]) for p in predictions]
    parsed = [(p, y) for p, y in pairs if p is not None]
    eps = 1e-6
    metrics = {"tag": tag, "n": len(rows), "n_parsed": len(parsed), "parse_rate": len(parsed) / max(len(rows), 1),
               "soft_brier": soft_brier(pairs),
               "strict_brier": sum(1.0 if p is None else (p - y) ** 2 for p, y in pairs) / max(len(pairs), 1),
               "brier_parsed_only": soft_brier(parsed) if parsed else None,
               "log_loss_parsed_only": (-sum(math.log(min(max(p, eps), 1 - eps)) if y else
                                             math.log(1 - min(max(p, eps), 1 - eps)) for p, y in parsed) / len(parsed))
               if parsed else None,
               "ece_parsed_only": expected_calibration_error(parsed) if parsed else None,
               "extremization_rate": (sum(1 for p, _ in parsed if p < 0.1 or p > 0.9) / len(parsed)) if parsed else None,
               "mean_probability": (sum(p for p, _ in parsed) / len(parsed)) if parsed else None,
               "base_rate": sum(y for _, y in pairs) / max(len(pairs), 1),
               "mean_completion_chars": sum(sum(p["completion_chars"]) / len(p["completion_chars"]) for p in predictions) / max(len(predictions), 1),
               "n_samples": n_samples, "temperature": args.eval_temperature, "seconds": round(time.time() - started, 1)}
    comparators = {}
    for key in COMPARATORS:
        values = [(p.get(key), p["resolution"]) for p in predictions if p.get(key) is not None]
        if values:
            comparators[key] = {"soft_brier": soft_brier(values), "n": len(values)}
    comparators["constant_0.5"] = {"soft_brier": 0.25, "n": len(rows)}
    metrics["comparators_same_rows"] = comparators
    return metrics


# --------------------------------------------------------------------------------------------- callbacks


def make_callbacks(args, run_dir: Path, validation_rows: list[dict], tokenizer, generator_factory):
    from transformers import TrainerCallback

    class JsonlLogger(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs:
                with (run_dir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": state.global_step, "time": time.time(), **logs}) + "\n")

    class Status(TrainerCallback):
        def on_save(self, a, state, control, **kwargs):
            if state.is_world_process_zero:
                write_status(run_dir, f"CHECKPOINT step={state.global_step}")

        def on_step_end(self, a, state, control, **kwargs):
            if state.is_world_process_zero and state.global_step % 10 == 0:
                write_status(run_dir, f"STEP {state.global_step}/{state.max_steps}")

    class Validation(TrainerCallback):
        def on_step_end(self, a, state, control, **kwargs):
            if not validation_rows or args.eval_every <= 0 or state.global_step % args.eval_every:
                return
            if not state.is_world_process_zero:
                return
            model = kwargs.get("model")
            try:
                generator = generator_factory(model)
                rows = validation_rows[:args.eval_size]
                metrics = evaluate(rows, generator, tokenizer, args, args.eval_samples,
                                   run_dir / "validation" / f"step-{state.global_step}.jsonl", f"step-{state.global_step}")
                metrics["step"] = state.global_step
                with (run_dir / "validation" / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                print(f"[validation step {state.global_step}] soft_brier={metrics['soft_brier']:.4f} "
                      f"parse_rate={metrics['parse_rate']:.3f} n={metrics['n']}", flush=True)
            except Exception as error:  # validation must never kill the run
                write_status(run_dir, f"VALIDATION_ERROR step={state.global_step} {type(error).__name__}: {error}")
                traceback.print_exc()

    return [JsonlLogger(), Status(), Validation()]


def write_status(run_dir: Path, line: str) -> None:
    with (run_dir / "status.txt").open("a") as handle:
        handle.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + line + "\n")


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = [Path(p) for p in glob.glob(str(run_dir / "checkpoints" / "checkpoint-*"))]
    candidates = [c for c in candidates if (c / "trainer_state.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda c: int(c.name.rsplit("-", 1)[-1]))


def library_versions() -> dict:
    versions = {}
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "vllm"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except Exception:
            versions[name] = None
    try:
        versions["git_sha"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                             text=True, timeout=20).stdout.strip() or None
    except Exception:
        versions["git_sha"] = None
    return versions


# --------------------------------------------------------------------------------------------- main


def load_model_and_tokenizer(args, device_map=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if args.dry_run else torch.bfloat16
    kwargs = {"revision": args.revision, "attn_implementation": args.attn}
    if device_map is not None:
        kwargs["device_map"] = device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kwargs)
    except TypeError:  # transformers < 5 spelling
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **kwargs)
    return model, tokenizer


def lora_config(args):
    from peft import LoraConfig
    return LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
                      task_type="CAUSAL_LM", bias="none")


def run_eval_only(args) -> int:
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = load_test_rows(args.test_set, args.test_limit)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if args.vllm_inprocess:
        generator = VLLMInprocessGenerator(args.model, args.adapter, args.tensor_parallel, args.max_model_len,
                                           args.lora_r, args.revision)
    else:
        model, tokenizer = load_model_and_tokenizer(args, device_map="auto" if not args.dry_run else None)
        if args.adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(args.adapter))
        generator = HFGenerator(model, tokenizer, args.eval_batch)
    metrics = evaluate(rows, generator, tokenizer, args, args.test_samples, run_dir / "test" / "predictions.jsonl",
                       "test")
    metrics["table1_reference"] = TABLE1
    metrics["adapter"] = str(args.adapter) if args.adapter else None
    metrics["model"] = args.model
    (run_dir / "test" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "table1_reference"}, indent=1))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        apply_dry_run_defaults(args)
    if args.eval_only:
        return run_eval_only(args)

    import torch
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world = args.num_processes or int(os.environ.get("WORLD_SIZE", "1"))
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation").mkdir(exist_ok=True)

    train_rows = read_jsonl(args.train, limit=args.prompts_per_step * args.max_steps if args.dry_run else None)
    assert_chronological(train_rows)
    validation_rows = read_jsonl(args.validation) if args.validation and Path(args.validation).exists() else []
    if args.dry_run:
        validation_rows = validation_rows[:args.eval_size]

    completions_per_step = args.prompts_per_step * args.group_size
    micro = world * args.per_device_batch
    if completions_per_step % micro:
        raise SystemExit(f"prompts_per_step*G = {completions_per_step} is not a multiple of "
                         f"num_processes*per_device_batch = {micro}")
    grad_accum = completions_per_step // micro
    max_steps = args.max_steps or math.ceil(len(train_rows) / args.prompts_per_step)

    if rank == 0:
        config = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  "derived": {"world_size": world, "gradient_accumulation_steps": grad_accum,
                              "completions_per_step": completions_per_step, "max_steps": max_steps,
                              "train_rows": len(train_rows), "validation_rows": len(validation_rows)},
                  "versions": library_versions(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
        write_status(run_dir, f"START model={args.model} steps={max_steps} world={world}")

    model, tokenizer = load_model_and_tokenizer(args)
    guardrails = GuardrailConfig(lambda_lang=args.lambda_lang, lambda_gib=args.lambda_gib,
                                 lambda_miss=args.lambda_miss, min_reasoning_words=args.min_reasoning_words,
                                 max_score_chars=args.max_score_chars)
    reward = RolloutLog(run_dir, rank, guardrails)
    reward.__name__ = "brier_with_guardrails"

    use_cuda = torch.cuda.is_available() and not args.dry_run
    config_kwargs = dict(
        output_dir=str(run_dir / "checkpoints"), seed=args.seed,
        learning_rate=args.lr, lr_scheduler_type="constant", warmup_steps=0, weight_decay=0.0,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, max_grad_norm=args.max_grad_norm,
        bf16=use_cuda, per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=grad_accum, num_generations=args.group_size,
        max_completion_length=args.max_completion_tokens, temperature=args.temperature, top_p=args.top_p,
        beta=args.beta, epsilon=args.clip, num_iterations=1,
        scale_rewards="none" if args.algo == "grpo-nostd" else "group", loss_type="grpo",
        entropy_coef=args.entropy_coef, shuffle_dataset=False, num_train_epochs=1, max_steps=max_steps,
        logging_steps=1, save_strategy="steps", save_steps=args.save_every, save_total_limit=args.save_total_limit,
        report_to=[], remove_unused_columns=False, log_completions=False, mask_truncated_completions=False,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if not args.no_gradient_checkpointing else None,
        chat_template_kwargs=chat_template_kwargs(args) or None, dataloader_drop_last=False,
        use_cpu=not use_cuda,
    )
    if args.vllm_server_url:
        config_kwargs.update(use_vllm=True, vllm_mode="server", vllm_server_base_url=args.vllm_server_url,
                             vllm_server_timeout=600.0)
    config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(model=model, reward_funcs=[reward], args=config, train_dataset=to_conversational(train_rows),
                          processing_class=tokenizer, peft_config=lora_config(args))

    def generator_factory(trained_model):
        if args.vllm_server_url:
            return ServerGenerator(args.vllm_server_url, args.eval_batch)
        return HFGenerator(trained_model, tokenizer, args.eval_batch)

    for callback in make_callbacks(args, run_dir, validation_rows, tokenizer, generator_factory):
        trainer.add_callback(callback)

    captured_advantages: list = []
    if args.dry_run:
        original = trainer._generate_and_score_completions

        def capture(inputs):
            output = original(inputs)
            captured_advantages.append(output["advantages"].detach().float().cpu().tolist())
            return output
        trainer._generate_and_score_completions = capture
        before = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}

    resume = latest_checkpoint(run_dir) if args.resume else None
    if rank == 0 and resume:
        write_status(run_dir, f"RESUME {resume}")
    try:
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    except Exception as error:
        if rank == 0:
            write_status(run_dir, f"RUN_FAILED {type(error).__name__}: {error}")
        raise

    final_dir = run_dir / "final"
    if trainer.accelerator.is_main_process:
        trainer.model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        write_status(run_dir, f"TRAINED steps={trainer.state.global_step} final={final_dir}")

    checks = {}
    if args.dry_run:
        checks = dry_run_checks(trainer, reward, captured_advantages, before, run_dir, args)

    if not args.skip_test and trainer.accelerator.is_main_process:
        write_status(run_dir, "TEST_EVAL start")
        if args.vllm_server_url and hasattr(trainer, "vllm_generation"):
            try:  # push the post-final-step weights; otherwise the server is one step stale
                trainer.vllm_generation.sync_weights()
            except Exception as error:
                write_status(run_dir, f"TEST_EVAL weight sync failed ({type(error).__name__}); server one step stale")
        rows = load_test_rows(args.test_set, args.test_limit)
        generator = generator_factory(trainer.model)
        metrics = evaluate(rows, generator, tokenizer, args, args.test_samples, run_dir / "test" / "predictions.jsonl", "test")
        metrics["table1_reference"] = TABLE1
        metrics["trained_steps"] = trainer.state.global_step
        metrics["model"] = args.model
        (run_dir / "test" / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"[test] soft_brier={metrics['soft_brier']:.4f} parse_rate={metrics['parse_rate']:.3f} n={metrics['n']} "
              f"comparators={json.dumps(metrics['comparators_same_rows'])}", flush=True)
        checks["test_eval"] = {"n": metrics["n"], "soft_brier": metrics["soft_brier"], "parse_rate": metrics["parse_rate"]}

    if trainer.accelerator.is_main_process:
        if args.dry_run:
            (run_dir / "dry-run-checks.json").write_text(json.dumps(checks, indent=2))
            print("DRY_RUN_CHECKS " + json.dumps(checks), flush=True)
            failed = [k for k, v in checks.items() if isinstance(v, dict) and v.get("ok") is False]
            write_status(run_dir, "DRY_RUN_FAILED " + ",".join(failed) if failed else "DRY_RUN_OK")
            if failed:
                return 1
        write_status(run_dir, "RUN_DONE")
    return 0


def dry_run_checks(trainer, reward: RolloutLog, captured_advantages, before, run_dir: Path, args) -> dict:
    import torch
    from peft import PeftModel
    checks: dict = {}
    # 1. reward parsing ran on every completion
    checks["reward_parsing"] = {"completions_scored": reward.total, "parsed": reward.parsed,
                                "parse_rate": reward.parsed / max(reward.total, 1), "ok": reward.total > 0}
    # 2. advantages = reward - group mean, no std division (compare the trainer's tensor with our own arithmetic)
    adv_ok, max_err, compared = True, 0.0, 0
    for rewards, advantages in zip(reward.calls, captured_advantages):
        if len(rewards) != len(advantages):
            adv_ok = False
            continue
        for start in range(0, len(rewards), args.group_size):
            group = rewards[start:start + args.group_size]
            mean = sum(group) / len(group)
            for r, a in zip(group, advantages[start:start + args.group_size]):
                max_err = max(max_err, abs((r - mean) - a))
                compared += 1
    adv_ok = adv_ok and compared > 0 and max_err < 1e-4
    checks["advantage_no_std"] = {"ok": adv_ok, "compared": compared, "max_abs_error": max_err,
                                  "generation_calls": len(captured_advantages)}
    # 3. finite loss on every logged step
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    checks["finite_loss"] = {"ok": bool(losses) and all(math.isfinite(l) for l in losses), "losses": losses,
                             "kl": [h.get("kl") for h in trainer.state.log_history if "kl" in h]}
    # 4. optimizer stepped: LoRA weights moved
    moved = 0
    total_delta = 0.0
    for name, param in trainer.model.named_parameters():
        if name in before:
            delta = (param.detach().float().cpu() - before[name].float().cpu()).abs().max().item()
            total_delta += delta
            moved += int(delta > 0)
    checks["optimizer_step"] = {"ok": moved > 0 and trainer.state.global_step == args.max_steps,
                                "trainable_tensors": len(before), "tensors_changed": moved,
                                "global_step": trainer.state.global_step, "sum_max_abs_delta": total_delta}
    # 5. checkpoint save/load: reload the latest checkpoint adapter on a fresh base and compare LoRA weights
    checkpoint = latest_checkpoint(run_dir)
    load_ok, mismatches = False, None
    if checkpoint is not None:
        base, _ = load_model_and_tokenizer(args)
        reloaded = PeftModel.from_pretrained(base, str(checkpoint))
        reloaded_params = {n: p for n, p in reloaded.named_parameters() if "lora_" in n}
        mismatches = 0
        for name, param in trainer.model.named_parameters():
            if "lora_" in name:
                other = reloaded_params.get(name)
                if other is None or not torch.allclose(param.detach().float().cpu(), other.detach().float().cpu(), atol=1e-6):
                    mismatches += 1
        load_ok = mismatches == 0 and len(reloaded_params) > 0
    checks["checkpoint_save_load"] = {"ok": load_ok, "checkpoint": str(checkpoint) if checkpoint else None,
                                      "lora_tensor_mismatches": mismatches,
                                      "resume_would_use": str(latest_checkpoint(run_dir))}
    # 6. chronology: prompts were consumed in file order
    expected = [r["id"] for r in read_jsonl(args.train, limit=len(reward.ids_seen))]
    checks["chronological_order"] = {"ok": reward.ids_seen == expected, "ids_seen": len(reward.ids_seen)}
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
