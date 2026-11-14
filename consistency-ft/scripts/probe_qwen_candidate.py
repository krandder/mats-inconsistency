#!/usr/bin/env python3
"""Bounded generation probe of the pinned, locally loaded Qwen candidate.

Three predeclared original-corpus film propositions; no resampling, weight update,
benchmark export, paid inference or claim of complete own-base anchor coverage.
Launch under ``timeout --kill-after=10s 900s`` for an enforced external wall bound;
Python signal handling can be delayed inside native kernels.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EPISODE_ID = "2028/AndChecker:11"
SEEDS = (2026091301, 2026091302, 2026091303)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (36 * 2**30, 36 * 2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (900, 910))

    def expired(signum, frame):
        raise TimeoutError("bounded candidate probe exhausted its declared time limit")

    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGXCPU, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.alarm(900)
    started = time.monotonic()
    plan = {"episode_id": EPISODE_ID, "role_seeds": SEEDS, "max_new_tokens_per_role": 32,
            "threads": 2, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "logits_processors": [], "model_mode": "eval", "use_cache": False,
            "softmax_dtype": "float32", "address_space_limit_bytes": 36 * 2**30,
            "python_alarm_seconds": 900, "cpu_time_soft_limit_seconds": 900,
            "requires_external_wall_supervisor": True,
            "selected_before_generation": True, "scientific_base_selected": False,
            "training_performed": False, "candidate_anchor_required_propositions": 2926}
    write_json(args.output / "plan.json", plan)
    records = []
    try:
        import torch
        from harness.run_native_checks import PROTOCOLS, classify_readout, parse_probability
        from training.qwen_checkpoint import load_qwen_text_checkpoint, tensor_sha256
        from training.qwen_tokenizer import QwenTextTokenizer

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        raw = (ROOT / "data/episodes/train.jsonl").read_bytes()
        episodes = [json.loads(line) for line in raw.splitlines() if line.strip()]
        episode, = [row for row in episodes if row["id"] == EPISODE_ID]
        if len(episode["prompts"]) != 3:
            raise ValueError("predeclared diagnostic episode changed")
        protocol = PROTOCOLS["numeric-v1"]
        tokenizer = QwenTextTokenizer(args.tokenizer_artifacts)
        tokenizer_identity = tokenizer.verify_runtime()
        plan.update({"train_file_sha256": digest_bytes(raw), "episode_sha256": digest_bytes(canonical(episode)),
                     "protocol": protocol, "protocol_sha256": digest_bytes(canonical(protocol)),
                     "tokenizer_identity": tokenizer_identity,
                     "eos_token_id": tokenizer.eos_token_id,
                     "model_config_eos_token_id": tokenizer.model_config["text_config"]["eos_token_id"],
                     "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                     "probe_source_sha256": digest_bytes(Path(__file__).read_bytes())})
        write_json(args.output / "plan.json", plan)
        load_started = time.monotonic()
        model, load_report = load_qwen_text_checkpoint(args.checkpoint, args.manifest,
                                                       attn_implementation="eager")
        load_report["load_wall_seconds"] = time.monotonic() - load_started
        write_json(args.output / "loaded-checkpoint.json", load_report)
        print(json.dumps({"candidate_weights_loaded": True, "text_parameters": load_report["text_parameters"],
                          "load_wall_seconds": load_report["load_wall_seconds"]}), flush=True)
        with (args.output / "token-events.jsonl").open("xb", buffering=0) as token_stream, \
                (args.output / "readouts.jsonl").open("xb", buffering=0) as readout_stream:
            for prompt, seed in zip(episode["prompts"], SEEDS):
                messages = (("system", protocol["system_instruction"]),
                            ("user", protocol["verbal_prompt"].format(proposition=prompt["text"])))
                rendered, prompt_ids = tokenizer.render_and_encode(messages)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                completion, logprobs = [], []
                role_started = time.monotonic()
                row = {"role": prompt["role"], "proposition_sha256": digest_bytes(prompt["text"].encode()),
                       "messages": messages, "messages_sha256": digest_bytes(canonical(messages)),
                       "rendered_prompt_sha256": digest_bytes(rendered.encode()), "prompt_token_ids": prompt_ids,
                       "seed": seed, "rng_before_sha256": tensor_sha256(generator.get_state()),
                       "finish_reason": "length", "readout_status": "pending"}
                try:
                    with torch.no_grad():
                        for index in range(plan["max_new_tokens_per_role"]):
                            ids = torch.tensor([list(prompt_ids) + completion], dtype=torch.long)
                            before_token = time.monotonic()
                            output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                           use_cache=False, logits_to_keep=1)
                            logits = output.logits[0, -1].float()
                            if tuple(logits.shape) != (248320,) or not torch.isfinite(logits).all():
                                raise ValueError("candidate logits are nonfinite or unexpected")
                            token = int(torch.multinomial(logits.softmax(-1), 1, generator=generator).item())
                            logprob = float(logits.log_softmax(-1)[token])
                            completion.append(token)
                            logprobs.append(logprob)
                            token_stream.write(canonical({"role": prompt["role"], "index": index,
                                "token_id": token, "logprob": logprob,
                                "wall_seconds": time.monotonic() - before_token,
                                "rng_after_sha256": tensor_sha256(generator.get_state())}) + b"\n")
                            if token == tokenizer.eos_token_id:
                                row["finish_reason"] = "eos"
                                break
                    response = tokenizer.decode_completion(completion)
                    row["response"] = response
                    row["readout_status"] = classify_readout(response) if row["finish_reason"] == "eos" else "truncated"
                    row["probability"] = parse_probability(response) if row["readout_status"] == "probability" else None
                except Exception as error:
                    row.update({"error_type": type(error).__name__, "error": str(error),
                                "readout_status": "runtime-or-decoder-error", "probability": None})
                    raise
                finally:
                    row.update({"completion_token_ids": completion, "token_logprobs": logprobs,
                                "rng_after_sha256": tensor_sha256(generator.get_state()),
                                "wall_seconds": time.monotonic() - role_started,
                                "recorded_at": datetime.now(timezone.utc).isoformat()})
                    records.append(row)
                    readout_stream.write(canonical(row) + b"\n")
                    print(json.dumps({"role": row["role"], "status": row["readout_status"],
                                      "completion_tokens": len(completion), "wall_seconds": row["wall_seconds"]}), flush=True)
        if tokenizer.verify_runtime() != tokenizer_identity:
            raise ValueError("tokenizer identity changed during probe")
        # Check actual loaded storage again; no .data/version-counter shortcut.
        for name, tensor in model.state_dict().items():
            if tensor_sha256(tensor) != load_report["text_tensors"][name]["sha256"]:
                raise ValueError("pretrained weights changed during generation: " + name)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report = {"status": "candidate-generation-probe-complete", "candidate_weights_loaded": True,
                  "candidate_forward_performed": True, "candidate_backward_performed": False,
                  "training_performed": False, "base_weights_unchanged": True,
                  "attempted_role_readouts": len(records),
                  "valid_probability_readouts": sum(r["readout_status"] == "probability" for r in records),
                  "generated_tokens": sum(len(r["completion_token_ids"]) for r in records),
                  "complete_training_anchor": False, "candidate_training_propositions_required": 2926,
                  "wall_seconds": time.monotonic() - started, "cpu_seconds": usage.ru_utime + usage.ru_stime,
                  "peak_rss_bytes_linux": usage.ru_maxrss * 1024,
                  "provider_inference_requests": 0, "new_paid_resources_allocated": False,
                  "scientific_base_selected": False,
                  "scope": "Three predeclared original film propositions; no cohort/anchor/training-quality claim."}
        write_json(args.output / "report.json", report)
        print(json.dumps(report), flush=True)
    except Exception as error:
        write_json(args.output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                   "error": str(error), "retained_readouts": len(records), "resampled": False,
                   "wall_seconds": time.monotonic() - started})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
