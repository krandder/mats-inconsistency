#!/usr/bin/env python3
"""Train the auditable one-label Boolean-pair experiment with LoRA.

This is supervised BCE plus an optional named Boolean consistency loss.  It
never samples actions, computes rewards, or calls an RL trainer.  The script is
intended for the archived filtered-pair data on a GPU worker; a tiny model
smoke test can run with the same interfaces.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from training.supervised_pairs import (FORM_ORDER, LABEL_FILES, PairExample, load_split,
                                        permuted_relation_penalties, prior_pull_penalty,
                                        relation_penalties, render_prompt, prompt_parts,
                                        probability_metrics)


def auxiliary_loss(arm: str, values: dict) -> object | None:
    """Unweighted auxiliary term for an arm; None for plain accuracy.

    consistency: named Boolean relation penalties; prior: pull to 0.5/0.25/0.75;
    permuted: the same relation penalties on a fixed permutation of form names.
    """
    if arm == "accuracy":
        return None
    if arm == "consistency":
        return relation_penalties(values)["total"]
    if arm == "prior":
        return prior_pull_penalty(values)
    if arm == "permuted":
        return permuted_relation_penalties(values)["total"]
    raise ValueError(f"unknown arm {arm}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    return {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(root.rglob("*")) if path.is_file()}


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


class RunLog:
    def __init__(self, root: Path, arm: str, config: dict):
        self.root, self.arm, self.config = root, arm, config
        self.events = root / "events.jsonl"
        self.live = root / "training-live.json"
        self.started = time.time()

    def emit(self, event: str, phase: str, **fields) -> None:
        row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "event": event, "phase": phase, "arm": self.arm, **fields}
        with self.events.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
        state = {"status": "running", "arm": self.arm, "updated_at": row["timestamp"],
                 "elapsed_seconds": time.time() - self.started, **fields}
        atomic_json(self.live, state)
        print(json.dumps(row, sort_keys=True), flush=True)


def _one_token(tokenizer, text: str, name: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{name} must be exactly one token, got {ids}")
    return ids[0]


def _batch_prompts(batch: list[PairExample]) -> list[str]:
    return [render_prompt(example, form) for example in batch for form in FORM_ORDER]


def _logits(model, tokenizer, prompts: list[str], *, device: torch.device, max_length: int):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model(**encoded, use_cache=False, return_dict=True, logits_to_keep=1)  # last position only: full-sequence logits are ~20 GB per batch
    logits = output.logits[:, -1, :]
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite model logits")
    return logits, lengths


def _logits_shared(model, tokenizer, batch: list[PairExample], *, device: torch.device, max_length: int):
    """Same readout as ``_logits`` with one prefix forward per pair.

    The pair's shared prefix (instructions, pack, both questions) runs once; its
    KV cache is repeated for the twelve form suffixes, which run right-padded in
    one batch.  Gradients flow through the cached keys/values.  A prefix longer
    than ``max_length`` minus the longest suffix is cut from the left, like the
    full-prompt path.  Returns logits (len(batch) * 12, vocab) in FORM_ORDER.
    """
    rows, lengths = [], []
    n = len(FORM_ORDER)
    for example in batch:
        parts = [prompt_parts(example, form) for form in FORM_ORDER]
        prefix = parts[0][0]
        if any(part[0] != prefix for part in parts):
            raise ValueError(f"{example.pair['pair_id']} forms do not share one prefix")
        suffix_ids = [tokenizer.encode(part[1], add_special_tokens=False) for part in parts]
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        width = max(map(len, suffix_ids))
        keep = max_length - width
        if len(prefix_ids) > keep:
            prefix_ids = prefix_ids[len(prefix_ids) - keep:]
        plen = len(prefix_ids)
        out = model(input_ids=torch.tensor([prefix_ids], device=device), use_cache=True,
                    return_dict=True, logits_to_keep=1)
        cache = out.past_key_values
        cache.batch_repeat_interleave(n)
        pad = tokenizer.pad_token_id
        ids = torch.tensor([row + [pad] * (width - len(row)) for row in suffix_ids], device=device)
        mask = torch.zeros((n, plen + width), dtype=torch.long, device=device)
        mask[:, :plen] = 1
        for i, row in enumerate(suffix_ids):
            mask[i, plen:plen + len(row)] = 1
        position_ids = torch.arange(plen, plen + width, device=device).unsqueeze(0).expand(n, -1)
        suffix_out = model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                           past_key_values=cache, use_cache=True, return_dict=True)
        last = torch.tensor([len(row) - 1 for row in suffix_ids], device=device)
        rows.append(suffix_out.logits[torch.arange(n, device=device), last, :])
        lengths.extend(plen + len(row) for row in suffix_ids)
        del out, cache, suffix_out
    logits = torch.cat(rows, 0)
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite model logits")
    return logits, lengths


def _forward(model, tokenizer, batch: list[PairExample], *, device: torch.device, max_length: int,
             shared_prefix: bool = False):
    if shared_prefix:
        return _logits_shared(model, tokenizer, batch, device=device, max_length=max_length)
    return _logits(model, tokenizer, _batch_prompts(batch), device=device, max_length=max_length)


def _probs(logits, yes_id: int, no_id: int):
    pair = logits[:, [yes_id, no_id]].float()
    return pair.softmax(-1)[:, 0]


def _metrics(model, tokenizer, examples: list[PairExample], full_examples: list[PairExample],
             *, yes_id: int, no_id: int, device: torch.device, max_length: int,
             batch_size: int, predictions_path: Path | None = None, shared_prefix: bool = False) -> dict:
    """Score one split in a single pass.

    ``examples`` (one-label targets) and ``full_examples`` (all twelve labels)
    carry identical prompts.  With ``predictions_path`` every (pair, form)
    probability is written with label, observed flag and family IDs.
    """
    model.eval()
    if len(examples) != len(full_examples) or any(x.pair["pair_id"] != y.pair["pair_id"]
                                                  for x, y in zip(examples, full_examples)):
        raise ValueError("observed and full-label splits must list the same pairs in the same order")
    observed_probabilities, observed_labels = [], []
    full_probabilities, full_labels = {f: [] for f in FORM_ORDER}, {f: [] for f in FORM_ORDER}
    relation_rows, prediction_rows = [], []
    truncations = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            full_batch = full_examples[start:start + batch_size]
            logits, lengths = _forward(model, tokenizer, batch, device=device, max_length=max_length,
                                       shared_prefix=shared_prefix)
            truncations += sum(length >= max_length for length in lengths)
            probs = _probs(logits, yes_id, no_id).view(len(batch), len(FORM_ORDER))
            for index, (example, full_example) in enumerate(zip(batch, full_batch)):
                observed_form = example.observed_form
                observed_probabilities.append(float(probs[index, FORM_ORDER.index(observed_form)]))
                observed_labels.append(int(example.labels[example.pair["observed_query_id"]]))
                values = {form: probs[index, col] for col, form in enumerate(FORM_ORDER)}
                relation = {k: float(v.detach().cpu()) for k, v in relation_penalties(values).items()}
                relation["prior_pull"] = float(prior_pull_penalty(values))
                relation["permuted_total"] = float(permuted_relation_penalties(values)["total"])
                relation_rows.append(relation)
                for col, form in enumerate(FORM_ORDER):
                    query_id = full_example.queries[form]["query_id"]
                    label = full_example.labels.get(query_id)
                    probability = float(probs[index, col])
                    if label is not None:
                        full_probabilities[form].append(probability)
                        full_labels[form].append(int(label))
                    prediction_rows.append({"pair_id": example.pair["pair_id"], "query_id": query_id,
                                            "form": form, "probability": probability, "label": label,
                                            "observed": form == observed_form,
                                            "p_family": example.pair.get("p_family"),
                                            "q_family": example.pair.get("q_family")})
    if predictions_path is not None:
        with predictions_path.open("w") as stream:
            for row in prediction_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    observed = probability_metrics(observed_probabilities, observed_labels)
    full = {f: probability_metrics(full_probabilities[f], full_labels[f]) for f in FORM_ORDER if full_probabilities[f]}
    relation = {k: sum(r[k] for r in relation_rows) / len(relation_rows) for k in relation_rows[0]}
    full_log_loss = sum(m["log_loss"] for m in full.values()) / len(full) if full else None
    return {"observed": observed, "full_by_form": full, "full_mean_log_loss": full_log_loss, "relation": relation,
            "consistency_mse": relation["total"], "observed_count": len(observed_labels),
            "full_count": sum(map(len, full_labels.values())), "truncated_sequences": truncations}


def _supervised_loss(probs_row, example: PairExample, *, train_labels: str, device: torch.device):
    """Mean BCE over this example's labeled forms under the selected label mode."""
    if train_labels == "one":
        forms = {example.observed_form: example.labels[example.pair["observed_query_id"]]}
    else:
        forms = example.labeled_forms
    if not forms:
        raise ValueError(f"{example.pair['pair_id']} has no trainable label")
    losses = [F.binary_cross_entropy(probs_row[FORM_ORDER.index(f)], torch.tensor(float(y), device=device))
              for f, y in forms.items()]
    return sum(losses) / len(losses)


def preflight(args) -> dict:
    """Tokenize every prompt in every split; report lengths and truncation, no model."""
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    yes_id, no_id = _one_token(tokenizer, " YES", "YES"), _one_token(tokenizer, " NO", "NO")
    report = {"max_length": args.max_length, "tokenizer_yes_id": yes_id, "tokenizer_no_id": no_id, "splits": {}}
    for split in ("train", "validation", "test"):
        labels_file = LABEL_FILES[args.train_labels] if split == "train" else LABEL_FILES["one"]
        examples = load_split(args.data_root, split, labels_file=labels_file)
        lengths = sorted(len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in _batch_prompts(examples))
        q = lambda x: lengths[min(len(lengths) - 1, int(x * len(lengths)))]
        report["splits"][split] = {"pairs": len(examples), "prompts": len(lengths),
                                   "labeled_forms": sum(len(e.labeled_forms) for e in examples),
                                   "median": q(.5), "p95": q(.95), "p995": q(.995), "max": lengths[-1],
                                   "over_max_length": sum(n > args.max_length for n in lengths)}
    return report


def train(args) -> dict:
    data_root, output = Path(args.data_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)
    random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    train = load_split(data_root, "train", labels_file=LABEL_FILES[args.train_labels])
    valid = load_split(data_root, "validation", labels_file="one-label-targets.jsonl")
    valid_full = load_split(data_root, "validation", labels_file="full-label-targets.jsonl")
    test = load_split(data_root, "test", labels_file="one-label-targets.jsonl")
    test_full = load_split(data_root, "test", labels_file="full-label-targets.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"  # never cut the trailing readout position
    yes_id, no_id = _one_token(tokenizer, " YES", "YES"), _one_token(tokenizer, " NO", "NO")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 local_files_only=True).to(device)
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    base = model
    config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        bias="none", task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    log = RunLog(output, args.arm, vars(args))
    source_manifest = tree_manifest(data_root)
    model_path = Path(args.model)
    model_manifest = tree_manifest(model_path) if model_path.is_dir() else {"identifier": str(args.model)}
    manifest = {"schema_version": 1, "arm": args.arm, "seed": seed, "config": vars(args),
                "source_manifest": source_manifest, "forms": list(FORM_ORDER),
                "model": args.model, "model_manifest": model_manifest,
                "tokenizer_yes_id": yes_id, "tokenizer_no_id": no_id,
                "torch": torch.__version__, "python": platform.python_version(),
                "hostname": platform.node(), "world_size": torch.cuda.device_count()}
    atomic_json(output / "run-manifest.json", manifest)
    log.emit("run_started", "prepare", train_pairs=len(train), validation_pairs=len(valid), test_pairs=len(test),
             full_text=True, explicit_relation_maps=True)
    sp = args.shared_prefix
    baseline = _metrics(base, tokenizer, valid, valid_full, yes_id=yes_id, no_id=no_id, device=device,
                        max_length=args.max_length, batch_size=args.eval_batch_size, shared_prefix=sp,
                        predictions_path=output / "predictions-validation-epoch-00.jsonl")
    if args.zero_shot_test:
        zero = _metrics(base, tokenizer, test, test_full, yes_id=yes_id, no_id=no_id, device=device,
                        max_length=args.max_length, batch_size=args.eval_batch_size, shared_prefix=sp,
                        predictions_path=output / "predictions-test-zero-shot.jsonl")
        atomic_json(output / "test-metrics-zero-shot.json", zero)
        log.emit("zero_shot_test_complete", "validate", test_full_mean_log_loss=zero["full_mean_log_loss"])
    history = [{"epoch": 0, "arm": "initial", **baseline}]
    atomic_json(output / "metrics.json", history)
    best_key, best_epoch, since_best, epoch = None, 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); order = list(train); random.Random(seed + epoch).shuffle(order)
        total, steps, t0 = 0.0, 0, time.time()
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            # --pairs-per-forward pairs (12 sequences each) per forward/backward with gradient
            # accumulation: a whole 8-pair batch of ~800-token prompts needs >100 GB of activations
            # without gradient checkpointing.
            weights = [float(e.pair["pair_weight"]) if args.pair_weights else 1.0 for e in batch]
            weight_total = sum(weights)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            k = args.pairs_per_forward
            for micro in range(0, len(batch), k):
                chunk, chunk_weights = batch[micro:micro + k], weights[micro:micro + k]
                logits, _ = _forward(model, tokenizer, chunk, device=device, max_length=args.max_length,
                                     shared_prefix=sp)
                rows = _probs(logits, yes_id, no_id).view(len(chunk), len(FORM_ORDER))
                scaled = 0.0
                for index, (example, weight) in enumerate(zip(chunk, chunk_weights)):
                    row = rows[index]
                    example_loss = _supervised_loss(row, example, train_labels=args.train_labels, device=device)
                    aux = auxiliary_loss(args.arm, {form: row[col] for col, form in enumerate(FORM_ORDER)})
                    if aux is not None:
                        example_loss = example_loss + args.lambda_consistency * aux
                    scaled = scaled + example_loss * (weight / weight_total)
                scaled.backward()
                loss_value += float(scaled.detach().cpu())
            loss = torch.tensor(loss_value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach().cpu()); steps += 1
            if steps == 1 or steps % args.log_every == 0:
                log.emit("step_progress", "train", epoch=epoch, step=steps,
                         total_steps=(len(order) + args.batch_size - 1) // args.batch_size,
                         loss=total / steps, epoch_seconds=time.time() - t0, progress_pct=100 * (epoch - 1 + start / len(order)) / args.epochs)
        model.eval(); metrics = _metrics(model, tokenizer, valid, valid_full, yes_id=yes_id, no_id=no_id,
                                         device=device, max_length=args.max_length, batch_size=args.eval_batch_size,
                                         predictions_path=output / f"predictions-validation-epoch-{epoch:02d}.jsonl",
                                         shared_prefix=sp)
        record = {"epoch": epoch, "arm": args.arm, "seconds": time.time() - t0, **metrics}
        history.append(record); atomic_json(output / "metrics.json", history)
        checkpoint = output / f"epoch-{epoch:02d}"; checkpoint.mkdir(exist_ok=True)
        model.save_pretrained(checkpoint); tokenizer.save_pretrained(checkpoint)
        torch.save({"epoch": epoch, "optimizer": optimizer.state_dict(), "rng": torch.random.get_rng_state()}, checkpoint / "training-state.pt")
        if args.select_on == "full":
            key = (metrics["full_mean_log_loss"], metrics["consistency_mse"])
        else:
            key = (metrics["observed"]["log_loss"], metrics["consistency_mse"])
        if best_key is None or key < best_key:
            best_key, best_epoch, since_best = key, epoch, 0
            (output / "best.txt").write_text(str(epoch) + "\n")
        else:
            since_best += 1
        log.emit("validation_complete", "validate", since_best=since_best, **record)
        if args.patience and since_best >= args.patience:
            log.emit("early_stopped", "train", epoch=epoch, best_epoch=best_epoch, patience=args.patience)
            break
    # Test is scored once, on the best validation checkpoint, never on the last epoch.
    best_dir = output / f"epoch-{best_epoch:02d}"
    if best_epoch != epoch:
        model.load_adapter(str(best_dir), adapter_name="best")
        model.set_adapter("best")
    final = _metrics(model, tokenizer, test, test_full, yes_id=yes_id, no_id=no_id, device=device,
                     max_length=args.max_length, batch_size=args.eval_batch_size,
                     predictions_path=output / "predictions-test-best.jsonl", shared_prefix=sp)
    atomic_json(output / "test-metrics.json", final)
    manifest.update({"best_epoch": best_epoch, "history_sha256": sha256_file(output / "metrics.json"),
                     "test_metrics_sha256": sha256_file(output / "test-metrics.json"),
                     "test_checkpoint": best_dir.name, "stopped_after_epoch": epoch})
    atomic_json(output / "run-manifest.json", manifest)
    log.emit("run_completed", "train", status="complete", best_epoch=best_epoch,
             validation_log_loss=best_key[0], test_observed=final["observed"])
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=("accuracy", "consistency", "prior", "permuted"), required=True,
                        help="accuracy: BCE only; consistency: + lambda * Boolean relation penalties; "
                             "prior: + lambda * mean (p - 0.5/0.25/0.75)^2; permuted: + lambda * relation "
                             "penalties on a fixed permutation of form names (seed 20260923)")
    parser.add_argument("--train-labels", choices=tuple(LABEL_FILES), default="one",
                        help="one: observed query only; full: all twelve forms; single: P and Q only")
    parser.add_argument("--pair-weights", action="store_true",
                        help="weight each pair's loss by its pair_weight (full-label hub correction)")
    parser.add_argument("--grad-checkpointing", action="store_true", help="trade compute for activation memory")
    parser.add_argument("--patience", type=int, default=0,
                        help="stop after N epochs without improving the selection tuple (0 = never)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="tokenize all prompts, report lengths and truncation, exit without a model")
    parser.add_argument("--shared-prefix", action="store_true",
                        help="one prefix forward per pair, KV cache repeated for the 12 form suffixes")
    parser.add_argument("--select-on", choices=("observed", "full"), default="observed",
                        help="validation key for the best epoch: observed-form log loss, or mean 12-form log loss")
    parser.add_argument("--zero-shot-test", action="store_true",
                        help="also score the untrained base model on test before training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=.05)
    parser.add_argument("--lambda-consistency", type=float, default=1.0,
                        help="weight of the auxiliary term for consistency, prior and permuted arms")
    parser.add_argument("--pairs-per-forward", type=int, default=1,
                        help="pairs per forward/backward inside a batch (gradient accumulation between)")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--log-every", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1 or args.patience < 0 or args.pairs_per_forward < 1:
        parser.error("epochs, batch size, and max length must be positive; patience nonnegative")
    if args.preflight_only:
        print(json.dumps(preflight(args), indent=2, sort_keys=True))
        return
    train(args)


if __name__ == "__main__":
    main()
