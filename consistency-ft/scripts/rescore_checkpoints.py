#!/usr/bin/env python3
"""Score every kept epoch adapter of a run on a split, offline.

No training. Loads the frozen base once, attaches each epoch adapter in turn,
and writes ``rescore-<split>-epoch-XX.json`` plus a predictions dump next to
the adapter.  Used to check whether validation-based early stopping picked the
right checkpoint when validation and test differ in composition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from train_supervised_pairs import _metrics, _one_token, atomic_json
from training.supervised_pairs import load_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory containing epoch-XX/ adapters")
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-length", type=int, default=1280)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    run = Path(a.run)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"; tok.truncation_side = "left"
    yes, no = _one_token(tok, " YES", "YES"), _one_token(tok, " NO", "NO")
    one = load_split(a.data_root, a.split, labels_file="one-label-targets.jsonl")
    full = load_split(a.data_root, a.split, labels_file="full-label-targets.jsonl")
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, local_files_only=True).to(a.device)
    base.config.use_cache = False
    epochs = sorted(d for d in run.iterdir() if d.is_dir() and d.name.startswith("epoch-"))
    model = None
    for d in epochs:
        out = run / f"rescore-{a.split}-{d.name}.json"
        if out.exists():
            print("skip", out.name); continue
        if model is None:
            model = PeftModel.from_pretrained(base, str(d), adapter_name=d.name)
        else:
            model.load_adapter(str(d), adapter_name=d.name)
        model.set_adapter(d.name)
        m = _metrics(model, tok, one, full, yes_id=yes, no_id=no, device=torch.device(a.device),
                     max_length=a.max_length, batch_size=a.batch_size,
                     predictions_path=run / f"predictions-{a.split}-{d.name}.jsonl")
        atomic_json(out, {"run": run.name, "split": a.split, "epoch": d.name, **m})
        print(run.name, d.name, a.split, "logloss %.4f brier %.4f" % (m["observed"]["log_loss"], m["observed"]["brier"]), flush=True)


if __name__ == "__main__":
    main()
