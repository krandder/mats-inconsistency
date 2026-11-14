#!/usr/bin/env python3
"""Check that trie-packed scoring (scripts/preq_student.py) equals one ordinary pass per prompt.

Takes N pairs from the stream (default 2 pairs = 24 prompts), computes P(YES) for every form
  (a) the ordinary way: each prompt alone, causal mask, logit(YES) - logit(NO) at the last position;
  (b) packed: the pair's prompts as one token trie with the tree mask and depth position ids,
and reports the largest difference in the logit, for the base model and for a LoRA model with
random nonzero adapters. Also runs one backward pass through the packed path. Exit 1 on mismatch.

  python scripts/preq_check_packing.py --stream .../stream.jsonl.gz --model .../Qwen3-0.6B [--device cuda]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from preq_student import Packer, build_batch, readout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp32", action="store_true", help="fp32 without autocast on cuda too (strict check)")
    ap.add_argument("--tol", type=float, default=None, help="max |logit diff| (default 2e-3 fp32, 0.25 bf16 cuda)")
    args = ap.parse_args()
    device = torch.device(args.device)
    cuda = device.type == "cuda" and not args.fp32  # cuda = bf16 + autocast, as in training
    dtype = torch.float32  # fp32 weights as in preq_student.py; "cuda" mode adds bf16 autocast
    tol = args.tol if args.tol is not None else (0.25 if cuda else 2e-3)

    groups = defaultdict(list)
    with gzip.open(args.stream, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["block"] == args.block:
                if r["pair_id"] not in groups and len(groups) == args.pairs:
                    continue
                groups[r["pair_id"]].append(r)
    tok = AutoTokenizer.from_pretrained(args.model)
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("YES", "NO"))
    rows = [r for g in groups.values() for r in g]
    ids = {r["idx"]: x[-1152:] for r, x in zip(rows, tok([r["prompt"] for r in rows], add_special_tokens=False)["input_ids"])}
    packer = Packer(ids)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    head = model.get_output_embeddings().weight
    wdiff = (head[yes].float() - head[no].float()).detach()

    def separate(m):
        out = []
        for r in rows:
            x = torch.tensor([ids[r["idx"]]], device=device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
                lg = m(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            out.append(float(lg[yes] - lg[no]))
        return torch.tensor(out)

    def packed(inner):
        packs = [packer.pack([r["idx"] for r in g]) for g in groups.values()]
        with torch.no_grad():
            return readout(inner, wdiff, build_batch(packs, device, dtype), cuda).cpu()

    report = {}
    a, b = separate(model), packed(model.model)
    report["base"] = float((a - b).abs().max())
    from peft import LoraConfig, get_peft_model
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    torch.manual_seed(0)
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)
    pm.eval()
    inner = pm.get_base_model().model
    a2, b2 = separate(pm), packed(inner)
    report["lora"] = float((a2 - b2).abs().max())
    report["lora_changed_logits_by"] = float((a2 - a).abs().max())
    pm.train()
    packs = [packer.pack([r["idx"] for r in g]) for g in groups.values()]
    z = readout(inner, wdiff, build_batch(packs, device, dtype), cuda)
    z.sum().backward()
    gn = sum(float(p.grad.float().norm() ** 2) for p in pm.parameters() if p.grad is not None) ** 0.5
    report.update(grad_norm=gn, prompts=len(rows), tokens_separate=sum(len(v) for v in ids.values()),
                  tokens_packed=sum(len(p[0]) for p in packs), tol=tol, dtype=str(dtype))
    ok = report["base"] <= tol and report["lora"] <= tol and gn == gn and gn > 0
    report["ok"] = ok
    print(json.dumps(report))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
