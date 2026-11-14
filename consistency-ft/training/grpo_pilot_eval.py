#!/usr/bin/env python
"""Held-out evaluation for the GRPO consistency pilot: one form per call, greedy, no projection.

Runs vLLM in-process on one GPU with the base model and, optionally, a LoRA adapter, over JSONL row files
(test.jsonl = all 12 forms of each test pair; singles-external.jsonl = 2026 Metaculus/Manifold questions).
The P and Q rows of test.jsonl are the Manifold single forecasts (the same prompts as singles-test.jsonl).
Writes <out>/<tag>/<file stem>.jsonl with the parsed probability and the reply tail for every row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.verbalized_baseline import parse_probability  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    # FlashInfer's sampler JIT-compiles with ninja/nvcc, which the cluster node lacks; greedy decoding does not need it.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    kw = dict(model=args.model, revision=args.revision, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enable_prefix_caching=True, seed=0)
    if args.adapter:
        kw.update(enable_lora=True, max_lora_rank=args.max_lora_rank, max_loras=1)
    llm = LLM(**kw)
    lora = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest(args.tag, 1, str(args.adapter))
    params = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens)
    (args.out / args.tag).mkdir(parents=True, exist_ok=True)
    for path in args.rows:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()][:args.limit]
        ids = [tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                                       add_generation_prompt=True, tokenize=True, return_dict=True,
                                       enable_thinking=False)["input_ids"] for r in rows]
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": list(i)} for i in ids], params, lora_request=lora, use_tqdm=False)
        with (args.out / args.tag / f"{path.stem}.jsonl").open("w") as fh:
            for r, o in zip(rows, outs):
                text = o.outputs[0].text
                p, why = parse_probability(text)
                keep = {k: v for k, v in r.items() if k not in ("system", "user")}
                fh.write(json.dumps({**keep, "p": p, "parse": why, "tokens": len(o.outputs[0].token_ids),
                                     "text": text[-600:]}) + "\n")
        print(f"[eval {args.tag}] {path.name}: {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)
    (args.out / args.tag / "DONE").write_text(json.dumps({"adapter": str(args.adapter), "model": args.model,
                                                          "revision": args.revision}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
