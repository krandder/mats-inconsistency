"""Generate verbalized probabilities with vLLM for one or more (split, variant) jobs.

One engine load serves every job.  Each job writes
``<out>/predictions-<split>-<variant>[-thinking].jsonl`` with the raw reply, the
parse result, one retry at temperature 0 for unparseable replies, and the label
(labels are read only to be stored for offline scoring; they never enter a prompt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from training.supervised_pairs import FORM_ORDER, load_split
from training import verbalized_baseline as vb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--job", action="append", required=True,
                    help="split:variant[:thinking][:pairs=N] e.g. validation:v2_plain_language")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--thinking-max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    import vllm

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    heartbeat = out / "heartbeat.json"
    def beat(**kw):
        heartbeat.write_text(json.dumps({"at": time.time(), **kw}))
    beat(phase="loading")
    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True, seed=args.seed)
    tok = llm.get_tokenizer()
    load_s = time.time() - t0
    (out / "engine.json").write_text(json.dumps({"vllm": vllm.__version__, "model": args.model, "max_model_len": args.max_model_len,
                                                 "gpu_memory_utilization": args.gpu_memory_utilization, "load_seconds": load_s}, indent=1))
    rates = vb.train_base_rates(args.data_root)

    def render(msgs, thinking):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)

    for job in args.job:
        parts = job.split(":")
        split, variant = parts[0], parts[1]
        thinking = "thinking" in parts[2:]
        limit = next((int(p.split("=")[1]) for p in parts[2:] if p.startswith("pairs=")), None)
        name = f"{split}-{variant}" + ("-thinking" if thinking else "") + (f"-pairs{limit}" if limit else "")
        path = out / f"predictions-{name}.jsonl"
        if path.exists():
            print("skip existing", path, flush=True); continue
        examples = load_split(args.data_root, split, labels_file="full-label-targets.jsonl")
        if limit:
            examples = examples[:limit]
        rows, prompts = [], []
        for ex in examples:
            for form in FORM_ORDER:
                msgs = vb.messages(ex, form, variant, rates=rates)
                if thinking:
                    msgs[-1]["content"] = msgs[-1]["content"].removesuffix(" /no_think")
                text = render(msgs, thinking)
                qid = ex.queries[form]["query_id"]
                rows.append({"pair_id": ex.pair["pair_id"], "query_id": qid, "form": form,
                             "observed": qid == ex.pair["observed_query_id"], "label": ex.labels[qid],
                             "p_id": ex.pair["p_id"], "q_id": ex.pair["q_id"],
                             "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(), "_msgs": msgs})
                prompts.append(text)
        if thinking:
            sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.thinking_max_tokens, seed=args.seed)
        else:
            mt = 20 if vb.VARIANTS[variant][4] is vb.ANSWER_DIRECT else args.max_tokens
            sp = SamplingParams(temperature=0.0, max_tokens=mt)
        beat(phase="generate", job=name, n=len(prompts))
        t1 = time.time()
        outs = llm.generate(prompts, sp)
        gen_s = time.time() - t1
        retry_idx = []
        for i, (row, o) in enumerate(zip(rows, outs)):
            c = o.outputs[0]
            p, why = vb.parse_probability(c.text)
            row.update(raw=c.text, finish_reason=c.finish_reason, prompt_tokens=len(o.prompt_token_ids),
                       output_tokens=len(c.token_ids), parsed_first=p, reason_first=why, retried=False,
                       raw_retry=None, p=p, reason=why)
            if p is None:
                retry_idx.append(i)
        beat(phase="retry", job=name, n=len(retry_idx))
        if retry_idx:
            rp = [render(rows[i]["_msgs"] + [{"role": "assistant", "content": rows[i]["raw"]},
                                            {"role": "user", "content": vb.RETRY_REMINDER}], False) for i in retry_idx]
            routs = llm.generate(rp, SamplingParams(temperature=0.0, max_tokens=20))
            for i, o in zip(retry_idx, routs):
                p, why = vb.parse_probability(o.outputs[0].text)
                rows[i].update(retried=True, raw_retry=o.outputs[0].text, p=p, reason=why)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for row in rows:
                row.pop("_msgs")
                f.write(json.dumps(row) + "\n")
        tmp.rename(path)
        missing = sum(r["p"] is None for r in rows)
        summary = {"job": name, "queries": len(rows), "generate_seconds": gen_s, "retries": len(retry_idx), "missing": missing,
                   "prompt_tokens": sum(r["prompt_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows)}
        with (out / "jobs.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        beat(phase="job-done", job=name)
    beat(phase="all-done")
    (out / "DONE").write_text(str(time.time()))


if __name__ == "__main__":
    main()
