#!/usr/bin/env python3
"""How far bf16 scoring is from fp32, for ordinary and packed prompts (GPU, informative, not a gate).

For 6 pairs of block 20 (72 prompts): logit(YES) - logit(NO) in full fp32 with one ordinary pass per prompt
(the reference), then the same with bf16 weights + autocast and with fp32 weights + bf16 autocast (what
preq_student.py runs), each both ordinary and trie-packed; prints the max and mean |logit| error vs the
reference and the max |probability| error. Run on the VM: python preq_check_precision.py
"""
import gzip, json, sys
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from preq_student import Packer, build_batch, readout

S, M = "/root/job/bundle/stream.jsonl.gz", "/root/models/Qwen3-0.6B"
groups = defaultdict(list)
with gzip.open(S, "rt") as f:
    for line in f:
        r = json.loads(line)
        if r["block"] == 20:
            if r["pair_id"] not in groups and len(groups) == 6:
                continue
            groups[r["pair_id"]].append(r)
tok = AutoTokenizer.from_pretrained(M)
yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("YES", "NO"))
rows = [r for g in groups.values() for r in g]
ids = {r["idx"]: x[-1152:] for r, x in zip(rows, tok([r["prompt"] for r in rows], add_special_tokens=False)["input_ids"])}
packer = Packer(ids)
dev = torch.device("cuda")
packs = [packer.pack([r["idx"] for r in g]) for g in groups.values()]


def run(wdtype, autocast, packed):
    m = AutoModelForCausalLM.from_pretrained(M, torch_dtype=wdtype).to(dev).eval()
    head = m.get_output_embeddings().weight
    wd = (head[yes].float() - head[no].float()).detach()
    with torch.no_grad():
        if packed:
            z = readout(m.model, wd, build_batch(packs, dev, torch.bfloat16 if wdtype == torch.bfloat16 else torch.float32), autocast).cpu()
        else:
            out = []
            for r in rows:
                x = torch.tensor([ids[r["idx"]]], device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                    lg = m(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                out.append(float(lg[yes] - lg[no]))
            z = torch.tensor(out)
    del m; torch.cuda.empty_cache()
    return z


truth = run(torch.float32, False, False)
res = {}
for name, cfg in {"bf16w_separate": (torch.bfloat16, True, False), "bf16w_packed": (torch.bfloat16, True, True),
                  "fp32w_autocast_separate": (torch.float32, True, False), "fp32w_autocast_packed": (torch.float32, True, True),
                  "fp32_packed": (torch.float32, False, True)}.items():
    z = run(*cfg)
    d = (z - truth).abs()
    pt, pz = torch.sigmoid(truth), torch.sigmoid(z)
    res[name] = {"max_logit": round(float(d.max()), 4), "mean_logit": round(float(d.mean()), 4), "max_prob": round(float((pt - pz).abs().max()), 4)}
print(json.dumps({"prompts": len(rows), **res}))
