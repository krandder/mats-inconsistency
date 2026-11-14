#!/usr/bin/env python3
"""Prequential study, step 2: online LoRA fine-tuning on the time-ordered stream (GPU worker).

For block k = 0, 1, ...: first train once on every item that became usable by the block time T_k
(items of earlier blocks only, each used exactly once, single pass), then score all rows of block k
with the current model. So every row is scored before any training on it, and nothing is used
before its time:

  arm  target                             usable at
  Z    no training                        -
  R    teacher raw answer (t_raw)         the pair's as_of (the forecast date)
  C    teacher kl-projected (t_proj)      the pair's as_of
  O    outcome label                      avail_outcome (end of the closing week of the events it needs)
  CO   0.5 * t_proj + 0.5 * label         avail_outcome

Student, prompt, readout and LoRA as in scripts/distill_student.py (except: fp32 base weights with bf16
autocast, where distill_student.py loaded bf16 weights; slightly closer to fp32): P(YES) = sigmoid(logit YES -
logit NO) at the first assistant token; LoRA r16 alpha32 dropout 0.05 on all linear layers; BCE
against the soft target; AdamW lr 2e-4, 50 warmup steps then constant (an online learner has no
known horizon); about 32 rows per optimizer step (whole groups of one pair's usable forms, added
until the step holds >= 32 rows); grad-norm clip 1.0; max length 1152 (left truncation).

Speed: the 12 prompts of a pair share long prefixes (preamble, question A, question B). The forms of
one pair are packed as a token trie in ONE sequence: each token attends to its trie ancestors only
(4D attention mask), position ids = depth, and each form is read out at its own last token. This is
the same computation as 12 separate passes (checked against them in scripts/preq_check_packing.py),
with ~3-4x fewer tokens. The forms of a pair share one LoRA-dropout mask on the shared prefix.

Outputs in --out: preds.csv (idx,p for every stream row, block by block), events.jsonl (one line
per block), result.json; ckpt.pt (adapter + optimizer, rewritten after each block) is removed at the
end. A rerun with the same --out resumes after the last checkpointed block.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation with 2 jobs per GPU
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ARMS = ("Z", "R", "C", "O", "CO")
LO = 0.01


def target(row: dict, arm: str) -> float:
    return {"R": row["t_raw"], "C": row["t_proj"], "O": float(row["label"]),
            "CO": 0.5 * row["t_proj"] + 0.5 * float(row["label"])}[arm]


def usable_date(row: dict, arm: str) -> str:
    return row["as_of"] if arm in ("R", "C") else row["avail_outcome"]


def schedule(rows: list[dict], arm: str, T: list[str]) -> dict[int, list[int]]:
    """block j -> stream indices trained just before scoring block j: the first j > block(row) with T_j >= usable."""
    out = defaultdict(list)
    for r in rows:
        u = usable_date(r, arm)
        for j in range(r["block"] + 1, len(T)):
            if T[j] >= u:
                out[j].append(r["idx"]); break
    return out


class Packer:
    """Packs the token sequences of one group (one pair's forms) into a trie."""

    def __init__(self, ids: list[list[int]]):
        self.ids = ids

    def pack(self, members: list[int]):
        tok, parent, depth, child = [], [], [], [{}]
        leaves = []
        for m in members:
            cur, par = 0, -1  # child[0] = root's children; node n's children at child[n + 1]
            for t in self.ids[m]:
                node = child[cur].get(t)
                if node is None:
                    node = len(tok); tok.append(t); parent.append(par)
                    depth.append(depth[par] + 1 if par >= 0 else 0)
                    child[cur][t] = node; child.append({})
                cur, par = node + 1, node
            leaves.append(par)
        return tok, parent, depth, leaves


def build_batch(packs, device, dtype):
    """packs: list of (tok, parent, depth, leaves). Right-padded batch with a 4D additive tree mask."""
    L = max(len(p[0]) for p in packs)
    L = (L + 7) // 8 * 8
    B = len(packs)
    ids = torch.zeros((B, L), dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    allow = np.zeros((B, L, L), dtype=bool)
    bidx, lidx = [], []
    for b, (tok, parent, depth, leaves) in enumerate(packs):
        n = len(tok)
        ids[b, :n] = torch.tensor(tok); pos[b, :n] = torch.tensor(depth)
        a = allow[b]
        for i in range(n):
            if parent[i] >= 0:
                a[i, :i] = a[parent[i], :i]
            a[i, i] = True
        for i in range(n, L):  # padding attends to itself only (keeps its hidden states finite)
            a[i, i] = True
        bidx += [b] * len(leaves); lidx += leaves
    mask = torch.zeros((B, 1, L, L), dtype=dtype)
    mask.masked_fill_(~torch.from_numpy(allow).unsqueeze(1), -1e30)  # finite in fp32 and in bf16 (autocast casts it)
    return (ids.to(device), mask.to(device), pos.to(device),
            torch.tensor(bidx, device=device), torch.tensor(lidx, device=device))


def readout(inner, wdiff, batch, autocast: bool):
    ids, mask, pos, bidx, lidx = batch
    with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=autocast):
        h = inner(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state
    z = h[bidx, lidx].float() @ wdiff
    if not torch.isfinite(z).all():
        raise ValueError("nonfinite logits")
    return z


def micro_batches(groups: list[list[int]], lens: dict, cap: int) -> list[list[int]]:
    """Split groups (already ordered) into micro-batches whose padded size B * max_len stays <= cap."""
    out, cur, mx = [], [], 0
    for gi in groups:
        m = max(mx, lens[gi])
        if cur and m * (len(cur) + 1) > cap:
            out.append(cur); cur, m = [], lens[gi]
        cur.append(gi); mx = m
    if cur:
        out.append(cur)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, help="stream.jsonl.gz from scripts/preq_data.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--step-rows", type=int, default=32)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=1152)
    # 4096: two jobs share a 40 GB A100; at 6144 each job reserved ~19.7 GB and two seed-1 jobs ran out of
    # memory. The loss is summed over a step's micro-batches and divided by the step's rows, so the cap
    # changes memory and speed only, not the update.
    ap.add_argument("--train-cap", type=int, default=4096, help="max padded tokens per training micro-batch")
    ap.add_argument("--score-cap", type=int, default=24576, help="max padded tokens per scoring micro-batch")
    ap.add_argument("--limit-pairs-per-block", type=int, default=None, help="smoke test")
    ap.add_argument("--max-blocks", type=int, default=None, help="smoke test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    dtype = torch.float32  # fp32 weights; on cuda the forward runs under bf16 autocast (see preq_check_precision.py)
    t0 = time.time()
    ev = (out / "events.jsonl").open("a")

    def emit(event, **kw):
        row = {"t": round(time.time() - t0, 1), "event": event, "arm": args.arm, "seed": args.seed, **kw}
        ev.write(json.dumps(row) + "\n"); ev.flush(); print(json.dumps(row), flush=True)

    with gzip.open(args.stream, "rt") as f:
        rows = [json.loads(l) for l in f]
    if args.limit_pairs_per_block:
        keep, seen = set(), defaultdict(list)
        for r in rows:
            if r["pair_id"] not in seen[r["block"]] and len(seen[r["block"]]) < args.limit_pairs_per_block:
                seen[r["block"]].append(r["pair_id"])
        keep = {p for v in seen.values() for p in v}
        rows = [r for r in rows if r["pair_id"] in keep]
    if args.max_blocks:
        rows = [r for r in rows if r["block"] < args.max_blocks]
    by_idx = {r["idx"]: r for r in rows}
    nblocks = max(r["block"] for r in rows) + 1
    T = [min(r["as_of"] for r in rows if r["block"] == b) for b in range(nblocks)]
    block_rows = defaultdict(list)
    for r in rows:
        block_rows[r["block"]].append(r["idx"])

    tok = AutoTokenizer.from_pretrained(args.model)
    yes, no = (tok.encode(w, add_special_tokens=False) for w in ("YES", "NO"))
    assert len(yes) == 1 and len(no) == 1, (yes, no)
    enc = tok([r["prompt"] for r in rows], add_special_tokens=False)["input_ids"]
    truncated = sum(len(x) > args.max_length for x in enc)
    ids = {r["idx"]: x[-args.max_length:] for r, x in zip(rows, enc)}
    packer = Packer(ids)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.config.use_cache = False
    head = model.get_output_embeddings().weight
    wdiff = (head[yes[0]].float() - head[no[0]].float()).detach()
    sched_rows = schedule(rows, args.arm, T) if args.arm != "Z" else {}

    if args.arm != "Z":
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        inner = model.get_base_model().model
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        warm = args.warmup_steps
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    else:
        inner = model.model

    # resume
    start_block, step = 0, 0
    ck = out / "ckpt.pt"
    preds_path = out / "preds.csv"
    if ck.exists():
        state = torch.load(ck, map_location=device, weights_only=False)
        start_block, step = state["next_block"], state["step"]
        if args.arm != "Z":
            set_peft_model_state_dict(model, state["adapter"])
            opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        done = {i for b in range(start_block) for i in block_rows[b]}
        kept = [l for l in preds_path.read_text().splitlines()[1:] if int(l.split(",")[0]) in done]
        preds_path.write_text("idx,p\n" + "".join(l + "\n" for l in kept))
        emit("resume", next_block=start_block, step=step, kept_preds=len(kept))
    elif (out / "result.json").exists():
        print("already done"); return
    else:
        preds_path.write_text("idx,p\n")
        emit("start", config=vars(args), rows=len(rows), blocks=nblocks, truncated=truncated,
             yes_id=yes[0], no_id=no[0], torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if cuda else "cpu",
             scheduled_train_rows=sum(len(v) for v in sched_rows.values()))

    def split_pack(members: list[int], cap: int) -> list[tuple[list[int], tuple]]:
        pk = packer.pack(members)
        if len(pk[0]) <= cap or len(members) == 1:
            return [(members, pk)]
        h = len(members) // 2
        return split_pack(members[:h], cap) + split_pack(members[h:], cap)

    def groups_of(idxs: list[int]) -> list[list[int]]:
        g = defaultdict(list)
        for i in idxs:
            g[by_idx[i]["pair_id"]].append(i)
        return [sorted(v) for v in g.values()]

    for k in range(start_block, nblocks):
        tb = time.time()
        info = {"block": k, "T": T[k]}
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        # 1) train once on the items that became usable by T_k
        todo = sched_rows.get(k, [])
        if todo:
            model.train()
            rng = random.Random(args.seed * 1000 + k)
            torch.manual_seed(args.seed * 1000 + k)
            groups = groups_of(todo); rng.shuffle(groups)
            # a group whose trie is longer than --train-cap is cut into sub-packs (only pairs with left-truncated
            # prompts, which lose their shared prefix: 1 of 3,652 pairs, 8,654 tokens); the step keeps the group
            units, unit_ids = [], []
            for g in groups:
                unit_ids.append([])
                for m, pk in split_pack(g, args.train_cap):
                    unit_ids[-1].append(len(units)); units.append((m, pk))
            ulens = {u: len(pk[0]) for u, (_, pk) in enumerate(units)}
            lens = {gi: sum(ulens[u] for u in us) for gi, us in enumerate(unit_ids)}
            # steps of >= step_rows rows, length-grouped inside windows of 24 groups, step order shuffled
            order = list(range(len(groups)))
            windows = [sorted(order[s:s + 24], key=lambda i: lens[i]) for s in range(0, len(order), 24)]
            steps, cur, n = [], [], 0
            for w in windows:
                for gi in w:
                    cur.append(gi); n += len(groups[gi])
                    if n >= args.step_rows:
                        steps.append(cur); cur, n = [], 0
            if cur:
                steps.append(cur)
            rng.shuffle(steps)
            tot_loss, tot_rows = 0.0, 0
            for st in steps:
                nrows = sum(len(groups[gi]) for gi in st)
                for mb in micro_batches([u for gi in st for u in unit_ids[gi]], ulens, args.train_cap):
                    batch = build_batch([units[u][1] for u in mb], device, dtype)
                    z = readout(inner, wdiff, batch, cuda)
                    tg = torch.tensor([target(by_idx[i], args.arm) for u in mb for i in units[u][0]],
                                      dtype=torch.float32, device=device)
                    loss = F.binary_cross_entropy_with_logits(z, tg, reduction="sum")
                    (loss / nrows).backward()
                    tot_loss += float(loss); tot_rows += len(tg)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
            info.update(train_rows=tot_rows, train_groups=len(groups), steps=len(steps), step=step,
                        train_loss=tot_loss / max(1, tot_rows), lr=sched.get_last_lr()[0],
                        train_s=round(time.time() - tb, 1))
        else:
            info.update(train_rows=0, step=step)
        # 2) score block k (all forms of every pair), before any training on it
        ts = time.time()
        model.eval()
        groups = groups_of(block_rows[k])
        packs = [packer.pack(g) for g in groups]
        order = sorted(range(len(groups)), key=lambda i: len(packs[i][0]))
        lens = {i: len(packs[i][0]) for i in order}
        probs = {}
        with torch.no_grad():
            for mb in micro_batches(order, lens, args.score_cap):
                z = readout(inner, wdiff, build_batch([packs[gi] for gi in mb], device, dtype), cuda)
                for i, v in zip([i for gi in mb for i in groups[gi]], torch.sigmoid(z).tolist()):
                    probs[i] = v
        bits = 0.0
        for i, p in probs.items():
            q = min(1 - LO, max(LO, p)); y = by_idx[i]["label"]
            bits -= math.log2(q if y else 1 - q)
        with preds_path.open("a") as f:
            f.write("".join(f"{i},{probs[i]:.6f}\n" for i in sorted(probs)))
        info.update(score_rows=len(probs), bits=bits, bits_per_row=bits / len(probs), score_s=round(time.time() - ts, 1),
                    packed_tokens=sum(lens.values()), unpacked_tokens=sum(len(ids[i]) for i in block_rows[k]),
                    max_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None,
                    block_s=round(time.time() - tb, 1))
        emit("block", **info)
        # 3) checkpoint (atomic)
        state = {"next_block": k + 1, "step": step}
        if args.arm != "Z":
            state.update(adapter=get_peft_model_state_dict(model), opt=opt.state_dict(), sched=sched.state_dict())
        torch.save(state, str(ck) + ".tmp"); os.replace(str(ck) + ".tmp", ck)

    n = sum(1 for _ in preds_path.open()) - 1
    assert n == len(rows), (n, len(rows))
    tot = 0.0
    for l in preds_path.read_text().splitlines()[1:]:
        i, p = l.split(","); q = min(1 - LO, max(LO, float(p))); y = by_idx[int(i)]["label"]
        tot -= math.log2(q if y else 1 - q)
    res = {"arm": args.arm, "seed": args.seed, "rows": n, "blocks": nblocks, "steps": step, "total_bits": tot,
           "bits_per_row": tot / n, "trained_rows": sum(len(v) for v in sched_rows.values()),
           "truncated_prompts": truncated, "seconds_this_session": time.time() - t0, "config": vars(args)}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    ck.unlink(missing_ok=True)
    emit("done", total_bits=tot, bits_per_row=tot / n, steps=step)


if __name__ == "__main__":
    main()
