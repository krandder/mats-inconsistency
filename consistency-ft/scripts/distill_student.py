#!/usr/bin/env python3
"""Consistency distillation, step 2: train one student arm with LoRA and predict (GPU worker).

One call per form: the student sees one v7b_explicit statement (answer line "YES/NO"), and its
probability is the P(YES) readout at the first assistant token: sigmoid(logit YES - logit NO).
The loss is binary cross-entropy against a soft target (the same for every arm; only the target
changes):

  Z   no training (zero-shot)
  R   the teacher's raw answer                  (rows without a teacher answer are dropped)
  C   the teacher's kl-projected (coherent) answer
  O   the real outcome of every form (all 12 labels)
  CO  0.5 * projected teacher answer + 0.5 * outcome (= the mean of the C and O losses)

Which teacher gives R and C is the --bundle (bundle/ = Qwen3-32B, bundle-235b/ = Qwen3-235B-A22B).

Early stopping on validation all-12 log loss (outcomes, clip 0.01), evaluated at step 0 and every
--eval-every optimizer steps; the best adapter is kept, and the test split is scored once with it.
Outputs in --out: events.jsonl, best/ (adapter), predictions-validation.jsonl,
predictions-test.jsonl (pair_id, query_id, form, observed, label, p), result.json.

Checkpoints (spot VMs get preempted): every --ckpt-minutes of wall time, after every evaluation and
when training ends, ckpt/step-NNNNNN.pt holds the LoRA adapter, the AdamW and LR-schedule state, the
position in the data (epoch, next micro-batch), the early-stopping state and the python/torch/CUDA
RNG states; ckpt/step-NNNNNN.json (written after the .pt is complete) holds its sha256. The best
adapter of each evaluation is best-STEP/ and its validation predictions val-preds-STEP.jsonl. Every
file is written to a temporary name and renamed, so a copy taken at any moment is usable. A run
started with an --out that holds a checkpoint resumes from the newest one whose sha256 checks
(same configuration required); the data order is a function of (seed, epoch), so the resumed run
sees the same batches as an uninterrupted one. A finished arm (result.json) is not rerun.

--merge-micro runs the --grad-accum micro-batches of one optimizer step as one forward/backward
pass, each row weighted 1/(grad_accum * micro-batch rows): the same gradient as the separate
passes, in fewer, larger kernels (faster on a GPU that small batches leave idle).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ARMS = ("Z", "R", "C", "O", "CO")
LO = 0.01
# arguments that must match for a checkpoint to be resumed (the rest only change speed or logging)
SIG_KEYS = ("model", "arm", "lr", "epochs", "micro_bs", "grad_accum", "eval_every", "patience", "warmup",
            "lora_r", "lora_alpha", "lora_dropout", "max_length", "seed", "limit_pairs", "max_steps")


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    if limit:
        keep = set(sorted({r["pair_id"] for r in rows})[:limit])
        rows = [r for r in rows if r["pair_id"] in keep]
    return rows


def target(row: dict, arm: str) -> float | None:
    if arm == "R":
        return row["t_raw"]
    if arm == "C":
        return row["t_proj"]
    if arm == "O":
        return float(row["label"])
    if arm == "CO":
        return 0.5 * row["t_proj"] + 0.5 * float(row["label"])
    raise ValueError(arm)


def encode(tok, rows: list[dict], max_length: int) -> tuple[list[list[int]], int]:
    ids = tok([r["prompt"] for r in rows], add_special_tokens=False)["input_ids"]
    cut = sum(len(x) > max_length for x in ids)
    return [x[-max_length:] for x in ids], cut


def collate(seqs: list[list[int]], pad: int, device) -> dict:
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):  # left padding: the readout position is the last column
        ids[i, width - len(s):] = torch.tensor(s)
        mask[i, width - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "position_ids": pos.to(device)}


def readout(model, batch: dict, yes: int, no: int) -> torch.Tensor:
    out = model(**batch, use_cache=False, return_dict=True, logits_to_keep=1)
    z = out.logits[:, -1, yes].float() - out.logits[:, -1, no].float()
    if not torch.isfinite(z).all():
        raise ValueError("nonfinite logits")
    return z


def predict(model, seqs, pad, yes, no, device, batch_size: int) -> list[float]:
    model.eval()
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    probs = [0.0] * len(seqs)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            z = readout(model, collate([seqs[i] for i in idx], pad, device), yes, no)
            for i, v in zip(idx, torch.sigmoid(z).tolist()):
                probs[i] = v
    return probs


def log_loss(probs: list[float], rows: list[dict]) -> float:
    tot = 0.0
    for p, r in zip(probs, rows):
        p = min(1 - LO, max(LO, p)); y = r["label"]
        tot -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return tot / len(rows)


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_preds(path: Path, rows: list[dict], probs: list[float]) -> None:
    atomic_text(path, "".join(json.dumps({k: r[k] for k in ("pair_id", "query_id", "form", "observed", "label")}
                                         | {"p": p}) + "\n" for r, p in zip(rows, probs)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def batches(lengths: list[int], micro: int, seed: int, epoch: int) -> list[list[int]]:
    """Length-grouped micro-batches: shuffle, sort within chunks of 64 micro-batches, shuffle batch order."""
    rng = random.Random(seed * 1000 + epoch)
    idx = list(range(len(lengths))); rng.shuffle(idx)
    chunk = micro * 64
    out = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s:s + chunk], key=lambda i: lengths[i])
        out += [part[j:j + micro] for j in range(0, len(part), micro)]
    rng.shuffle(out)
    return out


def latest_ckpt(ck_dir: Path) -> tuple[Path, dict] | None:
    """Newest checkpoint whose .json sidecar exists and whose .pt matches its sha256."""
    for side in sorted(ck_dir.glob("step-*.json"), reverse=True):
        meta = json.loads(side.read_text())
        pt = side.with_suffix(".pt")
        if pt.exists() and sha256_file(pt) == meta["sha256"]:
            return pt, meta
        print(f"checkpoint {pt.name} missing or sha256 mismatch, trying an older one", flush=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=1152)
    ap.add_argument("--eval-bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--merge-micro", action="store_true", help="one forward/backward per optimizer step (same gradient)")
    ap.add_argument("--ckpt-minutes", type=float, default=10.0, help="training checkpoint every N minutes of wall time")
    ap.add_argument("--keep-ckpts", type=int, default=2)
    ap.add_argument("--limit-pairs", type=int, default=None, help="smoke test: first N pairs of each split")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test: stop after N optimizer steps")
    ap.add_argument("--stop-after-steps", type=int, default=None,
                    help="resume test: exit (code 3) right after the checkpoint at this step, as if preempted")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        print(f"{out}: result.json exists, arm done"); return
    device = torch.device(args.device)
    random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.time()
    ev = (out / "events.jsonl").open("a")
    sig = {k: vars(args)[k] for k in SIG_KEYS}
    sig["model"] = Path(sig["model"]).name

    def emit(event, **kw):
        row = {"t": round(time.time() - t0, 1), "wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "event": event, "arm": args.arm, **kw}
        ev.write(json.dumps(row) + "\n"); ev.flush(); print(json.dumps(row), flush=True)

    bundle = Path(args.bundle)
    train = load_rows(bundle / "train.jsonl", args.limit_pairs)
    val = load_rows(bundle / "validation.jsonl", args.limit_pairs)
    test = load_rows(bundle / "test.jsonl", args.limit_pairs)
    tok = AutoTokenizer.from_pretrained(args.model)
    yes, no = (tok.encode(w, add_special_tokens=False) for w in ("YES", "NO"))
    assert len(yes) == 1 and len(no) == 1, (yes, no)
    yes, no = yes[0], no[0]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    val_ids, val_cut = encode(tok, val, args.max_length)
    test_ids, test_cut = encode(tok, test, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    emit("start", config=vars(args), train_rows=len(train), val_rows=len(val), test_rows=len(test),
         truncated={"validation": val_cut, "test": test_cut}, yes_id=yes, no_id=no,
         torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

    if args.arm == "Z":
        vp = predict(model, val_ids, pad, yes, no, device, args.eval_bs)
        best = {"step": 0, "val_log_loss": log_loss(vp, val)}
        emit("eval", step=0, val_log_loss=best["val_log_loss"])
        write_preds(out / "predictions-validation.jsonl", val, vp)
        tp = predict(model, test_ids, pad, yes, no, device, args.eval_bs)
        write_preds(out / "predictions-test.jsonl", test, tp)
        atomic_text(out / "result.json", json.dumps({"arm": "Z", "best": best, "train_rows": 0,
                                                     "test_log_loss_all12": log_loss(tp, test),
                                                     "seconds": time.time() - t0}, indent=1))
        emit("done", test_log_loss_all12=log_loss(tp, test))
        return

    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    tr = [(r, target(r, args.arm)) for r in train]
    dropped = sum(t is None for _, t in tr)
    tr = [(r, t) for r, t in tr if t is not None]
    tr_ids, tr_cut = encode(tok, [r for r, _ in tr], args.max_length)
    targets = torch.tensor([t for _, t in tr], dtype=torch.float32)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    lengths = [len(x) for x in tr_ids]
    steps_per_epoch = math.ceil(len(tr_ids) / args.micro_bs) // args.grad_accum
    total = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)) if s >= warm
        else (s + 1) / warm)

    ck_dir = out / "ckpt"; ck_dir.mkdir(exist_ok=True)
    # training state; `pos` = (epoch, index of the next micro-batch group in that epoch's batch list)
    st = {"step": 0, "epoch": 0, "k_next": 0, "best": None, "bad": 0, "run_loss": 0.0, "n_loss": 0,
          "phase": "train", "elapsed": 0.0, "stopped_early": False}

    def save_best(step, vp):
        tmp = out / f"best-{step}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        model.save_pretrained(tmp)
        dst = out / f"best-{step}"
        if dst.exists():
            shutil.rmtree(dst)
        os.replace(tmp, dst)
        write_preds(out / f"val-preds-{step}.jsonl", val, vp)

    def save_ckpt(reason):
        state = dict(st, elapsed=st["elapsed"] + (time.time() - t_seg), sig=sig, total=total,
                     adapter=get_peft_model_state_dict(model), opt=opt.state_dict(), sched=sched.state_dict(),
                     rng={"py": random.getstate(), "torch": torch.get_rng_state(),
                          "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None})
        name = f"step-{st['step']:06d}"
        tmp = ck_dir / (name + ".pt.tmp")
        with tmp.open("wb") as f:
            torch.save(state, f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, ck_dir / (name + ".pt"))
        atomic_text(ck_dir / (name + ".json"), json.dumps(
            {"step": st["step"], "phase": st["phase"], "best_step": st["best"]["step"], "reason": reason,
             "sha256": sha256_file(ck_dir / (name + ".pt")), "wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        # keep the newest --keep-ckpts checkpoints, and the best adapters they point to
        sides = sorted(ck_dir.glob("step-*.json"))
        for old in sides[:-args.keep_ckpts]:
            old.with_suffix(".pt").unlink(missing_ok=True); old.unlink(missing_ok=True)
        keep = {json.loads(s.read_text())["best_step"] for s in sides[-args.keep_ckpts:]}
        for d in out.glob("best-*"):
            if d.suffix != ".tmp" and d.is_dir() and int(d.name.split("-")[1]) not in keep:
                shutil.rmtree(d)
        for f in out.glob("val-preds-*.jsonl"):
            if int(f.stem.split("-")[2]) not in keep:
                f.unlink()
        emit("ckpt", step=st["step"], reason=reason, phase=st["phase"], best_step=st["best"]["step"])

    emit("train_setup", train_rows=len(tr_ids), dropped_no_target=dropped, truncated_train=tr_cut,
         steps_per_epoch=steps_per_epoch, total_steps=total, trainable=sum(p.numel() for p in params),
         mean_target=float(targets.mean()))

    def evaluate():
        vp = predict(model, val_ids, pad, yes, no, device, args.eval_bs)
        model.train()
        return vp, log_loss(vp, val)

    found = latest_ckpt(ck_dir)
    t_seg = time.time()
    if found:
        pt, meta = found
        state = torch.load(pt, map_location="cpu", weights_only=False)
        if state["sig"] != sig:
            raise SystemExit(f"checkpoint {pt} was made with another configuration: {state['sig']} vs {sig}")
        set_peft_model_state_dict(model, {k: v.to(device) for k, v in state["adapter"].items()})
        opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        random.setstate(state["rng"]["py"]); torch.set_rng_state(state["rng"]["torch"])
        if device.type == "cuda" and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        for k in st:
            st[k] = state[k]
        t0 = time.time() - st["elapsed"]
        if not (out / f"best-{st['best']['step']}").is_dir():
            raise SystemExit(f"best-{st['best']['step']} missing next to {pt}")
        emit("resume", from_ckpt=pt.name, step=st["step"], epoch=st["epoch"], k_next=st["k_next"], phase=st["phase"],
             best=st["best"], elapsed=round(st["elapsed"], 1))
        del state
    else:
        vp, v = evaluate()
        st["best"] = {"step": 0, "val_log_loss": v}
        save_best(0, vp)
        emit("eval", step=0, val_log_loss=v, best=True)
        save_ckpt("eval")
    last_ckpt = time.time()

    model.train()
    ga = args.grad_accum
    while st["phase"] == "train":
        mbs = batches(lengths, args.micro_bs, args.seed, st["epoch"])
        for k in range(0, len(mbs) - ga + 1, ga):
            if k < st["k_next"]:
                continue  # done before the checkpoint this run resumed from
            group = mbs[k:k + ga]
            if args.merge_micro:
                rows = [i for mb in group for i in mb]
                w = torch.tensor([1.0 / (ga * len(mb)) for mb in group for _ in mb], device=device)
                batch = collate([tr_ids[i] for i in rows], pad, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    z = readout(model, batch, yes, no)
                per = F.binary_cross_entropy_with_logits(z, targets[rows].to(device), reduction="none")
                (per * w).sum().backward()
                s = 0
                for mb in group:  # log the same per-micro-batch means as the unmerged loop
                    st["run_loss"] += float(per[s:s + len(mb)].mean()); st["n_loss"] += 1; s += len(mb)
            else:
                for mb in group:
                    batch = collate([tr_ids[i] for i in mb], pad, device)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        z = readout(model, batch, yes, no)
                    loss = F.binary_cross_entropy_with_logits(z, targets[mb].to(device))
                    (loss / ga).backward()
                    st["run_loss"] += float(loss); st["n_loss"] += 1
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            st["step"] += 1; st["k_next"] = k + ga
            step = st["step"]
            if step % 20 == 0:
                emit("train", step=step, epoch=st["epoch"], loss=st["run_loss"] / st["n_loss"], lr=sched.get_last_lr()[0])
                st["run_loss"], st["n_loss"] = 0.0, 0
            evaluated = False
            if step % args.eval_every == 0 or step == total:
                vp, v = evaluate()
                is_best = v < st["best"]["val_log_loss"]
                if is_best:
                    st["best"] = {"step": step, "val_log_loss": v}; st["bad"] = 0
                    save_best(step, vp)
                else:
                    st["bad"] += 1
                emit("eval", step=step, val_log_loss=v, best=is_best)
                evaluated = True
                if st["bad"] >= args.patience:
                    emit("early_stop", step=step); st["phase"] = "trained"; st["stopped_early"] = True
            if step >= total:
                st["phase"] = "trained"
            if st["phase"] != "train":
                break
            if evaluated or time.time() - last_ckpt >= args.ckpt_minutes * 60:
                save_ckpt("eval" if evaluated else "timer"); last_ckpt = time.time()
            if args.stop_after_steps and step >= args.stop_after_steps:
                save_ckpt("stop-after-steps")
                emit("simulated_stop", step=step); ev.close()
                raise SystemExit(3)
        else:
            st["epoch"] += 1; st["k_next"] = 0
            continue
    if not found or found[1]["phase"] == "train":
        save_ckpt("trained")

    # one test evaluation, with the best-validation adapter (step 0 = the zero-shot model with an identity adapter)
    step = st["step"]; best = st["best"]
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best_src = out / f"best-{best['step']}"
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, best_src).to(device)
    tp = predict(model, test_ids, pad, yes, no, device, args.eval_bs)
    write_preds(out / "predictions-test.jsonl", test, tp)
    # copies go through a .tmp name and a rename, so a sync snapshot never sees a half-written file
    shutil.copyfile(out / f"val-preds-{best['step']}.jsonl", out / "predictions-validation.jsonl.tmp")
    os.replace(out / "predictions-validation.jsonl.tmp", out / "predictions-validation.jsonl")
    for d in (out / "best.tmp", out / "best"):
        if d.exists():
            shutil.rmtree(d)
    shutil.copytree(best_src, out / "best.tmp")
    os.replace(out / "best.tmp", out / "best")
    res = {"arm": args.arm, "best": best, "steps_run": step, "total_steps": total, "train_rows": len(tr_ids),
           "dropped_no_target": dropped, "stopped_early": st["stopped_early"], "test_log_loss_all12": log_loss(tp, test),
           "seconds": time.time() - t0, "config": vars(args)}
    atomic_text(out / "result.json", json.dumps(res, indent=1))
    emit("done", **{k: res[k] for k in ("best", "steps_run", "test_log_loss_all12", "seconds")})
    for f in ck_dir.glob("step-*"):  # the adapter, predictions and events stay; optimizer state is no longer needed
        f.unlink()


if __name__ == "__main__":
    main()
