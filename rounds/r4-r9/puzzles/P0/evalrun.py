"""Evaluate one P0 model on frozen sets with p0eval.evalsuite; writes <out>/summary.json.

  python evalrun.py --model M-ref --sets m9,m13,m33,m33[0:100]@3000=m33h,m59 --probe v9 --out results/eval/M-ref
  python evalrun.py --model M1:0:best ...        (ckpt/<arm>_s<seed>[_best].pt)
  python evalrun.py --model MRfile:path/to.pt ...

A set spec is  base[a:b]@Tc=alias  (slice and T_cap override optional; alias defaults to the spec).
Strict fp32 unless --tf32.
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p0eval  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--sets", required=True)
p.add_argument("--probe", default="")
p.add_argument("--repeat", default="")
p.add_argument("--out", required=True)
p.add_argument("--frozen", default=os.path.join(HERE, "frozen"))
p.add_argument("--ckpt", default=os.path.join(HERE, "ckpt"))
p.add_argument("--tf32", action="store_true")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()

if a.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    p0eval.strict_fp32()
t0 = time.time()
all_sets, man = p0eval.load_frozen(a.frozen)
sets, order, bases = {}, [], set()
for spec in [s for s in re.split(r",(?![^\[]*\])", a.sets) if s]:
    m = re.fullmatch(r"([A-Za-z0-9_.\-]+)(?:\[(\d*):(\d*)\])?(?:@(\d+))?(?:=([A-Za-z0-9_.\-]+))?", spec)
    assert m, spec
    base, lo, hi, tc, alias = m.groups()
    bases.add(base)
    d = all_sets[base]
    if lo is not None:
        idx = np.arange(int(lo or 0), min(int(hi) if hi else d["meta"]["n"], d["meta"]["n"]))
        d = p0eval.subset(d, idx, alias or spec)
    if tc:
        d = {**d, "meta": {**d["meta"], "T_cap": int(tc)}}
    name = alias or spec
    sets[name] = d
    order.append(name)
if a.probe:
    sets[a.probe] = all_sets[a.probe]
bad = p0eval.verify_frozen({k: all_sets[k] for k in bases})
assert not bad, bad

if a.model == "M-ref" or a.model.startswith("MRfile:"):
    ad, width = p0eval.load_baseline(a.model, device=a.device)
    src = a.model
else:
    arm, seed, *w = a.model.split(":")
    which = w[0] if w else "final"
    ad, width = p0eval.load_baseline(arm, int(seed), a.device, a.ckpt, which)
    src = f"{arm}_s{seed}{'' if which == 'final' else '_best'}.pt"
summ = p0eval.evalsuite(ad, sets, a.out, a.device, width, order, probe_set=a.probe or None,
                        repeat_check=a.repeat or None)
summ.update(model=a.model, source=src, tf32=a.tf32, device=a.device, width=width,
            gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu",
            torch=torch.__version__, sec=round(time.time() - t0, 1), frozen_files=man["files"])
json.dump(summ, open(os.path.join(a.out, "summary.json.tmp"), "w"), indent=1, default=float)
os.replace(os.path.join(a.out, "summary.json.tmp"), os.path.join(a.out, "summary.json"))
print(json.dumps({"done": a.model, "sec": summ["sec"]}), flush=True)
