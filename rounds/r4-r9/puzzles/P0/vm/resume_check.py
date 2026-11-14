"""Smoke test: a train.py run killed after its first .partial and restarted must end with the same weights and log
as an uninterrupted run (CPU, 1 thread, tiny widths). Writes <out>/resume_check.json; exit 1 on mismatch.

  python vm/resume_check.py OUTDIR
"""
import json, os, subprocess, sys, time
import torch

JOB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1]
ck = os.path.join(out, "ckpt_resume")
os.makedirs(ck, exist_ok=True)
env = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
cases = {"maze": ["--task", "maze", "--width", "8", "--T", "3", "--mode", "bansal", "--alpha", "0.01"],
         "prefix": ["--task", "prefix", "--width", "16", "--T", "5", "--mode", "chunk", "--k", "1"]}
res = {}
for case, args in cases.items():
    base = [sys.executable, "train.py", "--out", ck, "--steps", "1100", "--val_every", "800", "--probe_at", "",
            "--device", "cpu", "--seed", "3"] + args
    subprocess.run(base + ["--name", f"RU_{case}"], cwd=JOB, env=env, check=True, capture_output=True)
    part = os.path.join(ck, f"RI_{case}.partial")
    pr = subprocess.Popen(base + ["--name", f"RI_{case}"], cwd=JOB, env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    while not os.path.exists(part) and pr.poll() is None:
        time.sleep(0.05)
    pr.kill(); pr.wait()
    killed_with_partial = os.path.exists(part)
    r = subprocess.run(base + ["--name", f"RI_{case}"], cwd=JOB, env=env, capture_output=True, text=True)
    a = torch.load(os.path.join(ck, f"RU_{case}.pt"), weights_only=False)
    b = torch.load(os.path.join(ck, f"RI_{case}.pt"), weights_only=False)
    md = max((a["model"][k] - b["model"][k]).abs().max().item() for k in a["model"])
    la = [{k: v for k, v in e.items() if k != "sec"} for e in a["log"]]
    lb = [{k: v for k, v in e.items() if k != "sec"} for e in b["log"]]
    res[case] = {"killed_with_partial": killed_with_partial, "resumed": '"resumed_at"' in r.stdout,
                 "max_weight_diff": md, "logs_equal": la == lb, "ok": killed_with_partial and md == 0.0 and la == lb}
res["all_ok"] = all(v["ok"] for v in res.values())
json.dump(res, open(os.path.join(out, "resume_check.json"), "w"), indent=1)
print(json.dumps(res))
sys.exit(0 if res["all_ok"] else 1)
