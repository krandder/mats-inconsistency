"""VM pilot: seconds per training step and counted FLOPs per step of one arm (single thread)."""
import json, sys, time, torch
torch.set_num_threads(1)
import p14
arm = sys.argv[1]
s = p14.System(arm, seed=0, fee=1.0, lam=0.01, tau=1.0, grace=10 ** 9)
st = p14.Stream(0)
for _ in range(5):
    s.train_step(st)
f0, t = s.flops.total, time.time()
n = 150
for _ in range(n):
    s.train_step(st)
dt = (time.time() - t) / n
fps = (s.flops.total - f0) / n
B = p14.STEPS_V * p14.per_step_flops("V")
steps = p14.ARMS[arm].get("steps") or int(B / fps) + 1
print(json.dumps({"arm": arm, "s_per_step": round(dt, 4), "flops_per_step": fps, "steps_to_B": steps,
                  "train_min": round(steps * dt / 60, 1)}), flush=True)
