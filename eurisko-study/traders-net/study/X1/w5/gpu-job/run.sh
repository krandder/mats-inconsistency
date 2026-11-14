#!/bin/bash
# X1 worker 5: (1) G1 smoke matmul (fresh reproduction), (2) wealth-net fresh seeds 4,5 to 23400 steps
# with the unmodified wealth-net/gpu code (same config as train-s1..s3), snapshots every 100 steps.
set -eu
python - <<PY
import json, os, time, torch
assert torch.cuda.is_available()
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize(); t = time.time()
for _ in range(50): y = x @ x
torch.cuda.synchronize(); dt = time.time() - t
r = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda, tflops_fp32=round(50*2*4096**3/dt/1e12, 2), host=os.uname().nodename)
json.dump(r, open(os.environ["OUT"] + "/smoke.json", "w"), indent=2); print(r)
PY
nvidia-smi -L > "$OUT/gpu.txt"
for s in 4 5; do
  python train.py --mode train --seed $s --steps 23400 --max-min 55 --threads 2 --out "$OUT" > "$OUT/train-s$s.log" 2>&1 &
done
wait
ls -la "$OUT" > "$OUT/listing.txt"
echo ok > "$OUT/done.txt"
