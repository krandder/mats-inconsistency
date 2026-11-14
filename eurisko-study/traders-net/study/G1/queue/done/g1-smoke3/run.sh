#!/bin/bash
# G1 queue smoke: prove torch sees the L4, time a matmul, write smoke.json to $OUT.
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
sleep 60
echo ok > $OUT/done.txt
