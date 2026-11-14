#!/usr/bin/env bash
# Pilot on the GPU (no trained results are looked at): step timing of penalty training, and the pre-training
# validation of the propagators on random-initialised networks.
set -uo pipefail
mkdir -p "$OUT"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
for D in 4 8; do for K in 2 4; do
  python train.py --depth $D --lam 3 --K $K --seed 9 --steps 12 --eval_every 6 --out $OUT/timing_d${D}_K${K} 2>&1 | tail -1 | cut -c1-300
done; done
python train.py --depth 8 --lam 0 --seed 9 --steps 200 --eval_every 100 --out $OUT/timing_d8_l0 2>&1 | tail -1 | cut -c1-200
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
python validate_init.py --out $OUT/validate_init --n 2097152 --seeds 0 1 2 --stress 1 2 --depths 1 2 4 6 8 --device cuda 2>&1 | cut -c1-200
