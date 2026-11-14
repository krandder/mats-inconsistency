#!/usr/bin/env bash
# Remote bootstrap for the Turtel 14B replication (runs once on the rented box, from setup.py).
# Installs the pinned stack, downloads the 14B, verifies 4 x H100 and a CUDA matmul on each, records the environment.
set -euo pipefail
ROOT=/workspace/turtel
mkdir -p $ROOT/{models,runs,code,data,hf}
cd $ROOT
tar -xzf code.tar.gz -C code
tar -xzf data.tar.gz -C data
export HF_HOME=$ROOT/hf PIP_DISABLE_PIP_VERSION_CHECK=1
PY=python3
command -v uv >/dev/null 2>&1 || $PY -m pip install -q uv
# vllm 0.28.0 (pins torch==2.13.0) is expected from the image; otherwise install it (pulls torch) before the rest.
if ! $PY -c "import vllm, sys; sys.exit(0 if vllm.__version__ == '0.28.0' else 1)" 2>/dev/null; then
  echo "vllm 0.28.0 not present; installing (this pulls torch 2.13.0)"
  uv pip install --system --python $PY 'vllm==0.28.0'
fi
uv pip install --system --python $PY -r code/training/requirements-grpo.in
$PY - <<'PY'
import json, os, pathlib, subprocess, time, torch
from huggingface_hub import snapshot_download, HfApi
root = pathlib.Path('/workspace/turtel')
manifest = json.loads((root / 'manifest.json').read_text())
n = torch.cuda.device_count()
assert n == 4, f'expected 4 GPUs, found {n}'
names = [torch.cuda.get_device_name(i) for i in range(n)]
assert all('H100' in name for name in names), names
for i in range(n):  # a real kernel on every GPU (driver/runtime mismatch shows up here, before any rental time is wasted)
    a = torch.randn(2048, 2048, device=f'cuda:{i}', dtype=torch.bfloat16)
    assert torch.isfinite((a @ a).float().sum()).item()
model_id = manifest['model']['id']; revision = manifest['model'].get('revision') or 'main'
local = root / 'models' / model_id.split('/')[-1]
snapshot_download(model_id, revision=revision, local_dir=str(local), allow_patterns=['*.json', '*.safetensors', '*.txt', '*.py', '*.model'])
sha = HfApi().model_info(model_id, revision=revision).sha
import transformers, peft, trl, accelerate, datasets, vllm
env = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpus': names, 'driver': subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.split('\n')[0],
       'model': model_id, 'revision_requested': revision, 'revision_sha': sha, 'model_dir': str(local),
       'versions': {'transformers': transformers.__version__, 'peft': peft.__version__, 'trl': trl.__version__, 'accelerate': accelerate.__version__, 'datasets': datasets.__version__, 'vllm': vllm.__version__},
       'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'hostname': os.uname().nodename}
(root / 'environment.json').write_text(json.dumps(env, indent=2))
print(json.dumps(env, indent=1))
PY
$PY -m pip freeze > pip-freeze.txt 2>/dev/null || uv pip freeze --system --python $PY > pip-freeze.txt
cd $ROOT/code && $PY -m unittest test_turtel_reward -q 2>&1 | tail -2
echo BOOTSTRAP_OK
