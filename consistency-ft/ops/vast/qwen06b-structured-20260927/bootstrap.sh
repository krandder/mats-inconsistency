#!/usr/bin/env bash
set -euo pipefail
cd /workspace/consistency
mkdir -p models runs code data
tar -xzf code.tar.gz -C code
tar -xzf data.tar.gz -C data
python -m pip install -q 'transformers==4.51.3' 'peft==0.15.2' 'huggingface-hub==0.30.2' 'safetensors==0.5.3' 'accelerate==1.6.0'
python - <<'PY'
import json,pathlib,torch
from huggingface_hub import snapshot_download
p=pathlib.Path('/workspace/consistency')
n=torch.cuda.device_count(); assert n>=1
assert all('H100' in torch.cuda.get_device_name(i) for i in range(n))
revision='6130ef31402718485ca4d80a6234f70d9a4cf362'
snapshot_download('Qwen/Qwen3-0.6B',revision=revision,local_dir=p/'models/Qwen3-0.6B')
(p/'environment.json').write_text(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(n)],'model':'Qwen/Qwen3-0.6B','revision':revision},indent=2))
PY
sha256sum models/Qwen3-0.6B/*.safetensors > model-sha256.txt
python -m pip freeze > pip-freeze.txt
echo BOOTSTRAP_OK
