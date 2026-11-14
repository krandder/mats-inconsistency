#!/bin/bash
# GCE startup script for cdistill-* VMs (runs as root at boot): Python venv with torch/transformers/peft,
# the Qwen3 student models named in instance metadata "cdistill-models", then /var/lib/cdistill/READY
# (or /var/lib/cdistill/FAILED).
exec >>/var/log/cdistill-startup.log 2>&1
set -x
mkdir -p /var/lib/cdistill /root/out /root/models
[[ -f /var/lib/cdistill/READY ]] && exit 0
fail() { echo "FAIL $*" >>/var/lib/cdistill/FAILED; exit 1; }
export DEBIAN_FRONTEND=noninteractive HOME=/root
for i in 1 2 3 4 5; do apt-get update -q && break; sleep 10; done
# a mirror download can hang for good (cdistill-v2-t1, 09-29): time out and retry
ok=
for i in 1 2 3 4; do
  timeout 300 apt-get -o Acquire::http::Timeout=30 -o Acquire::Retries=3 install -y -q python3-venv python3-pip rsync \
    && { ok=1; break; }
  dpkg --configure -a; sleep 10
done
[[ $ok ]] || fail apt
python3 -m venv /opt/venv || fail venv
/opt/venv/bin/pip install -q --upgrade pip || fail pip
/opt/venv/bin/pip install -q torch==2.7.1 transformers==4.51.3 peft==0.15.2 accelerate==1.6.0 \
  huggingface-hub==0.30.2 safetensors numpy || fail pip-install
MODELS=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/cdistill-models)
# a Hugging Face download can stall for good (cdistill-v2-a4, 09-29): time out and retry; downloads resume
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
for m in $MODELS; do
  ok=
  for i in 1 2 3 4 5; do
  timeout 600 /opt/venv/bin/python - "$m" <<'PY' && { ok=1; break; }
import sys, json
from huggingface_hub import snapshot_download, HfApi
m = sys.argv[1]
info = HfApi().model_info(f"Qwen/{m}")
p = snapshot_download(f"Qwen/{m}", revision=info.sha, local_dir=f"/root/models/{m}",
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
open(f"/root/models/{m}/REVISION", "w").write(info.sha + "\n")
PY
  sleep 5
  done
  [[ $ok ]] || fail "download $m"
done
/opt/venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" || fail cuda
{
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "torch=$(/opt/venv/bin/python -c 'import torch;print(torch.__version__, torch.version.cuda)')"
  for m in $MODELS; do echo "$m=$(cat /root/models/$m/REVISION)"; done
} >/var/lib/cdistill/READY.tmp && mv /var/lib/cdistill/READY.tmp /var/lib/cdistill/READY
