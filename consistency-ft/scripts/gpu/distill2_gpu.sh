#!/bin/bash
# Checkpointed spot VMs for the v2 distillation run (1.7B students, 235B teacher): distill_gpu.sh with
# JOB=distill2 (VMs cdistill-v2-GROUPn, own watchdog and spend, cap $25 at Spot prices, self-delete after 6h,
# ledger section in the distillation LEDGER.md, synced results in
# /home/agent/data/consistency-distill-v2-20260929/gpu/GROUP/current).
#   distill2_gpu.sh watchdog &                  # spend cap / age / idle guard for every cdistill-v2 VM
#   distill2_gpu.sh supervise a "Qwen3-1.7B" Qwen3-1.7B "C@32b R@32b" --grad-ckpt
#       one supervisor per group; it re-creates preempted VMs (a1, a2, ...) and resumes from the last
#       sha256-verified sync (checkpoints every --ckpt-minutes, syncs every CDISTILL_SYNC_MIN)
#   CDISTILL_MACHINE=g2-standard-8 distill2_gpu.sh supervise t ...   # L4 (the resume test)
#   distill2_gpu.sh status|gsync|restore|delete NAME ; distill2_gpu.sh spend|list
export CDISTILL_LABEL=cdistill-v2 CDISTILL_JOB=distill2 CDISTILL_SOURCE=consistency-distill-v2
export CDISTILL_OUT=${CDISTILL_OUT:-/home/agent/data/consistency-distill-v2-20260929}
export CDISTILL_LEDGER=${CDISTILL_LEDGER:-/home/agent/data/consistency-distill-20260929/gpu/LEDGER.md}
# Kelvin's 24 h experiment budget (09-29 19:17Z): count spend at the Spot price per region, cap $25 in real
# dollars (was $30 at the on-demand list price)
export CDISTILL_SPEND_BASIS=${CDISTILL_SPEND_BASIS:-spot}
export CDISTILL_CAP_USD=${CDISTILL_CAP_USD:-25} CDISTILL_MAX_RUN=${CDISTILL_MAX_RUN:-6h} CDISTILL_MAX_AGE_H=${CDISTILL_MAX_AGE_H:-6.05}
export CDISTILL_SYNC_MIN=${CDISTILL_SYNC_MIN:-4}
# spot capacity can be gone in every zone for a while (09-29 17:53Z): keep retrying every 5 min for 3 h
export CDISTILL_MAX_UP_FAILS=${CDISTILL_MAX_UP_FAILS:-36}
if [[ ${CDISTILL_MACHINE:-a2-highgpu-1g} == a2-highgpu-1g ]]; then
  # us-west4-b held a spot A100 for 2.7 h on 09-29 while us-central1 preempted three within 1-3 h;
  # then every a2-highgpu-1g zone (A100 40GB)
  export CDISTILL_ZONES=${CDISTILL_ZONES:-"us-west4-b us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-west1-b us-west3-b europe-west4-a europe-west4-b asia-northeast1-a asia-northeast1-c asia-northeast3-a asia-northeast3-b asia-southeast1-b asia-southeast1-c me-west1-b me-west1-c"}
else
  export CDISTILL_ZONES=${CDISTILL_ZONES:-"us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west4-a us-west4-c europe-west4-a europe-west4-b europe-west4-c"}
fi
exec "$(dirname "$0")/distill_gpu.sh" "$@"
