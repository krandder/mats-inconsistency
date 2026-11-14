#!/bin/bash
# Spot A100 VMs for the prequential study: distill_gpu.sh with the study's settings
# (VMs cdistill-preq-*, own watchdog and spend, cap $15 ceiling, self-delete after 4h, ledger section in
# the distillation LEDGER.md, outputs in /home/agent/data/prequential-20260929/gpu/NAME).
#   preq_gpu.sh up cdistill-preq-a Qwen3-0.6B
#   preq_gpu.sh start cdistill-preq-a Qwen3-0.6B "Z:1 C:1 R:1 O:1 CO:1 C:2 ..."
#   preq_gpu.sh status|sync|down|delete NAME ; preq_gpu.sh watchdog|spend|list
export CDISTILL_LABEL=cdistill-preq CDISTILL_JOB=preq CDISTILL_SOURCE=consistency-preq
export CDISTILL_OUT=${CDISTILL_OUT:-/home/agent/data/prequential-20260929}
export CDISTILL_LEDGER=${CDISTILL_LEDGER:-/home/agent/data/consistency-distill-20260929/gpu/LEDGER.md}
export CDISTILL_CAP_USD=${CDISTILL_CAP_USD:-15} CDISTILL_MAX_RUN=${CDISTILL_MAX_RUN:-4h} CDISTILL_MAX_AGE_H=${CDISTILL_MAX_AGE_H:-4.05}
exec "$(dirname "$0")/distill_gpu.sh" "$@"
