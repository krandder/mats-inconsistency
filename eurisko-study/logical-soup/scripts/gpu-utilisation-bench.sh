#!/usr/bin/env bash
set -euo pipefail
cd /root/soup
export BEND_HOME=/opt/bend2
out=/root/runs/gpu-utilisation
mkdir -p "$out"
date -u +%FT%TZ > "$out/started-at.txt"
nvidia-smi -q > "$out/nvidia-smi.txt"
cp /var/lib/soup/READY "$out/vm-ready.txt"
timeout 480 scripts/build-gpu.sh build/utilisation/base > "$out/build-base.log" 2>&1
python3 scripts/make-utilisation-variant.py > "$out/variant-source.txt"
BEND_SOURCE=build/utilisation/src/soup.bend timeout 480 scripts/build-gpu.sh build/utilisation/guard > "$out/build-guard.log" 2>&1
cp build/utilisation/*.build-info.txt "$out/"
python3 scripts/gpu-utilisation-bench.py "$out"
# Matched split-phase variants, after throughput runs to avoid profiler interference.
python3 scripts/make-scaling-profile.py > "$out/profile-source.txt"
BEND_SOURCE=build/scaling-profile/src/soup.bend timeout 480 scripts/build-gpu.sh build/utilisation/profile-base > "$out/build-profile-base.log" 2>&1
cp build/utilisation/src/metrics.bend build/scaling-profile/src/metrics.bend
BEND_SOURCE=build/scaling-profile/src/soup.bend timeout 480 scripts/build-gpu.sh build/utilisation/profile-guard > "$out/build-profile-guard.log" 2>&1
python3 scripts/gpu-utilisation-bench.py "$out" profiles
cp build/utilisation/*.build-info.txt "$out/"
date -u +%FT%TZ > "$out/finished-at.txt"
