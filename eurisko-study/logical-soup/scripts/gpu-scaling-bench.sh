#!/usr/bin/env bash
# Execute on a single rental; gpu-run.sh owns sync/deletion, GCE enforces TTL.
set -euo pipefail
cd /root/soup
export BEND_HOME=/opt/bend2
DEVICE=${1:?usage: gpu-scaling-bench.sh l4|a100}
case "$DEVICE" in l4|a100) ;; *) exit 2;; esac
OUT=${GPU_BENCH_OUT:-/root/runs/gpu-scaling-$DEVICE}
[[ ! -e "$OUT/started-at.txt" ]] || { echo "Refuse to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT"
date -u +%FT%TZ > "$OUT/started-at.txt"
cp /var/lib/soup/READY "$OUT/vm-ready.txt"
cp build/gpu-source-commit.txt "$OUT/source-base-commit.txt"
nvidia-smi -q > "$OUT/nvidia-smi.txt"
lscpu > "$OUT/cpu-info.txt"
# Use toolkit wrappers; do not select architecture-specific target binaries.
export PATH="/usr/local/cuda/bin:$PATH"
# Record profiler availability; no silent claim of hardware-counter profiling.
{ command -v nsys || true; command -v ncu || true; ls /opt/nvidia/nsight* /usr/local/cuda/bin/ns* 2>/dev/null || true; } > "$OUT/profilers.txt"
timeout --kill-after=20 480 scripts/build-gpu.sh build/scaling/soup > "$OUT/build.log" 2>&1
cp build/scaling/soup.build-info.txt "$OUT/"
python3 scripts/make-scaling-profile.py > "$OUT/profile-source.txt"
BEND_SOURCE=build/scaling-profile/src/soup.bend timeout --kill-after=20 480 scripts/build-gpu.sh build/scaling/profile > "$OUT/profile-build.log" 2>&1
cp build/scaling/profile.build-info.txt "$OUT/"
python3 scripts/gpu-scaling-bench.py "$OUT" "$DEVICE"
date -u +%FT%TZ > "$OUT/finished-at.txt"
