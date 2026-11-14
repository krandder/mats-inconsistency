#!/usr/bin/env bash
# Explicit CUDA lane; never silently fall back to CPU.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export BEND_HOME=${BEND_HOME:-$ROOT/../research/bend2-scratch/bend-2.0.29}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
CC=${CC:-clang-19}
OUT=${1:-$ROOT/build/gpu/soup}
[[ "$OUT" = /* ]] || OUT="$ROOT/$OUT"
[[ -f "$CUDA_HOME/include/nvrtc.h" ]] || { echo 'CUDA nvrtc headers required' >&2; exit 2; }
scripts/build-native.sh --out "$OUT" --emit-only
# One stderr line per CUDA runtime pass supplies dispatch evidence.
# A pass may launch several kernels; Nsight records individual launches.
# Instrument host-only gpu_pass; the embedded device program is unchanged.
python3 - "$OUT.c" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]); s=p.read_text()
a='static void gpu_pass(u32 f) {\n  gpu_run(f);'
assert s.count(a)==1
s=s.replace(a, 'static void gpu_pass(u32 f) {\n  struct timespec soup_a, soup_b;\n  clock_gettime(CLOCK_MONOTONIC, &soup_a);\n  gpu_run(f);')
a='    err_fail("device fault");\n  }\n}'
assert s.count(a)==1
s=s.replace(a, '    err_fail("device fault");\n  }\n  clock_gettime(CLOCK_MONOTONIC, &soup_b);\n  unsigned long long ns = (soup_b.tv_sec - soup_a.tv_sec) * 1000000000ll + soup_b.tv_nsec - soup_a.tv_nsec;\n  fprintf(stderr, "SOUP_CUDA_PASS frontier=%u wall_ns=%llu\\n", f, ns);\n}')
a='        cube_run(H, true);'
assert s.count(a)==1
s=s.replace(a, '        fprintf(stderr, "SOUP_CUDA_CUBE begin fid=%u\\n", (u32)term_aux(t));\n        cube_run(H, true);\n        fprintf(stderr, "SOUP_CUDA_CUBE end\\n");')
p.write_text(s)
PY
"$CC" -std=c11 -O3 -DBEND_CUDA=1 -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib64/stubs" "$OUT.c" -lpthread -lm -lcuda -lnvrtc -o "$OUT" > "$OUT.build.log" 2>&1 || { cat "$OUT.build.log" >&2; exit 1; }
"$OUT" --gpu-build > "$OUT.gpu-build.log" 2>&1 || { cat "$OUT.gpu-build.log" >&2; exit 1; }
{
  echo 'lane=cuda; runtime selection must use --gpu on'
  date -u +checked_at=%FT%TZ
  "$CC" --version
  sha256sum "$OUT" "$OUT.c" "${BEND_SOURCE:-src/soup.bend}" src/*.bend
} > "$OUT.build-info.txt"
echo "Built CUDA lane: $OUT"
