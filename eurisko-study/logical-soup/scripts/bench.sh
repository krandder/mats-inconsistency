#!/bin/bash
# VM throughput on the JS lane: instructions/s = ops / (t(F) - t(F=1)),
# best of 3 runs each, so bun startup and population build cancel out.
# usage: scripts/bench.sh [F]   (from the repo root, after make build/vm_bench.js)
F=${1:-4096}
cd "$(dirname "$0")/.."
best() { local b=999; for r in 1 2 3; do s=$(date +%s.%N); bun build/vm_bench.js "$@" >/dev/null; e=$(date +%s.%N); b=$(echo "t=$e-$s; if (t<$b) t else $b" | bc -l); done; echo $b; }
printf "%-8s %-6s %-10s %-12s %-10s %-10s %s\n" kind N F ops t_run_s t_base_s instr_per_s
for n in 8 12; do for k in 0 1 2; do
  name=$(echo "random arbloop momloop" | cut -d' ' -f$((k+1)))
  ops=$(bun build/vm_bench.js $n $F $k | sed 's/ops=\([0-9]*\).*/\1/')
  t=$(best $n $F $k); t0=$(best $n 1 $k)
  printf "%-8s %-6s %-10s %-12s %-10.3f %-10.3f %.0f\n" $name $((1<<n)) $F $ops $t $t0 $(echo "$ops/($t-$t0)" | bc -l)
done; done
