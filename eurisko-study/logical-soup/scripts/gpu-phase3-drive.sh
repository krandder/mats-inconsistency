#!/usr/bin/env bash
# Local driver: remote chunks pause while a consistent checkpoint is copied home.
set -euo pipefail
cd "$(dirname "$0")/.."
source infra/lib.sh
seed=${1:?seed}; [[ $seed =~ ^[123]$ ]] || exit 2
id=k16-f12-s$seed
out="$SOUP_DIR/runs/phase3/$id"
remote=/root/runs/phase3/$id
mkdir -p "$out" "$STATE/checkpoint-$id"
if [[ -f "$out/soup.dump" ]]; then python3 scripts/verify-checkpoint.py "$out" --repair; fi
rshell="ssh ${SSH_OPTS[*]}"
vm_ssh "$GPU_VM_IP" "sudo mkdir -p $remote"
# Only the locally verified snapshot and matching CSV are recovery inputs.
rsync -az --rsync-path='sudo rsync' -e "$rshell" "$out/" "$SSH_USER@$GPU_VM_IP:$remote/"
vm_root "$GPU_VM_IP" env BEND_HOME=/opt/bend2 scripts/build-gpu.sh build/phase3/soup
vm_ssh "$GPU_VM_IP" "sudo cp /root/soup/build/phase3/soup.build-info.txt $remote/build-info.txt"
vm_root "$GPU_VM_IP" env 'SOUP_CMD=build/phase3/soup --gpu on --threads 8' bash scripts/gpu-resume-smoke.sh > "$out/resume-smoke.log" 2>&1
grep -q SOUP_CUDA_PASS "$out/resume-smoke.log" || { echo "no CUDA dispatch in resume smoke" >&2; exit 1; }

while :; do
  done_epoch=$(python3 - "$out" <<'PY'
import pathlib,struct,sys
p=pathlib.Path(sys.argv[1])/'soup.dump'
print(struct.unpack_from('<I',p.read_bytes(),24)[0] if p.exists() else 0)
PY
)
  (( done_epoch < 5000 )) || break
  next=$((done_epoch+100)); ((next<=5000)) || next=5000
  chunk_rc=0
  vm_root "$GPU_VM_IP" env 'SOUP_CMD=build/phase3/soup --gpu on --threads 8' timeout --kill-after=30 30m scripts/run.sh --seed "$seed" --K '2^16' --F '2^12' --epochs "$next" --b 64 --free-income 0 --chunk 100 --out "$remote" >> "$out/worker.log" 2>&1 || chunk_rc=$?
  [[ $chunk_rc != 124 ]] || exit 75
  [[ $chunk_rc == 0 ]] || exit "$chunk_rc"
  # No remote writer exists here. Stage, validate, then promote the entire pair.
  stage="$STATE/checkpoint-$id"
  rsync -az --delete --rsync-path='sudo rsync' -e "$rshell" --exclude=worker.log "$SSH_USER@$GPU_VM_IP:$remote/" "$stage/"
  python3 scripts/verify-checkpoint.py "$stage" "$next"
  cp "$stage/soup.dump" "$out/soup.dump.new"
  for file in config.json metrics.csv timing.csv build-info.txt; do
    cp "$stage/$file" "$out/$file.new"
    mv "$out/$file.new" "$out/$file"
  done
  mv "$out/soup.dump.new" "$out/soup.dump"
  printf '%s seed=%s epoch=%s vm=%s\n' "$(date -u +%FT%TZ)" "$seed" "$next" "$GPU_VM_NAME"
done
