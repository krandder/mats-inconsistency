#!/usr/bin/env bash
# Runs on the L4 VM from /root/runs/$RID (kstar-frames, PREREG runs A-D at widths 128 and 64; run E came back from
# kstar1; width 32 runs on General).
set -u
PY=${PY:-/root/venv/bin/python}
export PY KF_DEVICE=cuda OMP_NUM_THREADS=2
mkdir -p out/runs out/logs
nvidia-smi -L
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
cd scratch
pipe() {
  n=$1; s=$2; L=$3; t=n${n}_s${s}_L${L}; c=$((L/2)); N=16384; [ $n = 128 ] && N=8192
  $PY r1_layers.py --n $n --seed $s --L $L --threads 2 --out ../out/runs > ../out/logs/A_$t.log 2>&1
  echo "A $t rc=$? $(date -u +%H:%M:%SZ)"
  $PY r2_big.py --n $n --seed $s --L $L --threads 2 --out ../out/runs > ../out/logs/B_$t.log 2>&1
  echo "B $t rc=$? $(date -u +%H:%M:%SZ)"
  $PY r2b_analyze.py --n $n --seed $s --L $L --threads 2 --runs ../out/runs > ../out/logs/Ban_$t.log 2>&1 &
  $PY r3_pairs.py --n $n --seed $s --L $L --threads 2 --runs ../out/runs > ../out/logs/C_$t.log 2>&1 &
  if [ $s != 402 ]; then
    $PY r4_mlp.py --n $n --seed $s --L $L --c $c --ntrain $N --threads 2 --runs ../out/runs > ../out/logs/D_${t}_c${c}_N${N}.log 2>&1
  fi
  wait
  echo "pipe $t done $(date -u +%H:%M:%SZ)"
}
extras() {
  n=$1; L=$2; t=n${n}_s400_L${L}; c=$((L/2)); N=16384; [ $n = 128 ] && N=8192
  for NT in 1024 4096; do
    $PY r4_mlp.py --n $n --seed 400 --L $L --c $c --ntrain $NT --threads 2 --runs ../out/runs > ../out/logs/D_${t}_c${c}_N${NT}.log 2>&1
  done
  if [ $L = 32 ]; then
    for cc in 8 24; do
      $PY r4_mlp.py --n $n --seed 400 --L 32 --c $cc --ntrain $N --threads 2 --runs ../out/runs > ../out/logs/D_${t}_c${cc}_N${N}.log 2>&1
    done
  fi
  echo "extras $t done $(date -u +%H:%M:%SZ)"
}
( for s in 400 401 402; do pipe 128 $s 32; done; extras 128 32 ) &
( for s in 400 401 402; do pipe 128 $s 16; done; extras 128 16 ) &
( for s in 400 401 402; do pipe 64 $s 32; done; extras 64 32; for s in 400 401 402; do pipe 64 $s 16; done; extras 64 16 ) &
wait
grep -h "done\|Traceback\|Error" ../out/logs/*.log | tail -40
du -sh ../out
echo RUN-GPU-DONE
