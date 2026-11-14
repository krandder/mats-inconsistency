#!/usr/bin/env bash
# L10 training on the spot L4 (called by work/run_on_l4.sh from the repo root). Outputs: /root/runs/$1/
set -uo pipefail
RUN=$1; STEPS=${2:-4000}
OUT=/root/runs/$RUN; mkdir -p "$OUT"
S=results/explore/l10-discrete-keys-20261020/run_l10.py
export PYTHONPATH=.
pip install -q jsonschema >/dev/null 2>&1 || true
nvidia-smi -L > "$OUT/gpu.txt" 2>&1 || true
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.__version__)" >> "$OUT/gpu.txt"
for e in E2 E3-collude; do python $S data --env $e --cache /root/cache > "$OUT/data_$e.txt" 2>&1 & done
wait
ARMS=(none bits1 bits2 bits3 bits4 bits6 bits8 bits12 bits16 bits24 bag1x16 bag2x16 bag4x16 bag8x16 cont1 cont2 cont4 cont8 cont16 prov_bits prov_full)
for e in E2 E3-collude; do
  for g in 0 1 2 3; do
    grp=(); for i in "${!ARMS[@]}"; do (( i % 4 == g )) && grp+=("${ARMS[$i]}"); done
    python $S train --env $e --arms "${grp[@]}" --seeds 0 1 --steps "$STEPS" --cache /root/cache --out "$OUT" \
      --tag "g$g" --threads 1 > "$OUT/stdout_${e}_g$g.txt" 2>&1 &
  done
done
wait
ls "$OUT/arms" | wc -l > "$OUT/DONE"
