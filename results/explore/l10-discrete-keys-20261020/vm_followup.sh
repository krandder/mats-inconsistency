#!/usr/bin/env bash
# L10 follow-up diagnostics on the spot L4 (called by work/run_on_l4.sh from the repo root). Outputs: /root/runs/$1/
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
ARMS=(stbits2 stbits4 stbits8 stbits16 wbits2 wbits4 wbits8 wbits16 none@12000 bits8@12000 wbits8@12000 cont4@12000 prov_bits@12000 prov_full@12000)
for e in E2 E3-collude; do
  for g in 0 1 2 3; do
    grp=(); for i in "${!ARMS[@]}"; do (( i % 4 == g )) && grp+=("${ARMS[$i]}"); done
    python $S train --env $e --arms "${grp[@]}" --seeds 0 --steps "$STEPS" --cache /root/cache --out "$OUT" \
      --tag "f$g" --threads 1 > "$OUT/stdout_${e}_f$g.txt" 2>&1 &
  done
done
wait
ls "$OUT/arms" | wc -l > "$OUT/DONE_followup"
