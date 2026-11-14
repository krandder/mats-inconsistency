#!/bin/bash
# P4 stage 3 on the trader-cat-p4 VM (run as root from /root/job, started by gpu.sh start NAME all):
#  1. base evals (id_8_32 T=24, L64/128 steps, L128/256, L256/512), GPU selection at t=16 and t=8
#     (P=512, AS m=5..200, FLIP/SOFT/PAY-Shapley ladder on 64 orders + 448 reference orders), F0 x seeds 0-2;
#  2. the market arms for seeds 0-2 on the CPU-selection picks in picks.json
#     (a selector whose picks equal the AS picks is not rerun: marked as a tie in the analysis);
#  3. seeds 3-4 for every arm if F0's L128 accuracy SD over seeds 0-2 is above 3 points on either base.
# Outputs: /root/out/{base,select,ft}/*.json (atomic via .tmp), logs/, failed.txt, JOB_DONE.
set -u
export JOB=${P4_JOB:-/root/job} OUT=${P4_OUT:-/root/out} PY=${P4_PY:-/opt/venv/bin/python} EXTRA=${P4_EXTRA:-}
export MRDIR=$JOB
cd $JOB
mkdir -p $OUT/logs $OUT/base $OUT/select $OUT/ft
touch $OUT/JOB_RUNNING; rm -f $OUT/JOB_DONE

job() {  # job NAME ARGS...: one ftmarket.py call -> $OUT/NAME.json
  local n=$1; shift
  [ -f "$OUT/$n.json" ] && return 0
  if $PY ftmarket.py "$@" $EXTRA --out "$OUT/$n.json.tmp" >>"$OUT/logs/${n//\//_}.log" 2>&1; then
    mv "$OUT/$n.json.tmp" "$OUT/$n.json"
  else
    echo "$(date -u +%FT%TZ) FAILED $n" >>"$OUT/failed.txt"
  fi
}
export -f job

arms() {  # arms S0 S1: job lines for the market arms, seeds S0..S1
  $PY - "$1" "$2" <<'PY'
import json, os, sys
pk = json.load(open('picks.json'))
for s in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    for b in ('A', 'Bds'):
        ck = f"{os.environ['JOB']}/pilot_{b}.pt"
        p = pk[b]
        print(f"ft/{b}_F0_s{s} finetune --ckpt {ck} --arm F0 --seed {s}")
        g = ','.join(map(str, p['AS']))
        print(f"ft/{b}_F-AS-mkt_s{s} finetune --ckpt {ck} --arm F-AS-mkt --groups {g} --seed {s}")
        print(f"ft/{b}_F-AS-aux_s{s} finetune --ckpt {ck} --arm F-AS-aux --groups {g} --seed {s}")
        if sorted(p['LOO']) != sorted(p['AS']):
            print(f"ft/{b}_F-LOO-mkt_s{s} finetune --ckpt {ck} --arm F-LOO-mkt --groups {','.join(map(str, p['LOO']))} --seed {s}")
        r = p['rand'][str(s)]
        if sorted(r) != sorted(p['AS']):
            print(f"ft/{b}_F-rand-mkt_s{s} finetune --ckpt {ck} --arm F-rand-mkt --groups {','.join(map(str, r))} --seed {s}")
PY
}

run_lines() { xargs -P "${P4_WORKERS:-4}" -L 1 bash -c 'job "$@"' _; }

echo "$(date -u +%FT%TZ) phase 1"
{
  for b in A Bds; do
    echo "base/$b evalbase --ckpt $JOB/pilot_$b.pt --sets id_8_32,L64,L128,L256"
  done
  for b in A Bds; do for t in 16 8; do
    echo "select/${b}_t$t select --ckpt $JOB/pilot_$b.pt --t $t --P 512 --m 5 10 20 50 100 200 --shapK 64 --refK 448 --chunk 1024 --seed 1"
  done; done
  for s in 0 1 2; do for b in A Bds; do echo "ft/${b}_F0_s$s finetune --ckpt $JOB/pilot_$b.pt --arm F0 --seed $s"; done; done
} | run_lines

echo "$(date -u +%FT%TZ) phase 2"
arms 0 2 | run_lines

more=$($PY - <<'PY'
import json, glob, os, numpy as np
out = 0
for b in ('A', 'Bds'):
    a = [json.load(open(f))['eval']['L128']['acc'] for f in sorted(glob.glob(os.environ['OUT'] + f'/ft/{b}_F0_s[0-2].json'))]
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    print(f'# F0 {b} L128 accs {a} sd {sd:.4f}', flush=True)
    out |= sd > 0.03
print(int(out))
PY
)
echo "$more"
if [ "$(echo "$more" | tail -1)" = 1 ]; then
  echo "$(date -u +%FT%TZ) phase 3: seeds 3-4 (F0 L128 SD > 3 points)"
  arms 3 4 | run_lines
fi
touch $OUT/JOB_DONE; rm -f $OUT/JOB_RUNNING
echo "$(date -u +%FT%TZ) ALL DONE"
