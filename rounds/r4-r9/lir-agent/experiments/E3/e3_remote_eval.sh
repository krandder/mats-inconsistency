#!/bin/bash
# Runs ON the L4 VM as root. Evaluation only, from checkpoints staged under /root/soup/ck/s$S/{roles,fused}/ARM/c*.pt
# and earlier partial evals under /root/soup/prev/s$S/ARM/eval (reused, same state seeds). Args: SEED [SEED ...]
# Two passes in priority order (pass 1: c000 and c100 pairs; pass 2: c050; pass 3: c025), two processes at a time.
set -x
DEADLINE_MIN=${DEADLINE_MIN:-100}
T0=$(date +%s); left() { echo $(( DEADLINE_MIN*60 - ($(date +%s)-T0) )); }
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
command -v python3-config >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-dev > /root/apt.log 2>&1
python3 -m pip install -q --target /root/pylib torch transformers==5.18.0 peft==0.21.2 accelerate==1.15.0 numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib:/root/soup/harness LIR_HARNESS=/root/soup/harness HF_HOME=/root/hf HF_HUB_OFFLINE=0 \
  TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')" > /root/dl.log 2>&1
export HF_HUB_OFFLINE=1
E=/root/soup/E3
EV="python3 $E/e3_eval.py --n-states ${NST:-120} --n-cands 64 --n-probe-states ${NPS:-100} --n-probe-plans 16"
job() {  # S ARM PAIRS
  local S=$1 A=$2 P=$3 K=roles X=""
  case $A in M0g) K=fused; X="--fused 1 --goal-mode visible";; B0s) K=fused; X="--fused 1 --preset hardsyco";; esac
  local CKD=/root/soup/ck/s$S/$K/$A; [[ -d $CKD ]] || { echo "no ckpt $CKD"; return; }
  local OUT=/root/runs/E3/s$S/$A/eval; mkdir -p $OUT
  local TL=$(( $(left) - 30 )); (( TL > 60 )) || return
  timeout $TL $EV $X --ckpt $CKD --out $OUT --skip-dir /root/soup/prev/s$S/$A/eval --pairs $P >> /root/runs/E3/s$S/$A/eval-$(date +%s).log 2>&1
}
run2() {  # run a list of "S A PAIRS" jobs, two at a time
  local n=0
  while read -r S A P; do
    [[ -n $S ]] || continue
    job $S $A $P &
    n=$((n+1)); if (( n % 2 == 0 )); then wait -n; n=$((n-1)); fi
  done
  wait
}
J1=""; J2=""; J3=""
for S in "$@"; do
  J1+="$S P c000:c000,c100:c100,c100:c000"$'\n'"$S M0g c000:c000,c100:c100,c100:c000"$'\n'"$S B0s c000:c000,c100:c100,c100:c000"$'\n'
  J1+="$S RAPf c100:c100"$'\n'"$S RAPc c100:c100,c100:c000"$'\n'
  J2+="$S P c050:c050,c050:c000"$'\n'"$S RAPc c050:c050,c050:c000"$'\n'"$S RAPf c050:c050"$'\n'"$S M0g c050:c050,c050:c000"$'\n'"$S B0s c050:c050,c050:c000"$'\n'
  J3+="$S P c025:c025,c025:c000"$'\n'"$S RAPc c025:c025,c025:c000"$'\n'"$S RAPf c025:c025"$'\n'"$S M0g c025:c025,c025:c000"$'\n'"$S B0s c025:c025,c025:c000"$'\n'
done
printf "%s" "$J1" | run2; echo "pass1 t=$(( $(date +%s)-T0 ))" >> /root/runs/E3/times-eval-$1.txt
printf "%s" "$J2" | run2; echo "pass2 t=$(( $(date +%s)-T0 ))" >> /root/runs/E3/times-eval-$1.txt
# pass 3 (c025) dropped for compute: see PREREG deviations
exit 0
