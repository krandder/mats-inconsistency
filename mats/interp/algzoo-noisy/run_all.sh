#!/usr/bin/env bash
# Interpretation battery on one noisy model: run_all.sh MODEL.npz TAG [full]
# noisesens (own sigma and 0.01), lowrank (+ per-step), DAG sweep + prefix search, family scorecard, backward; with 'full'
# also devpred (scenarios) and ARC's bar (estim; ADF budgets from ADF_KS). Sequential, one process, one BLAS thread.
# Skips any step whose output JSON already exists.
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=$1; T=$2; FULL=${3:-}
PY=/home/agent/.venvs/aggext/bin/python
S=$($PY -c "import numpy as np; print(float(np.load('$M')['sigma']))")
cd /home/agent/workspaces/mats/work/algzoo-noisy
run() { local out=$1; shift; [ -s "$out" ] && return 0; nice -n 10 $PY "$@"; }
run results/noisesens_${T}_own.json noisesens.py $M $S results/noisesens_${T}_own.json > logs/ra_${T}_ns.log 2>&1
run results/noisesens_${T}_s0.01.json noisesens.py $M 0.01 results/noisesens_${T}_s0.01.json >> logs/ra_${T}_ns.log 2>&1
run results/lowrank_${T}.json lowrank.py $M results/lowrank_${T}.json > logs/ra_${T}_lr.log 2>&1
run results/dag_${T}.json dag_run.py $M results/dag_${T}.json 13 15 prefixonly > logs/ra_${T}_dag.log 2>&1
run results/fam_${T}.json fam.py $M results/fam_${T}.json > logs/ra_${T}_fam.log 2>&1
run results/backward_${T}.json backward.py $M results/backward_${T}.json 13 64 > logs/ra_${T}_bw.log 2>&1
if [ "$FULL" = full ]; then
  run results/devpred_${T}.json devpred.py $M results/dag_${T}.json 0.01 results/devpred_${T}.json > logs/ra_${T}_dev.log 2>&1
  ADF_KS=${ADF_KS:-0,2} run results/estim_${T}.json estim.py $M results/estim_${T}.json 16 ${DELTA:-0.003} ${TRUTH_LOG2:-22} > logs/ra_${T}_est.log 2>&1
fi
echo RUN-ALL-DONE $T
