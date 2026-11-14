#!/usr/bin/env bash
# run one analysis command when the lane has a free slot (<= 4 active python processes)
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
active() { ps -eo stat,args | grep -E "python[^ ]* (-W ignore )?([^ ]*/)?(nnoise|mirun|task1|task3|e1|e2|e3|pfn_rare|pfn_diag|pfn_diag2|pfn_is|pfn_is2|dec|twist|holder|pfn_holder|pfn_holder2|sur_small|sur_pfn|devpred|dag|qw|bc|qstruct|qstruct_b|qbayes|qw_lin|qw_lin2|qpay|qbayes_reg|qsys|qshare|qshare2|qshare_dag|qint|pfn|qgroup|qinter|qinter_imp|qweights|qpatch|qmech)\.py" | grep -v grep | grep -v "bash " | awk '$1 !~ /T/' | wc -l; }
while [ $(active) -ge 4 ]; do sleep 17; done
"$@"
