#!/bin/bash
# run_exact.sh WORLD : evaluation side, on this host. Fetch the test sets, then the exact references (cexact.py with the
# U table from cU.py). No arm reads the result.
W=$1
cd /home/agent/workspaces/mats/work/deep-study/wf13/consistency
R=user@[ip-removed]; SSH="ssh -i $[ssh-key-path-removed] -o BatchMode=yes"
mkdir -p exactwork/w$W out/w$W
rsync -a -e "$SSH" $R:/cluster/home/user/wave13/consistency/data/w$W/{t1.pt,t2.pt,t3_0.pt,t3_1.pt,score.pt} exactwork/w$W/
NTHREADS=4 nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python -B cexact.py $W ../../wf7/probworld/ckpt/M_w${W}_M.pt M exactwork/w$W out/w$W/exact.npz
