#!/bin/bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PL=/home/agent/workspaces/eurisko-study/wealth-net/.pylib
R=/tmp/traders-net/T1/rerun
( cd $R/w1 && for x in a d c; do PYTHONPATH=$PL timeout 1800 python3 check_L1.py $x > $R/log_L1$x.txt 2>&1; echo "L1$x rc=$?" >> $R/done.txt; done ) &
( cd $R/w1 && PYTHONPATH=$PL python3 check_L2.py > $R/log_L2.txt 2>&1; echo "L2 rc=$?" >> $R/done.txt; PYTHONPATH=$PL python3 check_L3.py > $R/log_L3.txt 2>&1; echo "L3 rc=$?" >> $R/done.txt ) &
( cd $R/w3 && for s in 1 2 3; do PYTHONPATH=$PL python3 check_L6.py --seed $s > $R/log_L6s$s.txt 2>&1; echo "L6s$s rc=$?" >> $R/done.txt; done ) &
( cd $R/w3 && for s in 1 2 3; do PYTHONPATH=$PL python3 check_L7.py --seed $s > $R/log_L7s$s.txt 2>&1; echo "L7s$s rc=$?" >> $R/done.txt; done; python3 aggregate.py > $R/log_agg.txt 2>&1; echo "agg rc=$?" >> $R/done.txt ) &
( cd $R/w2 && PYTHONPATH=$PL python3 check_delay.py > $R/log_delay.txt 2>&1; echo "delay rc=$?" >> $R/done.txt; PYTHONPATH=$PL python3 check_delay.py --eta-scale --out results/delay_stable.json > $R/log_delays.txt 2>&1; echo "delay_stable rc=$?" >> $R/done.txt ) &
( cd $R/w2 && PYTHONPATH=$PL python3 check_li.py > $R/log_li.txt 2>&1; echo "li rc=$?" >> $R/done.txt; python3 summarize.py > $R/log_sum.txt 2>&1; echo "sum rc=$?" >> $R/done.txt ) &
wait
echo ALLDONE >> $R/done.txt
