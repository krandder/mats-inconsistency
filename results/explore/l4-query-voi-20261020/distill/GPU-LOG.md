# GPU-LOG

- 2026-10-20T23:32Z PENDING-CREATE vm=soup-gpu-l4-query-voi-20261020c-1020233217-13694 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=l4-query-voi-20261020c by=gpu-up.sh
- 2026-10-20T23:32Z CREATE vm=soup-gpu-l4-query-voi-20261020c-1020233217-13694 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=SPOT max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=l4-query-voi-20261020c by=gpu-up.sh
- 2026-10-20T23:33Z READY vm=soup-gpu-l4-query-voi-20261020c-1020233217-13694 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
- 2026-10-20T23:34Z RUN vm=soup-gpu-l4-query-voi-20261020c-1020233217-13694 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
- 2026-10-21T00:04Z RUN-END vm=soup-gpu-l4-query-voi-20261020c-1020233217-13694 rc=0
- 2026-10-21T00:05Z DELETE vm=soup-gpu-l4-query-voi-20261020c-1020233217-13694 zone=us-central1-a type=g2-standard-8 age_h=0.53 cost_usd=0.46 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
