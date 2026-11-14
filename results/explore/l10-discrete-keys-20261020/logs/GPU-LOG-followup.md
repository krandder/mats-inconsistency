# GPU-LOG

- 2026-10-21T00:01Z PENDING-CREATE vm=soup-gpu-l10-followup-1021000134-5054 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=2.26 group=explore-stage1 run=l10-followup by=gpu-up.sh
- 2026-10-21T00:02Z CREATE vm=soup-gpu-l10-followup-1021000134-5054 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=SPOT max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=2.26 spent_before_usd=0 group=explore-stage1 run=l10-followup by=gpu-up.sh
- 2026-10-21T00:03Z READY vm=soup-gpu-l10-followup-1021000134-5054 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
- 2026-10-21T00:03Z RUN vm=soup-gpu-l10-followup-1021000134-5054 timeout=7200s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
- 2026-10-21T00:22Z RUN-END vm=soup-gpu-l10-followup-1021000134-5054 rc=0
- 2026-10-21T00:23Z DELETE vm=soup-gpu-l10-followup-1021000134-5054 zone=us-central1-a type=g2-standard-8 age_h=0.34 cost_usd=0.30 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
