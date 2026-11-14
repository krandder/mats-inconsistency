# GPU-LOG

- 2026-10-20T23:39Z PENDING-CREATE vm=soup-gpu-l10-discrete-keys-1020233924-18042 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=l10-discrete-keys by=gpu-up.sh
- 2026-10-20T23:39Z CREATE vm=soup-gpu-l10-discrete-keys-1020233924-18042 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=SPOT max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=l10-discrete-keys by=gpu-up.sh
- 2026-10-20T23:41Z READY vm=soup-gpu-l10-discrete-keys-1020233924-18042 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
- 2026-10-20T23:41Z RUN vm=soup-gpu-l10-discrete-keys-1020233924-18042 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
- 2026-10-21T00:07Z RUN-END vm=soup-gpu-l10-discrete-keys-1020233924-18042 rc=0
- 2026-10-21T00:08Z DELETE vm=soup-gpu-l10-discrete-keys-1020233924-18042 zone=us-central1-a type=g2-standard-8 age_h=0.46 cost_usd=0.40 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
