# GPU-LOG (work/lyapunov-dimension/pbo)

Every VM used by this addendum. Cap $3 (coordinator, 29 Oct ~20:20Z). On-demand L4 (g2-standard-8, ceiling $0.87/h).

2026-10-29T20:25:33Z LAUNCH rid=lyapdim1 provisioning=STANDARD machine=g2-standard-8(L4) stage=/tmp/lyapdim_stage/lyapdim1-202533 cmd="bash scratch/gpu_job_pbo.sh lyapdim1"
    - 2026-10-29T20:25Z PENDING-CREATE vm=soup-gpu-lyapdim1-1029202536-5286 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=2.26 group=explore-stage1 run=lyapdim1 by=gpu-up.sh
    - 2026-10-29T20:25Z CREATE vm=soup-gpu-lyapdim1-1029202536-5286 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=2.26 spent_before_usd=0 group=explore-stage1 run=lyapdim1 by=gpu-up.sh
    - 2026-10-29T20:39Z READY vm=soup-gpu-lyapdim1-1029202536-5286 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T20:39Z RUN vm=soup-gpu-lyapdim1-1029202536-5286 timeout=7200s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T21:08Z RUN-END vm=soup-gpu-lyapdim1-1029202536-5286 rc=0
    - 2026-10-29T21:10Z DELETE vm=soup-gpu-lyapdim1-1029202536-5286 zone=us-central1-a type=g2-standard-8 age_h=0.72 cost_usd=0.63 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T21:10:17Z END rid=lyapdim1 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
