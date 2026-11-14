# GPU-LOG (work/nongauss-family)

Every VM used by this lane. Cap $5 (brief). On-demand L4 (g2-standard-8, ceiling $0.87/h).

2026-10-29T17:31:12Z LAUNCH rid=ngfam1 provisioning=STANDARD machine=g2-standard-8(L4) stage=/tmp/ngfam_stage/ngfam1-173112 cmd="bash scratch/gpu_job.sh ngfam1"
2026-10-29T17:52:05Z LAUNCH rid=ngfam2 provisioning=STANDARD machine=g2-standard-8(L4) stage=/tmp/ngfam_stage/ngfam2-175205 cmd="bash scratch/gpu_job2.sh ngfam2"
    - 2026-10-29T17:52Z PENDING-CREATE vm=soup-gpu-ngfam2-1029175207-27732 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=ngfam2 by=gpu-up.sh
    - 2026-10-29T17:52Z CREATE-FAILED vm=soup-gpu-ngfam2-1029175207-27732 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T17:52:46Z END rid=ngfam2 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
2026-10-29T18:09:47Z LAUNCH rid=ngfam2b provisioning=STANDARD machine=g2-standard-8(L4) stage=/tmp/ngfam_stage/ngfam2b-180947 cmd="bash scratch/gpu_job2.sh ngfam2b"
    - 2026-10-29T18:09Z PENDING-CREATE vm=soup-gpu-ngfam2b-1029180948-20822 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=ngfam2b by=gpu-up.sh
    - 2026-10-29T18:10Z CREATE-FAILED vm=soup-gpu-ngfam2b-1029180948-20822 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T18:10:34Z END rid=ngfam2b rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
2026-10-29T18:10:41Z LAUNCH rid=ngfam2c provisioning=STANDARD machine=g2-standard-8(L4) stage=/tmp/ngfam_stage/ngfam2c-181041 cmd="bash scratch/gpu_job2.sh ngfam2c"
2026-10-29T18:26:30Z LAUNCH rid=ngfam3 provisioning=STANDARD machine=g2-standard-8(L4) stage=/tmp/ngfam_stage/ngfam3-182630 cmd="bash scratch/gpu_job3.sh ngfam3"
2026-10-29T18:30:37Z LAUNCH rid=ngfam4 provisioning=STANDARD machine=g2-standard-8(L4) stage=/tmp/ngfam_stage/ngfam4-183037 cmd="bash scratch/gpu_job4.sh ngfam4"
    - 2026-10-29T18:10Z PENDING-CREATE vm=soup-gpu-ngfam2c-1029181042-20792 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=ngfam2c by=gpu-up.sh
    - 2026-10-29T18:11Z CREATE vm=soup-gpu-ngfam2c-1029181042-20792 zone=us-west1-b type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=ngfam2c by=gpu-up.sh
    - 2026-10-29T18:12Z READY vm=soup-gpu-ngfam2c-1029181042-20792 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T18:12Z RUN vm=soup-gpu-ngfam2c-1029181042-20792 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T18:30Z RUN-END vm=soup-gpu-ngfam2c-1029181042-20792 rc=0
    - 2026-10-29T18:31Z DELETE vm=soup-gpu-ngfam2c-1029181042-20792 zone=us-west1-b type=g2-standard-8 age_h=0.32 cost_usd=0.28 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T18:31:12Z END rid=ngfam2c rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
2026-10-29T18:43:10Z LAUNCH rid=ngfam5 provisioning=STANDARD machine=g2-standard-8(L4) stage=/tmp/ngfam_stage/ngfam5-184310 cmd="bash scratch/gpu_job5.sh ngfam5"
    - 2026-10-29T18:43Z PENDING-CREATE vm=soup-gpu-ngfam5-1029184315-19660 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=ngfam5 by=gpu-up.sh
    - 2026-10-29T18:47Z CREATE-FAILED vm=soup-gpu-ngfam5-1029184315-19660 type=g2-standard-8 zones=us-west1-b us-west1-a us-central1-c us-central1-a us-central1-b us-east4-a us-east4-c us-east1-c us-east1-d us-west4-a europe-west4-a europe-west4-b err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T18:47:10Z END rid=ngfam5 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
2026-10-29T18:53:46Z LAUNCH rid=ngfam5t1 provisioning=STANDARD machine=g2-standard-8(L4) stage=/tmp/ngfam_stage/ngfam5t1-185346 cmd="bash scratch/gpu_job5.sh ngfam5t1"
    - 2026-10-29T18:54Z PENDING-CREATE vm=soup-gpu-ngfam5t1-1029185349-8324 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=ngfam5t1 by=gpu-up.sh
    - 2026-10-29T18:58Z CREATE-FAILED vm=soup-gpu-ngfam5t1-1029185349-8324 type=g2-standard-8 zones=us-central1-c us-central1-a us-central1-b us-west1-a us-west1-b us-east4-a us-east4-c us-east1-c us-east1-d us-west4-a europe-west4-a europe-west4-b err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T18:58:15Z END rid=ngfam5t1 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
2026-10-29T19:00:16Z LAUNCH rid=ngfam5t2 provisioning=STANDARD machine=g2-standard-8(L4) stage=/tmp/ngfam_stage/ngfam5t2-190015 cmd="bash scratch/gpu_job5.sh ngfam5t2"
    - 2026-10-29T18:26Z PENDING-CREATE vm=soup-gpu-ngfam3-1029182632-18950 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=ngfam3 by=gpu-up.sh
    - 2026-10-29T18:28Z CREATE vm=soup-gpu-ngfam3-1029182632-18950 zone=us-central1-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=ngfam3 by=gpu-up.sh
    - 2026-10-29T18:29Z READY vm=soup-gpu-ngfam3-1029182632-18950 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T18:29Z RUN vm=soup-gpu-ngfam3-1029182632-18950 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T19:00Z RUN-END vm=soup-gpu-ngfam3-1029182632-18950 rc=0
    - 2026-10-29T19:01Z DELETE vm=soup-gpu-ngfam3-1029182632-18950 zone=us-central1-c type=g2-standard-8 age_h=0.53 cost_usd=0.46 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T19:01:20Z END rid=ngfam3 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 2026-10-29T19:00Z PENDING-CREATE vm=soup-gpu-ngfam5t2-1029190018-23575 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=ngfam5t2 by=gpu-up.sh
    - 2026-10-29T19:03Z CREATE vm=soup-gpu-ngfam5t2-1029190018-23575 zone=us-east4-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=ngfam5t2 by=gpu-up.sh
    - 2026-10-29T19:04Z READY vm=soup-gpu-ngfam5t2-1029190018-23575 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T19:04Z RUN vm=soup-gpu-ngfam5t2-1029190018-23575 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T19:29Z RUN-END vm=soup-gpu-ngfam5t2-1029190018-23575 rc=0
    - 2026-10-29T19:30Z DELETE vm=soup-gpu-ngfam5t2-1029190018-23575 zone=us-east4-c type=g2-standard-8 age_h=0.45 cost_usd=0.39 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T19:30:34Z END rid=ngfam5t2 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 2026-10-29T17:31Z PENDING-CREATE vm=soup-gpu-ngfam1-1029173114-1229 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=ngfam1 by=gpu-up.sh
    - 2026-10-29T17:31Z CREATE vm=soup-gpu-ngfam1-1029173114-1229 zone=us-central1-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=ngfam1 by=gpu-up.sh
    - 2026-10-29T17:33Z READY vm=soup-gpu-ngfam1-1029173114-1229 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T17:33Z RUN vm=soup-gpu-ngfam1-1029173114-1229 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T19:47Z RUN-END vm=soup-gpu-ngfam1-1029173114-1229 rc=0
    - 2026-10-29T19:48Z DELETE vm=soup-gpu-ngfam1-1029173114-1229 zone=us-central1-c type=g2-standard-8 age_h=2.27 cost_usd=1.97 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T19:48:52Z END rid=ngfam1 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 2026-10-29T18:30Z PENDING-CREATE vm=soup-gpu-ngfam4-1029183040-21896 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=ngfam4 by=gpu-up.sh
    - 2026-10-29T18:31Z CREATE vm=soup-gpu-ngfam4-1029183040-21896 zone=us-west1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=ngfam4 by=gpu-up.sh
    - 2026-10-29T18:32Z READY vm=soup-gpu-ngfam4-1029183040-21896 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T18:32Z RUN vm=soup-gpu-ngfam4-1029183040-21896 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T19:48Z RUN-END vm=soup-gpu-ngfam4-1029183040-21896 rc=0
    - 2026-10-29T19:49Z DELETE vm=soup-gpu-ngfam4-1029183040-21896 zone=us-west1-a type=g2-standard-8 age_h=1.29 cost_usd=1.12 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T19:49:19Z END rid=ngfam4 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)

## Totals (29 Oct, 19:51Z)

| VM | hours | cost ceiling |
|---|---|---|
| ngfam1 (R1-R4) | 2.27 | $1.97 |
| ngfam2c (Amendment 1) | 0.32 | $0.28 |
| ngfam3 (Amendment 3) | 0.53 | $0.46 |
| ngfam4 (T5a, T5b, T6) | 1.29 | $1.12 |
| ngfam5t2 (Amendment 4) | 0.45 | $0.39 |
| total | 4.86 | $4.22 (cap $5) |

Failed creations (stockouts, $0): ngfam2, ngfam2b, ngfam5, ngfam5t1. All VMs deleted (gcloud list empty, 19:51Z).
