2026-10-29T18:59:23Z LAUNCH rid=sfx-b1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_job.sh sfx-b1 10" (scale-free exp; lane cap $10)
    - 2026-10-29T18:59Z PENDING-CREATE vm=soup-gpu-sfx-b1-1029185926-11745 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.87 group=explore-stage1 run=sfx-b1 by=gpu-up.sh
    - 2026-10-29T19:00Z CREATE vm=soup-gpu-sfx-b1-1029185926-11745 zone=us-central1-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.87 spent_before_usd=0 group=explore-stage1 run=sfx-b1 by=gpu-up.sh
    - 2026-10-29T19:01Z READY vm=soup-gpu-sfx-b1-1029185926-11745 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T19:01Z RUN vm=soup-gpu-sfx-b1-1029185926-11745 timeout=18000s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T19:08Z RUN-END vm=soup-gpu-sfx-b1-1029185926-11745 rc=143
    - 2026-10-29T19:09Z DELETE vm=soup-gpu-sfx-b1-1029185926-11745 zone=us-central1-c type=g2-standard-8 age_h=0.15 cost_usd=0.13 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=143: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T19:10:00Z END rid=sfx-b1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-29T19:20:46Z LAUNCH rid=sfx-b2 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_job.sh sfx-b2 12" (scale-free exp; lane cap $10)
    - 2026-10-29T19:21Z PENDING-CREATE vm=soup-gpu-sfx-b2-1029192049-17954 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.87 group=explore-stage1 run=sfx-b2 by=gpu-up.sh
    - 2026-10-29T19:21Z CREATE vm=soup-gpu-sfx-b2-1029192049-17954 zone=us-central1-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.87 spent_before_usd=0 group=explore-stage1 run=sfx-b2 by=gpu-up.sh
    - 2026-10-29T19:23Z READY vm=soup-gpu-sfx-b2-1029192049-17954 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T19:23Z RUN vm=soup-gpu-sfx-b2-1029192049-17954 timeout=18000s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T19:31Z RUN-END vm=soup-gpu-sfx-b2-1029192049-17954 rc=0
    - 2026-10-29T19:32Z DELETE vm=soup-gpu-sfx-b2-1029192049-17954 zone=us-central1-c type=g2-standard-8 age_h=0.17 cost_usd=0.15 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T19:32:34Z END rid=sfx-b2 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-29T19:42:10Z LAUNCH rid=sfx-b3 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_job.sh sfx-b3 12" (scale-free exp; lane cap $10)
2026-10-29T20:01:26Z LAUNCH rid=sfx-b4 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_job.sh sfx-b4 12" (scale-free exp; lane cap $10)
- 20:03Z sfx-b3: recipe switched to batch 1024 x 10k steps (Amendment 5) for jobs started after 20:03Z; jobs2 emptied (E2 moved to sfx-b4); 12 running tok jobs stopped. 20:04Z sfx-b4 (asia-southeast1-a): files replaced before its job started (12 tok W0 + E4 + Amendment-3 arms; phase 2 = E2).
- 20:55Z sfx-b4: stopped E4 (9 running; e4_flat1_s1 had died of CUDA OOM) and set N=256 (2) runs (Amendment 7); E4 requeued as e4r_* (8k steps) at the head of phase 2; jobs2 now 138 lines.
2026-10-29T20:57:32Z LAUNCH rid=sfx-b5 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_job.sh sfx-b5 12" (scale-free exp; lane cap $10)
- 21:05Z sfx-b5 (asia-southeast1-a) runs the 168 light E1 jobs (phase 1) and 92 phase-2 jobs moved from sfx-b4 (Amendments 8-9); skip markers placed on sfx-b3 (168) and sfx-b4 (92).
- 22:07Z sfx-b5: 8 ablation jobs (Amendment 10) appended to phase 2.
    - 2026-10-29T19:42Z PENDING-CREATE vm=soup-gpu-sfx-b3-1029194215-11322 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.87 group=explore-stage1 run=sfx-b3 by=gpu-up.sh
    - 2026-10-29T19:52Z CREATE vm=soup-gpu-sfx-b3-1029194215-11322 zone=asia-southeast1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.87 spent_before_usd=0 group=explore-stage1 run=sfx-b3 by=gpu-up.sh
    - 2026-10-29T19:53Z READY vm=soup-gpu-sfx-b3-1029194215-11322 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T19:54Z RUN vm=soup-gpu-sfx-b3-1029194215-11322 timeout=18000s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T22:22Z RUN-END vm=soup-gpu-sfx-b3-1029194215-11322 rc=0
    - 2026-10-29T22:23Z DELETE vm=soup-gpu-sfx-b3-1029194215-11322 zone=asia-southeast1-a type=g2-standard-8 age_h=2.53 cost_usd=2.20 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T22:23:27Z END rid=sfx-b3 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
    - 2026-10-29T20:57Z PENDING-CREATE vm=soup-gpu-sfx-b5-1029205741-7109 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.87 group=explore-stage1 run=sfx-b5 by=gpu-up.sh
    - 2026-10-29T20:58Z CREATE vm=soup-gpu-sfx-b5-1029205741-7109 zone=asia-southeast1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.87 spent_before_usd=0 group=explore-stage1 run=sfx-b5 by=gpu-up.sh
    - 2026-10-29T20:59Z READY vm=soup-gpu-sfx-b5-1029205741-7109 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T21:00Z RUN vm=soup-gpu-sfx-b5-1029205741-7109 timeout=18000s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T23:07Z RUN-END vm=soup-gpu-sfx-b5-1029205741-7109 rc=0
    - 2026-10-29T23:08Z DELETE vm=soup-gpu-sfx-b5-1029205741-7109 zone=asia-southeast1-a type=g2-standard-8 age_h=2.16 cost_usd=1.88 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T23:08:46Z END rid=sfx-b5 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
- 23:17Z sfx-b4: the 10 E2 W1 fine-tunes had failed (base nets were on sfx-b3); bases copied from sfx-b3's sync and the jobs relaunched by hand as e2w1r_* (same recipe).
    - 2026-10-29T20:01Z PENDING-CREATE vm=soup-gpu-sfx-b4-1029200129-29392 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.87 group=explore-stage1 run=sfx-b4 by=gpu-up.sh
    - 2026-10-29T20:01Z CREATE vm=soup-gpu-sfx-b4-1029200129-29392 zone=asia-southeast1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.87 spent_before_usd=0 group=explore-stage1 run=sfx-b4 by=gpu-up.sh
    - 2026-10-29T20:03Z READY vm=soup-gpu-sfx-b4-1029200129-29392 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T20:04Z RUN vm=soup-gpu-sfx-b4-1029200129-29392 timeout=18000s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T23:37Z RUN-END vm=soup-gpu-sfx-b4-1029200129-29392 rc=0
    - 2026-10-29T23:38Z DELETE vm=soup-gpu-sfx-b4-1029200129-29392 zone=asia-southeast1-a type=g2-standard-8 age_h=3.60 cost_usd=3.13 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T23:38:18Z END rid=sfx-b4 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
- 23:46Z all sfx VMs deleted (b1-b5); gcloud instance list shows no soup-gpu-sfx-* (only another lane's cmb-e30). Total list-price ceiling: 0.13 + 0.15 + 2.20 + 3.13 + 1.88 = $7.49.
