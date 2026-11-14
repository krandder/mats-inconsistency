# GPU-LOG (work/cm-interp-B)

Every VM used by phase B. Cap $10 (coordinator, 2026-10-29, for the larger-model arm E21). On-demand L4 (g2-standard-8, ceiling $0.87/h).

2026-10-29T16:59:05Z LAUNCH rid=cmB-e21-w provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_e21.sh cmB-e21-w" (E21, amendment 28; cap $10 for this lane)
2026-10-29T16:59:09Z END rid=cmB-e21-w rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
- cmB-e21-w: rejected by gpu-run.sh before provisioning (run id must be lowercase); no VM created, $0. Instance list checked: no cm-interp-B VMs.
2026-10-29T17:00:26Z LAUNCH rid=cmb-e21-w provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_e21.sh cmb-e21-w" (E21, amendment 28; cap $10 for this lane)
    - 2026-10-29T17:00Z PENDING-CREATE vm=soup-gpu-cmb-e21-w-1029170028-27535 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=cmb-e21-w by=gpu-up.sh
    - 2026-10-29T17:01Z CREATE-FAILED vm=soup-gpu-cmb-e21-w-1029170028-27535 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T17:01:19Z END rid=cmb-e21-w rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
- cmb-e21-w (try 1): CREATE-FAILED, L4 stockout in us-central1-a/b/c; no VM created, $0.
- cmb-e21-w2: retry with GPU_ZONES across US regions (us-east4-a us-east4-c us-east1-b us-east1-c us-west1-a us-west1-b us-west4-a us-central1-a us-central1-b us-central1-c) after the us-central1 stockout.
2026-10-29T17:02:48Z LAUNCH rid=cmb-e21-w2 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_e21.sh cmb-e21-w2" (E21, amendment 28; cap $10 for this lane)
    - 2026-10-29T17:02Z PENDING-CREATE vm=soup-gpu-cmb-e21-w2-1029170251-10609 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=cmb-e21-w2 by=gpu-up.sh
    - 2026-10-29T17:07Z CREATE-FAILED vm=soup-gpu-cmb-e21-w2-1029170251-10609 type=g2-standard-8 zones=us-east4-a us-east4-c us-east1-b us-east1-c us-west1-a us-west1-b us-west4-a us-central1-a us-central1-b us-central1-c err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T17:07:31Z END rid=cmb-e21-w2 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
- cmb-e21-w2: CREATE-FAILED in all 10 zones (STOCKOUT / resource_availability); no VM created, $0. Instance list checked 17:08Z: no cm-interp-B VMs. Next: retry every 20 min (gpu/retry_e21.sh).
- 17:14Z: retries now include non-US L4 zones (coordinator's fallback 1). T4 is not supported by the shared launcher (gpu-up.sh fixes the machine type), so fallback 2 is not used.
2026-10-29T17:10:28Z LAUNCH rid=cmb-e21-w3 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_e21.sh cmb-e21-w3" (E21, amendment 28; cap $10 for this lane)
    - 2026-10-29T17:10Z PENDING-CREATE vm=soup-gpu-cmb-e21-w3-1029171031-27341 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=cmb-e21-w3 by=gpu-up.sh
    - 2026-10-29T17:15Z CREATE vm=soup-gpu-cmb-e21-w3-1029171031-27341 zone=us-west4-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=cmb-e21-w3 by=gpu-up.sh
    - 2026-10-29T17:17Z READY vm=soup-gpu-cmb-e21-w3-1029171031-27341 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T17:17Z RUN vm=soup-gpu-cmb-e21-w3-1029171031-27341 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T18:30Z RUN-END vm=soup-gpu-cmb-e21-w3-1029171031-27341 rc=0
    - 2026-10-29T18:31Z DELETE vm=soup-gpu-cmb-e21-w3-1029171031-27341 zone=us-west4-c type=g2-standard-8 age_h=1.26 cost_usd=1.10 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T18:31:46Z END rid=cmb-e21-w3 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-29T18:31:46Z attempt 3 (cmb-e21-w3): VM was created and the job ran; stopping retries
- cmb-e21-w3: soup-gpu-cmb-e21-w3-1029171031-27341, us-west4-c, on-demand L4, created ~17:15Z, deleted 18:31Z (age 1.26 h, ~$1.10 at the rate ceiling); all 15 nets finished and synced. Instance list checked 18:33Z: no cm-interp-B VMs. Lane total so far ~$1.10 of $40.
2026-10-29T22:51:27Z LAUNCH rid=cmb-e30-a provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_e30.sh cmb-e30-a" (E30, amendment 38; cap $3 for E30)
    - 2026-10-29T22:51Z PENDING-CREATE vm=soup-gpu-cmb-e30-a-1029225137-24989 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.28 group=explore-stage1 run=cmb-e30-a by=gpu-up.sh
    - 2026-10-29T22:52Z CREATE-FAILED vm=soup-gpu-cmb-e30-a-1029225137-24989 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T22:52:44Z END rid=cmb-e30-a rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
- cmb-e30-a: CREATE-FAILED, L4 stockout in us-central1-a/b/c; no VM created, $0. Next: gpu/retry_e30.sh across US and non-US L4 zones (coordinator's fallback 1).
2026-10-29T22:52:51Z LAUNCH rid=cmb-e30-a2 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_e30.sh cmb-e30-a2" (E30, amendment 38; cap $3 for E30)
- cmb-e30-a2: VM soup-gpu-cmb-e30-a2-1029225254-2407 created in us-west1-a, READY 22:59Z (on-demand L4); job deadline 9,900 s after start; expected cost <= ~$2.5.
    - 2026-10-29T22:53Z PENDING-CREATE vm=soup-gpu-cmb-e30-a2-1029225254-2407 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.28 group=explore-stage1 run=cmb-e30-a2 by=gpu-up.sh
    - 2026-10-29T22:57Z CREATE vm=soup-gpu-cmb-e30-a2-1029225254-2407 zone=us-west1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.28 spent_before_usd=0 group=explore-stage1 run=cmb-e30-a2 by=gpu-up.sh
    - 2026-10-29T22:58Z READY vm=soup-gpu-cmb-e30-a2-1029225254-2407 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T22:59Z RUN vm=soup-gpu-cmb-e30-a2-1029225254-2407 timeout=11400s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T01:12Z RUN-END vm=soup-gpu-cmb-e30-a2-1029225254-2407 rc=0
    - 2026-10-30T01:13Z DELETE vm=soup-gpu-cmb-e30-a2-1029225254-2407 zone=us-west1-a type=g2-standard-8 age_h=2.26 cost_usd=1.97 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T01:13:42Z END rid=cmb-e30-a2 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T01:13:42Z attempt 2 (cmb-e30-a2): VM was created and the job ran; stopping retries
- cmb-e30-a2: soup-gpu-cmb-e30-a2-1029225254-2407, us-west1-a, on-demand L4, ready 22:59Z, deleted 01:13Z (age 2.26 h, ~$1.97 at the rate ceiling); all 48 trainings finished (E30-DONE) and synced. Instance list checked 01:14Z: no cm-interp-B VMs. Lane total ~$3.07 of $40.
2026-10-30T02:05:39Z LAUNCH rid=cmb-e37a-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs.sh cmb-e37a-1 jobs.txt" (E37+E39 seeds 0-5, amendments 45 and 47; lane cap $40)
2026-10-30T02:06:55Z LAUNCH rid=cmb-e37b-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs.sh cmb-e37b-1 jobs.txt" (E37+E39 seeds 6-11, amendments 45 and 47; lane cap $40)
2026-10-30T02:08:11Z LAUNCH rid=cmb-e38-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs.sh cmb-e38-1 jobs.txt" (E38 size, amendment 46; lane cap $40)
2026-10-30T02:21:58Z LAUNCH rid=cmb-e40-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_t2.sh cmb-e40-1 jobs.txt" (E40 task 2, amendment 48; lane cap $40)
- cmb-e37a-1: on-demand L4 created (zone=asia-southeast1-a), READY 2026-10-30T02:27Z; running (fourth agent; E37-E40).
- cmb-e37b-1: on-demand L4 created (zone=us-central1-b), READY 2026-10-30T02:08Z; running (fourth agent; E37-E40).
- cmb-e38-1: on-demand L4 created (zone=us-central1-a), READY 2026-10-30T02:09Z; running (fourth agent; E37-E40).
- cmb-e40-1: on-demand L4 created (zone=us-west1-a), READY 2026-10-30T02:25Z; running (fourth agent; E37-E40).
2026-10-30T03:14:28Z LAUNCH rid=cmb-e43a-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e43a-1 jobs.txt train_br.py" (E43 branching seeds 0-6, amendment 51; lane cap $40)
2026-10-30T03:15:49Z LAUNCH rid=cmb-e43b-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e43b-1 jobs.txt train_br.py" (E43 branching seeds 7-11, amendment 51; lane cap $40)
- cmb-e43a-1: on-demand L4 created (zone=us-central1-a), READY 2026-10-30T03:16Z; running (E43 seeds 0-6, 42 trainings). cmb-e43b-1: on-demand L4 (zone=us-central1-a), READY 03:17Z; running (E43 seeds 7-11, 35 trainings).
    - 2026-10-30T02:22Z PENDING-CREATE vm=soup-gpu-cmb-e40-1-1030022202-3103 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e40-1 by=gpu-up.sh
    - 2026-10-30T02:24Z CREATE vm=soup-gpu-cmb-e40-1-1030022202-3103 zone=us-west1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e40-1 by=gpu-up.sh
    - 2026-10-30T02:25Z READY vm=soup-gpu-cmb-e40-1-1030022202-3103 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T02:25Z RUN vm=soup-gpu-cmb-e40-1-1030022202-3103 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T04:42Z RUN-END vm=soup-gpu-cmb-e40-1-1030022202-3103 rc=0
    - 2026-10-30T04:43Z DELETE vm=soup-gpu-cmb-e40-1-1030022202-3103 zone=us-west1-a type=g2-standard-8 age_h=2.31 cost_usd=2.01 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T04:43:14Z END rid=cmb-e40-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T04:43:14Z attempt 1 (cmb-e40-1): VM was created and the job ran; stopping retries
- cmb-e40-1: soup-gpu-cmb-e40-1-1030022202-3103, us-west1-a, on-demand L4, ready 02:25Z, deleted 04:43Z (age 2.31 h, ~$2.01 at the rate ceiling); all 48 E40 trainings finished and synced. Instance list checked 04:43Z: no cmb-e40 VM.
    - 2026-10-30T03:16Z PENDING-CREATE vm=soup-gpu-cmb-e43b-1-1030031553-843 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e43b-1 by=gpu-up.sh
    - 2026-10-30T03:16Z CREATE vm=soup-gpu-cmb-e43b-1-1030031553-843 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e43b-1 by=gpu-up.sh
    - 2026-10-30T03:17Z READY vm=soup-gpu-cmb-e43b-1-1030031553-843 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T03:17Z RUN vm=soup-gpu-cmb-e43b-1-1030031553-843 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T05:11Z RUN-END vm=soup-gpu-cmb-e43b-1-1030031553-843 rc=0
    - 2026-10-30T05:12Z DELETE vm=soup-gpu-cmb-e43b-1-1030031553-843 zone=us-central1-a type=g2-standard-8 age_h=1.93 cost_usd=1.68 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T05:13:02Z END rid=cmb-e43b-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T05:13:02Z attempt 1 (cmb-e43b-1): VM was created and the job ran; stopping retries
- cmb-e43b-1: soup-gpu-cmb-e43b-1-1030031553-843, us-central1-a, on-demand L4, ready 03:17Z, deleted about 05:13Z (age 1.93 h, ~$1.68 at the rate ceiling); all 35 E43 trainings (seeds 7-11) finished and synced. Instance list checked 05:20Z: no cmb-e43b VM.
    - 2026-10-30T03:14Z PENDING-CREATE vm=soup-gpu-cmb-e43a-1-1030031433-8071 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e43a-1 by=gpu-up.sh
    - 2026-10-30T03:14Z CREATE vm=soup-gpu-cmb-e43a-1-1030031433-8071 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e43a-1 by=gpu-up.sh
    - 2026-10-30T03:16Z READY vm=soup-gpu-cmb-e43a-1-1030031433-8071 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T03:16Z RUN vm=soup-gpu-cmb-e43a-1-1030031433-8071 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T05:33Z RUN-END vm=soup-gpu-cmb-e43a-1-1030031433-8071 rc=0
    - 2026-10-30T05:34Z DELETE vm=soup-gpu-cmb-e43a-1-1030031433-8071 zone=us-central1-a type=g2-standard-8 age_h=2.32 cost_usd=2.02 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T05:35:02Z END rid=cmb-e43a-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T05:35:02Z attempt 1 (cmb-e43a-1): VM was created and the job ran; stopping retries
- cmb-e43a-1: soup-gpu-cmb-e43a-1-1030031433-8071, us-central1-a, on-demand L4, ready 03:16Z, deleted about 05:34Z; all 42 E43 trainings (seeds 0-6) finished and synced. Instance list checked 05:35Z: no cmb-e43a VM.
    - 2026-10-30T02:07Z PENDING-CREATE vm=soup-gpu-cmb-e37b-1-1030020658-29569 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e37b-1 by=gpu-up.sh
    - 2026-10-30T02:07Z CREATE vm=soup-gpu-cmb-e37b-1-1030020658-29569 zone=us-central1-b type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e37b-1 by=gpu-up.sh
    - 2026-10-30T02:08Z READY vm=soup-gpu-cmb-e37b-1-1030020658-29569 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T02:08Z RUN vm=soup-gpu-cmb-e37b-1-1030020658-29569 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T05:54Z RUN-END vm=soup-gpu-cmb-e37b-1-1030020658-29569 rc=0
    - 2026-10-30T05:55Z DELETE vm=soup-gpu-cmb-e37b-1-1030020658-29569 zone=us-central1-b type=g2-standard-8 age_h=3.78 cost_usd=3.29 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T05:55:18Z END rid=cmb-e37b-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T05:55:18Z attempt 1 (cmb-e37b-1): VM was created and the job ran; stopping retries
- cmb-e37b-1: soup-gpu-cmb-e37b-1-1030020658-29569, us-central1-b, on-demand L4, ready 02:08Z, deleted 05:55Z (age 3.78 h, ~$3.29 at the rate ceiling); all 66 E37 and 6 E39 trainings (seeds 6-11) finished and synced. Instance list checked 05:55Z: no cmb-e37b VM.
    - 2026-10-30T02:08Z PENDING-CREATE vm=soup-gpu-cmb-e38-1-1030020813-31218 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e38-1 by=gpu-up.sh
    - 2026-10-30T02:08Z CREATE vm=soup-gpu-cmb-e38-1-1030020813-31218 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e38-1 by=gpu-up.sh
    - 2026-10-30T02:09Z READY vm=soup-gpu-cmb-e38-1-1030020813-31218 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T02:10Z RUN vm=soup-gpu-cmb-e38-1-1030020813-31218 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T06:02Z RUN-END vm=soup-gpu-cmb-e38-1-1030020813-31218 rc=0
    - 2026-10-30T06:03Z DELETE vm=soup-gpu-cmb-e38-1-1030020813-31218 zone=us-central1-a type=g2-standard-8 age_h=3.90 cost_usd=3.39 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T06:03:21Z END rid=cmb-e38-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T06:03:21Z attempt 1 (cmb-e38-1): VM was created and the job ran; stopping retries
- cmb-e38-1: soup-gpu-cmb-e38-1-1030020813-31218, us-central1-a, on-demand L4, ready 02:09Z, deleted about 06:03Z; all 60 E38 trainings finished and synced. Instance list checked 06:04Z: no cmb-e38 VM.
    - 2026-10-30T02:05Z PENDING-CREATE vm=soup-gpu-cmb-e37a-1-1030020546-32308 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e37a-1 by=gpu-up.sh
    - 2026-10-30T02:25Z CREATE vm=soup-gpu-cmb-e37a-1-1030020546-32308 zone=asia-southeast1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e37a-1 by=gpu-up.sh
    - 2026-10-30T02:27Z READY vm=soup-gpu-cmb-e37a-1-1030020546-32308 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T02:27Z RUN vm=soup-gpu-cmb-e37a-1-1030020546-32308 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T06:03Z RUN-END vm=soup-gpu-cmb-e37a-1-1030020546-32308 rc=0
    - 2026-10-30T06:04Z DELETE vm=soup-gpu-cmb-e37a-1-1030020546-32308 zone=asia-southeast1-a type=g2-standard-8 age_h=3.64 cost_usd=3.17 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T06:04:30Z END rid=cmb-e37a-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T06:04:30Z attempt 1 (cmb-e37a-1): VM was created and the job ran; stopping retries
- cmb-e37a-1: soup-gpu-cmb-e37a-1-1030020546-32308, asia-southeast1-a, on-demand L4, ready 02:27Z, deleted about 06:04Z; all 66 E37 and 6 E39 trainings (seeds 0-5) finished and synced. Instance list checked 06:04Z: no cm-interp-B VM left.
2026-10-30T06:50:21Z LAUNCH rid=cmb-e44a-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e44a-1 jobs.txt train_hy.py" (E44 hybrid inits pairs 1-3, amendment 52; lane cap $40)
2026-10-30T06:51:43Z LAUNCH rid=cmb-e44b-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e44b-1 jobs.txt train_hy.py" (E44 hybrid inits pairs 4-6, amendment 52; lane cap $40)
- cmb-e44a-1: on-demand L4 (us-central1-a), READY 2026-10-30T06:52Z; cmb-e44b-1: on-demand L4 (us-central1-b), READY 06:53Z; running E44 (36 trainings each).
2026-10-30T06:58:41Z LAUNCH rid=cmb-e46a-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e46a-1 jobs.txt train_pt.py" (E46 perturbed inits part 1, amendment 54; lane cap $40)
2026-10-30T07:00:03Z LAUNCH rid=cmb-e46b-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e46b-1 jobs.txt train_pt.py" (E46 perturbed inits part 2, amendment 54; lane cap $40)
- cmb-e46a-1: on-demand L4 (us-west1-a), READY 2026-10-30T07:02Z; cmb-e46b-1: on-demand L4 (us-central1-c), READY 07:02Z; running E46 (33 trainings each).
    - 2026-10-30T06:50Z PENDING-CREATE vm=soup-gpu-cmb-e44a-1-1030065024-5085 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e44a-1 by=gpu-up.sh
    - 2026-10-30T06:50Z CREATE vm=soup-gpu-cmb-e44a-1-1030065024-5085 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e44a-1 by=gpu-up.sh
    - 2026-10-30T06:52Z READY vm=soup-gpu-cmb-e44a-1-1030065024-5085 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T06:52Z RUN vm=soup-gpu-cmb-e44a-1-1030065024-5085 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T08:49Z RUN-END vm=soup-gpu-cmb-e44a-1-1030065024-5085 rc=0
    - 2026-10-30T08:50Z DELETE vm=soup-gpu-cmb-e44a-1-1030065024-5085 zone=us-central1-a type=g2-standard-8 age_h=1.98 cost_usd=1.72 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T08:50:09Z END rid=cmb-e44a-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T08:50:09Z attempt 1 (cmb-e44a-1): VM was created and the job ran; stopping retries
- cmb-e44a-1: us-central1-a, on-demand L4, ready 06:52Z, deleted about 08:50Z; all 36 E44 trainings (pairs 1-3) finished and synced. Instance list checked 08:50Z: no cmb-e44a VM.
    - 2026-10-30T06:51Z PENDING-CREATE vm=soup-gpu-cmb-e44b-1-1030065144-10654 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e44b-1 by=gpu-up.sh
    - 2026-10-30T06:52Z CREATE vm=soup-gpu-cmb-e44b-1-1030065144-10654 zone=us-central1-b type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e44b-1 by=gpu-up.sh
    - 2026-10-30T06:53Z READY vm=soup-gpu-cmb-e44b-1-1030065144-10654 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T06:53Z RUN vm=soup-gpu-cmb-e44b-1-1030065144-10654 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T08:50Z RUN-END vm=soup-gpu-cmb-e44b-1-1030065144-10654 rc=0
    - 2026-10-30T08:51Z DELETE vm=soup-gpu-cmb-e44b-1-1030065144-10654 zone=us-central1-b type=g2-standard-8 age_h=1.97 cost_usd=1.71 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T08:51:18Z END rid=cmb-e44b-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T08:51:18Z attempt 1 (cmb-e44b-1): VM was created and the job ran; stopping retries
    - 2026-10-30T06:58Z PENDING-CREATE vm=soup-gpu-cmb-e46a-1-1030065844-22823 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e46a-1 by=gpu-up.sh
    - 2026-10-30T07:00Z CREATE vm=soup-gpu-cmb-e46a-1-1030065844-22823 zone=us-west1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e46a-1 by=gpu-up.sh
    - 2026-10-30T07:02Z READY vm=soup-gpu-cmb-e46a-1-1030065844-22823 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T07:02Z RUN vm=soup-gpu-cmb-e46a-1-1030065844-22823 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T08:53Z RUN-END vm=soup-gpu-cmb-e46a-1-1030065844-22823 rc=0
    - 2026-10-30T08:54Z DELETE vm=soup-gpu-cmb-e46a-1-1030065844-22823 zone=us-west1-a type=g2-standard-8 age_h=1.88 cost_usd=1.64 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T08:54:26Z END rid=cmb-e46a-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T08:54:26Z attempt 1 (cmb-e46a-1): VM was created and the job ran; stopping retries
    - 2026-10-30T07:00Z PENDING-CREATE vm=soup-gpu-cmb-e46b-1-1030070005-22637 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e46b-1 by=gpu-up.sh
    - 2026-10-30T07:00Z CREATE vm=soup-gpu-cmb-e46b-1-1030070005-22637 zone=us-central1-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e46b-1 by=gpu-up.sh
    - 2026-10-30T07:02Z READY vm=soup-gpu-cmb-e46b-1-1030070005-22637 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T07:02Z RUN vm=soup-gpu-cmb-e46b-1-1030070005-22637 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T08:54Z RUN-END vm=soup-gpu-cmb-e46b-1-1030070005-22637 rc=0
    - 2026-10-30T08:54Z DELETE vm=soup-gpu-cmb-e46b-1-1030070005-22637 zone=us-central1-c type=g2-standard-8 age_h=1.89 cost_usd=1.64 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T08:55:02Z END rid=cmb-e46b-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T08:55:02Z attempt 1 (cmb-e46b-1): VM was created and the job ran; stopping retries
- cmb-e44b-1 (us-central1-b), cmb-e46a-1 (us-west1-a), cmb-e46b-1 (us-central1-c): finished and deleted about 08:53-08:55Z; all 36 E44 (pairs 4-6) and 66 E46 trainings synced. Instance list checked 08:55Z: no cm-interp-B VM left.
2026-10-30T09:39:19Z LAUNCH rid=cmb-e48a-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e48a-1 jobs.txt train_lin.py" (E48 world 1 sq/sgnsq (amendment 56))
2026-10-30T09:39:21Z LAUNCH rid=cmb-e48b-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e48b-1 jobs.txt train_t2lin.py" (E48 world 2 D=2 prod/sgnprod (amendment 56))
2026-10-30T09:39:23Z LAUNCH rid=cmb-e48c-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e48c-1 jobs.txt train_t2lin.py" (E48 world 2 D=1 prod/sgnprod (amendment 56))
2026-10-30T09:49:07Z LAUNCH rid=cmb-e49-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e49-1 jobs.txt train_pre.py" (E49 pre-attention MLP, both worlds (amendment 57))
    - 2026-10-30T09:39Z PENDING-CREATE vm=soup-gpu-cmb-e48c-1-1030093924-17421 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e48c-1 by=gpu-up.sh
    - 2026-10-30T09:39Z CREATE vm=soup-gpu-cmb-e48c-1-1030093924-17421 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e48c-1 by=gpu-up.sh
    - 2026-10-30T09:41Z READY vm=soup-gpu-cmb-e48c-1-1030093924-17421 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T09:41Z RUN vm=soup-gpu-cmb-e48c-1-1030093924-17421 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T10:52Z RUN-END vm=soup-gpu-cmb-e48c-1-1030093924-17421 rc=0
    - 2026-10-30T10:53Z DELETE vm=soup-gpu-cmb-e48c-1-1030093924-17421 zone=us-central1-a type=g2-standard-8 age_h=1.21 cost_usd=1.05 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T10:53:10Z END rid=cmb-e48c-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T10:53:10Z attempt 1 (cmb-e48c-1): VM was created and the job ran; stopping retries
    - 2026-10-30T09:39Z PENDING-CREATE vm=soup-gpu-cmb-e48b-1-1030093922-15605 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e48b-1 by=gpu-up.sh
    - 2026-10-30T09:40Z CREATE vm=soup-gpu-cmb-e48b-1-1030093922-15605 zone=us-central1-b type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e48b-1 by=gpu-up.sh
    - 2026-10-30T09:41Z READY vm=soup-gpu-cmb-e48b-1-1030093922-15605 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T09:41Z RUN vm=soup-gpu-cmb-e48b-1-1030093922-15605 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T10:52Z RUN-END vm=soup-gpu-cmb-e48b-1-1030093922-15605 rc=0
    - 2026-10-30T10:53Z DELETE vm=soup-gpu-cmb-e48b-1-1030093922-15605 zone=us-central1-b type=g2-standard-8 age_h=1.22 cost_usd=1.06 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T10:53:52Z END rid=cmb-e48b-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T10:53:52Z attempt 1 (cmb-e48b-1): VM was created and the job ran; stopping retries
    - 2026-10-30T09:39Z PENDING-CREATE vm=soup-gpu-cmb-e48a-1-1030093921-26175 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e48a-1 by=gpu-up.sh
    - 2026-10-30T09:39Z CREATE vm=soup-gpu-cmb-e48a-1-1030093921-26175 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e48a-1 by=gpu-up.sh
    - 2026-10-30T09:40Z READY vm=soup-gpu-cmb-e48a-1-1030093921-26175 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T09:41Z RUN vm=soup-gpu-cmb-e48a-1-1030093921-26175 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T11:00Z RUN-END vm=soup-gpu-cmb-e48a-1-1030093921-26175 rc=0
    - 2026-10-30T11:01Z DELETE vm=soup-gpu-cmb-e48a-1-1030093921-26175 zone=us-central1-a type=g2-standard-8 age_h=1.35 cost_usd=1.17 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T11:01:32Z END rid=cmb-e48a-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T11:01:32Z attempt 1 (cmb-e48a-1): VM was created and the job ran; stopping retries
- cmb-e48a-1 (us-central1-a, 1.35 h, $1.17), cmb-e48b-1 (us-central1-b, 1.22 h, $1.06), cmb-e48c-1 (us-central1-a, 1.21 h, $1.05): E48, all 72 trainings finished; deleted; instance list checked 11:01Z (only cmb-e49-1 running). Lane total $28.62 of $40.
    - 2026-10-30T09:49Z PENDING-CREATE vm=soup-gpu-cmb-e49-1-1030094909-29662 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e49-1 by=gpu-up.sh
    - 2026-10-30T09:50Z CREATE vm=soup-gpu-cmb-e49-1-1030094909-29662 zone=us-central1-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e49-1 by=gpu-up.sh
    - 2026-10-30T09:51Z READY vm=soup-gpu-cmb-e49-1-1030094909-29662 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T09:52Z RUN vm=soup-gpu-cmb-e49-1-1030094909-29662 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T11:50Z RUN-END vm=soup-gpu-cmb-e49-1-1030094909-29662 rc=0
    - 2026-10-30T11:51Z DELETE vm=soup-gpu-cmb-e49-1-1030094909-29662 zone=us-central1-c type=g2-standard-8 age_h=2.01 cost_usd=1.75 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T11:51:13Z END rid=cmb-e49-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T11:51:13Z attempt 1 (cmb-e49-1): VM was created and the job ran; stopping retries
- cmb-e49-1 (us-central1-c, 2.01 h, $1.75): E49, all 36 trainings finished; deleted; instance list checked 11:51Z (no cmb VM left). Lane total $30.37 of $40.
2026-10-30T12:05:53Z LAUNCH rid=cmb-e50-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e50-1 jobs.txt train_noln.py" (E50 block-1 LayerNorm ablation (amendment 58))
    - 2026-10-30T12:05Z PENDING-CREATE vm=soup-gpu-cmb-e50-1-1030120555-15413 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e50-1 by=gpu-up.sh
    - 2026-10-30T12:07Z CREATE vm=soup-gpu-cmb-e50-1-1030120555-15413 zone=us-west1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e50-1 by=gpu-up.sh
    - 2026-10-30T12:09Z READY vm=soup-gpu-cmb-e50-1-1030120555-15413 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T12:09Z RUN vm=soup-gpu-cmb-e50-1-1030120555-15413 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T14:02Z RUN-END vm=soup-gpu-cmb-e50-1-1030120555-15413 rc=0
    - 2026-10-30T14:03Z DELETE vm=soup-gpu-cmb-e50-1-1030120555-15413 zone=us-west1-a type=g2-standard-8 age_h=1.92 cost_usd=1.67 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T14:03:06Z END rid=cmb-e50-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T14:03:06Z attempt 1 (cmb-e50-1): VM was created and the job ran; stopping retries
- cmb-e50-1 (us-west1-a, 1.92 h, $1.67): E50, all 36 trainings finished; deleted; instance list checked 14:03Z (no cmb VM left). Lane total $32.04 of $40.
2026-10-30T16:53:57Z LAUNCH rid=cmb-e58-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e58-1 jobs.txt train_frz.py" (E58 frozen designed embeddings, causal check (amendment 66))
    - 2026-10-30T16:54Z PENDING-CREATE vm=soup-gpu-cmb-e58-1-1030165359-21171 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e58-1 by=gpu-up.sh
    - 2026-10-30T17:00Z CREATE vm=soup-gpu-cmb-e58-1-1030165359-21171 zone=asia-southeast1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e58-1 by=gpu-up.sh
    - 2026-10-30T17:01Z READY vm=soup-gpu-cmb-e58-1-1030165359-21171 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T17:01Z RUN vm=soup-gpu-cmb-e58-1-1030165359-21171 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T18:45Z RUN-END vm=soup-gpu-cmb-e58-1-1030165359-21171 rc=0
    - 2026-10-30T18:46Z DELETE vm=soup-gpu-cmb-e58-1-1030165359-21171 zone=asia-southeast1-a type=g2-standard-8 age_h=1.77 cost_usd=1.54 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T18:46:04Z END rid=cmb-e58-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T18:46:04Z attempt 1 (cmb-e58-1): VM was created and the job ran; stopping retries
- cmb-e58-1 (asia-southeast1-a, 1.77 h, $1.54): E58, all 30 trainings finished; deleted; instance list checked 18:46Z (no cmb VM left). Lane total $33.58 of $40.
2026-10-30T21:07:40Z LAUNCH rid=cmb-e61-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e61-1 jobs.txt train_uni.py" (E61 unif at width 128 and depth 3 (amendment 69))
    - 2026-10-30T21:07Z PENDING-CREATE vm=soup-gpu-cmb-e61-1-1030210743-15690 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e61-1 by=gpu-up.sh
    - 2026-10-30T21:08Z CREATE vm=soup-gpu-cmb-e61-1-1030210743-15690 zone=us-east4-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e61-1 by=gpu-up.sh
    - 2026-10-30T21:09Z READY vm=soup-gpu-cmb-e61-1-1030210743-15690 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-30T21:10Z RUN vm=soup-gpu-cmb-e61-1-1030210743-15690 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-30T22:42Z RUN-END vm=soup-gpu-cmb-e61-1-1030210743-15690 rc=0
    - 2026-10-30T22:43Z DELETE vm=soup-gpu-cmb-e61-1-1030210743-15690 zone=us-east4-a type=g2-standard-8 age_h=1.57 cost_usd=1.37 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-30T22:43:46Z END rid=cmb-e61-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-30T22:43:46Z attempt 1 (cmb-e61-1): VM was created and the job ran; stopping retries
- cmb-e61-1 (us-east4-a, 1.57 h, $1.37): E61, all 20 trainings finished; deleted; instance list checked 22:51Z (no cmb VM left). Lane total $34.95 of $40.
2026-10-31T04:39:40Z LAUNCH rid=cmb-e64b-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e64b-1 jobs.txt train_e64b.py" (E64b split and splitnc (amendment 72 addendum 1))
    - 2026-10-31T04:39Z PENDING-CREATE vm=soup-gpu-cmb-e64b-1-1031043943-2680 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e64b-1 by=gpu-up.sh
    - 2026-10-31T04:42Z CREATE vm=soup-gpu-cmb-e64b-1-1031043943-2680 zone=us-west1-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e64b-1 by=gpu-up.sh
    - 2026-10-31T04:43Z READY vm=soup-gpu-cmb-e64b-1-1031043943-2680 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-31T04:43Z RUN vm=soup-gpu-cmb-e64b-1-1031043943-2680 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-31T06:09Z RUN-END vm=soup-gpu-cmb-e64b-1-1031043943-2680 rc=0
    - 2026-10-31T06:10Z DELETE vm=soup-gpu-cmb-e64b-1-1031043943-2680 zone=us-west1-c type=g2-standard-8 age_h=1.47 cost_usd=1.28 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-31T06:10:58Z END rid=cmb-e64b-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-31T06:10:58Z attempt 1 (cmb-e64b-1): VM was created and the job ran; stopping retries
- cmb-e64b-1 (us-west1-c, 1.47 h, $1.28): E64b split and splitnc, all 24 trainings finished; deleted; instance list checked 06:11Z (no cmb VM left). Lane total $36.23 of $40.
2026-10-31T06:29:03Z LAUNCH rid=cmb-e69-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e69-1 jobs.txt train_h4.py" (E69 4-head nets (amendment 77))
    - 2026-10-31T06:29Z PENDING-CREATE vm=soup-gpu-cmb-e69-1-1031062905-12623 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e69-1 by=gpu-up.sh
    - 2026-10-31T06:40Z CREATE vm=soup-gpu-cmb-e69-1-1031062905-12623 zone=asia-southeast1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e69-1 by=gpu-up.sh
    - 2026-10-31T06:41Z READY vm=soup-gpu-cmb-e69-1-1031062905-12623 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-31T06:42Z RUN vm=soup-gpu-cmb-e69-1-1031062905-12623 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-31T07:26Z RUN-END vm=soup-gpu-cmb-e69-1-1031062905-12623 rc=0
    - 2026-10-31T07:27Z DELETE vm=soup-gpu-cmb-e69-1-1031062905-12623 zone=asia-southeast1-a type=g2-standard-8 age_h=0.80 cost_usd=0.70 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-31T07:27:24Z END rid=cmb-e69-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-31T07:27:24Z attempt 1 (cmb-e69-1): VM was created and the job ran; stopping retries
- cmb-e69-1 (asia-southeast1-a, 0.80 h, $0.70): E69 4-head nets, all 12 trainings finished; deleted; instance list checked 07:35Z (no cmb VM left). Lane total $36.93 of $40.
2026-10-31T07:42:14Z LAUNCH rid=cmb-e70-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs_x.sh cmb-e70-1 jobs.txt train_e70.py" (E70 b1h4 and ownmag (amendment 78))
    - 2026-10-31T07:42Z PENDING-CREATE vm=soup-gpu-cmb-e70-1-1031074216-23038 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.02 group=explore-stage1 run=cmb-e70-1 by=gpu-up.sh
    - 2026-10-31T07:42Z CREATE vm=soup-gpu-cmb-e70-1-1031074216-23038 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.02 spent_before_usd=0 group=explore-stage1 run=cmb-e70-1 by=gpu-up.sh
    - 2026-10-31T07:43Z READY vm=soup-gpu-cmb-e70-1-1031074216-23038 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-31T07:44Z RUN vm=soup-gpu-cmb-e70-1-1031074216-23038 timeout=18600s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-31T08:59Z RUN-END vm=soup-gpu-cmb-e70-1-1031074216-23038 rc=0
    - 2026-10-31T09:00Z DELETE vm=soup-gpu-cmb-e70-1-1031074216-23038 zone=us-central1-a type=g2-standard-8 age_h=1.29 cost_usd=1.12 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-31T09:01:00Z END rid=cmb-e70-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-31T09:01:00Z attempt 1 (cmb-e70-1): VM was created and the job ran; stopping retries
- cmb-e70-1 (us-central1-a, 1.29 h, $1.12): E70 b1h4 and ownmag, all 24 trainings finished; deleted; instance list checked 09:02Z (no cmb VM left). Lane total $38.05 of $40.
