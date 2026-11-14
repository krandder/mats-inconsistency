# GPU-LOG (work/arc-noisy)

Every VM used by this lane. Cap $50 (raised from $20 by Kelvin, 2026-10-29). On-demand L4 (g2-standard-8, ceiling $0.87/h).

2026-10-29T01:19:10Z LAUNCH rid=arcn-truth1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_truth1.sh arcn-truth1"
    - 2026-10-29T01:19Z PENDING-CREATE vm=soup-gpu-arcn-truth1-1029011913-1334 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.87 group=explore-stage1 run=arcn-truth1 by=gpu-up.sh
    - 2026-10-29T01:19Z CREATE vm=soup-gpu-arcn-truth1-1029011913-1334 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.87 spent_before_usd=0 group=explore-stage1 run=arcn-truth1 by=gpu-up.sh
    - 2026-10-29T01:21Z READY vm=soup-gpu-arcn-truth1-1029011913-1334 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T01:21Z RUN vm=soup-gpu-arcn-truth1-1029011913-1334 timeout=18000s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T01:42Z RUN-END vm=soup-gpu-arcn-truth1-1029011913-1334 rc=143
    - 2026-10-29T01:43Z DELETE vm=soup-gpu-arcn-truth1-1029011913-1334 zone=us-central1-a type=g2-standard-8 age_h=0.39 cost_usd=0.34 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=143: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T01:43:54Z END rid=arcn-truth1 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
- arcn-truth1: soup-gpu-arcn-truth1-1029011913-1334, us-central1-a, on-demand L4, created ~01:19Z, deleted ~01:43Z (stopped early: kernel memory-bound at 260 s per net-sigma; 3 truth files kept). Cost ~0.4 h x $0.87 = ~$0.35. Instance list for this lane empty after deletion (checked 01:43Z).
2026-10-29T01:44:17Z LAUNCH rid=arcn-truth2a provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_truth2a.sh arcn-truth2a"
2026-10-29T01:44:37Z LAUNCH rid=arcn-truth2b provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_truth2b.sh arcn-truth2b"
2026-10-29T02:17:58Z LAUNCH rid=arcn-samp1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_samp1.sh arcn-samp1"
    - 2026-10-29T02:18Z PENDING-CREATE vm=soup-gpu-arcn-samp1-1029021802-19997 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.00 group=explore-stage1 run=arcn-samp1 by=gpu-up.sh
    - 2026-10-29T02:19Z CREATE vm=soup-gpu-arcn-samp1-1029021802-19997 zone=us-central1-b type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.00 spent_before_usd=0 group=explore-stage1 run=arcn-samp1 by=gpu-up.sh
    - 2026-10-29T02:20Z READY vm=soup-gpu-arcn-samp1-1029021802-19997 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T02:20Z RUN vm=soup-gpu-arcn-samp1-1029021802-19997 timeout=14400s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T02:25Z RUN-END vm=soup-gpu-arcn-samp1-1029021802-19997 rc=0
    - 2026-10-29T02:26Z DELETE vm=soup-gpu-arcn-samp1-1029021802-19997 zone=us-central1-b type=g2-standard-8 age_h=0.13 cost_usd=0.11 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T02:26:53Z END rid=arcn-samp1 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
2026-10-29T02:26:56Z LAUNCH rid=arcn-samp2 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_samp1.sh arcn-samp2"
2026-10-29T02:41:12Z LAUNCH rid=arcn-samp3 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_samp3.sh arcn-samp3"
2026-10-29T02:45:08Z LAUNCH rid=arcn-truth3 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_truth3.sh arcn-truth3"
    - 2026-10-29T02:45Z PENDING-CREATE vm=soup-gpu-arcn-truth3-1029024512-25907 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=arcn-truth3 by=gpu-up.sh
    - 2026-10-29T02:45Z CREATE-FAILED vm=soup-gpu-arcn-truth3-1029024512-25907 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T02:46:09Z END rid=arcn-truth3 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - arcn-truth3: VM creation failed (L4 stockout in us-central1-a/b/c), no VM created, $0
2026-10-29T02:46:54Z LAUNCH rid=arcn-truth3b provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_truth3.sh arcn-truth3b"
    - 2026-10-29T02:41Z PENDING-CREATE vm=soup-gpu-arcn-samp3-1029024116-697 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=2.26 group=explore-stage1 run=arcn-samp3 by=gpu-up.sh
    - 2026-10-29T02:41Z CREATE vm=soup-gpu-arcn-samp3-1029024116-697 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=2.26 spent_before_usd=0 group=explore-stage1 run=arcn-samp3 by=gpu-up.sh
    - 2026-10-29T02:43Z READY vm=soup-gpu-arcn-samp3-1029024116-697 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T02:43Z RUN vm=soup-gpu-arcn-samp3-1029024116-697 timeout=7200s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T02:47Z RUN-END vm=soup-gpu-arcn-samp3-1029024116-697 rc=0
    - 2026-10-29T02:48Z DELETE vm=soup-gpu-arcn-samp3-1029024116-697 zone=us-central1-a type=g2-standard-8 age_h=0.10 cost_usd=0.09 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T02:48:22Z END rid=arcn-samp3 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 2026-10-29T02:47Z PENDING-CREATE vm=soup-gpu-arcn-truth3b-1029024658-14914 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=arcn-truth3b by=gpu-up.sh
    - 2026-10-29T02:49Z CREATE-FAILED vm=soup-gpu-arcn-truth3b-1029024658-14914 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-c us-west1-a us-west1-b europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T02:50:10Z END rid=arcn-truth3b rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - arcn-samp3: ran 02:41-02:48Z (0.10 h), failed (scipy missing in its script), VM deleted by cleanup; $0.09
2026-10-29T03:11:28Z LAUNCH rid=arcn-samp4 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_samp4.sh arcn-samp4"
    - 2026-10-29T02:27Z PENDING-CREATE vm=soup-gpu-arcn-samp2-1029022659-7385 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.00 group=explore-stage1 run=arcn-samp2 by=gpu-up.sh
    - 2026-10-29T02:27Z CREATE vm=soup-gpu-arcn-samp2-1029022659-7385 zone=us-central1-b type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.00 spent_before_usd=0 group=explore-stage1 run=arcn-samp2 by=gpu-up.sh
    - 2026-10-29T02:28Z READY vm=soup-gpu-arcn-samp2-1029022659-7385 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T02:28Z RUN vm=soup-gpu-arcn-samp2-1029022659-7385 timeout=14400s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T03:34Z RUN-END vm=soup-gpu-arcn-samp2-1029022659-7385 rc=0
    - 2026-10-29T03:36Z DELETE vm=soup-gpu-arcn-samp2-1029022659-7385 zone=us-central1-b type=g2-standard-8 age_h=1.13 cost_usd=0.98 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T03:36:15Z END rid=arcn-samp2 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 2026-10-29T01:44Z PENDING-CREATE vm=soup-gpu-arcn-truth2a-1029014420-23150 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.87 group=explore-stage1 run=arcn-truth2a by=gpu-up.sh
    - 2026-10-29T01:45Z CREATE vm=soup-gpu-arcn-truth2a-1029014420-23150 zone=us-central1-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.87 spent_before_usd=0 group=explore-stage1 run=arcn-truth2a by=gpu-up.sh
    - 2026-10-29T01:46Z READY vm=soup-gpu-arcn-truth2a-1029014420-23150 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T01:46Z RUN vm=soup-gpu-arcn-truth2a-1029014420-23150 timeout=18000s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T03:53Z RUN-END vm=soup-gpu-arcn-truth2a-1029014420-23150 rc=0
    - 2026-10-29T03:54Z DELETE vm=soup-gpu-arcn-truth2a-1029014420-23150 zone=us-central1-c type=g2-standard-8 age_h=2.14 cost_usd=1.86 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T03:54:32Z END rid=arcn-truth2a rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 2026-10-29T03:11Z PENDING-CREATE vm=soup-gpu-arcn-samp4-1029031134-18719 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=arcn-samp4 by=gpu-up.sh
    - 2026-10-29T03:13Z CREATE vm=soup-gpu-arcn-samp4-1029031134-18719 zone=us-west1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=arcn-samp4 by=gpu-up.sh
    - 2026-10-29T03:14Z READY vm=soup-gpu-arcn-samp4-1029031134-18719 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T03:15Z RUN vm=soup-gpu-arcn-samp4-1029031134-18719 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T04:03Z RUN-END vm=soup-gpu-arcn-samp4-1029031134-18719 rc=0
    - 2026-10-29T04:04Z DELETE vm=soup-gpu-arcn-samp4-1029031134-18719 zone=us-west1-a type=g2-standard-8 age_h=0.84 cost_usd=0.73 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T04:04:57Z END rid=arcn-samp4 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 2026-10-29T01:44Z PENDING-CREATE vm=soup-gpu-arcn-truth2b-1029014440-13678 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.87 group=explore-stage1 run=arcn-truth2b by=gpu-up.sh
    - 2026-10-29T01:45Z CREATE vm=soup-gpu-arcn-truth2b-1029014440-13678 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.87 spent_before_usd=0 group=explore-stage1 run=arcn-truth2b by=gpu-up.sh
    - 2026-10-29T01:48Z READY vm=soup-gpu-arcn-truth2b-1029014440-13678 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T01:48Z RUN vm=soup-gpu-arcn-truth2b-1029014440-13678 timeout=18000s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T04:31Z RUN-END vm=soup-gpu-arcn-truth2b-1029014440-13678 rc=0
    - 2026-10-29T04:32Z DELETE vm=soup-gpu-arcn-truth2b-1029014440-13678 zone=us-central1-a type=g2-standard-8 age_h=2.78 cost_usd=2.42 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T04:32:50Z END rid=arcn-truth2b rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)

## Summary (04:45Z)

| run | VM | zone | hours | cost (ceiling) | outcome |
|---|---|---|---|---|---|
| arcn-truth1 | soup-gpu-arcn-truth1-1029011913-1334 | us-central1-a | 0.39 | $0.34 | stopped early (slow kernel), 3 truth files kept |
| arcn-truth2a | soup-gpu-arcn-truth2a-1029014420-23150 | us-central1-c | 2.14 | $1.86 | dev truths, noiseless controls, eval nets 18-23 |
| arcn-truth2b | soup-gpu-arcn-truth2b-1029014440-13678 | us-central1-a | 2.78 | $2.42 | eval nets 0-17 |
| arcn-samp1 | soup-gpu-arcn-samp1-1029021802-19997 | us-central1-b | 0.13 | $0.11 | crashed on a device bug (MC files kept) |
| arcn-samp2 | soup-gpu-arcn-samp2-1029022659-7385 | us-central1-b | 1.13 | $0.98 | sampler v1 (all sigma) + antithetic v1 |
| arcn-samp3 | soup-gpu-arcn-samp3-1029024116-697 | us-central1-a | 0.10 | $0.09 | failed (scipy missing) |
| arcn-truth3, arcn-truth3b | none created | - | 0 | $0 | L4 stockout in 13 zones |
| arcn-samp4 | soup-gpu-arcn-samp4-1029031134-18719 | us-west1-a | 0.84 | $0.73 | sampler v2 (after the code review) |

Total 7.51 VM-hours, $6.53 at the $0.87/h ceiling (cap $50). Every VM deleted by gpu-run.sh cleanup; `gcloud compute
instances list --filter="name~arcn"` returned no instances at 04:33Z.
2026-10-29T05:48:20Z LAUNCH rid=arcn-tdev1 provisioning=STANDARD machine=g2-standard-8(L4) job-loop timeout=36000s (tails part 2)
2026-10-29T05:52:28Z JOB rid=arcn-tdev1 name=x3a cmd="python3 -c "print(1)"; /root/venv/bin/python tails/x3.py 100,101,102,103 0.001 26 $OUT/x3"
2026-10-29T05:52:41Z JOB rid=arcn-tdev1 name=x1a cmd="for s in 100 101; do for L in 32 8; do /root/venv/bin/python tails/x1.py $s $L 0.001 32 $OUT/x1; done; done"
2026-10-29T06:01:38Z JOB rid=arcn-tdev1 name=x1b cmd="for s in 100 101; do /root/venv/bin/python tails/x1.py $s 4 0.001 32 $OUT/x1; done"
2026-10-29T06:03:53Z JOB rid=arcn-tdev1 name=x2a cmd="for L in 4 8 32; do /root/venv/bin/python tails/x2.py 100 $L $OUT/x1/x1_100_L${L}_s0.001.json $OUT/x2 8 2; done"
2026-10-29T06:05:05Z JOB rid=arcn-tdev1 name=bench cmd="/root/venv/bin/python tails/bench.py"
2026-10-29T06:30:05Z JOB rid=arcn-tdev1 name=k1thr cmd="/root/venv/bin/python tails/thresholds.py 102,103,104,105 $OUT/thr"
2026-10-29T06:30:20Z JOB rid=arcn-tdev1 name=k2truth cmd="for s in 102 103 104 105; do /root/venv/bin/python tails/truth.py $s 0.001 $OUT/thr $OUT/truth; done"
2026-10-29T06:30:34Z JOB rid=arcn-tdev1 name=k3mode cmd="for s in 102 103 104 105; do /root/venv/bin/python tails/mode.py $s 0.001 32 $OUT/thr $OUT/mode 8; done"
2026-10-29T06:30:58Z JOB rid=arcn-tdev1 name=k4mom cmd="/root/venv/bin/python tails/x3.py 102,103,104,105 0.001 22 $OUT/x3k22"
2026-10-29T06:31:20Z JOB rid=arcn-tdev1 name=k5tsmc cmd="for s in 102 103; do for L in 4 8 32; do /root/venv/bin/python tails/x2.py $s $L $OUT/thr/thr_$s.json $OUT/x2k 8 2; done; done"
2026-10-29T06:31:38Z JOB rid=arcn-tdev1 name=k6is cmd="for s in 102 103 104 105; do /root/venv/bin/python tails/x4.py $s $OUT/thr $OUT/truth $OUT/x4 4 2; done"
2026-10-29T06:31:48Z LAUNCH rid=arcn-tdev2 provisioning=STANDARD machine=g2-standard-8(L4) job-loop timeout=36000s (tails part 2)
2026-10-29T06:34:46Z JOB rid=arcn-tdev2 name=e1thr cmd="/root/venv/bin/python tails/thresholds.py 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 $OUT/thr"
2026-10-29T06:35:01Z JOB rid=arcn-tdev2 name=e2truth001 cmd="for s in $(seq 0 19); do /root/venv/bin/python tails/truth.py $s 0.001 $OUT/thr $OUT/truth; done"
2026-10-29T06:35:17Z JOB rid=arcn-tdev2 name=e3truth003 cmd="for s in $(seq 0 19); do /root/venv/bin/python tails/truth.py $s 0.003 $OUT/thr $OUT/truth; done"
2026-10-29T06:35:31Z JOB rid=arcn-tdev2 name=e4truth01 cmd="for s in $(seq 0 19); do /root/venv/bin/python tails/truth.py $s 0.01 $OUT/thr $OUT/truth; done"
2026-10-29T06:40:13Z JOB rid=arcn-tdev1 name=m1xc cmd="for sg in 0.001 0.003 0.01; do /root/venv/bin/python tails/mcx.py 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 $sg thr_eval $OUT/mc 23 xc 1; done"
2026-10-29T06:40:30Z JOB rid=arcn-tdev1 name=m2m20 cmd="for sg in 0.001 0.003 0.01; do /root/venv/bin/python tails/mcx.py 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 $sg thr_eval $OUT/mc 20 m20 4; done"
2026-10-29T06:40:46Z JOB rid=arcn-tdev1 name=m3kd cmd="for sg in 0.001 0.003 0.01; do /root/venv/bin/python tails/kerdock_tail.py 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 $sg thr_eval $OUT/kd 16 16; done"
2026-10-29T06:43:42Z JOB rid=arcn-tdev1 name=m4m22 cmd="for sg in 0.001 0.003 0.01; do /root/venv/bin/python tails/mcx.py 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 $sg thr_eval $OUT/mc 22 m22 4; done"
2026-10-29T06:46:09Z JOB rid=arcn-tdev1 name=n1truth01 cmd="for s in $(seq 0 19); do /root/venv/bin/python tails/truth.py $s 0.01 thr_eval $OUT/truth; done"
2026-10-29T06:46:09Z MOVE job e4truth01 (sigma 0.01 truth) from arcn-tdev2 (replaced by a no-op) to arcn-tdev1 as n1truth01
2026-10-29T06:56:08Z JOB rid=arcn-tdev1 name=m25mode cmd="for sg in 0.001 0.003 0.01; do for s in $(seq 0 19); do /root/venv/bin/python tails/mode.py $s $sg 32 thr_eval $OUT/mode 8; done; done"
2026-10-29T07:08:53Z JOB rid=arcn-tdev2 name=e5truth01b cmd="for s in $(seq 10 19); do /root/venv/bin/python tails/truth.py $s 0.01 thr_eval $OUT/truth; done"
2026-10-29T07:08:53Z SPLIT sigma 0.01 truth: seeds 0-9 on arcn-tdev1 (n1truth01), 10-19 on arcn-tdev2 (e5truth01b)
2026-10-29T08:34:07Z STOP rid=arcn-tdev1
    - 2026-10-29T05:48Z PENDING-CREATE vm=soup-gpu-arcn-tdev1-1029054823-13747 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.22 group=explore-stage1 run=arcn-tdev1 by=gpu-up.sh
    - 2026-10-29T05:48Z CREATE vm=soup-gpu-arcn-tdev1-1029054823-13747 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.22 spent_before_usd=0 group=explore-stage1 run=arcn-tdev1 by=gpu-up.sh
    - 2026-10-29T05:50Z READY vm=soup-gpu-arcn-tdev1-1029054823-13747 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T05:50Z RUN vm=soup-gpu-arcn-tdev1-1029054823-13747 timeout=36000s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T08:34Z RUN-END vm=soup-gpu-arcn-tdev1-1029054823-13747 rc=0
    - 2026-10-29T08:35Z DELETE vm=soup-gpu-arcn-tdev1-1029054823-13747 zone=us-central1-a type=g2-standard-8 age_h=2.76 cost_usd=2.40 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T08:35:17Z END rid=arcn-tdev1 rc=127 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
2026-10-29T08:37:51Z STOP rid=arcn-tdev2
    - 2026-10-29T06:32Z PENDING-CREATE vm=soup-gpu-arcn-tdev2-1029063155-32447 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=5.22 group=explore-stage1 run=arcn-tdev2 by=gpu-up.sh
    - 2026-10-29T06:33Z CREATE vm=soup-gpu-arcn-tdev2-1029063155-32447 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=5.22 spent_before_usd=0 group=explore-stage1 run=arcn-tdev2 by=gpu-up.sh
    - 2026-10-29T06:34Z READY vm=soup-gpu-arcn-tdev2-1029063155-32447 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T06:34Z RUN vm=soup-gpu-arcn-tdev2-1029063155-32447 timeout=36000s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T08:37Z RUN-END vm=soup-gpu-arcn-tdev2-1029063155-32447 rc=0
    - 2026-10-29T08:38Z DELETE vm=soup-gpu-arcn-tdev2-1029063155-32447 zone=us-central1-a type=g2-standard-8 age_h=2.10 cost_usd=1.83 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T08:39:02Z END rid=arcn-tdev2 rc=127 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)

## Part 2 summary (08:40Z)

| run | VM | zone | hours | cost (ceiling) | outcome |
|---|---|---|---|---|---|
| arcn-tdev1 | soup-gpu-arcn-tdev1-1029054823-13747 | us-central1-a | 2.76 | $2.40 | exploration, kill tests (dev 102-105), Monte Carlo cross-checks and moments, Kerdock, mode splitting, truth sigma 0.01 seeds 0-9 |
| arcn-tdev2 | soup-gpu-arcn-tdev2-1029063155-32447 | us-central1-a | 2.10 | $1.83 | evaluation thresholds, truths sigma 0.001 and 0.003, sigma 0.01 seeds 10-19 |

Part 2: 4.86 VM-hours, $4.23 at the $0.87/h ceiling. Lane total (parts 1 + 2): 12.4 VM-hours, $10.76 of the $50 cap.
Both VMs stopped through the job loop's STOP file and deleted by gpu-run.sh; `gcloud compute instances list
--filter="name~arcn"` was empty at 08:39Z. (The "rc=127" on the END lines comes from the launcher's post-run step; both
runs ended "GPU-RUN EXIT 0".)
2026-10-29T09:28:25Z LAUNCH rid=arcn-tconf provisioning=STANDARD machine=g2-standard-8(L4) job-loop timeout=10800s (tails part 2)
2026-10-29T09:32:49Z JOB rid=arcn-tconf name=c1thr cmd="/root/venv/bin/python tails/thresholds.py 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43 $OUT/thr"
2026-10-29T09:33:05Z JOB rid=arcn-tconf name=c2truth cmd="for s in $(seq 24 43); do /root/venv/bin/python tails/truth.py $s 0.001 $OUT/thr $OUT/truth 4,8; done"
2026-10-29T09:33:19Z JOB rid=arcn-tconf name=c3mc cmd="/root/venv/bin/python tails/mcx.py 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43 0.001 $OUT/thr $OUT/mc 23 xc 1"
2026-10-29T09:52:20Z STOP rid=arcn-tconf
    - 2026-10-29T09:28Z PENDING-CREATE vm=soup-gpu-arcn-tconf-1029092831-24646 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=arcn-tconf by=gpu-up.sh
    - 2026-10-29T09:30Z CREATE vm=soup-gpu-arcn-tconf-1029092831-24646 zone=us-central1-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=arcn-tconf by=gpu-up.sh
    - 2026-10-29T09:31Z READY vm=soup-gpu-arcn-tconf-1029092831-24646 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T09:32Z RUN vm=soup-gpu-arcn-tconf-1029092831-24646 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T09:52Z RUN-END vm=soup-gpu-arcn-tconf-1029092831-24646 rc=0
    - 2026-10-29T09:53Z DELETE vm=soup-gpu-arcn-tconf-1029092831-24646 zone=us-central1-c type=g2-standard-8 age_h=0.37 cost_usd=0.32 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T09:53:25Z END rid=arcn-tconf rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
| arcn-tconf | soup-gpu-arcn-tconf-1029092831-24646 | us-central1-c | 0.37 | $0.32 | confirmatory test (nets 24-43): thresholds, truths at L = 4, 8, Monte Carlo cross-check |

Part 2 with the confirmatory run: 5.23 VM-hours, $4.55. Lane total (parts 1 + 2): $11.08 of the $50 cap. Instance list
empty again at 09:55Z.
