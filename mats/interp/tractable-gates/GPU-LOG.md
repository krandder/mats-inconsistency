# GPU log (work/tractable-gates)

Cap $3 (main's approval, 2026-10-31 ~01:37Z: one on-demand L4 for the PFN arm, 6 activations x 3 seeds, full 90k recipe).
Launcher: gpu/launch.sh (adapted from cm-interp-B/gpu/launch_x.sh) via work/run_on_l4.sh; the VM auto-deletes.
Job deadlines: no new training after 2.0 h, hard stop at 2.67 h; VM timeout 3 h (ceiling $0.87/h -> at most ~$2.6).

2026-10-31T01:49:09Z LAUNCH rid=tg-pfn-1 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu_jobs.sh tg-pfn-1 jobs.txt train_act.py 7200 9600" (PFN activation arm, 18 trainings, cap $3)
    - 2026-10-31T01:49Z PENDING-CREATE vm=soup-gpu-tg-pfn-1-1031014916-19599 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=tg-pfn-1 by=gpu-up.sh
    - 2026-10-31T01:50Z CREATE vm=soup-gpu-tg-pfn-1-1031014916-19599 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=tg-pfn-1 by=gpu-up.sh
    - 2026-10-31T01:51Z READY vm=soup-gpu-tg-pfn-1-1031014916-19599 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-31T01:51Z RUN vm=soup-gpu-tg-pfn-1-1031014916-19599 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-31T03:05Z RUN-END vm=soup-gpu-tg-pfn-1-1031014916-19599 rc=0
    - 2026-10-31T03:07Z DELETE vm=soup-gpu-tg-pfn-1-1031014916-19599 zone=us-central1-a type=g2-standard-8 age_h=1.27 cost_usd=1.10 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-31T03:07:05Z END rid=tg-pfn-1 rc=0 (VM deleted by gpu-run.sh cleanup; instance list checked separately)
2026-10-31T03:07:05Z attempt 1 (tg-pfn-1): VM was created and the job ran; stopping retries
- tg-pfn-1 (us-central1-a, 1.27 h, $1.10 at the $0.87/h ceiling): all 18 PFN trainings finished (gpu-run.out EXIT 0); VM deleted 03:07Z; instance list checked 03:27Z (gcloud: 14 instances in the project, none named soup-gpu*). Lane total $1.10 of $3.
