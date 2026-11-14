# GPU-LOG (work/sparse-codes)

Every VM used by this lane. Cap $10 (brief, 2026-10-29). On-demand L4 (g2-standard-8, ceiling $0.87/h).

2026-10-29T16:35:38Z LAUNCH rid=sc-runA provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobA.sh sc-runA"
2026-10-29T16:35:40Z LAUNCH rid=sc-runB provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobB.sh sc-runB"
2026-10-29T16:35:45Z END rid=sc-runA rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
2026-10-29T16:35:45Z END rid=sc-runB rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - sc-runA, sc-runB: rejected before provisioning (uppercase run id invalid); no VM created, $0
2026-10-29T16:35:56Z LAUNCH rid=sc-runa provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobA.sh sc-runa"
2026-10-29T16:35:59Z LAUNCH rid=sc-runb provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobB.sh sc-runb"
    - 2026-10-29T16:36Z PENDING-CREATE vm=soup-gpu-sc-runa-1029163600-16281 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.00 group=explore-stage1 run=sc-runa by=gpu-up.sh
    - 2026-10-29T16:38Z CREATE-FAILED vm=soup-gpu-sc-runa-1029163600-16281 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T16:38:26Z END rid=sc-runa rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
2026-10-29T16:39:51Z LAUNCH rid=sc-runa2 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobA.sh sc-runa2"
2026-10-29T16:41:56Z LAUNCH rid=sc-runc provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobC.sh sc-runc"
2026-10-29T16:51:39Z LAUNCH rid=sc-a3big provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobA3big.sh sc-a3big"
2026-10-29T16:51:41Z LAUNCH rid=sc-a3small provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobA3small.sh sc-a3small"
2026-10-29T16:51:43Z LAUNCH rid=sc-b2 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobB2.sh sc-b2"
    - 2026-10-29T16:36Z PENDING-CREATE vm=soup-gpu-sc-runb-1029163603-6760 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.00 group=explore-stage1 run=sc-runb by=gpu-up.sh
    - 2026-10-29T16:40Z CREATE vm=soup-gpu-sc-runb-1029163603-6760 zone=us-central1-b type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.00 spent_before_usd=0 group=explore-stage1 run=sc-runb by=gpu-up.sh
    - 2026-10-29T16:41Z READY vm=soup-gpu-sc-runb-1029163603-6760 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T16:42Z RUN vm=soup-gpu-sc-runb-1029163603-6760 timeout=14400s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T16:52Z DELETE vm=soup-gpu-sc-runb-1029163603-6760 zone=us-central1-b type=g2-standard-8 age_h=0.23 cost_usd=0.20 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=130: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T16:52:27Z END rid=sc-runb rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 2026-10-29T16:51Z PENDING-CREATE vm=soup-gpu-sc-a3small-1029165143-29421 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.87 group=explore-stage1 run=sc-a3small by=gpu-up.sh
    - 2026-10-29T16:52Z CREATE-FAILED vm=soup-gpu-sc-a3small-1029165143-29421 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c us-central1-f us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west4-a us-west4-c europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c err="ERROR: (gcloud.compute.instances.create) Could not fetch resource:  - Invalid value for field 'resource.machineType': 'https://compute.googleapis.com/compute/v1/projects/GCP_PROJECT/zones/us-central1-"
2026-10-29T16:52:34Z END rid=sc-a3small rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 2026-10-29T16:51Z PENDING-CREATE vm=soup-gpu-sc-b2-1029165146-1844 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.87 group=explore-stage1 run=sc-b2 by=gpu-up.sh
    - 2026-10-29T16:52Z CREATE-FAILED vm=soup-gpu-sc-b2-1029165146-1844 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c us-central1-f us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west4-a us-west4-c europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c err="ERROR: (gcloud.compute.instances.create) Could not fetch resource:  - Invalid value for field 'resource.machineType': 'https://compute.googleapis.com/compute/v1/projects/GCP_PROJECT/zones/us-central1-"
2026-10-29T16:52:36Z END rid=sc-b2 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 2026-10-29T16:51Z PENDING-CREATE vm=soup-gpu-sc-a3big-1029165141-29147 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.87 group=explore-stage1 run=sc-a3big by=gpu-up.sh
    - 2026-10-29T16:52Z CREATE-FAILED vm=soup-gpu-sc-a3big-1029165141-29147 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c us-central1-f us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west4-a us-west4-c europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c err="ERROR: (gcloud.compute.instances.create) Could not fetch resource:  - Invalid value for field 'resource.machineType': 'https://compute.googleapis.com/compute/v1/projects/GCP_PROJECT/zones/us-central1-"
2026-10-29T16:52:36Z END rid=sc-a3big rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - sc-a3big, sc-a3small, sc-b2 (16:51Z): CREATE-FAILED (us-central1-a/b/c stocked out, then us-central1-f has no g2 type, which ends the zone loop); no VM created, $0
2026-10-29T16:54:02Z LAUNCH rid=sc-a3big provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobA3big.sh sc-a3big"
2026-10-29T16:54:04Z LAUNCH rid=sc-a3small provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobA3small.sh sc-a3small"
2026-10-29T16:54:06Z LAUNCH rid=sc-b2 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobB2.sh sc-b2"
    - sc-runc: stopped by me at 16:56Z (E4 closure overflow bug: NaN; chain weights not saved). Fixed in run_e4.py; rerun as sc-c2.
2026-10-29T16:56:18Z LAUNCH rid=sc-c2 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobC2.sh sc-c2"
    - 2026-10-29T16:42Z PENDING-CREATE vm=soup-gpu-sc-runc-1029164158-13195 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.00 group=explore-stage1 run=sc-runc by=gpu-up.sh
    - 2026-10-29T16:43Z CREATE vm=soup-gpu-sc-runc-1029164158-13195 zone=us-central1-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.00 spent_before_usd=0 group=explore-stage1 run=sc-runc by=gpu-up.sh
    - 2026-10-29T16:44Z READY vm=soup-gpu-sc-runc-1029164158-13195 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T16:44Z RUN vm=soup-gpu-sc-runc-1029164158-13195 timeout=14400s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T16:56Z RUN-END vm=soup-gpu-sc-runc-1029164158-13195 rc=255
    - 2026-10-29T16:57Z DELETE vm=soup-gpu-sc-runc-1029164158-13195 zone=us-central1-c type=g2-standard-8 age_h=0.23 cost_usd=0.20 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=255: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T16:57:19Z END rid=sc-runc rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - sc-runa2: stopped by me at 16:57Z after arc n = 64 (v1 consistency run; v2 supersedes; budget)
    - 2026-10-29T16:40Z PENDING-CREATE vm=soup-gpu-sc-runa2-1029163954-27422 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.00 group=explore-stage1 run=sc-runa2 by=gpu-up.sh
    - 2026-10-29T16:40Z CREATE vm=soup-gpu-sc-runa2-1029163954-27422 zone=us-central1-b type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.00 spent_before_usd=0 group=explore-stage1 run=sc-runa2 by=gpu-up.sh
    - 2026-10-29T16:41Z READY vm=soup-gpu-sc-runa2-1029163954-27422 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T16:41Z RUN vm=soup-gpu-sc-runa2-1029163954-27422 timeout=14400s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T16:57Z RUN-END vm=soup-gpu-sc-runa2-1029163954-27422 rc=255
    - 2026-10-29T16:58Z DELETE vm=soup-gpu-sc-runa2-1029163954-27422 zone=us-central1-b type=g2-standard-8 age_h=0.29 cost_usd=0.25 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=255: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T16:58:36Z END rid=sc-runa2 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
2026-10-29T17:05:08Z LAUNCH rid=sc-e5 provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobE5.sh sc-e5"
    - 2026-10-29T17:05Z PENDING-CREATE vm=soup-gpu-sc-e5-1029170511-27614 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=sc-e5 by=gpu-up.sh
    - 2026-10-29T17:10Z CREATE-FAILED vm=soup-gpu-sc-e5-1029170511-27614 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west4-a europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T17:10:26Z END rid=sc-e5 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 2026-10-29T16:54Z PENDING-CREATE vm=soup-gpu-sc-a3small-1029165408-15666 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.00 group=explore-stage1 run=sc-a3small by=gpu-up.sh
    - 2026-10-29T17:10Z CREATE-FAILED vm=soup-gpu-sc-a3small-1029165408-15666 type=g2-standard-8 zones=us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west4-a europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T17:10:26Z END rid=sc-a3small rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 17:15Z packing (coordinator's request): sc-a3small's three nets (arc n=64 s65, n=128 s65, n=64 s66) run as extra processes on the sc-b2 VM; E5 (n=256; chain n=64, 128, 64-seed66) as extra processes on the sc-a3big VM. Outputs land in those runs' dirs and sync when the main job ends. sc-a3small and sc-e5 were never created (L4 stockout in all 16 zones), $0.
    - 17:18Z E5 extras on sc-a3big stopped by me: E5 n=256 hit CUDA OOM (GPU shared by 3 processes) and the small chain risked the main n=256 job; E5 moves to its own VM (sc-e5b).
2026-10-29T17:18:26Z LAUNCH rid=sc-e5b provisioning=STANDARD machine=g2-standard-8(L4) cmd="bash gpu/jobE5b.sh sc-e5b"
    - 2026-10-29T17:18Z PENDING-CREATE vm=soup-gpu-sc-e5b-1029171828-4481 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=2.70 group=explore-stage1 run=sc-e5b by=gpu-up.sh
    - 2026-10-29T17:22Z CREATE-FAILED vm=soup-gpu-sc-e5b-1029171828-4481 type=g2-standard-8 zones=us-central1-a us-central1-c us-east1-c us-central1-b us-east4-a us-east4-c us-east1-b us-east1-d us-west1-a us-west1-b us-west4-a europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c err="WARNING: Some requests generated warnings:  - Disk size: '100 GB' is larger than image size: '50 GB'. You might need to resize the root repartition manually if the operating system does not support au"
2026-10-29T17:22:35Z END rid=sc-e5b rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 17:23Z sc-b2: the extra n=128 process held 15 GB and the main tord n=64 net hit CUDA OOM (main job continued with the next net). Extra n=128 killed; it and tord n=64 will be rerun with a per-process memory cap.
    - 17:24Z E5 (n=256 with MEMFRAC 0.25; chain n=64, 128, 64-seed66 with MEMFRAC 0.18) started as capped extra processes on sc-b2 (sc-e5b could not get an L4: stockout in all zones, $0).
    - 2026-10-29T16:56Z PENDING-CREATE vm=soup-gpu-sc-c2-1029165621-17616 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=3.13 group=explore-stage1 run=sc-c2 by=gpu-up.sh
    - 2026-10-29T16:59Z CREATE vm=soup-gpu-sc-c2-1029165621-17616 zone=us-east1-c type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=3.13 spent_before_usd=0 group=explore-stage1 run=sc-c2 by=gpu-up.sh
    - 2026-10-29T17:00Z READY vm=soup-gpu-sc-c2-1029165621-17616 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T17:00Z RUN vm=soup-gpu-sc-c2-1029165621-17616 timeout=10800s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T17:23Z RUN-END vm=soup-gpu-sc-c2-1029165621-17616 rc=0
    - 2026-10-29T17:25Z DELETE vm=soup-gpu-sc-c2-1029165621-17616 zone=us-east1-c type=g2-standard-8 age_h=0.42 cost_usd=0.37 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T17:25:05Z END rid=sc-c2 rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 17:26Z sc-b2: E5 n=256 extra stopped again before any result (memory: 5 processes, 15 GB used); to be restarted when the two n=64 extras finish.
    - 17:39Z sc-b2: E5 chain hit its own 4 GB cap (MEMFRAC 0.18) at n=64 and n=128 before any result; stopped. E5 memory is reduced in code (FIT 2^16, per-k codes, 64-feature blocks) and restarted.
    - 2026-10-29T16:54Z PENDING-CREATE vm=soup-gpu-sc-a3big-1029165405-457 type=g2-standard-8 rate_usd_h=0.87 reserve_usd=4.00 group=explore-stage1 run=sc-a3big by=gpu-up.sh
    - 2026-10-29T16:55Z CREATE vm=soup-gpu-sc-a3big-1029165405-457 zone=us-central1-a type=g2-standard-8 gpu=1xnvidia-l4 model=STANDARD max_run=6h rate_usd_h=0.87 (UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower) reserve_usd=4.00 spent_before_usd=0 group=explore-stage1 run=sc-a3big by=gpu-up.sh
    - 2026-10-29T16:56Z READY vm=soup-gpu-sc-a3big-1029165405-457 ip=[ip-removed] gpu=NVIDIA L4, 580.178.04 nvcc=Build cuda_12.9.r12.9/compiler.35813241_0 clang=Ubuntu clang version 18.1.3 (1ubuntu1) bun=1.3.14 bend=v2.0.29@c8691aa 
    - 2026-10-29T16:56Z RUN vm=soup-gpu-sc-a3big-1029165405-457 timeout=14400s cmd="bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/null\ \&\&\ apt-get\ install\ -y\ -q\ python3-venv\ python3-pip\ \>/dev/null\ \&\&"
    - 2026-10-29T18:01Z RUN-END vm=soup-gpu-sc-a3big-1029165405-457 rc=0
    - 2026-10-29T18:02Z DELETE vm=soup-gpu-sc-a3big-1029165405-457 zone=us-central1-a type=g2-standard-8 age_h=1.12 cost_usd=0.97 (rate ceiling x age) sync=skipped reason="gpu-run.sh finished rc=0: bash -c export\ DEBIAN_FRONTEND=noninteractive\;\ apt-get\ update\ -q\ \>/dev/nu"
2026-10-29T18:02:31Z END rid=sc-a3big rc=0 (VM deleted by gpu-run.sh cleanup; verify with gcloud list)
    - 18:06Z sc-b2: extra processes: E3-dense rerun chain (n=256, 64, 64-s66; MEMFRAC 0.15); scheduler starts the n=128 net (0.2) and E5 n=256 then n=128 (0.25) when the n=64 extras finish. E5 n=64 s65 done; E5 n=128/256 earlier attempts hit their caps.
    - 18:31Z time trim scheduled on sc-b2: ordered tanh n=256 skipped when it starts; arc sigma=0.001 n=256 stopped once the extras finish (both secondary).
    - 18:32Z sc-b2: the n=128 net hit its 4.4 GB cap in E2 at layer 1 and was restarted with MEMFRAC 0.35.
    - 18:44Z sc-b2: E5 n=256 hit its cap at cut 24 (cuts 8 and 16 saved); the n=128 net hit its 7.7 GB cap again. Restarted n=128 as E1+E2 only (MEMFRAC 0.45). Queued on the VM after E5 n=128: E3-dense n=256 (0.3), then E5 n=256 cut 24 only (0.4).
    - 18:54Z sc-b2: the n=128 net hit its 9.9 GB cap in E2 (code matrix at F = 16n). Queued after stage 4: n=128 E1+E2 with F <= 8n, MEMFRAC 0.6.
    - 18:55Z sc-b2: took manual control of the VM's lifecycle to protect the extra processes (the main job's last net could end the job early). The local launcher, run_on_l4 and gpu-run.sh were stopped with SIGKILL (no cleanup); the main job's ssh session is left running. I will rsync /root/runs and delete the VM myself when the extras finish, and verify the instance list. Backstops: the VM's 6 h max-run self-delete (~22:55Z) and the watchdog.
    - 19:25Z sc-b2 (manual): extras finished; /root/runs synced to /tmp/l4runs/sc-b2/runs; VM soup-gpu-sc-b2-1029165410-10925 deleted; instance list for it now: ''
    - 19:27Z lane total (rate ceiling x age): runb 0.20, runa2 0.25, runc 0.20, c2 0.37, a3big 0.97, b2 ~2.18 (16:55-19:25Z) = about $4.2 of the $10 cap. Instance list for soup-gpu-sc-* empty at 19:26Z.
