#!/usr/bin/env python3
"""Bounded benchmarks, exact ledger/CSV checks, and split-stage diagnostics."""
import csv, datetime, json, pathlib, re, subprocess, sys, time
p=pathlib.Path(sys.argv[1]); device=sys.argv[2]; results=[]
# A100 command has 24 min total; individual runs capped to leave deletion time.
deadline=time.monotonic()+(1260 if device=='a100' else 2400)

def run(name,binary,lane,k,n,cap=900):
    remaining=deadline-time.monotonic()
    if remaining<15: return None
    cmd=['stdbuf','-oL','-eL',binary,'--gpu','on' if lane=='gpu' else 'off','--threads','8',str(k),str(n),'1','256','12']
    (p/(name+'-start.json')).write_text(json.dumps(dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,epochs_requested=n))+'\n')
    t=time.monotonic()
    with (p/(name+'.raw')).open('wb') as out,(p/(name+'.stderr')).open('wb') as err:
        try: rc=subprocess.run(cmd,stdout=out,stderr=err,timeout=min(cap,remaining-5)).returncode
        except subprocess.TimeoutExpired: rc=124
    dt=time.monotonic()-t
    lines=(p/(name+'.raw')).read_text().splitlines()
    phases=[list(map(int,x.split(',')[1:])) for x in lines if x.startswith('PROFILE,')]
    host=[list(map(int,x.split(',')[1:])) for x in lines if x.startswith('HOST_PROFILE,')]
    data='\n'.join(x for x in lines if not x.startswith(('PROFILE,','HOST_PROFILE,')))+'\n'
    (p/(name+'.csv')).write_text(data)
    rows=list(csv.DictReader(data.splitlines()))
    r=dict(name=name,device=device,lane=lane,logk=k,logq=12,fuel=256,epochs_requested=n,epochs_completed=len(rows),wall_s=dt,exit_code=rc)
    err=(p/(name+'.stderr')).read_text()
    ns=[int(x) for x in re.findall(r'wall_ns=(\d+)',err)]
    r.update(cuda_passes=len(ns),cuda_wall_s=sum(ns)/1e9)
    if rc==0:
        assert len(rows)==n,(name,len(rows))
        w=(1<<k)*1048576
        for x in rows:
            assert int(x['wealth'])==w+int(x['trader_pnl'])-int(x['used'])-int(x['burned']),(name,x)
            w=int(x['wealth'])
        ops=sum(int(x['ops']) for x in rows)
        r.update(instructions=ops,instructions_per_s=ops/dt,ledger='PASS')
        if lane=='gpu': assert ns,'silent CPU fallback'
    if host:
        r['host_phase_columns']=['epoch','prices_settle_ms','replication_ms','metrics_ms']
        r['host_phases']=host
    if phases:
        r['phase_columns']=['epoch','prepare_ms','boundary_ms','traders_ms','aggregate_ms','host_finish_ms']
        r['phases']=phases
    results.append(r);(p/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(r),flush=True)
    return r

# Short profiling first leaves useful diagnostics even if a large run fails.
for lane in ['gpu','cpu']:
    run('profile-'+lane+'-k18','build/scaling/profile',lane,18,1,180)
# Main measurements: GPU first ensures a number even if the CPU hits the deadline.
for k in ([18,20] if device=='l4' else [20]):
    for lane in ['gpu','cpu']:
        run(lane+'-k'+str(k),'build/scaling/soup',lane,k,10)
# Compare all successful overlapping prefixes, including split profile output.
for k in [18,20]:
    valid=[r for r in results if r['logk']==k and r['exit_code']==0]
    if valid:
        ref=valid[0]; a=(p/(ref['name']+'.csv')).read_text().splitlines()
        for r in valid:
            b=(p/(r['name']+'.csv')).read_text().splitlines();n=min(len(a),len(b))
            assert a[:n]==b[:n],(ref['name'],r['name'],'CSV mismatch')
            r['csv_reference']=ref['name'];r['csv_identical_prefix']=True
(p/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
# Optional bounded profiler capture. Its overhead is excluded from main timings.
import shutil
nsys=shutil.which('nsys')
if nsys and deadline-time.monotonic()>130:
    cmd=[nsys,'profile','--trace=cuda','--export=sqlite','--sample=none','--cpuctxsw=none','--cuda-um-cpu-page-faults=true','--cuda-um-gpu-page-faults=true','--force-overwrite=true','-o',str(p/'nsys'), 'build/scaling/soup','--gpu','on','--threads','8','18','1','1','256','12']
    with (p/'nsys.stdout').open('w') as out,(p/'nsys.stderr').open('w') as err:
        try: subprocess.run(cmd,stdout=out,stderr=err,timeout=120)
        except subprocess.TimeoutExpired: pass

# A runtime pass launches up to four kernels. Pick the actual longest kernel
# from Nsight Systems rather than confusing pass and kernel launch indices.
if shutil.which('ncu') and deadline-time.monotonic()>100:
    quick=deadline-time.monotonic()<190
    with (p/'counter-runner.log').open('w') as log:
        try:
            subprocess.run([sys.executable,'scripts/gpu-profile-counter.py',str(p)]+(['quick'] if quick else []),stdout=log,stderr=log,timeout=100 if quick else 190)
        except subprocess.TimeoutExpired:
            log.write('counter helper timed out\n')
