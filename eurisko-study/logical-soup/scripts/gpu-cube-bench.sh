#!/usr/bin/env bash
# One L4, bounded workload; infra/gpu-run.sh owns deletion and result sync.
set -euo pipefail
cd /root/soup
export BEND_HOME=/opt/bend2
OUT=${GPU_BENCH_OUT:-/root/runs/gpu-cube}
[[ ! -e "$OUT/started-at.txt" ]] || { echo "Refuse to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT"
cp /var/lib/soup/READY "$OUT/vm-ready.txt"
cp build/gpu-source-commit.txt "$OUT/source-commit.txt"
date -u +%FT%TZ > "$OUT/started-at.txt"
CUDA_OK=0
if timeout --kill-after=20 600 scripts/build-gpu.sh > "$OUT/gpu-build.log" 2>&1; then CUDA_OK=1; fi
cp build/gpu/soup.*log "$OUT/" || true
if ((CUDA_OK)); then
  CPU='build/gpu/soup --gpu off --threads 8'
  cp build/gpu/soup.build-info.txt "$OUT/"
else
  echo 'CUDA build failed; CPU and JS benchmarks remain diagnostic.' | tee "$OUT/gpu-failure.txt"
  timeout --kill-after=20 480 scripts/build-native.sh > "$OUT/cpu-build.log" 2>&1
  CPU='build/native/soup --threads 8'
fi
bun "$BEND_HOME/bend2/main.ts" src/soup.bend -o build/cube.js
export CPU CUDA_OK
python3 - "$OUT" <<'PY'
import csv,json,os,pathlib,resource,shlex,subprocess,sys,time
p=pathlib.Path(sys.argv[1]); results=[]
baseline_js='build/pool-on8tb-base/soup.js'
if not pathlib.Path(baseline_js).is_file(): baseline_js='build/cube.js'
(p/'js-source.txt').write_text(baseline_js+'\n')
lanes={'js':['bun',baseline_js],'cpu':shlex.split(os.environ['CPU'])}
if os.environ['CUDA_OK']=='1': lanes['gpu']=['build/gpu/soup','--gpu','on','--threads','8']
for k,n,q in [(8,10,2),(14,10,12),(16,10,12)]:
    baseline=None
    reference_lane=None
    for lane,cmd in lanes.items():
        name=f'{lane}-k{k}'
        env=os.environ.copy()
        # K16 metrics traverse a long list; Bun's default JSC stack overflows.
        # Both OS and JSC limits must grow. This changes capacity, not output.
        if lane=='js':
            soft,hard=resource.getrlimit(resource.RLIMIT_STACK)
            resource.setrlimit(resource.RLIMIT_STACK,(128*1024*1024,hard))
            env['JSC_maxPerThreadStackUsage']='67108864'
        t=time.monotonic()
        with (p/f'{name}.csv').open('wb') as out,(p/f'{name}.stderr').open('wb') as err:
            try: rc=subprocess.run(cmd+list(map(str,[k,n,1,256,q])),stdout=out,stderr=err,timeout=240,env=env).returncode
            except subprocess.TimeoutExpired: rc=124
        dt=time.monotonic()-t
        result=dict(lane=lane,logk=k,logq=q,epochs=n,fuel=256,wall_s=dt,exit_code=rc)
        if rc==0:
            data=(p/f'{name}.csv').read_bytes()
            rows=list(csv.DictReader(data.decode().splitlines()))
            assert len(rows)==n
            w=(1<<k)*1048576
            for r in rows:
                assert int(r['wealth'])==w+int(r['trader_pnl'])-int(r['used'])-int(r['burned']),r
                w=int(r['wealth'])
            ops=sum(int(r['ops']) for r in rows)
            if baseline is None:
                baseline=data
                reference_lane=lane
            result.update(instructions=ops,instructions_per_s=ops/dt,ledger='PASS',csv_identical=data==baseline,csv_reference_lane=reference_lane)
            assert data==baseline, name+' differs from JS'
            if lane=='gpu':
                passes=(p/f'{name}.stderr').read_text().count('SOUP_CUDA_PASS')
                result['cuda_passes']=passes
                assert passes>0,'GPU silently skipped'
        if rc!=0:
            rows=list(csv.DictReader((p/f'{name}.csv').read_text().splitlines()))
            completed=[r for r in rows if r.get('ops') and None not in r.values()]
            result['completed_epochs']=len(completed)
            result['completed_instructions']=sum(int(r['ops']) for r in completed)
            result['completed_instructions_per_wall_s']=result['completed_instructions']/dt
            result['cuda_passes']=(p/f'{name}.stderr').read_text().count('SOUP_CUDA_PASS')
        results.append(result)
        (p/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)
PY
date -u +%FT%TZ > "$OUT/finished-at.txt"
