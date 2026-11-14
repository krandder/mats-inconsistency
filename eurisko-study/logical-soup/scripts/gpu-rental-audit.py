#!/usr/bin/env python3
"""Verify one rental is gone and record conservative operation-based duration."""
import datetime,json,os,pathlib,subprocess,sys
name=sys.argv[1];p=pathlib.Path(sys.argv[2]);p.mkdir(parents=True,exist_ok=True)
def query(kind,flt,fmt):
    return json.loads(subprocess.check_output(['gcloud','compute',kind,'list','--project=GCP_PROJECT','--filter='+flt,'--format=json('+fmt+')']))
instances=query('instances','name='+name,'name,status');disks=query('disks','name='+name,'name,zone')
for key,value in [('instances',instances),('disks',disks)]: (p/f'deletion-{key}.json').write_text(json.dumps(value,indent=2)+'\n')
assert instances==disks==[],f'{name}: rental or disk remains'
ops=query('operations','targetLink~'+name,'operationType,insertTime,startTime,endTime,targetLink,status,error')
(p/'operations.json').write_text(json.dumps(ops,indent=2)+'\n')
starts=[x for x in ops if x['operationType']=='insert' and not x.get('error')]
if not starts: sys.exit('No successful insert found; do not infer zero cost')
ends=[x for x in ops if x['operationType'] in ['delete','compute.instances.preempted'] and x.get('endTime') and not x.get('error')]
assert ends,'No completed deletion/preemption receipt'
start=min(datetime.datetime.fromisoformat(x['insertTime']) for x in starts);end=max(datetime.datetime.fromisoformat(x['endTime']) for x in ends)
duration=(end-start).total_seconds();now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
log=pathlib.Path(os.environ.get('GPU_LOG_PATH','GPU-LOG.md'))
creates=[line for line in log.read_text().splitlines() if ' CREATE ' in line and f'vm={name} ' in line]
if not creates: creates=[line for line in log.read_text().splitlines() if ' PENDING-CREATE ' in line and f'vm={name} ' in line]
assert len(creates)==1, 'Need one CREATE or PENDING-CREATE receipt to establish machine type'
fields=dict(x.split('=',1) for x in creates[0].split() if '=' in x)
script=pathlib.Path(__file__).resolve().parents[1]/'infra/lib.sh'
rate=float(subprocess.check_output(['bash','-c','source "$1"; rate_for "$2"','audit',str(script),fields['type']]))
a=dict(vm=name,machine_type=fields['type'],duration_s=duration,rate_usd_h=rate,cost_ceiling_usd=duration*rate/3600,rate_verified=False,checked_at=now,instances=[],disks=[])
(p/'cost-audit.json').write_text(json.dumps(a,indent=2)+'\n')
with log.open('a') as f:f.write(f'- {now} COST-AUDIT vm={name} duration_s={duration:.3f} rate_usd_h={rate} cost_usd={a["cost_ceiling_usd"]:.6f} basis=successful-insert-to-deletion-or-preemption-end UNVERIFIED-ceiling instances=[] disks=[]\n')
print(json.dumps(a))
