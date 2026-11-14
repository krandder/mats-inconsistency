#!/usr/bin/env python3
"""Final read-only no-rental check after all seed supervisors have exited."""
import datetime,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);p=root/'runs/phase3/infra';p.mkdir(parents=True,exist_ok=True)
checks={}
for kind,flt in [('instances','labels.soup-group=pool-sddpv'),('disks','name~^soup-gpu-phase3-k16-f12-s')]:
 checks[kind]=json.loads(subprocess.check_output(['gcloud','compute',kind,'list','--project=GCP_PROJECT','--filter='+flt,'--format=json(name,zone)']))
now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
(p/'final-deletion.json').write_text(json.dumps(dict(checked_at=now,**checks),indent=2)+'\n')
assert checks==dict(instances=[],disks=[]),checks
with (root/'GPU-LOG.md').open('a') as f:f.write(f'- {now} VERIFIED group=pool-sddpv instances=[] disks=[] after all seed supervisors exited; evidence=runs/phase3/infra/final-deletion.json\n')
