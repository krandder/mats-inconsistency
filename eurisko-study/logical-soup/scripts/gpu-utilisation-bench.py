#!/usr/bin/env python3
"""Paired, alternating-order whole-process trials of one isolated host change."""
import csv,json,pathlib,re,subprocess,sys,time
p=pathlib.Path(sys.argv[1]);profiles=len(sys.argv)>2
results=json.loads((p/'summary.json').read_text()) if profiles else []
trials=[(18,1,0)] if profiles else [(16,10,0),(18,10,0),(18,10,1),(18,10,2)]
for k,epochs,trial in trials:
 for variant in (['base','guard'] if trial%2==0 else ['guard','base']):
  prefix='profile-' if profiles else '';name=f'{prefix}{variant}-k{k}-{trial}'
  cmd=['stdbuf','-oL','-eL',f'build/utilisation/{prefix}{variant}','--gpu','on','--threads','8',str(k),str(epochs),'1','256','12']
  with (p/(name+'.raw')).open('w') as out,(p/(name+'.stderr')).open('w') as err:
   started=time.monotonic();r=subprocess.run(cmd,stdout=out,stderr=err,timeout=240);elapsed=time.monotonic()-started
  assert r.returncode==0,(name,r.returncode)
  lines=(p/(name+'.raw')).read_text().splitlines();data='\n'.join(x for x in lines if not x.startswith(('PROFILE,','HOST_PROFILE,')))+'\n'
  (p/(name+'.csv')).write_text(data);rows=list(csv.DictReader(data.splitlines()));assert len(rows)==epochs
  w=(1<<k)*1048576
  for i,row in enumerate(rows):
   assert int(row['wealth'])==w+int(row['trader_pnl'])-int(row['used'])-int(row['burned']),(name,i)
   w=int(row['wealth'])
  ops=sum(int(row['ops']) for row in rows);ns=[int(x) for x in re.findall(r'wall_ns=(\d+)',(p/(name+'.stderr')).read_text())];assert ns,'No CUDA dispatch'
  result=dict(name=name,variant=variant,logk=k,logq=12,fuel=256,epochs=epochs,trial=trial,wall_s=elapsed,instructions=ops,instructions_per_s=ops/elapsed,cuda_wall_s=sum(ns)/1e9,cuda_passes=len(ns),ledger='PASS',phases=[x for x in lines if x.startswith(('PROFILE,','HOST_PROFILE,'))])
  results.append(result);(p/'summary.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(result),flush=True)
for k in [16,18]:
 runs=[r for r in results if r['logk']==k];ref=(p/(runs[0]['name']+'.csv')).read_text().splitlines()
 for r in runs:
  data=(p/(r['name']+'.csv')).read_text().splitlines();n=min(len(ref),len(data));assert ref[:n]==data[:n],r['name'];r['csv_equal_prefix']=True
(p/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
