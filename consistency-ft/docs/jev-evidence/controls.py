import json,time,re,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent
key=__import__('os').environ['TYPESAFE_API_KEY']
states={
'implies':'An integer is sampled uniformly from 1 through 10 inclusive. Event A: number is 1 or 2. Event B: number is at most 5.',
'uncertain':'A deployment has intermittent errors. A recent config change and increased load are both plausible causes, but no cause has been verified. Event A: the config change caused errors. Event B: increased load caused errors.'}
rows=[]
for repeat in range(3):
 for name,state in states.items():
  questions={k:{'type':'noul','instructions':v} for k,v in {'a':'Does event A occur?','not_a':'Does event A not occur?','b':'Does event B occur?','and':'Do both event A and event B occur?','or':'Does at least one of event A and event B occur (inclusive OR)?'}.items()}
  questions['joint']={'type':'choice','instructions':'Which mutually exclusive joint outcome occurs?','criteria':{'both':'A occurs and B occurs','a_only':'A occurs and B does not occur','b_only':'A does not occur and B occurs','neither':'A does not occur and B does not occur'}}
  payload={'model':'jev-latest','state':state,'questions':questions}
  req=urllib.request.Request('https://api.typesafe.ai/v1/systemone',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
  t=time.perf_counter()
  with urllib.request.urlopen(req,timeout=40) as r:out=json.load(r)
  rows.append({'name':name,'repeat':repeat,'request':payload,'response':out,'seconds':time.perf_counter()-t})
  (root/'control-results.json').write_text(json.dumps(rows,indent=2))
  print(name,repeat,json.dumps(out['answers']),flush=True)
