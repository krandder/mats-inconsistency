import json,time,re,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent
key=__import__('os').environ['TYPESAFE_API_KEY']
states={
'implies':'An integer is sampled uniformly from 1 through 10 inclusive. Event A: number is 1 or 2. Event B: number is at most 5.',
'uncertain':'A deployment has intermittent errors. A recent config change and increased load are both plausible causes, but no cause has been verified. Event A: the config change caused errors. Event B: increased load caused errors.'}
rows=[]
for repeat in range(3):
 for name,state in {'expanded':'An integer is sampled uniformly at random from 1 through 10 inclusive.'}.items():
  questions={k:{'type':'noul','instructions':v} for k,v in {'a':'Is the sampled integer either 1 or 2?','not_a':'Is the sampled integer neither 1 nor 2?','b':'Is the sampled integer at most 5?','and':'Is the sampled integer both (either 1 or 2) and (at most 5)?','or':'Is the sampled integer either (1 or 2) or (at most 5), including cases where both conditions hold?'}.items()}
  questions['joint']={'type':'choice','instructions':'Which mutually exclusive category contains the sampled integer?','criteria':{'both':'Integer is 1 or 2','a_only':'Integer is 1 or 2 and is greater than 5 (impossible)','b_only':'Integer is 3, 4, or 5','neither':'Integer is 6, 7, 8, 9, or 10'}}
  payload={'model':'jev-latest','state':state,'questions':questions}
  req=urllib.request.Request('https://api.typesafe.ai/v1/systemone',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
  t=time.perf_counter()
  with urllib.request.urlopen(req,timeout=40) as r:out=json.load(r)
  rows.append({'name':name,'repeat':repeat,'request':payload,'response':out,'seconds':time.perf_counter()-t})
  (root/'expanded-control-results.json').write_text(json.dumps(rows,indent=2))
  print(name,repeat,json.dumps(out['answers']),flush=True)
