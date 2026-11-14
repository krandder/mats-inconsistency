import json,statistics as st,collections
from pathlib import Path
p=Path(__file__).resolve().parent
rows=json.loads((p/'probe-results.json').read_text());groups=collections.defaultdict(list)
for r in rows:groups[r['name']].append(r)
s={}
for name,rs in groups.items():
 d={'n':len(rs),'median_seconds':st.median(r['seconds'] for r in rs),'input_tokens':[r['response'].get('usage',{}).get('input_tokens') for r in rs]}
 if name.startswith('logic'):
  d['observations']=[]
  for r in rs:
   v={k:a['noul'] for k,a in r['response']['answers'].items()}
   v['complement_error']=v['a']+v['not_a']-1
   v['inclusion_exclusion_error']=v['or']-(v['a']+v['b']-v['and'])
   v['conjunction_upper_violation']=max(0,v['and']-min(v['a'],v['b']))
   v['conjunction_lower_violation']=max(0,max(0,v['a']+v['b']-1)-v['and'])
   d['observations'].append(v)
 else:
  probs=collections.defaultdict(list);scores=[];conf=[]
  for r in rs:
   a=next(iter(r['response']['answers'].values()))
   # JSON preserves request order, but response keys may not; choose target explicitly.
   a=r['response']['answers'].get('target',a)
   if a['type']=='score':
    mass=collections.defaultdict(float)
    for k,v in a['probabilities'].items():mass[str(a['legend'][k])]+=v
    for k,v in mass.items():probs[k].append(v)
    scores.append(a['score'])
   else:
    for k,v in a['probabilities'].items():probs[k].append(v)
   conf.append(a['confidence'])
  d['probabilities']={k:{'mean':st.mean(v),'min':min(v),'max':max(v)} for k,v in probs.items()}
  d['confidence']=conf
  if scores:d['scores']=scores
 s[name]=d
summary={'calls':len(rows),'models':sorted(set(r['response'].get('model','ERROR') for r in rows)),'input_tokens':sum(r['response'].get('usage',{}).get('input_tokens',0) for r in rows),'groups':s}
(p/'probe-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
