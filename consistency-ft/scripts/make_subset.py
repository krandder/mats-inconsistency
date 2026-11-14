import json,pathlib,sys
src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2]); ns=[int(x) for x in sys.argv[3].split(',')]
for split,n in zip(('train','validation','test'),ns):
    d=dst/split; d.mkdir(parents=True,exist_ok=True)
    rows=lambda f:[json.loads(l) for l in (src/split/f).read_text().splitlines() if l.strip()]
    pairs=rows('pairs.jsonl')[:n]; ids={p['pair_id'] for p in pairs}; props={p['p_id'] for p in pairs}|{p['q_id'] for p in pairs}
    q=[r for r in rows('queries.jsonl') if r['pair_id'] in ids]; qids={r['query_id'] for r in q}
    out={'pairs.jsonl':pairs,'queries.jsonl':q,'propositions.jsonl':[r for r in rows('propositions.jsonl') if r['id'] in props],
     'one-label-targets.jsonl':[r for r in rows('one-label-targets.jsonl') if r['query_id'] in qids],
     'full-label-targets.jsonl':[r for r in rows('full-label-targets.jsonl') if r['query_id'] in qids]}
    if (src/split/'prompts.jsonl').exists():
        out['prompts.jsonl']=[r for r in rows('prompts.jsonl') if r['pair_id'] in ids]
    for f,r in out.items(): (d/f).write_text(''.join(json.dumps(x)+'\n' for x in r))
