#!/usr/bin/env python3
"""Isolated experiment: skip genome hashes for ancestry and inactive cells."""
import pathlib,shutil
root=pathlib.Path(__file__).resolve().parents[1];out=root/'build/utilisation/src'
out.mkdir(parents=True,exist_ok=True)
for p in (root/'src').glob('*.bend'):shutil.copy2(p,out/p.name)
p=out/'metrics.bend';s=p.read_text();old='      keep.if(U32.is_ge(w, ONE()), T.pick(k, a, code.hash(code, 0)), acc)'
new='      pop.id.live(U32.is_ge(w, ONE()), k, code, a, acc)'
helper="""def pop.id.kind(k: Bool, code: T.Vec, +a: U32, acc: List<&2, U32>) -> List<&2, U32>:
  match k:
    case True{}:
      a <> acc
    case False{}:
      code.hash(code, 0) <> acc

def pop.id.live(live: Bool, k: Bool, code: T.Vec, +a: U32, acc: List<&2, U32>) -> List<&2, U32>:
  match live:
    case False{}:
      acc
    case True{}:
      pop.id.kind(k, code, a, acc)

"""
s=s.replace('def pop.ids(',helper+'def pop.ids(',1)
assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
print(out/'soup.bend')
