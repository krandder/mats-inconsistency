#!/usr/bin/env python3
"""Generate a diagnostic split-cube variant; never edit production Bend source."""
import pathlib, shutil
root=pathlib.Path(__file__).resolve().parent.parent
out=root/'build/scaling-profile/src'
out.mkdir(parents=True,exist_ok=True)
for p in (root/'src').glob('*.bend'): shutil.copy2(p,out/p.name)
p=out/'soup.bend';s=p.read_text();a=s.index('def epoch.go(');b=s.index('\ndef epoch(',a)
s=s[:a]+'''def profile.finish(+e: U32, settled: SP, qs: List<&2, Blk>, +qv: T.Vec, +os: T.Vec, hs: T.Vec, fv: T.Vec, lv: T.Vec, +c: Cfg, sh: S.Sh, sub: S.GL, ops: Nat, +began: Nat, +settled_at: Nat) -> IO(EpOut):
  match settled:
    case SP{p, s1}:
      do IO<EpOut>:
        reproduced : SP = rep(Cfg.logk(c), 0, 1, p, c, e)
        +repped_at : Nat <- IO.now()
        result : EpOut = epoch.fin(e, reproduced, s1, qs, qv, brier(qv, os), U32.to_nat(U32.shln(1, Cfg.dq(c))), Mx.questions(Cfg.dq(c), qv, os, hs, fv, lv), c, sh, sub, ops)
        end : Nat <- IO.now()
        IO.print(String.join(["HOST_PROFILE", U32.show(e), Nat.show(Nat.sub(settled_at, began)), Nat.show(Nat.sub(repped_at, settled_at)), Nat.show(Nat.sub(end, repped_at))], ","))
        return result

def profile.runs(p: T.Pop, pr: T.Vec, hs: T.Vec, lv: T.Vec, c: Cfg) -> V.RPop:
  run_pop.at!(p, 0, 1, pr, hs, lv, c)

def profile.orders(r: V.RPop, c: Cfg) -> Agg:
  agg_pop!(r, c)

def epoch.go(+e: U32, p: T.Pop, +qs: List<&2, Blk>, +pr: T.Vec, +c: Cfg, sh: S.Sh, sub: S.GL, +ops: Nat) -> IO(EpOut):
  do IO<EpOut>:
    +start : Nat <- IO.now()
    +hs : T.Vec = T.Vec.of_list(Cfg.dq(c), blks.hist(qs))
    fv : T.Vec = T.Vec.of_list(Cfg.dq(c), blks.fam(qs))
    +lv : T.Vec = T.Vec.of_list(Cfg.dq(c), blks.live(qs))
    +q2 : List<&2, Blk> = blks.adv(qs)
    +os : T.Vec = T.Vec.of_list(Cfg.dq(c), blks.out(q2))
    +prepared : Nat <- IO.now()
    +boundary : Nat <- IO.now()
    +r : V.RPop = profile.runs(p, pr, hs, lv, c)
    +ran : Nat <- IO.now()
    +g : Agg = profile.orders(r, c)
    +aggregated : Nat <- IO.now()
    settled : SP = settle_pop(r, px.agg(False{}, Cfg.liq(c), g), os, c)
    qv : T.Vec = px.agg(True{}, Cfg.liq(c), g)
    settled_at : Nat <- IO.now()
    result : EpOut <- profile.finish(e, settled, q2, qv, os, hs, fv, lv, c, sh, sub, ops, aggregated, settled_at)
    finished : Nat <- IO.now()
    IO.print(String.join(["PROFILE", U32.show(e), Nat.show(Nat.sub(prepared, start)), Nat.show(Nat.sub(boundary, prepared)), Nat.show(Nat.sub(ran, boundary)), Nat.show(Nat.sub(aggregated, ran)), Nat.show(Nat.sub(finished, aggregated))], ","))
    return result
'''+s[b:];p.write_text(s)
print(p)
