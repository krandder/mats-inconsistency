#!/usr/bin/env python3
"""Check a snapshot/CSV pair; --repair removes only rows beyond its epoch."""
import csv,json,pathlib,struct,sys
args=sys.argv[1:];repair='--repair' in args
args=[a for a in args if a!='--repair'];p=pathlib.Path(args[0])
blob=(p/'soup.dump').read_bytes();c=json.loads((p/'config.json').read_text())
magic,version,logk,logq,logcode,seed,epoch=struct.unpack_from('<7I',blob)
assert (magic,version,logk,logq,logcode,seed)==(1347768147,1,c['logk'],c['logq'],6,c['seed']),'snapshot header/config mismatch'
assert c['free_income']==0,'this ledger verifier requires zero free income'
stride=(1<<logcode)+12;expected=28+c['K']*stride+c['F']*52+120
assert len(blob)==expected,(len(blob),expected,'snapshot size')
if len(args)>1:assert epoch==int(args[1]),(epoch,args[1])
with (p/'metrics.csv').open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
assert len(rows)>=epoch,(len(rows),epoch)
if not repair:assert len(rows)==epoch,(len(rows),epoch)
rows=rows[:epoch];w=c['K']*c['w0']
for i,r in enumerate(rows):
    assert int(r['epoch'])==i,(i,r['epoch'])
    assert int(r['wealth'])==w+int(r['trader_pnl'])-int(r['used'])-int(r['burned']),(i,'ledger')
    w=int(r['wealth'])
state_wealth=sum(struct.unpack_from('<I',blob,28+i*stride+(1<<logcode))[0] for i in range(c['K']))
assert state_wealth==w,(state_wealth,w,'snapshot/CSV wealth mismatch')
assert struct.unpack_from('<Q',blob,len(blob)-8)[0]==int(rows[-1]['cum_ops']),'snapshot/CSV instruction mismatch'
if repair:
    with (p/'metrics.csv').open('w') as f:writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    with (p/'timing.csv').open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;timings=list(reader)
    timings=[r for r in timings if int(r['first_epoch'])+int(r['epochs'])<=epoch]
    with (p/'timing.csv').open('w') as f:writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(timings)
print(f'checkpoint PASS: {epoch} epochs, exact ledger, snapshot wealth/instructions')
