"""List or rent an N x H100 offer for the Qwen3-0.6B fixed-prompt run.

python rent.py list N        -> cheapest reliable offers
python rent.py rent MACHINE N  -> rent the current offer on that machine (offer ids churn); record contract, credit and budget cap (never the instance key)
Budget cap = min($55, live credit - $5) at rent time (Kelvin, 2026-09-23); 8 h hard cap.
"""
import json, pathlib, sys, time
from api import call
P = pathlib.Path(__file__).parent
IMAGE = 'pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel'
def offers(n):
    q = {"gpu_name": {"in": ["H100 SXM"]}, "num_gpus": {"eq": n}, "rentable": {"eq": True}, "rented": {"eq": False},
         "reliability2": {"gte": 0.98}, "inet_down": {"gte": 200}, "disk_space": {"gte": 120}, "type": "on-demand", "limit": 200}
    return sorted(call('bundles/', 'POST', q)['offers'], key=lambda o: o['dph_total'])
if sys.argv[1] == 'list':
    for o in offers(int(sys.argv[2]))[:12]:
        print(o['id'], 'machine', o.get('machine_id'), o['gpu_name'], o['num_gpus'], round(o['dph_total'], 3), round(o.get('reliability2', 0), 4),
              o.get('geolocation'), round(o.get('inet_down', 0)), o.get('cuda_max_good'))
    raise SystemExit
machine, n = int(sys.argv[2]), int(sys.argv[3])
o = next(o for o in offers(n) if o.get('machine_id') == machine); offer = o['id']
credit = call('users/current/')['credit']
cap = min(55.0, credit - 5.0)
if cap < 10: raise SystemExit(f'credit {credit:.2f} too low')
body = {"client_id": "me", "image": IMAGE, "disk": 120, "label": "qwen06b-fixed-prompts-20260923",
        "runtype": "ssh_direct", "onstart": "touch /root/onstart-ran", "env": {}}
r = call(f'asks/{offer}/', 'PUT', body)
rec = {"created_at": time.time(), "offer_id": offer, "machine_id": machine, "image": IMAGE, "gpu": o['gpu_name'], "num_gpus": n,
       "quoted_dph": o['dph_total'], "geolocation": o.get('geolocation'), "reliability": o.get('reliability2'),
       "credit_at_rent": credit, "budget_usd": cap, "max_hours": 8, "credit_floor": 5.0,
       "new_contract": r.get('new_contract'), "success": r.get('success')}
(P / 'instance.json').write_text(json.dumps(rec, indent=1))
print(json.dumps(rec))
