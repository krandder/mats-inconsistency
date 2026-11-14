"""List or rent a 1 x H100 offer for the Qwen3-0.6B structured-pairs run (27 Sep 2026).

python rent.py list          -> cheapest reliable offers
python rent.py rent MACHINE  -> rent the current offer on that machine; record contract, credit and caps (never a key)
Budget cap = min($35, live credit - $5) at rent time (Kelvin, 27 Sep, pool-da26t); 6 h hard cap.
"""
import json, pathlib, sys, time
from api import call
P = pathlib.Path(__file__).parent
IMAGE = 'pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel'
GPUS = ["H100 SXM", "H100 NVL", "H100 PCIE"]
def offers():
    q = {"gpu_name": {"in": GPUS}, "num_gpus": {"eq": 1}, "rentable": {"eq": True}, "rented": {"eq": False},
         "reliability2": {"gte": 0.98}, "inet_down": {"gte": 200}, "disk_space": {"gte": 80}, "type": "on-demand", "limit": 200}
    return sorted(call('bundles/', 'POST', q)['offers'], key=lambda o: o['dph_total'])
if sys.argv[1] == 'list':
    for o in offers()[:12]:
        print(o['id'], 'machine', o.get('machine_id'), o['gpu_name'], o['num_gpus'], round(o['dph_total'], 3), round(o.get('reliability2', 0), 4),
              o.get('geolocation'), round(o.get('inet_down', 0)), o.get('cuda_max_good'))
    raise SystemExit
machine = int(sys.argv[2])
if (P / 'instance.json').exists() and not (P / 'destroyed.json').exists():
    raise SystemExit('instance.json exists and was not destroyed; refusing a second rent')
o = next(o for o in offers() if o.get('machine_id') == machine); offer = o['id']
credit = call('users/current/')['credit']
cap = min(35.0, credit - 5.0)
if cap < 10: raise SystemExit(f'credit {credit:.2f} too low')
body = {"client_id": "me", "image": IMAGE, "disk": 80, "label": "qwen06b-structured-20260927",
        "runtype": "ssh_direct", "onstart": "touch /root/onstart-ran", "env": {}}
r = call(f'asks/{offer}/', 'PUT', body)
rec = {"created_at": time.time(), "offer_id": offer, "machine_id": machine, "image": IMAGE, "gpu": o['gpu_name'], "num_gpus": 1,
       "quoted_dph": o['dph_total'], "geolocation": o.get('geolocation'), "reliability": o.get('reliability2'),
       "credit_at_rent": credit, "budget_usd": cap, "max_hours": 6, "credit_floor": 5.0,
       "new_contract": r.get('new_contract'), "success": r.get('success')}
(P / 'instance.json').write_text(json.dumps(rec, indent=1))
print(json.dumps(rec))
