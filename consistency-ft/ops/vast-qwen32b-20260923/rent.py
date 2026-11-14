"""Rent one 1xH100 offer for the Qwen3-32B verbalized baseline and record the contract."""
import json, pathlib, sys, time
from api import call
P = pathlib.Path(__file__).parent
offer = int(sys.argv[1]); image = sys.argv[2]
q = {"gpu_name": {"in": ["H100 SXM", "H100 NVL"]}, "num_gpus": {"eq": 1}, "rentable": {"eq": True}, "rented": {"eq": False},
     "type": "on-demand", "limit": 500}
o = next(o for o in call('bundles/', 'POST', q)['offers'] if o['id'] == offer)
body = {"client_id": "me", "image": image, "disk": 150, "label": "qwen32b-verbalized-baseline-20260923",
        "runtype": "ssh_direc ssh_proxy", "onstart": "touch /root/onstart-ran", "env": {}}
r = call(f'asks/{offer}/', 'PUT', body)
rec = {"created_at": time.time(), "offer_id": offer, "image": image, "gpu": o['gpu_name'], "quoted_dph": o['dph_total'],
       "geolocation": o.get('geolocation'), "reliability": o.get('reliability2'), "budget_usd": 30, "max_hours": 8,
       "new_contract": r.get('new_contract'), "success": r.get('success')}
(P / 'instance.json').write_text(json.dumps(rec, indent=1))
print(json.dumps(rec))
