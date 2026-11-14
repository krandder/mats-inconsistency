#!/usr/bin/env python3
"""Search Vast offers for the manifest's GPU spec and rent the cheapest acceptable one.

Usage: rent.py <lane_dir> [--go] [--dry-search]
  --dry-search  print the matching offers and exit (no rental)
  --go          actually create the instance; without it the script only prints what it would rent

Writes <lane_dir>/instance.json {created_at, offer, response.new_contract, quoted_dph, budget_usd, max_hours}
(the same shape data/vast-wave1/instance.json has) and refuses to rent twice.
Request shapes follow vast-python's `search offers` (POST /bundles/) and `create instance` (PUT /asks/{id}/).
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from api import call  # noqa: E402


def search(manifest):
    gpu = manifest['gpu']
    query = {'verified': {'eq': True}, 'external': {'eq': False}, 'rentable': {'eq': True}, 'rented': {'eq': False},
             'num_gpus': {'eq': gpu['num_gpus']}, 'gpu_name': {'eq': gpu['gpu_name']},
             'gpu_ram': {'gte': gpu['min_gpu_ram_mb']}, 'disk_space': {'gte': gpu['min_disk_gb']},
             'cuda_max_good': {'gte': gpu['min_cuda']}, 'inet_down': {'gte': gpu['min_inet_down_mbps']},
             'reliability2': {'gte': gpu['min_reliability']}, 'dph_total': {'lte': gpu['max_dph_total']},
             'order': [['dph_total', 'asc']], 'type': 'on-demand', 'limit': 20,
             'allocated_storage': manifest['disk_gb']}
    offers = call('bundles/', 'POST', query).get('offers', [])
    keep = ('id', 'dph_total', 'gpu_name', 'num_gpus', 'gpu_ram', 'disk_space', 'cuda_max_good', 'driver_version',
            'inet_down', 'inet_up', 'reliability2', 'host_id', 'machine_id', 'geolocation', 'verification')
    return [{k: o.get(k) for k in keep} for o in offers]


def main(argv):
    lane = pathlib.Path(argv[1])
    manifest = json.loads((lane / 'manifest.json').read_text())
    go = '--go' in argv
    instance_file = lane / 'instance.json'
    if instance_file.exists() and not (lane / 'destroyed.json').exists():
        raise SystemExit(f'{instance_file} exists and no destroyed.json: refusing to rent twice')
    offers = search(manifest)
    (lane / 'offers.json').write_text(json.dumps({'at': time.time(), 'offers': offers}, indent=1))
    for o in offers[:10]:
        print(f"offer {o['id']}: ${o['dph_total']:.3f}/h {o['num_gpus']}x{o['gpu_name']} disk {o['disk_space']:.0f}GB "
              f"cuda {o['cuda_max_good']} driver {o['driver_version']} down {o['inet_down']:.0f} rel {o['reliability2']:.3f}")
    if not offers:
        raise SystemExit('no offer matches the manifest spec')
    if '--dry-search' in argv or not go:
        print('no rental (pass --go to rent the first offer)')
        return 0
    offer = offers[0]
    max_hours = manifest['max_hours']
    if offer['dph_total'] * max_hours > manifest['budget_usd']:
        raise SystemExit(f"cheapest offer ${offer['dph_total']:.2f}/h x {max_hours}h exceeds budget ${manifest['budget_usd']}")
    credit = call('users/current/').get('credit')
    if credit is not None and credit < offer['dph_total'] * max_hours:
        raise SystemExit(f'credit ${credit:.2f} is below the worst case ${offer["dph_total"] * max_hours:.2f}')
    payload = {'client_id': 'me', 'image': manifest['image'], 'env': {}, 'disk': manifest['disk_gb'],
               'label': manifest['name'], 'runtype': 'ssh', 'onstart': 'mkdir -p ' + manifest['remote_root'],
               'python_utf8': True, 'lang_utf8': True, 'use_jupyter_lab': False, 'image_login': None,
               'price': None, 'extra': None, 'force': False, 'cancel_unavail': False, 'template_hash_id': None,
               'user': None}
    response = call(f"asks/{offer['id']}/", 'PUT', payload)
    record = {'created_at': time.time(), 'offer_id': offer['id'], 'offer': offer, 'response': response,
              'quoted_dph': offer['dph_total'], 'gpu': f"{offer['num_gpus']} x {offer['gpu_name']}",
              'budget_usd': manifest['budget_usd'], 'max_hours': max_hours, 'credit_before': credit}
    instance_file.write_text(json.dumps(record, indent=1))
    instance_file.chmod(0o600)
    if not response.get('success'):
        raise SystemExit(f'rental failed: {response}')
    print(f"rented contract {response.get('new_contract')} at ${offer['dph_total']:.3f}/h (credit before ${credit})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
