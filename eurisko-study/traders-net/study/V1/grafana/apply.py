#!/usr/bin/env python3
"""Install or update the traders-net dashboard in the fleet Grafana (GKE agent-fleet, ns observability).

Runs where kubectl reaches the cluster (deploy-host); use ./apply.sh, which feeds it its inputs.
Idempotent: each step checks the live object first and changes only what is missing.
Touches only: Service traders-runs, ConfigMap grafana-traders-net-dashboard (both new), one added
container in Deployment [removed], one datasource and one provider appended to ConfigMap
grafana-provisioning, one volume + mount in Deployment grafana. Nothing else.
"""
import json, subprocess, sys

NS = 'observability'
FILES = json.loads(sys.stdin.readline())          # apply.sh prepends the inputs as one JSON line
dashboard_only = '--dashboard-only' in sys.argv


def kc(*args, input=None):
    return subprocess.run(['kubectl', '-n', NS, *args], input=input, text=True, check=True, capture_output=True).stdout


def get(kind):
    return json.loads(kc('get', kind, '-o', 'json'))


def apply(obj):
    print(kc('apply', '-f', '-', input=json.dumps(obj)).strip())


apply({'apiVersion': 'v1', 'kind': 'ConfigMap',
       'metadata': {'name': 'grafana-traders-net-dashboard', 'namespace': NS, 'labels': {'app.kubernetes.io/part-of': 'traders-net'}},
       'data': {'traders-net.json': FILES['dashboard']}})
if dashboard_only:
    sys.exit(0)

apply({'apiVersion': 'v1', 'kind': 'Service',
       'metadata': {'name': 'traders-runs', 'namespace': NS, 'labels': {'app.kubernetes.io/part-of': 'traders-net'}},
       'spec': {'selector': {'app': '[removed]'},
                'ports': [{'name': 'traders-runs', 'port': 8767, 'targetPort': 8767, 'protocol': 'TCP'}]}})

gw = get('deploy/[removed]')
names = [c['name'] for c in gw['spec']['template']['spec']['containers']]
if 'socat-8767' not in names:
    patch = [{'op': 'test', 'path': '/metadata/resourceVersion', 'value': gw['metadata']['resourceVersion']},
             {'op': 'add', 'path': '/spec/template/spec/containers/-', 'value': {
                 'name': 'socat-8767', 'image': 'alpine/socat:1.8.0.0',
                 'args': ['TCP-LISTEN:8767,fork,reuseaddr', 'PROXY:localhost:%s:8767,proxyport=1055' % FILES['host']],
                 'resources': {'limits': {'memory': '64Mi'}, 'requests': {'cpu': '5m', 'memory': '16Mi'}}}}]
    print(kc('patch', 'deploy/[removed]', '--type=json', '-p', json.dumps(patch)).strip())

cm = get('cm/grafana-provisioning')
ds, prov = cm['data']['datasources.yaml'], cm['data']['dashboards.yaml']
patch = [{'op': 'test', 'path': '/metadata/resourceVersion', 'value': cm['metadata']['resourceVersion']}]
if 'uid: traders-net-runs' not in ds:
    patch.append({'op': 'replace', 'path': '/data/datasources.yaml', 'value': ds.rstrip('\n') + '\n' + FILES['datasource']})
if 'name: traders-net' not in prov:
    patch.append({'op': 'replace', 'path': '/data/dashboards.yaml', 'value': prov.rstrip('\n') + '\n' + FILES['provider']})
if len(patch) > 1:
    print(kc('patch', 'cm/grafana-provisioning', '--type=json', '-p', json.dumps(patch)).strip())

gf = get('deploy/grafana')
spec = gf['spec']['template']['spec']
patch = [{'op': 'test', 'path': '/metadata/resourceVersion', 'value': gf['metadata']['resourceVersion']}]
if 'traders-net-dashboard' not in [v['name'] for v in spec['volumes']]:
    patch.append({'op': 'add', 'path': '/spec/template/spec/volumes/-', 'value': {
        'name': 'traders-net-dashboard', 'configMap': {'name': 'grafana-traders-net-dashboard'}}})
    patch.append({'op': 'add', 'path': '/spec/template/spec/containers/0/volumeMounts/-', 'value': {
        'name': 'traders-net-dashboard', 'mountPath': '/var/lib/grafana/dashboards-traders-net', 'readOnly': True}})
if len(patch) > 1:        # the pod template changes, so Grafana restarts (Recreate) and reads the new datasource
    print(kc('patch', 'deploy/grafana', '--type=json', '-p', json.dumps(patch)).strip())
else:
    print('grafana deployment already has the volume; restart it yourself if the datasource changed')
