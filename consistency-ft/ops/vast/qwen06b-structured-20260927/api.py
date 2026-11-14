"""Vast.ai API; the key comes from the fleet vault (secret get [removed]) and is never printed or written."""
import json, subprocess, urllib.request
KEY = subprocess.run(['/home/agent/.local/bin/secret', 'get', 'vast-api-key-01'], capture_output=True, text=True, check=True).stdout.strip()
def call(path, method='GET', data=None):
    r = urllib.request.Request('https://console.vast.ai/api/v0/' + path, data=None if data is None else json.dumps(data).encode(),
                               headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'}, method=method)
    with urllib.request.urlopen(r, timeout=60) as f:
        return json.load(f)
