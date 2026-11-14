"""Minimal Vast.ai REST client (same shape as data/vast-wave1/api.py, which rented and destroyed a 4xH100 on 20 Sep).

Every lifecycle script imports ``call`` from here; nothing in this directory contacts the API at import time.
"""
import json
import pathlib
import urllib.request

API_KEY_FILE = pathlib.Path('/home/user/.config/vastai/api_key')
BASE = 'https://console.vast.ai/api/v0/'


def call(path, method='GET', data=None, timeout=60):
    key = API_KEY_FILE.read_text().strip()
    request = urllib.request.Request(BASE + path, data=None if data is None else json.dumps(data).encode(),
                                     headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
                                     method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)
