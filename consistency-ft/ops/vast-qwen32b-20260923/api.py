import pathlib,json,urllib.request
KEY=pathlib.Path('/home/user/.config/vastai/api_key').read_text().strip()
def call(path,method='GET',data=None):
 r=urllib.request.Request('https://console.vast.ai/api/v0/'+path,data=None if data is None else json.dumps(data).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'},method=method)
 with urllib.request.urlopen(r,timeout=60) as f:return json.load(f)
