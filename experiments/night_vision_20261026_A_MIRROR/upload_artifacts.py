"""Remote CPU-only uploader; write-only secret config stays outside published workspace."""
import pathlib,json,sys,urllib.request,urllib.error,time,concurrent.futures,hashlib
P=pathlib.Path(__file__).resolve().parent;cfg=json.loads(pathlib.Path(sys.argv[1]).read_text());BASE=cfg['url'].rstrip('/');TOKEN=cfg['token'];DEADLINE=cfg['deadline_unix']
def request(method,path,data=None,headers=None):
 h={'Authorization':'Bearer '+TOKEN};h.update(headers or {});r=urllib.request.urlopen(urllib.request.Request(BASE+path,data=data,headers=h,method=method),timeout=100);meta={k.lower():v for k,v in r.headers.items()};r.read();r.close();return meta
def register(items,complete=False,transport=None):
 body=json.dumps({'assets':items,'complete':complete,'transport':transport}).encode();return request('POST','/manifest',body,{'Content-Type':'application/json'})
def upload(a):
 file=P/'export'/a['name'];assert file.stat().st_size==a['bytes']
 for attempt in range(3):
  try:
   assert time.time()<DEADLINE,'Provider transfer deadline';meta=request('HEAD','/asset/'+a['name'])
   if meta.get('x-complete')=='1':return {'name':a['name'],'status':'ALREADY_VERIFIED'}
   offset=int(meta['x-offset'])
   with file.open('rb') as f:
    f.seek(offset)
    while offset<a['bytes']:
     assert time.time()<DEADLINE,'Provider transfer deadline';b=f.read(min(16*1024*1024,a['bytes']-offset));assert b
     meta=request('PUT','/asset/'+a['name'],b,{'Content-Type':'application/octet-stream','X-Offset':str(offset)});offset=int(meta['x-offset'])
   print(json.dumps({'uploaded':a['name'],'bytes':offset}),flush=True);return {'name':a['name'],'status':'COMPLETE_REMOTE_AND_LOCAL_HASH_VERIFIED'}
  except Exception as e:
   if attempt==2:return {'name':a['name'],'status':'FAILED','error':repr(e)}
   time.sleep(3)
def run(items,complete=False,transport=None):
 register(items,complete,transport)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:result=list(pool.map(upload,items))
 return result
results=[]
pref=P/'PREFETCH_AVAILABLE.json'
if pref.exists() and not (P/'export/TRANSPORT_READY.json').exists():
 seen=set();items=[]
 for a in json.loads(pref.read_text()):
  if '/best' in a['role'] or a['role'].startswith('mirror_tiny'):
   if a['name'] not in seen:items.append(a);seen.add(a['name'])
 if items:results+=run(items)
while time.time()<DEADLINE:
 ready=P/'export/TRANSPORT_READY.json'
 if ready.exists():
  transport=json.loads(ready.read_text());results+=run(transport['assets'],True,transport);break
 time.sleep(10)
else:results.append({'status':'FINAL_MANIFEST_NOT_AVAILABLE_BEFORE_PROVIDER_DEADLINE'})
(P/'upload_receipt.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'uploader_complete':True,'results':results}),flush=True)
