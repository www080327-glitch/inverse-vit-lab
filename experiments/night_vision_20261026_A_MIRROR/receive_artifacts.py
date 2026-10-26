"""Ephemeral write-only receiver for declared research assets. No file reads/API execution."""
import http.server,pathlib,json,secrets,hmac,hashlib,threading,time,os,sys
ROOT=pathlib.Path(sys.argv[1]).resolve();PRIVATE=pathlib.Path(sys.argv[2]).resolve();PORT=int(sys.argv[3]);ROOT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
TOKEN=secrets.token_hex(32);(PRIVATE/'receiver_secret.json').write_text(json.dumps({'token':TOKEN,'port':PORT}));os.chmod(PRIVATE/'receiver_secret.json',0o600)
DEADLINE=time.time()+3600;ASSETS={};FINAL=[];LOCKS={};GLOBAL=threading.Lock()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def persist(name,d):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def readiness():
 if FINAL and all((ROOT/(n+'.verified.json')).exists() for n in FINAL):persist('LOCAL_ATOMIC_TRANSFER_READY.json',{'status':'COMPLETE_ALL_ASSETS_SHA256_VERIFIED','assets':FINAL,'created_unix':time.time()})
class Handler(http.server.BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args):pass
 def respond(self,status,offset=None,complete=False):
  self.send_response(status);self.send_header('Content-Length','0')
  if offset is not None:self.send_header('X-Offset',str(offset))
  self.send_header('X-Complete','1' if complete else '0');self.end_headers()
 def authorized(self):return time.time()<DEADLINE and hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+TOKEN)
 def do_GET(self):self.respond(404)
 def do_HEAD(self):
  if not self.authorized():return self.respond(404)
  if self.path=='/health':return self.respond(200)
  name=self.path.removeprefix('/asset/')
  if name not in ASSETS:return self.respond(404)
  if (ROOT/(name+'.verified.json')).exists():return self.respond(200,ASSETS[name]['bytes'],True)
  p=ROOT/(name+'.uploading');self.respond(200,p.stat().st_size if p.exists() else 0)
 def do_POST(self):
  if not self.authorized() or self.path!='/manifest':return self.respond(404)
  n=int(self.headers.get('Content-Length','0'))
  if n<=0 or n>2*1024*1024:return self.respond(413)
  try:
   d=json.loads(self.rfile.read(n));items=d['assets'];assert 0<len(items)<=500
   for a in items:
    name=a['name'];assert name==pathlib.Path(name).name and '..' not in name and all(c.isalnum() or c in '._-' for c in name);assert 0<a['bytes']<2*1024**3;assert len(a['sha256'])==64 and all(c in '0123456789abcdef' for c in a['sha256'])
   assert sum(a['bytes'] for a in items)<8*1024**3
   with GLOBAL:
    for a in items:
     if a['name'] in ASSETS:assert ASSETS[a['name']]['sha256']==a['sha256'] and ASSETS[a['name']]['bytes']==a['bytes']
     ASSETS[a['name']]=a;LOCKS.setdefault(a['name'],threading.Lock())
    if d.get('complete'):
     FINAL[:]=[a['name'] for a in items];persist('expected_assets_final.json',d)
   persist('registered_assets.json',list(ASSETS.values()));readiness();self.respond(200)
  except Exception:self.respond(400)
 def do_PUT(self):
  if not self.authorized():return self.respond(404)
  name=self.path.removeprefix('/asset/')
  if name not in ASSETS:return self.respond(404)
  meta=ASSETS[name];n=int(self.headers.get('Content-Length','0'));offset=int(self.headers.get('X-Offset','-1'))
  if n<=0 or n>16*1024*1024 or offset<0 or offset+n>meta['bytes']:return self.respond(413)
  with LOCKS[name]:
   p=ROOT/(name+'.uploading');have=p.stat().st_size if p.exists() else 0
   if (ROOT/(name+'.verified.json')).exists():return self.respond(200,meta['bytes'],True)
   if have!=offset:return self.respond(409,have)
   self.connection.settimeout(90)
   try:
    with p.open('ab') as f:
     left=n
     while left:
      b=self.rfile.read(min(1024*1024,left))
      if not b:raise EOFError()
      f.write(b);left-=len(b)
     f.flush()
    have=p.stat().st_size
    if have==meta['bytes']:
     digest=sha(p)
     if digest!=meta['sha256']:return self.respond(422,have)
     p.replace(ROOT/name);persist(name+'.verified.json',{'bytes':have,'sha256':digest,'verified':True});readiness();print(json.dumps({'completed_asset':name,'bytes':have}),flush=True)
    self.respond(200,have,have==meta['bytes'])
   except (OSError,EOFError):self.close_connection=True
server=http.server.ThreadingHTTPServer(('127.0.0.1',PORT),Handler);server.timeout=2
while time.time()<DEADLINE:server.handle_request()
server.server_close()
