"""Bounded remote downloads into the independent night directory only."""
import pathlib,sys,os,time,json,hashlib,struct,urllib.request,subprocess,traceback
P=pathlib.Path(__file__).resolve().parent;os.chdir(P)
assert sys.platform=='linux' and os.environ.get('NIGHT_AUTHORIZED_FREE_DCU')=='1', 'Explicit owned-free-DCU authorization required'
END=time.time()+600;REV='2fc06364715b967f1860aea9cf38778875588b17'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(n,d):
 p=P/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def fetch(u,p,n,interval=None):
 headers={'Range':f'bytes={interval[0]}-{interval[1]}'} if interval else {}
 with urllib.request.urlopen(urllib.request.Request(u,headers=headers),timeout=30) as r:
  if interval:assert r.status==206 and r.headers.get('Content-Range','').startswith(f'bytes {interval[0]}-{interval[1]}/')
  count=0;t=p.with_suffix('.part')
  with t.open('wb') as f:
   while True:
    assert time.time()<END,'Preparation deadline'
    b=r.read(1048576)
    if not b:break
    count+=len(b);assert count<=n;f.write(b)
  assert count==n;t.replace(p)
try:
 dump('assets_status.json',{'status':'RUNNING','deadline_unix':END})
 base='https://hf-mirror.com/Qwen/Qwen3.5-0.8B/resolve/'+REV+'/model.safetensors-00001-of-00001.safetensors'
 if not (P/'vision.safetensors').exists():
  fetch(base+'?night_header=1',P/'header_length.bin',8,(0,7));n=struct.unpack('<Q',(P/'header_length.bin').read_bytes())[0];assert n<1000000
  fetch(base+'?night_header=2',P/'official_header.json',n,(8,7+n));d=json.loads((P/'official_header.json').read_text());v={k:x for k,x in d.items() if k.startswith('model.visual.')}
  lo=min(x['data_offsets'][0] for x in v.values());hi=max(x['data_offsets'][1] for x in v.values());assert hi-lo==201185792
  fetch(base+'?night_vision=1',P/'vision.raw',hi-lo,(8+n+lo,8+n+hi-1))
  for k,x in v.items():x['data_offsets']=[i-lo for i in x['data_offsets']]
  h=json.dumps({k.removeprefix('model.visual.'):x for k,x in v.items()},separators=(',',':')).encode();h+=b' '*((-len(h))%8)
  with (P/'vision.safetensors').open('wb') as out,(P/'vision.raw').open('rb') as src:
   out.write(struct.pack('<Q',len(h)));out.write(h)
   for b in iter(lambda:src.read(1048576),b''):out.write(b)
 assert sha(P/'vision.safetensors')=='7c8c31f06432f8e9d402869e840b037e8d121503c51c236655f26fbaf9a7cb26'
 if not (P/'cifar-10-python.tar.gz').exists():
  fetch('https://hf-mirror.com/datasets/VerisimilitudeX/cifar10/resolve/main/cifar-10-python.tar.gz?night=1',P/'cifar-10-python.tar.gz',170498071,(0,170498070))
 assert sha(P/'cifar-10-python.tar.gz')=='6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce'
 (P/'model').mkdir(exist_ok=True)
 for src,dst in [('official_model_config.json','config.json'),('official_preprocessor_config.json','preprocessor_config.json')]:
  (P/'model'/dst).write_bytes((P/src).read_bytes())
 subprocess.run([sys.executable,'-m','venv','--system-site-packages','env'],check=True,timeout=30)
 with (P/'dependency_install.log').open('w') as log:
  subprocess.run([str(P/'env/bin/python'),'-m','pip','install','--only-binary=:all:','--disable-pip-version-check','--no-cache-dir','--index-url','https://pypi.tuna.tsinghua.edu.cn/simple','--timeout','15','--retries','1','transformers==5.2.0'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
 dump('assets_status.json',{'status':'COMPLETE','revision':REV,'vision_sha256':sha(P/'vision.safetensors'),'cifar_sha256':sha(P/'cifar-10-python.tar.gz'),'finished_unix':time.time()})
except BaseException as e:
 dump('assets_status.json',{'status':'BLOCKED','error':repr(e),'traceback':traceback.format_exc()});raise
