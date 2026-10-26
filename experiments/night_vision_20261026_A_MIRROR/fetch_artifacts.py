"""Download/verify/materialize an immutable snapshot; never imports ML libraries."""
import argparse,pathlib,json,hashlib,urllib.request,zipfile,shutil,os,time,stat
HERE=pathlib.Path(__file__).resolve().parent
REPO='www080327-glitch/inverse-vit-lab';TAG='night-vision-a-mirror-20261026'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def safe(root,name):
 q=pathlib.PurePosixPath(name)
 if q.is_absolute() or '..' in q.parts:raise ValueError('Unsafe artifact path')
 p=root.joinpath(*q.parts)
 if root.resolve() not in p.resolve().parents:raise ValueError('Path outside snapshot')
 return p
def fetch(a,cache):
 p=cache/a['name']
 if p.exists() and p.stat().st_size==a['bytes'] and sha(p)==a['sha256']:return p
 partial=p.with_suffix(p.suffix+'.download');url=f'https://github.com/{REPO}/releases/download/{TAG}/{a["name"]}'
 for attempt in range(3):
  try:
   offset=partial.stat().st_size if partial.exists() else 0
   req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
   with urllib.request.urlopen(req,timeout=60) as r:
    if offset and (r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-')):raise RuntimeError('Server did not provide a safe resumed range')
    with partial.open('ab' if offset else 'wb') as f:
     for b in iter(lambda:r.read(8*1024*1024),b''):
      f.write(b)
      if f.tell()>a['bytes']:raise ValueError('Artifact byte limit exceeded')
   if partial.stat().st_size!=a['bytes'] or sha(partial)!=a['sha256']:raise ValueError('Artifact size/SHA256 mismatch')
   partial.replace(p);return p
  except Exception:
   if attempt==2:raise
   time.sleep(2)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--local-assets',type=pathlib.Path);parser.add_argument('--cache',type=pathlib.Path,default=pathlib.Path('.artifact-downloads'));parser.add_argument('--destination',type=pathlib.Path,default=pathlib.Path('downloaded_snapshot'));parser.add_argument('--verify-only',action='store_true');a=parser.parse_args()
 manifest=json.loads((HERE/'ARTIFACTS.json').read_text());items=manifest['assets']+manifest.get('supplements',[]);cache=(a.local_assets or a.cache).resolve();cache.mkdir(exist_ok=True,parents=True)
 files={}
 for item in items:
  p=cache/item['name'] if a.local_assets else fetch(item,cache)
  assert p.exists() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['name'];files[item['name']]=p
 if a.verify_only:print(json.dumps({'verified_assets':len(items),'bytes':sum(x['bytes'] for x in items)}));return
 dest=a.destination.resolve();stage=dest.with_name('.'+dest.name+'.incoming');assert not dest.exists() and not stage.exists(),'Use a new destination; existing snapshots are never replaced';stage.mkdir(parents=True)
 for item in items:
  file=files[item['name']]
  if item.get('encoding')=='zip':
   root=stage/'posthoc_pre_post_16' if item in manifest.get('supplements',[]) else stage;root.mkdir(exist_ok=True)
   with zipfile.ZipFile(file) as z:
    assert z.testzip() is None
    for info in z.infolist():
     if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unexpected ZIP symlink')
     out=safe(root,info.filename);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(z.read(info))
  else:
   for name in item['paths']:
    out=safe(stage,name);out.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(file,out)
    except OSError:shutil.copyfile(file,out)
 export=json.loads((HERE/'DECODER_EXPORT_FOR_ALIGNMENT.json').read_text())
 def relative(v):
  if isinstance(v,dict):return {k:relative(x) for k,x in v.items()}
  if isinstance(v,list):return [relative(x) for x in v]
  return v.removeprefix('snapshot/') if isinstance(v,str) else v
 (stage/'DECODER_EXPORT_FOR_ALIGNMENT.json').write_text(json.dumps(relative(export),indent=2)+'\n')
 shutil.copyfile(HERE/'STATISTICS_TRAIN_PARTITION.json',stage/'STATISTICS_TRAIN_PARTITION.json')
 (stage/'READY_LOCAL.json').write_text(json.dumps({'status':'COMPLETE_ALL_ASSETS_SHA256_VERIFIED','assets':len(items),'bytes':sum(x['bytes'] for x in items)},indent=2)+'\n');stage.rename(dest);print(str(dest))
if __name__=='__main__':main()
