"""Resume slow official download with eight bounded HTTP range requests (no mirrors)."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    root=Path('data/p1/cifar10'); root.mkdir(parents=True,exist_ok=True)
    archive=root/'cifar-10-python.tar.gz'
    if archive.exists(): raise FileExistsError(archive)
    partial=root/'cifar-10-python.tar.download'
    offset=partial.stat().st_size if partial.exists() else 0
    url='https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
    size=170498071; chunk=(size-offset+7)//8; began=time.time()
    def fetch(i):
        start=offset+i*chunk; end=min(size-1,start+chunk-1)
        path=root/f'range_{i}.part'
        req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req,timeout=60) as response, path.open('wb') as output:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                raise RuntimeError('Server did not honor exact requested range')
            while True:
                block=response.read(256*1024)
                if not block: break
                output.write(block)
        if path.stat().st_size!=end-start+1: raise RuntimeError('Incomplete range')
        print(json.dumps(dict(chunk=i,bytes=path.stat().st_size,seconds=time.time()-began)),flush=True)
        return path
    with ThreadPoolExecutor(max_workers=8) as pool: parts=list(pool.map(fetch,range(8)))
    joined=root/'cifar.joined'
    with joined.open('wb') as output:
        for path in ([partial] if offset else [])+parts:
            with path.open('rb') as f:
                while block:=f.read(1024*1024): output.write(block)
    md5=hashlib.md5(joined.read_bytes()).hexdigest()
    if md5!='c58f30108f718f92721af3b95e74349a': raise ValueError('Official MD5 failed')
    joined.replace(archive)
    record=dict(url=url,existing_bytes=offset,total_bytes=size,md5=md5,parallel_requests=8,seconds=time.time()-began)
    Path('evidence/p1/download.json').write_text(json.dumps(record,indent=2))
    for path in ([partial] if offset else [])+parts: path.unlink()
    from .data import prepare
    prepare(root)


if __name__=='__main__': main()
