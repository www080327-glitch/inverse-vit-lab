"""Free sampling from prompt + BOS only. No training/data/retrieval imports."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from .model import PixelQwen,generate,generate_cached_one,CLASSES,PROMPTS
from .probability import unpatchify


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def pixels(x):
    if not np.isfinite(x).all() or (np.abs(x)>1).any():raise ValueError('No display clipping may hide invalid feedback')
    return np.rint((x.transpose(1,2,0)+1)*127.5).astype(np.uint8)


def sample_model(m,out,classes,seeds,checkpoint_label='',batch_size=8,metadata=None,cache_single=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'process').mkdir()
    start=time.time();allp=[];allz=[];allk=[]
    (out/'partial').mkdir()
    if cache_single:batch_size=1
    for i in range(0,len(seeds),batch_size):
        if cache_single:p,z,k=generate_cached_one(m,classes[i],seeds[i])
        else:p,z,k=generate(m,classes[i:i+batch_size],seeds[i:i+batch_size])
        allp.append(p);allz.append(z);allk.append(k)
        np.savez(out/'partial'/f'{i:03d}.npz',patches=p.numpy(),z=z.numpy(),components=k.numpy())
        print(json.dumps(dict(kind='sample_progress',completed=min(i+batch_size,len(seeds)),total=len(seeds),seconds=time.time()-start)),flush=True)
    p=torch.cat(allp);z=torch.cat(allz);k=torch.cat(allk);raw=unpatchify(p).numpy()
    np.save(out/'patches.npy',p.numpy());np.save(out/'z.npy',z.numpy());np.save(out/'components.npy',k.numpy());np.save(out/'raw.npy',raw)
    # prefix[n,:j] is precisely what was fed back before predicting j; future canvas is display only.
    trajectory=np.zeros((64,len(seeds),3,32,32),dtype=np.float32)
    for j in range(64):
        canvas=torch.zeros_like(p);canvas[:,:j+1]=p[:,:j+1];trajectory[j]=unpatchify(canvas).numpy()
    np.save(out/'trajectory.npy',trajectory)
    grid=Image.new('RGB',(8*144,10*168),(245,245,245));gd=ImageDraw.Draw(grid)
    native=Image.new('RGB',(8*32,10*32),(127,127,127));rows=[]
    for i,(c,s) in enumerate(zip(classes,seeds)):
        name=f'{i:03d}_{CLASSES[c]}_seed{s}.png';im=Image.fromarray(pixels(raw[i]));im.save(out/name)
        frames=[Image.fromarray(pixels(trajectory[j,i])).resize((128,128),Image.Resampling.NEAREST) for j in range(64)]
        frames[0].save(out/'process'/f'{i:03d}.gif',save_all=True,append_images=frames[1:],duration=80,loop=0)
        if len(seeds)==80:
            x,y=(i%8)*144,(i//8)*168;grid.paste(im.resize((128,128),Image.Resampling.NEAREST),(x,y))
            gd.text((x,y+130),f'{CLASSES[c]} {s}',fill=(0,0,0));native.paste(im,((i%8)*32,(i//8)*32))
        rows.append(f'<figure><img width="32" height="32" src="{name}"><img width="128" height="128" src="{name}"><figcaption>{CLASSES[c]} / {s}</figcaption><a href="process/{i:03d}.gif">64 Patch 过程</a></figure>')
    if len(seeds)==80:grid.save(out/'atlas.png');native.save(out/'atlas_native.png')
    stats=dict(checkpoint=checkpoint_label,training=metadata or {'scope':'milestone preview'},classes=classes,seeds=seeds,prompts=[PROMPTS[c] for c in classes],
               t_mix=1.,t_noise=1.,cache=cache_single,cache_scope='fresh B1 native state per sample' if cache_single else 'full prefix recomputation',feedback='exact tanh(z) float32, no post-hoc clipping',
               start='prompt + own image_BOS; no real patch prefix',seconds=time.time()-start,
               raw_min=float(raw.min()),raw_max=float(raw.max()),oob_fraction=float((np.abs(raw)>1).mean()),
               saturation_abs_gt_099=float((np.abs(raw)>.99).mean()),exact_boundary_fraction=float((np.abs(raw)==1).mean()),
               z_min=float(z.min()),z_max=float(z.max()),count=len(seeds))
    (out/'sampling.json').write_text(json.dumps(stats,indent=2))
    scope='正式训练' if metadata and metadata.get('formal_training') else '本地短测或中期预览，不代表正式训练结果'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P2 全量自由采样</title><style>body{font:16px system-ui;max-width:1400px;margin:auto;padding:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}figure{margin:8px}img{image-rendering:pixelated;margin:4px}</style><h1>P2 全量自由采样</h1><h2>'+scope+'</h2><p>图像可辨认性须查看全部样本；范围合法不代表画质通过。无挑图，无真实前缀。</p><pre>'+html.escape(json.dumps(stats,ensure_ascii=False,indent=2))+'</pre><main>'+''.join(rows)+'</main>')
    print(json.dumps(stats),flush=True);return stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],required=True);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--cache-evidence',help='Admission JSON for this exact checkpoint/device; native B1 inference only')
    a=p.parse_args();torch.set_num_threads(6)
    if not os.environ.get('P2_GUARDED'):raise RuntimeError('External cumulative budget guard required')
    run=Path(a.checkpoint).parent
    m=PixelQwen(run/'qwen_config',pretrained=False,tokenizer_source=run/'tokenizer')
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True);m.load_state_dict(payload['model'],strict=True);m.to(a.device).eval()
    digest=file_sha(a.checkpoint)
    if a.cache_evidence:
        admitted=json.loads(Path(a.cache_evidence).read_text())
        if admitted.get('status')!='PASS' or admitted.get('device')!=a.device or admitted.get('checkpoint_sha256')!=digest:
            raise ValueError('Cache admission does not match checkpoint/device')
    config=json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else {}
    sample_model(m,a.out,[c for c in range(10) for _ in range(8)],list(range(610230,610238))*10,
                 checkpoint_label=digest,batch_size=a.batch_size,cache_single=bool(a.cache_evidence),
                 metadata=dict(run=run.name,optimizer_steps=payload['step'],formal_training=config.get('status')=='LOCKED_FROM_ACTUAL_PROBE'))


if __name__=='__main__':main()
