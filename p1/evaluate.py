"""Post-training NLL and offline copy diagnostics. Never imported by sample.py."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from .budget import Budget
from .data import load_split, CLASSES
from .sample import load_model, png_array
from .train import validate


def nn_scan(raw, train_u8, class_ids, train_labels):
    x=raw.reshape(len(raw),-1).astype(np.float32)
    results={}
    for name, query in [('raw',x),('display_clipped',x.clip(-1,1))]:
        best=np.full(len(x),np.inf); ix=np.full(len(x),-1)
        same=np.full(len(x),np.inf); six=np.full(len(x),-1)
        qnorm=(query*query).sum(1)
        for start in range(0,len(train_u8),512):
            y=2*((train_u8[start:start+512].astype(np.float32)+.5)/256)-1
            y=y.reshape(len(y),-1)
            d=np.maximum(0,qnorm[:,None]+(y*y).sum(1)[None]-2*query@y.T)/x.shape[1]
            local=d.argmin(1); value=d[np.arange(len(x)),local]
            update=value<best; best[update]=value[update]; ix[update]=start+local[update]
            allowed=np.asarray(class_ids)[:,None]==train_labels[start:start+len(y)][None]
            d[~allowed]=np.inf; local=d.argmin(1); value=d[np.arange(len(x)),local]
            update=value<same; same[update]=value[update]; six[update]=start+local[update]
        results[name]=dict(nearest_train_local_index=ix.tolist(),squared_distance_per_channel=best.tolist(),
                           nearest_same_class_index=six.tolist(),same_class_distance=same.tolist())
    return results


def contact_sheet(raw, stats, out):
    # Pure display of already sampled arrays. Only nearest-neighbor rescaling.
    sheet=Image.new('RGB',(8*208+100,10*188+50),'#eeeeee'); draw=ImageDraw.Draw(sheet)
    draw.text((12,10),'P1 full free samples | native 32 + nearest 128 | Tmix=Tnoise=1',fill='black')
    draw.text((12,28),f'Raw out-of-range {stats["oob_fraction"]:.2%} | range [{stats["raw_min"]:.3f}, {stats["raw_max"]:.3f}] | PNG display clipping only',fill='#b00020')
    for i,arr in enumerate(raw):
        r,c=divmod(i,8); x=100+c*208; y=50+r*188
        im=Image.fromarray(png_array(arr))
        sheet.paste(im,(x,y+30)); sheet.paste(im.resize((128,128),Image.Resampling.NEAREST),(x+40,y+30))
        draw.text((x,y+8),f'seed {stats["seeds"][i]}',fill='black')
        if c==0: draw.text((8,y+70),CLASSES[r],fill='black')
    sheet.save(out/'atlas.png')
    native=Image.new('RGB',(8*32,10*32))
    for i,arr in enumerate(raw): native.paste(Image.fromarray(png_array(arr)),((i%8)*32,(i//8)*32))
    native.save(out/'atlas_native.png')


def render(sampling, data):
    root=Path(sampling); raw=np.load(root/'raw.npy'); trajectory=np.load(root/'trajectory.npy',mmap_mode='r')
    stats=json.loads((root/'sampling.json').read_text())
    if len(raw)!=80: raise ValueError('Expected complete 80 sample atlas')
    contact_sheet(raw,stats,root)
    process=root/'process'; process.mkdir(exist_ok=True)
    chosen=[8*c for c in range(8)]  # Fixed before results: classes0..7, first seed.
    for i in chosen:
        frames=[Image.fromarray(png_array(trajectory[t,i])).resize((256,256),Image.Resampling.NEAREST) for t in range(64)]
        frames[0].save(process/f'{i:03d}_all64.gif',save_all=True,append_images=frames[1:],duration=120,loop=0)
        sheet=Image.new('RGB',(8*140,8*158),'#eeeeee'); draw=ImageDraw.Draw(sheet)
        for t,im in enumerate(frames):
            x=(t%8)*140; y=(t//8)*158
            sheet.paste(im.resize((128,128),Image.Resampling.NEAREST),(x,y+20))
            draw.text((x,y+2),f'patch {t+1}/64',fill='black')
        sheet.save(process/f'{i:03d}_all64.png')
    train=np.load(Path(data)/'train_images.npy',mmap_mode='r'); labels=np.load(Path(data)/'train_labels.npy')
    start=time.perf_counter(); neighbors=nn_scan(raw,train,stats['classes'],labels)
    neighbors['cpu_seconds']=time.perf_counter()-start
    global_train=np.load(Path(data)/'splits.npz')['train']
    for mode in ('raw','display_clipped'):
        neighbors[mode]['nearest_official_train_index']=global_train[neighbors[mode]['nearest_train_local_index']].tolist()
        neighbors[mode]['nearest_same_class_official_index']=global_train[neighbors[mode]['nearest_same_class_index']].tolist()
    (root/'nearest_neighbors.json').write_text(json.dumps(neighbors,indent=2))
    comparison=Image.new('RGB',(8*420,10*190),'#eeeeee'); draw=ImageDraw.Draw(comparison)
    for i in range(80):
        x=(i%8)*420; y=(i//8)*190
        ix=neighbors['raw']['nearest_train_local_index'][i]
        si=neighbors['raw']['nearest_same_class_index'][i]
        ims=[png_array(raw[i]),train[ix].transpose(1,2,0),train[si].transpose(1,2,0)]
        for j,im in enumerate(ims): comparison.paste(Image.fromarray(im).resize((128,128),Image.Resampling.NEAREST),(x+j*136,y+36))
        draw.text((x,y+3),f'{CLASSES[stats["classes"][i]]} seed={stats["seeds"][i]}',fill='black')
        draw.text((x,y+20),f'Generated | nearest {ix} | same class {si}',fill='black')
    comparison.save(root/'nearest_training.png')
    # Distances are diversity/copy diagnostics only, never a quality score.
    diversity=[]
    for c in range(10):
        a=raw[8*c:8*c+8].reshape(8,-1)
        d=((a[:,None]-a[None])**2).mean(-1)
        diversity.append(dict(class_name=CLASSES[c],unique_raw_arrays=len({v.tobytes() for v in a}),
                              mean_pairwise_raw_distance=float(d[np.triu_indices(8,1)].mean()),
                              semantic_diversity='MANUAL_REVIEW_REQUIRED; noise variance is not semantic diversity'))
    matched=raw.reshape(10,8,-1)
    cross=[float(((matched[c]-matched[(c+1)%10])**2).mean()) for c in range(10)]
    component=np.load(root/'components.npy')
    (root/'diversity.json').write_text(json.dumps(dict(within_class=diversity,
         adjacent_class_same_seed_raw_distance=cross,mixture_component_counts=np.bincount(component.flatten(),minlength=8).tolist(),
         caveat='Pixel differences and component use do not prove semantic diversity or correct classes'),indent=2))
    imgs=sorted(root.glob('[0-9][0-9][0-9]_*.png'))
    cells=''.join(f'<figure><figcaption>{stats["class_names"][i]} · seed {stats["seeds"][i]}</figcaption><img src="{p.name}" width="32" height="32"><img src="{p.name}" width="128" height="128"></figure>' for i,p in enumerate(imgs))
    processes=''.join(f'<div><p>{CLASSES[stats["classes"][i]]} · seed {stats["seeds"][i]} · all 64 patches</p><img src="process/{i:03d}_all64.gif" width="256"><a href="process/{i:03d}_all64.png">全部64帧</a></div>' for i in chosen)
    range_note=f'<p style="color:#a01818"><strong>原始采样越界 {stats["oob_fraction"]:.2%}</strong>，浮点范围 [{stats["raw_min"]:.3f}, {stats["raw_max"]:.3f}]。PNG显示时有裁剪；模型回灌始终使用未裁剪值。像素差异不等于语义多样性。</p>'
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P1 全量自由采样</title><style>body{font:15px system-ui;margin:24px;background:#f2f2f2}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}figure{margin:0;padding:12px;background:white}img{image-rendering:pixelated;vertical-align:middle;margin:4px}figcaption{margin-bottom:10px}</style><h1>P1：10类 × 8固定seed</h1><p>类别+BOS起步，原始GMM采样回灌；不是teacher-forced重建。主温度1/1；原始float已保留。32×32原图与最近邻放大；不挑图。</p>'+range_note+'<main>'+cells+'</main><h2>固定8个样本的全部逐块过程</h2>'+processes+'<h2>采样结束后的离线最近训练图对照</h2><p>每组：生成图 / 全训练集最近图 / 同类别最近图。检索不参与生成；距离不是画质分数。</p><a href="nearest_training.png">完整80组对照</a><p><a href="sampling.json">原始采样配置和越界比例</a></p>')


def curves(run):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run); rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for kind,key,label in [('train','train_nll','train batch'),('validation','val_nll','fixed val1000')]:
        points=[r for r in rows if r['kind']==kind]
        axs[0].plot([r['step'] for r in points],[r[key] for r in points],label=label)
    pts=[r for r in rows if r['kind']=='train']
    axs[1].plot([r['step'] for r in pts],[r['lr'] for r in pts])
    axs[0].set(ylabel='Continuous NLL / pixel channel',xlabel='step'); axs[0].legend()
    axs[1].set(ylabel='learning rate',xlabel='step')
    fig.tight_layout(); fig.savefig(run/'training_curves.png',dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',default='runs/p1/main'); p.add_argument('--data',default='data/p1/cifar10')
    p.add_argument('--sampling'); p.add_argument('--device',default='mps'); a=p.parse_args()
    if a.sampling: render(a.sampling,a.data); return
    run=Path(a.run); seal=run/'selection_sealed.json'
    if seal.exists() or (run/'final_evaluation.json').exists(): raise FileExistsError('Final test is single-shot; refuse repeat')
    weights={name:dict(path=str(run/(name+'_weights.pt')),sha256=hashlib.sha256((run/(name+'_weights.pt')).read_bytes()).hexdigest()) for name in ('best','last')}
    seal.write_text(json.dumps(dict(rule='minimum fixed val1000 NLL; no visual selection; selection closed before test',weights=weights),indent=2))
    b=Budget(); results={}
    try:
        with b.charge('eval','final_full_validation_and_single_test'):
            vi,vl=load_split(a.data,'val')
            for name in ('best','last'):
                model,payload=load_model(weights[name]['path'],a.device)
                results[name]=dict(step=payload['step'],val_count=len(vi),val_nll=validate(model,vi,vl,a.device,budget=b))
                del model
            model,_=load_model(weights['best']['path'],a.device)
            selection=np.load(Path(a.data)/'splits.npz')['selection_val']
            correct=validate(model,vi,vl,a.device,selection,b)
            wrong=validate(model,vi,(vl+1)%10,a.device,selection,b)
            results['class_condition_control']=dict(count=1000,correct_nll=correct,rotated_class_nll=wrong,delta=wrong-correct,
                interpretation='validation-only condition diagnostic, not visual quality and not used for checkpoint choice')
            del model
            ti,tl=load_split(a.data,'test'); model,payload=load_model(weights['best']['path'],a.device)
            results['test_best']=dict(step=payload['step'],count=len(ti),nll=validate(model,ti,tl,a.device,budget=b),selection_use=False)
        (run/'final_evaluation.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results),flush=True)
    finally: b.close()
    curves(run)


if __name__=='__main__': main()
