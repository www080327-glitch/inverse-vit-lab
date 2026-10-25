"""Standalone class+seed sampling. No data, training, retrieval or drawing imports."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from .model import Config, PixelAR, generate, unpatchify

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
ZH = dict(zip(['飞机','汽车','鸟','猫','鹿','狗','青蛙','马','船','卡车'], range(10)))


def load_model(path, device='mps'):
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no implicit CPU fallback')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model = PixelAR(Config(**payload['model_config']))
    model.load_state_dict(payload['model'])
    return model.to(device).eval(), payload


def png_array(chw):
    return np.rint(np.clip((chw.transpose(1,2,0)+1)/2, 0, 1)*255).astype(np.uint8)


def sample_to_dir(checkpoint, out, classes, seeds, device='mps', t_mix=1., t_noise=1.):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    model, payload = load_model(checkpoint, device)
    if device == 'mps': torch.mps.synchronize()
    start = time.perf_counter()
    trajectory = []
    patches, components = generate(model, classes, seeds, t_mix, t_noise,
                                    on_step=lambda n,p: trajectory.append(p.numpy().copy()))
    if device == 'mps': torch.mps.synchronize()
    seconds = time.perf_counter()-start
    raw = unpatchify(patches, model.config.grid, model.config.patch_size).numpy()
    np.save(out/'raw.npy', raw); np.save(out/'patches.npy', patches.numpy())
    np.save(out/'components.npy', components.numpy())
    # Every prefix saved; zero canvas is a display background only, never fed to model.
    canvases=[]
    for p in trajectory:
        full = np.zeros(tuple(patches.shape), dtype=np.float32); full[:, :p.shape[1]] = p
        canvases.append(unpatchify(torch.from_numpy(full), model.config.grid, model.config.patch_size).numpy())
    np.save(out/'trajectory.npy', np.stack(canvases))
    for i, (cls,seed) in enumerate(zip(classes,seeds)):
        Image.fromarray(png_array(raw[i])).save(out/f'{i:03d}_{CLASSES[cls]}_seed{seed}.png')
    stats = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                 step=payload['step'], model_config=asdict(model.config), parameters=model.parameter_count(),
                 classes=classes,class_names=[CLASSES[c] for c in classes],seeds=seeds, device=device,
                 t_mix=t_mix,t_noise=t_noise, feedback='raw_unclipped', kv_cache=False,
                 rng='one torch CPU Generator per image; multinomial then randn per patch',
                 seconds=seconds,images_per_second=len(seeds)/seconds,seconds_per_image=seconds/len(seeds),
                 raw_min=float(raw.min()),raw_max=float(raw.max()),oob_fraction=float((np.abs(raw)>1).mean()),
                 oob_per_image=(np.abs(raw)>1).mean((1,2,3)).tolist(),all_finite=bool(np.isfinite(raw).all()))
    (out/'sampling.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats),flush=True)
    return stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True); p.add_argument('--out',required=True)
    p.add_argument('--device',default='mps',choices=['cpu','mps'])
    p.add_argument('--class-name',default='ship'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num-images',type=int,default=8); p.add_argument('--atlas',action='store_true')
    p.add_argument('--t-mix',type=float,default=1.); p.add_argument('--t-noise',type=float,default=1.)
    p.add_argument('--budget-root',default='evidence/p1')
    args=p.parse_args()
    if args.atlas:
        classes=[c for c in range(10) for _ in range(8)]; seeds=list(range(610230,610238))*10
    else:
        c=ZH[args.class_name] if args.class_name in ZH else CLASSES.index(args.class_name)
        classes=[c]*args.num_images; seeds=list(range(args.seed,args.seed+args.num_images))
    # Budget is execution support, not part of the probability model. Standalone copy includes it.
    from .budget import Budget
    budget=Budget(args.budget_root)
    try:
        with budget.charge('eval','standalone_sampling'):
            if budget.remaining('eval') < 120: raise TimeoutError('Insufficient safe sampling reserve')
            sample_to_dir(args.checkpoint,args.out,classes,seeds,args.device,args.t_mix,args.t_noise)
    finally: budget.close()


if __name__=='__main__': main()
