"""Bounded single-process training and deterministic validation. All P1 only."""
import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from .budget import Budget
from .data import load_split, dequantize
from .model import Config, PixelAR, gmm_nll, generate, unpatchify


def sync(device):
    if str(device)=='mps': torch.mps.synchronize()


def atomic_torch(obj, path):
    path=Path(path); temp=path.with_suffix('.tmp'); torch.save(obj,temp); temp.replace(path)


@torch.inference_mode()
def validate(model, images, labels, device, indices=None, budget=None):
    was_training=model.training; model.eval()
    rng=torch.Generator().manual_seed(91023)
    total=0.; seen=0
    if indices is not None: images, labels=images[indices],labels[indices]
    try:
        for start in range(0,len(images),64):
            if budget is not None and budget.remaining('eval')<10: raise TimeoutError('evaluation budget')
            targets=dequantize(images[start:start+64],rng).to(device)
            ids=labels[start:start+64].to(device)
            loss=gmm_nll(model.teacher(ids,targets),targets)
            value=float(loss.item())
            if not math.isfinite(value): raise FloatingPointError('Nonfinite validation')
            total += value*len(ids); seen+=len(ids)
        sync(device)
        return total/seen
    finally: model.train(was_training)


def lr_at(step, c):
    if step <= c['warmup_steps']: return c['lr']*step/max(1,c['warmup_steps'])
    progress=min(1.,(step-c['warmup_steps'])/max(1,c['steps']-c['warmup_steps']))
    return c['lr']*(.1+.9*.5*(1+math.cos(math.pi*progress)))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    parser.add_argument('--resume'); parser.add_argument('--budget-root',default='evidence/p1')
    parser.add_argument('--stop-after',type=int,help='Graceful bounded stop for resume verification; does not change schedule')
    a=parser.parse_args(); c=json.loads(Path(a.config).read_text())
    out=Path(c['out']); out.mkdir(parents=True,exist_ok=True)
    if (out/'last_checkpoint.pt').exists() and not a.resume: raise FileExistsError('Refuse overwrite; use explicit resume')
    device=c['device']; torch.set_num_threads(4)
    if device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1': raise RuntimeError('CPU fallback prohibited')
    budget=Budget(a.budget_root)
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,stop)
    torch.manual_seed(c['seed']); np.random.seed(c['seed'])
    model=PixelAR(Config(**c['model'])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],betas=tuple(c['betas']),weight_decay=c['weight_decay'],foreach=False)
    rng=torch.Generator().manual_seed(c['seed']+1)
    train_images,train_labels=load_split(c['data'],'train')
    if c.get('smoke'): train_images,train_labels=train_images[:64],train_labels[:64]
    val_images,val_labels=load_split(c['data'],'val')
    selection=np.load(Path(c['data'])/'splits.npz')['selection_val']
    if c.get('smoke'): selection=selection[:64]
    data_manifest=json.loads((Path(c['data'])/'manifest.json').read_text())
    contract=dict(config=c,data_sha256=data_manifest['files'])
    step=0; best=math.inf; cumulative=0.; order=torch.empty(0,dtype=torch.long); cursor=0
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=True)
        if ck['contract'] != contract: raise ValueError('Resume configuration/data mismatch')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        step,best,cumulative=ck['step'],ck['best_val'],ck['train_seconds']
        rng.set_state(ck['rng']); torch.set_rng_state(ck['torch_rng'])
        if device=='mps': torch.mps.set_rng_state(ck['mps_rng'])
        order,cursor=ck['order'],ck['cursor']
    (out/'config.json').write_text(json.dumps(c,indent=2))
    (out/'process.json').write_text(json.dumps(dict(pid=os.getpid(),running=True,start_unix=time.time(),stop=f'kill -TERM {os.getpid()}'),indent=2))
    wall=time.perf_counter(); train_seconds=cumulative; times=[]; max_alloc=0; max_driver=0
    log=(out/'metrics.jsonl').open('a',buffering=1)
    def event(obj):
        obj.update(step=step,wall_seconds=time.perf_counter()-wall)
        log.write(json.dumps(obj)+'\n'); print(json.dumps(obj),flush=True)
    def save(best_model=False):
        sync(device)
        payload=dict(model_config=asdict(model.config),model={k:v.detach().cpu() for k,v in model.state_dict().items()},step=step)
        atomic_torch(payload,out/('best_weights.pt' if best_model else 'last_weights.pt'))
        state=dict(payload,optimizer=optimizer.state_dict(),contract=contract,best_val=best,train_seconds=train_seconds,
                   rng=rng.get_state(),torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if device=='mps' else None,
                   order=order,cursor=cursor)
        atomic_torch(state,out/('best_checkpoint.pt' if best_model else 'last_checkpoint.pt'))
    reason='max_steps'
    try:
        if step==0:
            with budget.charge('eval',out.name+'_initial_validation'):
                initial=validate(model,val_images,val_labels,device,selection,budget)
            event(dict(kind='validation',val_nll=initial,initial=True))
        model.train()
        while step<c['steps']:
            if a.stop_after and step>=a.stop_after: reason='requested_step_stop'; break
            if stopped or (out/'STOP').exists(): reason='safe_stop'; break
            if train_seconds>=c['max_seconds']-5 or budget.remaining('train')<10: reason='time_budget'; break
            with budget.charge('train',out.name+'_train_chunk'):
                # Persist timing once per logging chunk; signal observed at every step.
                for _ in range(min(c['log_every'],c['steps']-step)):
                    if a.stop_after and step>=a.stop_after: break
                    if stopped or (out/'STOP').exists() or train_seconds>=c['max_seconds']-5 or budget.remaining('train')<10: break
                    start=time.perf_counter()
                    if cursor+c['batch_size']>len(order):
                        order=torch.randperm(len(train_images),generator=rng); cursor=0
                    ix=order[cursor:cursor+c['batch_size']]; cursor+=len(ix)
                    target=dequantize(train_images[ix],rng).to(device); labels=train_labels[ix].to(device)
                    for group in optimizer.param_groups: group['lr']=lr_at(step+1,c)
                    optimizer.zero_grad(set_to_none=True)
                    loss=gmm_nll(model.teacher(labels,target),target)
                    if not torch.isfinite(loss).item(): raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip'],error_if_nonfinite=True)
                    optimizer.step(); sync(device)
                    elapsed=time.perf_counter()-start; train_seconds+=elapsed; times.append(elapsed); step+=1
                    if device=='mps':
                        max_alloc=max(max_alloc,torch.mps.current_allocated_memory()); max_driver=max(max_driver,torch.mps.driver_allocated_memory())
            event(dict(kind='train',train_nll=float(loss.item()),grad_norm=float(grad.item()),lr=optimizer.param_groups[0]['lr'],
                       train_seconds=train_seconds,seconds_per_step=float(np.mean(times[-c['log_every']:])),
                       mps_allocated_peak_observed=max_alloc,mps_driver_peak_observed=max_driver))
            if step % c['val_every']==0 or step==c['steps']:
                with budget.charge('eval',out.name+'_validation'):
                    value=validate(model,val_images,val_labels,device,selection,budget)
                improved=value<best
                if improved: best=value; save(True)
                event(dict(kind='validation',val_nll=value,best_val=best,improved=improved))
            if c.get('preview_every') and step%c['preview_every']==0:
                with budget.charge('eval',out.name+'_fixed_preview'):
                    patches,_=generate(model,[0,2,6,8],[610230]*4)
                    raw=unpatchify(patches).numpy(); np.save(out/f'preview_{step:06d}.npy',raw)
                    oob=float((np.abs(raw)>1).mean())
                model.train(); event(dict(kind='preview',oob_fraction=oob))
                if oob>.2: reason='severe_oob_stop'; break
            if step%c['save_every']==0: save()
        save()
        result=dict(reason=reason,step=step,best_val=best,train_seconds=train_seconds,wall_seconds=time.perf_counter()-wall,
                    parameters=model.parameter_count(),train_count=len(train_images),batch_size=c['batch_size'],
                    median_step_seconds=float(np.median(times[5:] or times)),mps_allocated_peak_observed=max_alloc,mps_driver_peak_observed=max_driver,
                    device=device,fp32=True,smoke=bool(c.get('smoke')),budget=budget.state)
        (out/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc),step=step),indent=2))
        save(); raise
    finally:
        log.close(); budget.close()
        (out/'process.json').write_text(json.dumps(dict(pid=os.getpid(),running=False,end_unix=time.time(),step=step),indent=2))


if __name__=='__main__': main()
