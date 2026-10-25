"""One full-parameter schedule; bounded probe first, then locked main config."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import torch
from .checkpoint import save,load_weights,set_rng,reuse_snapshot
from .data import images,text_tokens,text_batch
from .guard import atomic_json
from .model import PixelQwen
from .probability import dequantize,nll
from .verify import sync,gradient_audit,representatives


def amp(device):return torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext()


def factor(step,c):
    if step<=c['warmup_steps']:return step/max(1,c['warmup_steps'])
    t=min(1.,(step-c['warmup_steps'])/max(1,c['steps']-c['warmup_steps']))
    return .1+.9*(1+math.cos(math.pi*t))/2


@torch.inference_mode()
def validate(m,vi,vl,vt,c):
    m.eval();rng=torch.Generator().manual_seed(91023);total=0.;seen=0;protected=0
    import numpy as np
    ix=np.load('data/p1/cifar10/splits.npz')['selection_val'][:c['selection_images']]
    for start in range(0,len(ix),c['microbatch']):
        check_deadline()
        ids=ix[start:start+c['microbatch']];x,count=dequantize(vi[ids],rng);protected+=count
        with amp(c['device']):loss=nll(m.teacher(vl[ids].tolist(),x.to(m.device)),x.to(m.device))
        total+=float(loss)*len(ids);seen+=len(ids)
    trng=torch.Generator().manual_seed(91024);tl=[]
    for _ in range(c['text_val_batches']):
        check_deadline()
        with amp(c['device']):tl.append(float(m.text_loss(text_batch(vt,trng,length=c['text_length']).to(m.device),c['text_ce_chunk'])))
    m.train();return dict(val_nll=total/seen,text_val_ce=sum(tl)/len(tl),val_count=seen,endpoint_protections=protected)


def check_deadline():
    if 'P2_DEADLINE_UNIX' not in os.environ:raise RuntimeError('Budget watchdog required')
    deadline=min(float(os.environ['P2_DEADLINE_UNIX']),float(os.environ.get('P2_TRAIN_DEADLINE_UNIX',os.environ['P2_DEADLINE_UNIX'])))
    if time.time()>deadline:raise TimeoutError('Training reserve reached; leave time for final sampling/evaluation/save/release')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume')
    p.add_argument('--stop-after',type=int);a=p.parse_args();c=json.loads(Path(a.config).read_text())
    if not c.get('steps') or not os.environ.get('P2_GUARDED'):raise ValueError('Resolved steps and external budget guard required')
    # The approved half-card allocation supplies four CPU cores. Avoid
    # oversubscribing it with the previous hardcoded six worker threads.
    torch.set_num_threads(4);torch.manual_seed(c['seed']);device=c['device'];out=Path(c['out'])
    if out.exists() and any(out.iterdir()) and not a.resume:raise FileExistsError('Use a new run directory or explicit resume')
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/'config.json',c)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    m=PixelQwen(c['source']).to(device)
    if c['gradient_checkpointing']:m.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    opt=m.optimizer(c['backbone_lr'],c['interface_lr']);ti,tl=images('train');vi,vl=images('val');texts=text_tokens(m.tokenizer)
    atomic_json(out/'parameters.json',m.parameter_manifest())
    rng=torch.Generator().manual_seed(c['seed']+1);trng=torch.Generator().manual_seed(c['seed']+2)
    contract=dict(config=c,data_manifest_sha256=hashlib.sha256(Path('data/p1/cifar10/manifest.json').read_bytes()).hexdigest(),
                  text_manifest_sha256=hashlib.sha256(Path('evidence/p2_qwen/text_manifest.json').read_bytes()).hexdigest())
    step=0;best=math.inf;order=torch.empty(0,dtype=torch.long);cursor=0;epoch=0
    image_examples=0;text_token_count=0;times=[];by_task={'image':[],'text':[]};protected=0;loop_seconds=0.
    bad_gradient_steps=0;bad_validation_checks=0;initial_pixel_val=None
    if a.resume:
        s=torch.load(a.resume,map_location='cpu',weights_only=True)
        if s['contract']!=contract:raise ValueError('Resume contract mismatch')
        load_weights(m,Path(a.resume).parent/s['weights_file']);opt.load_state_dict(s['optimizer']);set_rng(s['rng'])
        step,best,order,cursor,epoch=s['step'],s['best'],s['order'],s['cursor'],s['epoch']
        rng.set_state(s['data_rng']);trng.set_state(s['text_rng'])
        image_examples=s['image_examples'];text_token_count=s['text_token_count'];loop_seconds=s['loop_seconds']
        bad_gradient_steps=s.get('bad_gradient_steps',0);bad_validation_checks=s.get('bad_validation_checks',0);initial_pixel_val=s.get('initial_pixel_val')
    # Same AdamW equations/state and schedule; multi-tensor kernels replace
    # hundreds of small Python-dispatched updates. No fused-step state conversion.
    # The documented extra buffer (~3GB) fits the measured half-card headroom.
    if device=='cuda':
        for group in opt.param_groups:group.update(foreach=True,fused=False)
    start=time.time();atomic_json(out/'process.json',dict(pid=os.getpid(),running=True,start_unix=start))
    log=(out/'metrics.jsonl').open('a',buffering=1)
    def event(**r):
        r.update(step=step,wall_seconds=time.time()-start)
        budget_path=os.environ.get('P2_BUDGET_LEDGER')
        if budget_path and Path(budget_path).exists():
            ledger=json.loads(Path(budget_path).read_text())
            if ledger.get('active') and 'instance_id' in ledger['active']:
                r['billed_card_hours']=(ledger['closed_seconds']+time.time()-ledger['active']['start_unix'])/3600
                r['remaining_card_hours']=max(0,10-r['billed_card_hours'])
        log.write(json.dumps(r)+'\n');print(json.dumps(r),flush=True)
    event(kind='execution',resume=a.resume,optimizer_backend='foreach' if device=='cuda' else 'original',
          cpu_threads=torch.get_num_threads(),effective_image_batch=c['microbatch']*c['accumulation'])
    latest_snapshot=None
    def snapshot(tag,resume_state=True):
        nonlocal latest_snapshot
        sync(device);t=time.time()
        if resume_state and latest_snapshot is not None and latest_snapshot[1]==step and latest_snapshot[0]!=tag:
            reuse_snapshot(out,latest_snapshot[0],tag,step)
            event(kind='save',tag=tag,seconds=time.time()-t,reused_transaction=latest_snapshot[0]);return
        save(m,opt,out,tag,dict(step=step,best=best,order=order,cursor=cursor,epoch=epoch,
             data_rng=rng.get_state(),text_rng=trng.get_state(),contract=contract,image_examples=image_examples,
             text_token_count=text_token_count,loop_seconds=loop_seconds,
             bad_gradient_steps=bad_gradient_steps,bad_validation_checks=bad_validation_checks,initial_pixel_val=initial_pixel_val,
             deadline_unix=float(os.environ['P2_DEADLINE_UNIX']),budget_ledger=os.environ.get('P2_BUDGET_LEDGER'),
             budget_snapshot=json.loads(Path(os.environ['P2_BUDGET_LEDGER']).read_text()) if os.environ.get('P2_BUDGET_LEDGER') else None),resume_state=resume_state)
        if resume_state:latest_snapshot=(tag,step)
        event(kind='save',tag=tag,seconds=time.time()-t)
    milestones=set([max(1,int(c['steps']*p)) for p in (.1,.5,1.)]);reason='max_steps'
    try:
        m.train()
        if step==0:
            initial_start=time.time()
            initial=validate(m,vi,vl,texts['validation'],c)
            initial_pixel_val=initial['val_nll']
            event(kind='validation',initial=True,seconds=time.time()-initial_start,**initial)
        while step<c['steps']:
            check_deadline()
            if stopped or (out/'STOP').exists():reason='signal_or_STOP';break
            if epoch>=c['max_epochs']:reason='max_epochs';break
            task='text' if (step+1)%(c['image_steps_per_text_step']+1)==0 else 'image'
            before=representatives(m) if step==0 else None
            for g in opt.param_groups:g['lr']=g['peak_lr']*factor(step+1,c)
            opt.zero_grad(set_to_none=True);t=time.time();loss_total=0.
            repeats=c['accumulation'] if task=='image' else 1
            for _ in range(repeats):
                # Complete the current accumulation transaction before observing a normal deadline.
                # Cursor/RNG in recovery state always correspond to committed optimizer steps.
                if task=='image':
                    if cursor+c['microbatch']>len(order):order=torch.randperm(len(ti),generator=rng);cursor=0;epoch+=1
                    ix=order[cursor:cursor+c['microbatch']];cursor+=len(ix)
                    x,pr=dequantize(ti[ix],rng);protected+=pr;x=x.to(device)
                    with amp(device):loss=nll(m.teacher(tl[ix].tolist(),x),x)/repeats
                    image_examples+=len(ix)
                else:
                    ids=text_batch(texts['train'],trng,c['text_batch'],c['text_length']).to(device)
                    with amp(device):loss=m.text_loss(ids,c['text_ce_chunk'])
                    text_token_count+=ids.shape[0]*(ids.shape[1]-1)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();loss_total+=float(loss.detach())
            if step==0 or (task=='text' and step<c['image_steps_per_text_step']+1):
                atomic_json(out/f'gradient_audit_{task}.json',gradient_audit(m) if task=='image' else [
                    dict(name=n,finite=bool(torch.isfinite(pv.grad).all()),absmax=float(pv.grad.abs().max()))
                    for n,pv in m.named_parameters() if pv.grad is not None])
            grad=torch.nn.utils.clip_grad_norm_([pv for pv in m.parameters() if pv.requires_grad],c['gradient_clip'],error_if_nonfinite=True)
            bad_gradient_steps=bad_gradient_steps+1 if float(grad)>c.get('exploding_gradient_norm',10000.) else 0
            opt.step();sync(device);duration=time.time()-t;loop_seconds+=duration;times.append(duration);by_task[task].append(duration);step+=1
            if before is not None:
                updates=[]
                for n,pv in m.named_parameters():
                    if n in before:
                        b=before[n];after=pv.detach().flatten()[:len(b)].cpu()
                        updates.append(dict(name=n,absmax=float((after-b).abs().max())))
                atomic_json(out/'first_step_updates.json',updates)
                for layer in range(24):
                    prefix=f'qwen.model.language_model.layers.{layer}.'
                    if not any(r['absmax']>0 for r in updates if r['name'].startswith(prefix)):
                        raise AssertionError(f'Layer {layer} did not update on first step')
            event(kind='train',task=task,loss=loss_total,grad_norm=float(grad),seconds=duration,
                  image_examples=image_examples,text_tokens=text_token_count,epoch=epoch,
                  backbone_lr=opt.param_groups[0]['lr'],interface_lr=opt.param_groups[1]['lr'])
            if bad_gradient_steps>=c.get('exploding_gradient_patience',10):reason='persistent_gradient_explosion';break
            if step%c['val_every']==0 or step in milestones:
                t=time.time();vals=validate(m,vi,vl,texts['validation'],c);event(kind='validation',seconds=time.time()-t,**vals)
                if initial_pixel_val is not None:
                    worse=vals['val_nll']>initial_pixel_val+c.get('pixel_val_regression_margin',.5)
                    bad_validation_checks=bad_validation_checks+1 if worse else 0
                if vals['val_nll']<best:best=vals['val_nll'];snapshot('best')
                if bad_validation_checks>=c.get('pixel_val_regression_patience',3):reason='persistent_pixel_val_regression';break
            if step%c['save_every']==0 and step<c['steps']:snapshot('last')
            if step in milestones and c.get('preview',False):
                snapshot(f'milestone_{step:06d}',resume_state=False)
                from .sample import sample_model
                sample_model(m,out/f'preview_{step:06d}',[0,2,6,8],[610230]*4,checkpoint_label=f'step_{step}')
                m.train()
        snapshot('last')
    except (TimeoutError,KeyboardInterrupt) as exc:
        reason=type(exc).__name__;snapshot('last')
    except BaseException as exc:
        reason='failure';atomic_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),step=step));snapshot('last');raise
    finally:
        result=dict(reason=reason,step=step,loop_seconds=loop_seconds,wall_seconds=time.time()-start,
          image_examples=image_examples,text_tokens=text_token_count,endpoint_protections=protected,
          per_task_seconds=by_task,observed_cuda_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
        atomic_json(out/'result.json',result);atomic_json(out/'process.json',dict(pid=os.getpid(),running=False,step=step));log.close()


if __name__=='__main__':main()
