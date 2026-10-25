"""Atomic pure inference weights and complete optimizer/data/RNG recovery state."""
import json
import errno
import hashlib
import os
from pathlib import Path
import random
import tempfile
import time
import torch
from .prepare import REVISION


def durable_sync(file):
    # A second fsync returning zero does not prove failed writeback was replayed.
    # The caller must replay the complete immutable local copy after an NFS EIO.
    os.fsync(file.fileno())


def discard_cache(file):
    if hasattr(os,'posix_fadvise'):
        try:os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL,errno.ENOSYS,errno.EOPNOTSUPP):raise


def atomic_save(obj,path):
    # Serialize once on the container's local disk. Stream to the project NFS,
    # sync bounded chunks, and verify the complete copy before atomic publication.
    # Discard clean cache pages so two large files do not exhaust the 16GB host quota.
    path=Path(path);pending=path.with_name(path.name+'.pending');started=time.time()
    with tempfile.TemporaryDirectory(prefix='p2-checkpoint-',dir=os.environ.get('P2_CHECKPOINT_SCRATCH')) as directory:
        local=Path(directory)/path.name
        with local.open('wb') as f:
            torch.save(obj,f);f.flush();durable_sync(f);discard_cache(f)
        for attempt in range(2):
            try:
                digest=hashlib.sha256();written=0;since_sync=0
                with local.open('rb') as source,pending.open('wb') as target:
                    for block in iter(lambda:source.read(4*1024*1024),b''):
                        target.write(block);digest.update(block);written+=len(block);since_sync+=len(block)
                        if since_sync>=256*1024*1024:
                            target.flush();durable_sync(target);discard_cache(target);discard_cache(source);since_sync=0
                    target.flush();durable_sync(target);discard_cache(target);discard_cache(source)
                copied=hashlib.sha256()
                with pending.open('rb') as f:
                    for block in iter(lambda:f.read(4*1024*1024),b''):copied.update(block)
                    discard_cache(f)
                if pending.stat().st_size!=written or copied.digest()!=digest.digest():
                    raise IOError('Checkpoint copy failed full size/SHA256 verification; prior checkpoint preserved')
                break
            except OSError as exc:
                if exc.errno not in (errno.EIO,errno.ETIMEDOUT) or attempt==1:raise
                print(json.dumps(dict(event='checkpoint_copy_retry',attempt=attempt+1,error=str(exc))),flush=True)
                time.sleep(1)
        pending.replace(path)
        receipt=dict(path=str(path),bytes=written,sha256=digest.hexdigest(),seconds=time.time()-started,
                     storage='local serialization -> chunk-synced NFS -> full SHA256 -> atomic replace',copy_attempts=attempt+1,copy_max_attempts=2)
        receipt_path=path.with_name(path.name+'.integrity.json');tmp=receipt_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(receipt_path)
        print(json.dumps(dict(event='checkpoint_durable',**receipt)),flush=True)


def publish_link(source,target):
    source,target=Path(source),Path(target);pending=target.with_name('.'+target.name+'.link')
    if pending.exists():pending.unlink()
    os.link(source,pending);pending.replace(target)
    receipt=source.with_name(source.name+'.integrity.json')
    if receipt.exists():
        data=json.loads(receipt.read_text());data.update(path=str(target),alias_of=str(source))
        receipt_target=target.with_name(target.name+'.integrity.json');tmp=receipt_target.with_suffix('.tmp')
        tmp.write_text(json.dumps(data,indent=2));tmp.replace(receipt_target)


def cleanup_generations(out):
    referenced=set()
    for tag in ('best','last'):
        state=out/f'{tag}_state.pt'
        if state.exists():referenced.add(torch.load(state,map_location='cpu',weights_only=True,mmap=True)['weights_file'])
    for old in out.glob('*_step_*_weights.pt'):
        if old.name not in referenced:old.unlink()


def reuse_snapshot(out,source_tag,target_tag,step):
    """Two labels for the exact same optimizer transaction need only one disk copy."""
    out=Path(out);state=torch.load(out/f'{source_tag}_state.pt',map_location='cpu',weights_only=True,mmap=True)
    if state['step']!=step:raise ValueError('Cannot reuse a different optimizer transaction')
    publish_link(out/state['weights_file'],out/f'{target_tag}_weights.pt')
    publish_link(out/f'{source_tag}_state.pt',out/f'{target_tag}_state.pt')
    cleanup_generations(out)


def rng_state():
    return dict(cpu=torch.get_rng_state(),python=random.getstate(),
                mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng(state):
    torch.set_rng_state(state['cpu']);random.setstate(state['python'])
    if state['mps'] is not None and torch.backends.mps.is_available():torch.mps.set_rng_state(state['mps'])
    if state['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


def save(model,optimizer,out,tag,training,resume_state=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    weights={n:p.detach().cpu() for n,p in model.state_dict().items() if n!='qwen.lm_head.weight'}
    weights['qwen.lm_head.weight']=weights['qwen.model.language_model.embed_tokens.weight']
    payload=dict(format='p2-qwen-v1',source_repo='Qwen/Qwen3.5-0.8B',source_revision=REVISION,
                 step=training['step'],model=weights,
                 visual_status='preserved, no gradient, NOT_EVALUATED',pixel_model='tanh_shared_component_diag_gmm_k8')
    # Recovery states reference an immutable weight generation. A failed state
    # write can therefore never pair the prior state with newer overwritten weights.
    weight_name=f'{tag}_step_{training["step"]:06d}_weights.pt' if resume_state else f'{tag}_weights.pt'
    atomic_save(payload,out/weight_name)
    if resume_state:
        publish_link(out/weight_name,out/f'{tag}_weights.pt')
    # Shared underlying storage is deduplicated by torch.save; embedding/lm_head remains tied on load.
    if resume_state:
        state=dict(training,optimizer=optimizer.state_dict(),rng=rng_state(),weights_file=weight_name)
        atomic_save(state,out/f'{tag}_state.pt')
        # Keep only generations referenced by the now-published state. The previous
        # generation remains intact until both new files have been durably saved.
        cleanup_generations(out)
    model.tokenizer.save_pretrained(out/'tokenizer')
    model.qwen.config.save_pretrained(out/'qwen_config')


def load_weights(model,path):
    p=torch.load(path,map_location='cpu',weights_only=True)
    if p['source_revision']!=REVISION:raise ValueError('Source revision mismatch')
    model.load_state_dict(p['model'],strict=True)
    return p
