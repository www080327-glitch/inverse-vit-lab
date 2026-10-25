"""Real interrupted/resumed optimization compared with uninterrupted run; tiny MPS only."""
import json
from pathlib import Path
import torch
from .run import run


def main():
    base=json.loads(Path('configs/p1/base.json').read_text())
    base.update(smoke=True,steps=4,warmup_steps=1,log_every=1,val_every=4,save_every=1,preview_every=0,batch_size=4,max_seconds=60)
    base['model'].update(dim=32,depth=2,heads=4,ff_dim=64)
    for name in ('resume_control','resume_interrupted'):
        cfg=dict(base,out='runs/p1/'+name)
        Path('configs/p1/'+name+'.json').write_text(json.dumps(cfg,indent=2))
    run('resume_control',['-m','p1.train','--config','configs/p1/resume_control.json'],120)
    run('resume_first2',['-m','p1.train','--config','configs/p1/resume_interrupted.json','--stop-after','2'],120)
    first=torch.load('runs/p1/resume_interrupted/last_checkpoint.pt',map_location='cpu',weights_only=True)
    assert first['step']==2
    run('resume_to4',['-m','p1.train','--config','configs/p1/resume_interrupted.json','--resume','runs/p1/resume_interrupted/last_checkpoint.pt'],120)
    control=torch.load('runs/p1/resume_control/last_checkpoint.pt',map_location='cpu',weights_only=True)
    resumed=torch.load('runs/p1/resume_interrupted/last_checkpoint.pt',map_location='cpu',weights_only=True)
    errors={k:float((v-resumed['model'][k]).abs().max()) for k,v in control['model'].items()}
    passed=all(v==0 for v in errors.values()) and control['step']==resumed['step']==4 and torch.equal(control['rng'],resumed['rng'])
    result=dict(passed=passed,model_max_difference=max(errors.values()),steps=4,interrupted_after=2,device='mps',
                scope='tiny architecture MPS, real optimizer updates, same dropout/data RNG schedule; not a second full main training')
    Path('evidence/p1/resume_verification.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))
    if not passed: raise AssertionError(result)


if __name__=='__main__': main()
