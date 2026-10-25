"""Sequential bounded P1 execution. Does not rerun P0, use remote compute, or retry training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(label, args, timeout=19000):
    log=Path('evidence/p1')/(label+'.log')
    with log.open('w') as output:
        env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',VECLIB_MAXIMUM_THREADS='4')
        proc=subprocess.Popen([sys.executable]+args,stdout=output,stderr=subprocess.STDOUT,env=env)
        print(json.dumps(dict(event='start',label=label,pid=proc.pid,log=str(log),unix=time.time())),flush=True)
        try: code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try: proc.wait(timeout=60)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            raise
        print(json.dumps(dict(event='end',label=label,code=code,unix=time.time())),flush=True)
        if code: raise RuntimeError(label+' failed; read '+str(log))


def main():
    if not Path('data/p1/cifar10/manifest.json').exists(): raise FileNotFoundError('Prepare and verify official data first')
    run('tests_split',['-m','pytest','p1/test_p1.py','-q','-k','official_split','--junitxml=evidence/p1/tests_split.xml'],60)
    run('resume_verification',['-m','p1.verify_resume'],600)
    run('smoke',['-m','p1.train','--config','configs/p1/smoke.json'],600)
    run('smoke_sampling',['-m','p1.sample','--checkpoint','runs/p1/smoke/last_weights.pt','--class-name','ship','--seed','610230','--num-images','2','--out','runs/p1/smoke_free'],120)
    for batch in (32,64): run(f'pilot{batch}',['-m','p1.train','--config',f'configs/p1/pilot{batch}.json'],600)
    pilots={b:json.loads(Path(f'runs/p1/pilot{b}/result.json').read_text()) for b in (32,64)}
    batch=max(pilots,key=lambda b:b/pilots[b]['median_step_seconds'])
    rows=[json.loads(s) for s in Path(f'runs/p1/pilot{batch}/metrics.jsonl').read_text().splitlines()]
    speeds=[r['seconds_per_step'] for r in rows if r['kind']=='train'][1:]
    conservative=max(speeds)*1.2
    budget=json.loads(Path('evidence/p1/budget.json').read_text())
    remaining=budget['train_limit']-budget['train_seconds']
    steps=min(30000,int((remaining-300)/conservative)//1000*1000)
    if steps<1000: raise TimeoutError('Insufficient budget for formal full-data training')
    config=json.loads(Path('configs/p1/base.json').read_text())
    config.update(batch_size=batch,steps=steps,max_seconds=min(17600,remaining-120),log_every=50)
    path=Path('configs/p1/main.json'); path.write_text(json.dumps(config,indent=2))
    record=dict(pilots=pilots,selected_batch=batch,selection_rule='maximum images/sec; reserve 20% step slowdown and 300s',
                conservative_step_seconds=conservative,planned_steps=steps,estimated_train_seconds=steps*pilots[batch]['median_step_seconds'],
                remaining_train_seconds=remaining,unix=time.time(),configuration=config,
                sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,Path('P1_PLAN.md'),Path('data/p1/cifar10/manifest.json')]+sorted(Path('p1').glob('*.py'))})
    Path('evidence/p1/frozen_before_main.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(dict(event='main_plan',batch=batch,steps=steps,estimated_train_seconds=record['estimated_train_seconds'],conservative_train_seconds=steps*conservative)),flush=True)
    run('main_training',['-m','p1.train','--config',str(path)],19000)
    # Final test cannot influence choice, sealed before opening official test tensors.
    run('final_nll',['-m','p1.evaluate'],1800)
    run('atlas',['-m','p1.sample','--checkpoint','runs/p1/main/best_weights.pt','--atlas','--out','runs/p1/atlas'],600)
    stats=json.loads(Path('runs/p1/atlas/sampling.json').read_text())
    if stats['oob_fraction']>.2:
        Path('evidence/p1/severe_oob.json').write_text(json.dumps(stats,indent=2))
        print('Severe raw sampling OOB; stop additional GPU sampling and report without method change',flush=True)
    else:
        run('independent',['-m','p1.independent','--checkpoint','runs/p1/main/best_weights.pt','--reference','runs/p1/atlas'],300)
    run('render_and_nearest',['-m','p1.evaluate','--sampling','runs/p1/atlas'],1200)
    print(json.dumps(dict(event='all_execution_completed',unix=time.time())),flush=True)


if __name__=='__main__': main()
