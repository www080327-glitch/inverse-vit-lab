"""Prepare and run a data-free subprocess using actual trained primary weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--reference',required=True)
    p.add_argument('--out',default='scratch/p1_independent'); a=p.parse_args()
    root=Path.cwd(); isolated=Path(a.out).resolve(); isolated.mkdir(parents=True,exist_ok=False)
    stats=json.loads((Path(a.reference)/'sampling.json').read_text())
    if stats['oob_fraction']>.2:
        result=dict(status='NOT_RUN',reason='Predeclared severe raw OOB threshold exceeded; no additional GPU sampling')
        (root/'evidence/p1/independent_sampling.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result)); return
    (isolated/'p1').mkdir()
    for name in ('__init__.py','model.py','sample.py','budget.py'):
        shutil.copy2(root/'p1'/name,isolated/'p1'/name)
    shutil.copy2(a.checkpoint,isolated/'weights.pt')
    guard="""import os,sys,runpy
from pathlib import Path
source=Path(SOURCE)
isolated=Path.cwd().resolve()
budget_dir=source/'evidence'/'p1'
allowed_budget={budget_dir/'budget.json',budget_dir/'budget.tmp',budget_dir/'gpu.lock'}
blocked={'p1.data','p1.train','p1.evaluate','p1.independent','data','e3_data','visuals','PIL.ImageDraw','torchvision'}
class NoData:
 def find_spec(self,fullname,path=None,target=None):
  if fullname in blocked: raise RuntimeError('Blocked data/drawing import: '+fullname)
sys.meta_path.insert(0,NoData())
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  path=Path(os.fsdecode(args[0])).resolve()
  original_project=(path.is_relative_to(source) and not path.is_relative_to(isolated)
                    and not path.is_relative_to(source/'.venv') and path not in allowed_budget)
  if original_project or path.name in {'data.py','e3_data.py','visuals.py'}:
   raise RuntimeError('Blocked training/target access: '+str(path))
sys.addaudithook(audit)
sys.argv=['p1.sample','--checkpoint','weights.pt','--atlas','--out','output','--device','mps','--budget-root',BUDGET]
runpy.run_module('p1.sample',run_name='__main__')
""".replace('SOURCE',repr(str(root))).replace('BUDGET',repr(str(root/'evidence/p1')))
    (isolated/'run_guarded.py').write_text(guard)
    proc=subprocess.run([sys.executable,'run_guarded.py'],cwd=isolated,capture_output=True,text=True,timeout=180)
    (isolated/'subprocess.log').write_text(proc.stdout+'\n'+proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stderr)
    actual=np.load(isolated/'output/raw.npy'); reference=np.load(Path(a.reference)/'raw.npy')
    difference=float(np.abs(actual-reference).max())
    # Repeat identical batching: cross-batch rounding is not part of the replay guarantee.
    result=dict(passed=bool(np.allclose(actual,reference,atol=1e-6,rtol=1e-6)),max_abs_difference=difference,
                comparison='same MPS device and batch80, all 80 fixed class/seed pairs, CPU RNG per image',
                inputs=['model code','pure weights','10 class IDs','8 fixed seeds per class'],
                forbidden=['target images','training images','drawing modules','retrieval'],
                files=[str(q.relative_to(isolated)) for q in isolated.rglob('*') if q.is_file()],
                checkpoint_sha256=hashlib.sha256((isolated/'weights.pt').read_bytes()).hexdigest())
    (root/'evidence/p1/independent_sampling.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))
    if not result['passed']: raise AssertionError('Independent sample differs; inspect batch numeric effects')


if __name__=='__main__': main()
