"""Guarded E3 inference proof for any explicitly supplied run directory."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from runtime import sha256


def verify_inference(root, run_dir, evidence, sandbox_name):
    root, run_dir, evidence = Path(root).resolve(), Path(run_dir).resolve(), Path(evidence).resolve()
    sandbox = root / "scratch" / sandbox_name
    sandbox.mkdir(parents=True, exist_ok=False)
    for name in ["model.py", "runtime.py", "sample.py"]:
        shutil.copy2(root / name, sandbox / name)
    shutil.copy2(run_dir / "weights.pt", sandbox / "weights.pt")
    shutil.copy2(root / "data/e3/heldout_conditions.npy", sandbox / "conditions.npy")
    code = '''import sys,os,runpy,json
blocked=['data','e3_data','e3_metrics','e3_evaluate','e3_independent','visuals','PIL.ImageDraw','PIL.ImageDraw2']
class Guard:
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==m or fullname.startswith(m+'.') for m in blocked):
            raise RuntimeError('Forbidden inference import: '+fullname)
sys.meta_path.insert(0,Guard())
forbidden=FORBIDDEN
def audit(event,args):
    if event=='open' and isinstance(args[0],str):
        path=os.path.realpath(args[0])
        if path==forbidden or path.startswith(forbidden+os.sep):
            raise RuntimeError('Forbidden dataset access')
sys.addaudithook(audit)
sys.argv=['sample.py','--weights','weights.pt','--conditions','conditions.npy','--out','out','--device','mps']
runpy.run_path('sample.py',run_name='__main__')
assert not any(m in sys.modules for m in blocked)
print(json.dumps({'guarded_inference':'PASS','data_open_guard':True,'renderer_import_guard':True}))
'''.replace("FORBIDDEN", repr(str(root / "data")))
    with open(evidence / "standalone.log", "w") as output:
        result = subprocess.run([sys.executable, "-c", code], cwd=sandbox,
            env={**os.environ, "PYTHONPATH": str(sandbox)}, stdout=output, stderr=subprocess.STDOUT, timeout=90)
    if result.returncode:
        raise RuntimeError("Independent inference failed")
    actual = np.load(sandbox / "out/outputs.npy", allow_pickle=False)
    expected = np.load(run_dir / "heldout_after/outputs.npy", allow_pickle=False)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    receipt = {"status": "PASS", "count": len(actual), "max_abs_error": float(np.abs(actual-expected).max()),
        "weights_sha256": sha256(run_dir / "weights.pt"), "source_run": str(run_dir.relative_to(root)),
        "dataset_open_guard": True, "renderer_import_guard": True, "tolerance": {"atol": 1e-5, "rtol": 1e-5}}
    (evidence / "standalone.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt
