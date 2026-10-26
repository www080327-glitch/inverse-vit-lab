"""No training: default numerical parity plus vector-condition portability."""
import json
import os
from pathlib import Path
import types
import zipfile
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="2"
if hasattr(os,"sched_getaffinity"):
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
from heads import Head
BASE=Path(__file__).resolve().parent
historical=types.ModuleType("c15_historical_heads")
with zipfile.ZipFile(BASE/"C15_DELIVERY.zip") as z:
    exec(compile(z.read("C15/heads.py"),"historical_heads.py","exec"),historical.__dict__)
report={}
for method in ("mse","diag","full","ear"):
    torch.manual_seed(33)
    old=historical.Head(method,8,4)
    new=Head(method,8,4)
    new.load_state_dict(old.state_dict(),strict=True)
    c=torch.tensor([0,1,2,3])
    x_old=old.sample(c,31,torch.Generator().manual_seed(91))
    x_new=new.sample(c,31,torch.Generator().manual_seed(91))
    assert torch.equal(x_old,x_new)
    x_vector=new.sample(torch.nn.functional.one_hot(c,4).float(),31,torch.Generator().manual_seed(91))
    assert torch.equal(x_new,x_vector)
    target=torch.randn(4,8)
    old_loss=old.loss(c,target,torch.Generator().manual_seed(18))
    new_loss=new.loss(c,target,torch.Generator().manual_seed(18))
    assert torch.equal(old_loss,new_loss)
    model=Head(method,8,4,condition_dim=16).double()
    context=torch.randn(3,16,dtype=torch.double)
    output=model.sample(context,17,torch.Generator().manual_seed(41))
    assert output.shape==(3,17,8) and output.dtype==torch.double
    assert torch.isfinite(output).all()
    loss=model.loss(context,torch.randn(3,8,dtype=torch.double),torch.Generator().manual_seed(82))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    report[method]={"default_sampling_bitwise_parity":True,"default_loss_bitwise_parity":True,
                    "onehot_equals_id_sample":True,"continuous_condition_sample_shape":[3,17,8],
                    "float64_and_gradient":"pass"}
(BASE/"extended/results/portability_checks.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
