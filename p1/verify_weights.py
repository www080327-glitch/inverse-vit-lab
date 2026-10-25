"""Actual trained primary model: save/reload distribution parameters, data-free."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from .budget import Budget
from .sample import load_model


def main():
    checkpoint=Path('runs/p1/main/best_weights.pt')
    budget=Budget()
    try:
        with budget.charge('eval','actual_trained_main_distribution_roundtrip'):
            a,payload=load_model(checkpoint,'mps')
            rng=torch.Generator().manual_seed(7123)
            prefix=torch.randn(2,17,48,generator=rng).to('mps')
            ids=torch.tensor([0,8],device='mps')
            with torch.inference_mode(): first=tuple(v.cpu() for v in a(ids,prefix))
            with tempfile.TemporaryDirectory(prefix='p1_weights_roundtrip_',dir='scratch') as temporary:
                path=Path(temporary)/'weights.pt'
                torch.save(dict(model_config=asdict(a.config),model={k:v.cpu() for k,v in a.state_dict().items()},step=payload['step']),path)
                b,_=load_model(path,'mps')
                with torch.inference_mode(): second=tuple(v.cpu() for v in b(ids,prefix))
                differences=[float((x-y).abs().max()) for x,y in zip(first,second)]
            result=dict(passed=all(d==0 for d in differences),max_difference_by_field=differences,
                        fields=['mixture_logits','means','log_scales'],shapes=[list(v.shape) for v in first],
                        finite=all(bool(torch.isfinite(v).all()) for v in first),step=payload['step'],
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        device='mps',parameters=a.parameter_count(),prefix_source='CPU Gaussian RNG seed7123; no image/data',
                        note='Main-model distribution save/reload; all80 fixed-seed full generation replay checked separately')
            np.savez('evidence/p1/main_roundtrip_parameters.npz',prefix=prefix.cpu().numpy(),
                     logits=first[0].numpy(),means=first[1].numpy(),log_scales=first[2].numpy())
            Path('evidence/p1/main_roundtrip.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
            if not result['passed'] or not result['finite']: raise AssertionError(result)
    finally: budget.close()


if __name__=='__main__': main()
