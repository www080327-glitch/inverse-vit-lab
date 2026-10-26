"""CPU limits must be set before importing numerical libraries."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import json
import platform
import sys
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_default_device('cpu')
torch.use_deterministic_algorithms(True)
ROOT = Path(__file__).resolve().parent

def initialize(seed=20261026):
    # torch.manual_seed also calls accelerator seed helpers. Seed CPU only.
    torch.random.default_generator.manual_seed(seed)
    np.random.seed(seed)
    return dict(seed=seed, device='cpu', torch=torch.__version__, numpy=np.__version__,
                python=sys.version, platform=platform.platform(), intraop_threads=2,
                interop_threads=1, accelerator_tensor_compute=False,seed_api='CPU default_generator only')

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
