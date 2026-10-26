# C15: conditional multimodal continuous heads

**Primary deliverable:** [extended/HANDOFF_ZH.md](extended/HANDOFF_ZH.md), [extended/RESULT.json](extended/RESULT.json), [extended/C15_DELIVERY.zip](extended/C15_DELIVERY.zip). The initial 1800-step study below is retained unchanged, including poor or collapsed distributions. A matched 9000-step extension of every method and seed was fixed after inspecting that initial evidence; cumulative training includes both stages and remains below 1800 seconds.

Compare deterministic MSE, shared-component diagonal GMM, full-covariance GMM,
and an EAR-style noise-conditioned MLP on frozen 1D/2D/8D distributions.

Read [HANDOFF_ZH.md](HANDOFF_ZH.md) for measured conclusions after execution,
[SOURCES.md](SOURCES.md) for paper/code cross-checks, and
[RESULT.json](RESULT.json) for machine-readable results.
[results/runs.json](results/runs.json) preserves every seed/condition and failure.
[C15_DELIVERY.zip](C15_DELIVERY.zip) contains all source and output artifacts.

Run from this directory:
```bash
python -m venv /tmp/c15-env
. /tmp/c15-env/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python run.py
```

The hosted run uses standard public GitHub Actions Linux, no accelerator.
CPU affinity is constrained to two CPUs and numerical-library threads to two.
Training-loop cumulative budget is 1800 seconds, fixed 1800 steps/model,
two seeds; no checkpoint tuning or retries. Existing budget records prevent a
silent repeat. Create an explicitly separate output location for a new study.
No large weights, images or datasets are downloaded.

Portable API:
```python
from heads import Head
import torch
h = Head('ear', d=8, k=4)
x = h.sample(condition=torch.tensor([0, 1]), n=128,
             generator=torch.Generator().manual_seed(42))
assert x.shape == (2,128,8)
```

Outputs are untrained until run.py trains them. No trained checkpoints or raw
datasets are committed, following repository contribution rules. Reproducibility
comes from exact code/config, analytic specifications, dataset seeds, versions,
and complete metrics/figures; no claims of bitwise cross-platform identity.
