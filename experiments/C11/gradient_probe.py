"""Optional selected-layer PyTorch probe. Not imported by the dependency-free toy runner.
No automatic registration, no optimizer step, and no full-model default.
Call before training backward on a sparse diagnostic step. Requires PyTorch.
"""
import math
import time

def selected_gradient_stats(model, loss_by_task, *, parameter_names, step,
                            every=100, loss_weights=None, max_elements=100_000):
    """Compare task gradients on explicitly selected shared parameters.

    loss_by_task: dict[str, scalar Tensor], unscaled and already reduced with
    documented masks/denominators. Separate graphs at the same parameter state.
    Returns only JSON-compatible scalars, never serializes gradient tensors.
    The caller owns RNG isolation, diagnostic batches and graph lifetime.
    Do not pass GradScaler-scaled losses. Do not run under no_grad/inference_mode.
    DDP statistics are rank-local; aggregate only with explicit collective logic.
    """
    if every < 1:
        raise ValueError("every must be positive")
    if step % every:
        return None
    names = list(parameter_names)
    if not names:
        raise ValueError("explicit shared parameter names required")
    if len(set(names)) != len(names):
        raise ValueError("duplicate parameter names")
    params = dict(model.named_parameters())
    selected = [params[name] for name in names]
    if any(not p.requires_grad for p in selected):
        raise ValueError("selected parameters must require gradients")
    count = sum(p.numel() for p in selected)
    if count > max_elements:
        raise ValueError("selected gradient memory cap exceeded")
    weights = loss_weights or {}
    import torch  # Optional dependency; no install/download side effect.
    t0 = time.perf_counter()
    grads, info = {}, {}
    for task, loss in loss_by_task.items():
        if loss.ndim != 0 or not loss.requires_grad:
            raise ValueError("each task loss must be a differentiable scalar")
        gs = torch.autograd.grad(loss, selected, retain_graph=True,
                                 create_graph=False, allow_unused=True)
        # .grad buffers are not changed. Store only selected layers, briefly.
        gs = [None if g is None else g.detach().float() for g in gs]
        grads[task] = gs
        sq = sum(float(g.square().sum()) for g in gs if g is not None)
        n = math.sqrt(max(0.0, sq))
        weight = float(weights.get(task, 1.0))
        info[task] = dict(loss_raw=float(loss.detach()), weight=weight,
                         shared_norm_raw=n, shared_norm_weighted=abs(weight)*n,
                         unused_selected_parameters=sum(g is None for g in gs))
    pairs = []
    tasks = list(grads)
    for i, a in enumerate(tasks):
        for b in tasks[i+1:]:
            dot = sum(float(ga.mul(gb).sum()) for ga, gb in
                      zip(grads[a], grads[b]) if ga is not None and gb is not None)
            den = info[a]["shared_norm_raw"] * info[b]["shared_norm_raw"]
            pairs.append(dict(task_a=a, task_b=b, dot_raw=dot,
                              cosine_raw=None if den < 1e-20 else dot/den))
    del grads
    # CUDA kernels/synchronization can make wall time attribution imprecise.
    return dict(step=int(step), selected_names=names, selected_elements=count,
                seconds=time.perf_counter()-t0, tasks=info, pairs=pairs,
                scope="selected parameters; rank-local; not update attribution")
