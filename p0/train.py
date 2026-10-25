"""Bounded, logged local FP32 MSE training. Last checkpoint is always evaluated."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from data import load_dataset
from model import InverseViT
from runtime import atomic_save, device_for, environment, read_config, save_weights, sha256, synchronize
from sample import infer, save_images
from metrics import reconstruction


def train(a):
    torch.set_num_threads(a.cpu_threads)
    device = device_for(a.device)
    config = read_config(a.config)
    out, dataset = Path(a.out), Path(a.data)
    if a.resume:
        if not out.is_dir():
            raise ValueError("Resume output directory missing")
    else:
        out.mkdir(parents=True, exist_ok=False)
    raw = load_dataset(dataset)
    data_hash = sha256(dataset / "dataset.npz")
    torch.manual_seed(a.seed)
    model = InverseViT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, foreach=False)
    generator = torch.Generator().manual_seed(a.seed+1)
    start_step, prior_seconds = 0, 0.
    if a.resume:
        saved = torch.load(a.resume, map_location="cpu", weights_only=True)
        if saved["config"] != asdict(config) or saved["data_sha256"] != data_hash or saved["seed"] != a.seed:
            raise ValueError("Resume config/data/seed mismatch")
        if saved["batch_size"] != a.batch_size or saved["lr"] != a.lr:
            raise ValueError("Resume batch/lr mismatch; no implicit tuning")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["sampler_rng"])
        start_step, prior_seconds = saved["step"], saved["training_seconds"]
    conditions = torch.from_numpy(raw["conditions"]).to(device)
    targets = torch.from_numpy(raw["images"]).to(device)
    if not a.resume:
        initial = infer(model.eval(), raw["conditions"])
        save_images(initial, out / "before")
        (out / "initial_metrics.json").write_text(json.dumps(reconstruction(initial, raw["images"], raw["masks"]), indent=2)+"\n")
        save_weights(model, out / "initial_weights.pt")
    plan = {"config": asdict(config), "parameters": model.parameter_count(), "seed": a.seed,
            "device": str(device), "dtype": "float32", "lr": a.lr, "weight_decay": .01,
            "batch_size": a.batch_size, "max_steps": a.steps, "max_seconds_this_invocation": a.max_seconds,
            "data": str(dataset), "data_sha256": data_hash, "count": len(conditions),
            "stop_target_mse": a.stop_mse, "min_steps": a.min_steps, "resume": a.resume,
            "environment": environment(), "pid": os.getpid(), "stop_command": f"kill -TERM {os.getpid()}",
            "selection": "last step; never best-looking checkpoint", "initial_and_final_evaluation_seconds_excluded": True,
            "periodic_evaluation_and_checkpoint_time_included": True}
    invocation = f"invocation_{start_step:06d}"
    (out / f"{invocation}.json").write_text(json.dumps(plan, indent=2)+"\n")
    (out / "active_process.json").write_text(json.dumps({"pid": os.getpid(), "state": "running", "stop_command": plan["stop_command"]})+"\n")
    interrupted = []
    def stop(signum, frame):
        interrupted.append(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    synchronize(device)
    started = time.monotonic()
    stop_reason = "max_steps"
    gradient_report = None
    step = start_step
    log = open(out / "loss.jsonl", "a", buffering=1)
    model.train()
    while step < a.steps:
        if interrupted or time.monotonic()-started >= a.max_seconds:
            stop_reason = "signal" if interrupted else "time_limit"
            break
        ids = torch.randperm(len(conditions), generator=generator)[:a.batch_size].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(conditions[ids])
        loss = torch.mean((prediction-targets[ids])**2)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; run failed")
        loss.backward()
        if gradient_report is None:
            groups = {"condition_projection": model.condition_projection, "backbone": model.blocks, "pixel_head": model.pixel_head}
            gradient_report = {}
            for name, group in groups.items():
                grads = [p.grad for p in group.parameters() if p.requires_grad]
                norm = float(torch.sqrt(sum((g*g).sum() for g in grads if g is not None)).item())
                finite = all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
                if not finite or norm <= 0:
                    raise AssertionError(f"Invalid gradients in {name}")
                gradient_report[name] = {"finite": finite, "l2_norm": norm}
            (out / f"{invocation}_gradients.json").write_text(json.dumps(gradient_report, indent=2)+"\n")
        optimizer.step()
        step += 1
        if step == start_step+1 or step % a.log_every == 0:
            synchronize(device)
            row = {"step": step, "batch_mse_before_update": float(loss.item()),
                   "training_seconds": prior_seconds+time.monotonic()-started}
            if step % a.eval_every == 0:
                model.eval()
                pred = infer(model, raw["conditions"])
                model.train()
                row["full_train_mse_after_update"] = float(np.mean((pred-raw["images"])**2))
            log.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
            if step >= a.min_steps and row.get("full_train_mse_after_update", float("inf")) <= a.stop_mse:
                stop_reason = "predeclared_train_mse_target"
                break
        if step % a.checkpoint_every == 0:
            atomic_save({"config": asdict(config), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "step": step, "training_seconds": prior_seconds+time.monotonic()-started,
                         "seed": a.seed, "sampler_rng": generator.get_state(), "data_sha256": data_hash,
                         "batch_size": a.batch_size, "lr": a.lr}, out / "checkpoint.pt")
    synchronize(device)
    elapsed = prior_seconds+time.monotonic()-started
    atomic_save({"config": asdict(config), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "step": step, "training_seconds": elapsed, "seed": a.seed, "sampler_rng": generator.get_state(),
                 "data_sha256": data_hash, "batch_size": a.batch_size, "lr": a.lr}, out / "checkpoint.pt")
    save_weights(model.eval(), out / "weights.pt")
    final = infer(model, raw["conditions"])
    save_images(final, out / "after")
    result = {"status": "COMPLETED", "stop_reason": stop_reason, "step": step, "training_seconds": elapsed,
              "parameters": model.parameter_count(), "device": str(device), "dtype": "float32",
              "final": reconstruction(final, raw["images"], raw["masks"]),
              "weights_sha256": sha256(out / "weights.pt"), "data_sha256": data_hash}
    (out / "training_result.json").write_text(json.dumps(result, indent=2)+"\n")
    (out / "active_process.json").write_text(json.dumps({"pid": os.getpid(), "state": "finished"})+"\n")
    log.close()
    print(json.dumps(result), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["cpu", "mps"], required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--stop-mse", type=float, default=.001)
    p.add_argument("--min-steps", type=int, default=300)
    p.add_argument("--resume")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("Positive finite resource limits required")
    train(args)
