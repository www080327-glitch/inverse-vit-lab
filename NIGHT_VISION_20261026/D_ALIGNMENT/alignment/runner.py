"""A-only real-model runner. This module never opens SSH or manages other jobs."""
import gc
import json
import os
from pathlib import Path
import signal
import time
import torch
from . import VERSION
from .core import Alignment, MODES, SEQUENCE_PROTOCOL, fingerprint
from .backends import QwenBackend, load_contract, load_decoder, verify_model_files
from .data import Dataset, attribute_items, replay_sets
from .io import Budget, digest, locked_file, read_json, restore_checkpoint, save_checkpoint, sha, write_json, write_jsonl


def manifest(path):
    path = Path(path).resolve()
    c = read_json(path)
    if c["schema"] != "alignment.manifest.v1" or c["version"] != VERSION:
        raise ValueError("Unsupported manifest schema/version")
    if c["geometry"] != {"tokens": 64, "width": 1024, "pre_grid_thw": [1, 16, 16], "post_grid_hw": [8, 8]}:
        raise ValueError("v1 requires exactly 64 post-merger positions of width 1024")
    t = c["training"]
    if t["scope"] not in ("full", "last4") or t["mode"] not in MODES:
        raise ValueError("Unsupported scope/conditioning mode")
    if c["resource_lock"]["status"] == "locked" and c["resource_lock"]["scope"] != t["scope"]:
        raise ValueError("Resource lock and actual update scope differ")
    if not 1 <= t["steps"] <= 200 or t["batch_size"] != 1 or t["gradient_accumulation"] != 1:
        raise ValueError("v1 supports 1..200 optimizer steps, batch=1, accumulation=1")
    if not 1 <= c["limits"]["wall_seconds_per_interface"] <= 3600:
        raise ValueError("Per-interface limit 1..3600 seconds")
    if not t["steps"] <= c["limits"]["maximum_steps_per_interface"] <= 200:
        raise ValueError("Requested optimizer steps exceed A's declared step limit")
    if c["limits"]["source_tokens"] != 2048 or c["limits"]["output_features"] != 64:
        raise ValueError("v1 requires fixed source/output bounds 2048/64")
    if c["data"]["counts"] != {"train": 640, "val": 80}:
        raise ValueError("Formal B snapshot must preserve the fixed 640/80 split")
    if c["evaluation"]["modes"] != list(MODES) or not 1 <= c["evaluation"]["val_samples"] <= 80:
        raise ValueError("Evaluate all three modes on the same predeclared validation prefix")
    if c["replay"]["every"] < 1 or c["replay"]["weight"] <= 0 or not 1 <= c["replay"]["max_new_tokens"] <= 256:
        raise ValueError("Independent instruction replay and bounded evaluation are required")
    if not 0 <= t["pixel_loss_weight"] <= 1 or not 0 < t["lr"] <= .01 or t["weight_decay"] < 0:
        raise ValueError("Invalid optimization/loss settings")
    if not 1 <= t["checkpoint_every"] <= 200 or not 0 < t["max_grad_norm"] <= 100:
        raise ValueError("Invalid checkpoint interval or gradient clipping")
    if c["decoder"]["input_space"] != c["normalization"]["decoder_space"]:
        raise ValueError("D input numeric contract mismatch")
    if c["decoder"]["normalization_sha256"] != digest(c["normalization"]):
        raise ValueError("D normalization contract must be explicitly hash-locked")
    if c["encoder"]["representation_sha256"] != digest(c["representation"]):
        raise ValueError("E representation mismatch")
    if c["decoder"]["representation_sha256"] != digest(c["representation"]):
        raise ValueError("D representation mismatch")
    r = c["representation"]
    if (r["repository"], r["revision"]) != (c["model"]["repository"], c["model"]["revision"]):
        raise ValueError("M and visual feature provenance mismatch")
    if r["feature_kind"] != "post_merger_pooler_output" or r["order"] != "merged_row_major":
        raise ValueError("Only raw official post-merger row-major targets are supported")
    if r["encoder_dtype"] != "float32" or r["preprocessing"] != "square_rgb_256_pil_bicubic_official_processor":
        raise ValueError("v1 source preprocessing/encoder precision must match the cached target contract")
    return c, path.parent


def validate_inputs(c, root):
    if c["resource_lock"]["executor"] != "A" or c["resource_lock"]["status"] != "locked":
        raise ValueError("A must first lock resources and scope in the manifest")
    data = Dataset(c["data"]["snapshot"], root, c["representation"], c["data"]["counts"])
    locked_file(c["decoder_export"], root)
    model_dir = verify_model_files(c["model"], root)
    if sha(model_dir / "preprocessor_config.json") != c["representation"]["processor_sha256"]:
        raise ValueError("E/D processor provenance differs from official M processor files")
    replay_train, replay_eval = replay_sets(c["replay"], root)
    if c["normalization"]["kind"] == "channel_standardize":
        if c["normalization"]["statistics_split_sha256"] != data.commit["split_sha256"]:
            raise ValueError("D statistics must use the same authoritative train split")
    # Hash every external data artifact before a real run; no silent changing files.
    for row in data.rows:
        for key in ("image", "source_features", "rgb_target"):
            locked_file(row[key], data.root)
    for key in ("weights", "source"):
        locked_file(c["decoder"][key], root)
    for spec in c["decoder"]["dependency_sources"]:
        locked_file(spec, root)
    if c["encoder"].get("weights"):
        locked_file(c["encoder"]["weights"], root)
    load_contract(c["normalization"], root)  # tiny statistics only; no model
    return data, replay_train, replay_eval


def require_A(executor, device):
    if executor != "A" or device != "cuda":
        raise ValueError("Real-model commands require --executor A --device cuda (ROCm uses torch cuda). CPU/MPS real runs are blocked")
    if not torch.cuda.is_available():
        raise RuntimeError("A's accelerator is unavailable; no fallback")


def build(c, root, interface, device):
    torch.manual_seed(c["seed"])
    torch.cuda.manual_seed_all(c["seed"])
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[c["training"]["dtype"]]
    backbone = QwenBackend(c["model"], c["encoder"], root, device, dtype,
                           c["training"]["scope"], c["training"]["gradient_checkpointing"], c["representation"])
    # Fingerprint the identical official M BEFORE the MLP exists. E and D are
    # independently fingerprinted below. No interface-specific M initialization.
    initial = fingerprint(backbone.qwen.model.language_model)
    decoder = load_decoder(c["decoder"], c["representation"], root, device)
    contract = load_contract(c["normalization"], root).to(device)
    engine = Alignment(backbone, decoder, contract, interface, c["mlp_width"]).to(device)
    # FP32 master parameters and Adam states; optional BF16 autocast inside M.
    return engine, initial


def instruction_eval(engine, rows, output, max_tokens, budget):
    engine.eval()
    results = []
    for row in rows:
        budget.check()
        response = engine.backbone.instruction_response(row, max_tokens)
        # Responses + original rubric are the before/after artifact. A reviewer
        # must judge instruction following; no CE-only retention assertion.
        results.append({"id": row["id"], "prompt": row["prompt"], "response": response,
                        "rubric": row["rubric"], "instruction_judgment": "PENDING_REVIEW"})
    write_jsonl(output, results)


def save_prediction(engine, z, path):
    from safetensors.torch import save_file
    save_file({"features_model_space": z.detach().cpu().contiguous(),
               "features_raw": engine.contract.raw(z).detach().cpu().contiguous()}, str(path),
              metadata={"sequence_protocol": json.dumps(SEQUENCE_PROTOCOL),
                        "contract_buffers_sha256": fingerprint(engine.contract),
                        "decoder_space": engine.contract.decoder_space})
    write_json(str(path) + ".json", {"sequence_protocol": SEQUENCE_PROTOCOL,
               "features_sha256": sha(path), "interface": engine.interface, "tokens": 64,
               "generation": "free_autoregressive_no_true_history"})


def evaluate(engine, data, c, output, budget):
    from PIL import Image
    from safetensors.torch import save_file
    engine.eval()
    output.mkdir()
    metrics, queue = [], []
    device = next(engine.parameters()).device
    for mode in c["evaluation"]["modes"]:
        for i, row in enumerate(data.splits["val"][:c["evaluation"]["val_samples"]]):
            budget.check()
            # References are loaded only AFTER all 64 predictions are finished.
            condition = data.condition(row, mode)
            z = engine.generate(condition, budget.check)
            with torch.no_grad():
                prediction = engine.decode(z).float()
            raw, rgb = data.target(row, device, include_rgb=True)
            if prediction.shape != rgb.shape:
                raise ValueError("Decoded and scoring image dimensions mismatch")
            key = f"{mode}_{i:03d}"
            feature_path = output / f"{key}.safetensors"
            save_prediction(engine, z, feature_path)
            pixels = (prediction[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
            png = output / f"{key}.png"
            Image.fromarray(pixels).save(png)
            metrics.append({"key": key, "image_id": row["image_id"], "mode": mode,
                "conditioning": {"text": condition.text, "source_image_allowed": mode == "redraw"},
                "generation": "free_autoregressive_64_no_true_history", "generated_tokens": int(z.shape[1]),
                "feature_mse_free": float((z - engine.contract.normalize(raw)).square().mean()),
                "pixel_mse_auxiliary": float((prediction - rgb).square().mean()),
                "stage": "stage2_source_redraw" if mode == "redraw" else "stage3_original_Qwen_description_bridge" if mode == "description" else "class_word_baseline"})
            attrs = attribute_items(row) if mode != "class_only" else {"subject": row["class_word"]}
            queue.append({"key": key, "image_id": row["image_id"], "mode": mode,
                          "prediction": str(png.resolve()), "prediction_sha256": sha(png),
                          "condition_text": condition.text, "explicit_attributes": attrs,
                          "description_uncertainty": row["description"].get("uncertainty", []) if mode != "class_only" else [],
                          "scores": {k: None for k in attrs}, "judge": None})
            write_json(output / "metrics_partial.json", metrics)
    write_json(output / "metrics.json", {"rows": metrics, "attribute_evaluation": "PENDING_JUDGMENTS",
        "scope": "Deterministic feature prediction. No diversity test. Pixel MSE is auxiliary for description/class_only.",
        "validation_selection": "Fixed first N rows of the predeclared validation snapshot; no quality selection"})
    write_jsonl(output / "attribute_queue.jsonl", queue)


def run_one(c, root, interface, output, device, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = digest({"manifest": c, "interface": interface})
    budget = Budget(c["limits"]["wall_seconds_per_interface"])
    def alarm(signum, frame):
        raise TimeoutError("Per-interface wall-clock limit reached")
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, budget.seconds)
    step, log, engine = 0, [], None
    status = "FAILED_OR_INCOMPLETE"
    try:
        data, replay_train, replay_eval = validate_inputs(c, root)
        engine, initial = build(c, root, interface, device)
        optimizer = torch.optim.AdamW([p for p in engine.parameters() if p.requires_grad],
                    lr=c["training"]["lr"], weight_decay=c["training"]["weight_decay"])
        frozen_before = {"E": fingerprint(engine.backbone.encoder), "D": fingerprint(engine.decoder)}
        trainable = {n: p.numel() for n, p in engine.named_parameters() if p.requires_grad}
        write_json(output / "lock.json", {"run_identity": identity, "manifest": c,
             "interface": interface, "scope": c["training"]["scope"], "initial_M_sha256": initial,
             "sequence_protocol": SEQUENCE_PROTOCOL,
             "frozen_before": frozen_before, "trainable_parameters": trainable,
             "runtime": engine.backbone.runtime, "real_model_training_complete": False})
        if resume:
            meta = restore_checkpoint(engine, optimizer, resume, identity)
            step = meta["step"]
            log = meta["extra"].get("training_log", [])
            # A resumed evaluation must compare against the ORIGINAL baseline,
            # not a mislabeled already-updated model.
            before = meta["extra"]["instruction_before"]
            write_jsonl(output / "instruction_before.jsonl", before)
        else:
            instruction_eval(engine, replay_eval, output / "instruction_before.jsonl",
                             c["replay"]["max_new_tokens"], budget)
        torch.manual_seed(c["seed"] + step)
        torch.cuda.manual_seed_all(c["seed"] + step)
        rows = data.splits["train"]
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(c["seed"])).tolist()
        from .io import jsonl
        before = jsonl(output / "instruction_before.jsonl")
        while step < c["training"]["steps"]:
            budget.check()
            # Explicit optimizer step schedule; same examples, replays, count and
            # order for both interfaces, including after restore.
            row = rows[order[step % len(rows)]]
            torch.manual_seed(c["seed"] + step)
            torch.cuda.manual_seed_all(c["seed"] + step)
            engine.train()
            optimizer.zero_grad(set_to_none=True)
            raw, rgb = data.target(row, device, c["training"]["pixel_loss_weight"] > 0)
            loss, parts = engine.losses(data.condition(row, c["training"]["mode"]), raw, rgb,
                                       c["training"]["pixel_loss_weight"])
            replay = loss.new_zeros(())
            replay_id = None
            if step % c["replay"]["every"] == 0:
                item = replay_train[(step // c["replay"]["every"]) % len(replay_train)]
                replay_id = item["id"]
                replay = engine.backbone.replay_loss(item)
                loss = loss + c["replay"]["weight"] * replay
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            params = [p for p in engine.parameters() if p.requires_grad]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise ValueError("Nonfinite gradient")
            if any(p.grad is not None for p in engine.decoder.parameters()) or any(p.grad is not None for p in engine.backbone.encoder.parameters()):
                raise ValueError("Frozen-module gradient violation")
            upstream = sum(float(p.grad.detach().float().square().sum())
                           for p in engine.backbone.parameters() if p.requires_grad and p.grad is not None)
            if upstream <= 0:
                raise ValueError("M received no gradient")
            torch.nn.utils.clip_grad_norm_(params, c["training"]["max_grad_norm"])
            optimizer.step()
            step += 1
            log.append({"step": step, "image_id": row["image_id"], "replay_id": replay_id,
                        "loss": float(loss.detach()), "replay_ce": float(replay.detach()),
                        "M_gradient_squared_norm": upstream,
                        **{k: float(v.detach()) for k, v in parts.items()}})
            write_json(output / "training_log.json", log)
            if step % c["training"]["checkpoint_every"] == 0 or step == c["training"]["steps"]:
                save_checkpoint(engine, optimizer, output / f"step_{step:04d}", step, identity,
                                {"training_log": log, "instruction_before": before})
        budget.check()
        frozen_after = {"E": fingerprint(engine.backbone.encoder), "D": fingerprint(engine.decoder)}
        if frozen_after != frozen_before:
            raise ValueError("Frozen E/D weights changed")
        evaluate(engine, data, c, output / "evaluation", budget)
        instruction_eval(engine, replay_eval, output / "instruction_after.jsonl",
                         c["replay"]["max_new_tokens"], budget)
        status = "TRAINING_AND_FREE_EVALUATION_COMPLETE_ATTRIBUTES_PENDING"
        write_json(output / "frozen_check.json", {"before": frozen_before, "after": frozen_after, "unchanged": True})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        write_json(output / "status.json", {"status": status, "optimizer_steps": step,
            "expected_steps": c["training"]["steps"], "elapsed_seconds": time.monotonic() - budget.start,
            "real_model_training_complete": step == c["training"]["steps"],
            "evaluation_complete": status.startswith("TRAINING_AND"), "instruction_retention": "UNJUDGED",
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "description_training_performed": c["training"]["mode"] == "description" and step > 0,
            "diversity_validated": False})
    return {"interface": interface, "steps": step, "initial_M_sha256": initial,
            "example_schedule": [{k: r[k] for k in ("image_id", "replay_id")} for r in log],
            "frozen": frozen_before, "identity": identity}


def run_pair(c, root, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "pair_lock.json", {"manifest_sha256": digest(c), "scope": c["training"]["scope"],
               "status": "RUNNING", "interfaces": ["direct", "mlp"]})
    results = []
    for interface in ("direct", "mlp"):
        results.append(run_one(c, root, interface, output / interface, device))
        gc.collect()
        torch.cuda.empty_cache()
    matched = all(results[0][k] == results[1][k] for k in ("steps", "initial_M_sha256", "example_schedule", "frozen"))
    write_json(output / "pair_result.json", {"matched": matched, "results": results,
        "scope": c["training"]["scope"], "manifest_sha256": digest(c),
        "claim": "Paired implementation run only. Interpret stage2/stage3 separately; attribute/instruction review required."})
    if not matched:
        raise ValueError("Interfaces are not a matched experiment")


def independent_decode(c, root, features, output):
    """D and statistics only: no M, E, images, IDs or dataset targets are loaded."""
    from safetensors.torch import load_file, save_file
    payload = load_file(str(features))
    if set(payload) != {"features_model_space", "features_raw"}:
        raise ValueError("Independent decode accepts exported feature-only files")
    contract = load_contract(c["normalization"], root)
    z = payload["features_model_space"]
    if z.ndim != 3 or z.shape[1:] != (64, 1024) or not torch.isfinite(z).all():
        raise ValueError("Complete finite feature sequence required")
    if not torch.allclose(contract.raw(z), payload["features_raw"], atol=1e-5, rtol=1e-5):
        raise ValueError("Exported normalization contract mismatch")
    decoder = load_decoder(c["decoder"], c["representation"], root, "cpu")
    with torch.no_grad():
        image = decoder(contract.for_decoder(z))
    if Path(output).exists():
        raise ValueError("Refuse to replace decode output")
    save_file({"prediction_rgb_unclipped": image.cpu().contiguous()}, str(output))
