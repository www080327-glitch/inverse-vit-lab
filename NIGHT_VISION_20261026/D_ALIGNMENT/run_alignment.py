#!/usr/bin/env python3
"""CPU development by default; real execution is explicitly reserved for A."""
import argparse
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                  TOKENIZERS_PARALLELISM="false", PYTORCH_ENABLE_MPS_FALLBACK="0")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    test = sub.add_parser("cpu-test", help="Random tensors only, 1 thread, <=300 s")
    test.add_argument("--output", type=Path, default=Path(__file__).parent / "evidence" / "cpu_checks.json")
    join = sub.add_parser("join-snapshots")
    join.add_argument("--images", required=True, type=Path)
    join.add_argument("--descriptions", required=True, type=Path)
    join.add_argument("--output", required=True, type=Path)
    score = sub.add_parser("score-attributes")
    for flag in ("queue", "judgments", "output"):
        score.add_argument("--" + flag, type=Path, required=True)
    for command in ("validate", "run-pair", "train", "generate", "decode"):
        a = sub.add_parser(command)
        a.add_argument("--manifest", type=Path, required=True)
        if command in ("run-pair", "train", "generate"):
            a.add_argument("--executor", required=True, choices=["A"])
            a.add_argument("--device", required=True, choices=["cuda"])
        if command in ("train", "generate"):
            a.add_argument("--interface", required=True, choices=["direct", "mlp"])
        if command in ("train", "generate", "run-pair", "decode"):
            a.add_argument("--output", required=True, type=Path)
        if command == "train":
            a.add_argument("--resume", type=Path)
        if command == "generate":
            a.add_argument("--checkpoint", required=True, type=Path)
            a.add_argument("--request", required=True, type=Path, help="Only mode,text; image allowed only for redraw")
        if command == "decode":
            a.add_argument("--features", required=True, type=Path)
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.command == "cpu-test":
        import signal
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU tests exceeded 300 seconds")))
        signal.alarm(300)
        from tests.cpu_checks import run
        result = run(args.output)
        signal.alarm(0)
        print(result)
        return
    if args.command == "join-snapshots":
        from alignment.data import join_snapshots
        print(join_snapshots(args.images, args.descriptions, args.output))
        return
    if args.command == "score-attributes":
        from alignment.data import score_attributes
        print(score_attributes(args.queue, args.judgments, args.output))
        return
    from alignment.runner import manifest, validate_inputs, require_A, run_pair, run_one, independent_decode
    from alignment.io import read_json, write_json, digest, restore_checkpoint, Budget
    c, root = manifest(args.manifest)
    # A may configure local decoder package locations explicitly; never hardcode
    # another session's checkpoint or mutate any external directory.
    for path in c["local_python_paths"]:
        sys.path.insert(0, str((root / path).resolve()))
    if args.command == "validate":
        data, a, b = validate_inputs(c, root)
        print({"status": "INPUTS_VALIDATED_NO_MODEL_LOADED", "counts": data.commit["counts"],
               "instruction_train": len(a), "instruction_eval": len(b)})
    elif args.command == "decode":
        independent_decode(c, root, args.features, args.output)
    else:
        require_A(args.executor, args.device)
        if args.command == "run-pair":
            run_pair(c, root, args.output, args.device)
        elif args.command == "train":
            run_one(c, root, args.interface, args.output, args.device, args.resume)
        else:
            from alignment.runner import build, save_prediction
            from alignment.core import Condition
            request = read_json(args.request)
            # Extra keys fail before loading the real model. No generic kwargs,
            # target feature field, target image field, image IDs or retrieval.
            condition = Condition(**request)
            engine, _ = build(c, root, args.interface, args.device)
            restore_checkpoint(engine, None, args.checkpoint,
                               digest({"manifest": c, "interface": args.interface}), restore_rng=False)
            if args.output.exists():
                raise ValueError("Refuse to replace generated features")
            z = engine.generate(condition, Budget(c["limits"]["wall_seconds_per_interface"]).check)
            save_prediction(engine, z, args.output)


if __name__ == "__main__":
    main()
