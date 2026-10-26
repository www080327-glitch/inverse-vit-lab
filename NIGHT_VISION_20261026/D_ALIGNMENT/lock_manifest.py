#!/usr/bin/env python3
"""A's metadata-only lock step. Does not instantiate a model or contact hardware."""
import argparse
from pathlib import Path
from alignment.io import read_json, write_json, sha, digest, locked_file
from alignment.data import snapshot


def lock(args):
    c = read_json(args.template)
    if args.output.exists():
        raise ValueError("Refuse to overwrite a locked manifest")
    if not args.attest_original:
        raise ValueError("A must attest that the model is official original Qwen, with no P2 updates")
    model = args.model.resolve()
    c["model"]["path"] = str(model)
    c["model"]["files"] = {p.relative_to(model).as_posix(): sha(p) for p in sorted(model.rglob("*"))
        if p.is_file() and p.suffix in (".safetensors", ".json", ".jinja", ".txt", ".model")}
    if "config.json" not in c["model"]["files"] or not any(s.endswith(".safetensors") for s in c["model"]["files"]):
        raise ValueError("Complete local official model required")
    data, _, _ = snapshot(args.snapshot)
    if data["kind"] != "joined" or data["counts"] != {"train": 640, "val": 80}:
        raise ValueError("A complete immutable joined 640/80 snapshot is required")
    c["representation"] = data["representation"]
    export = read_json(args.decoder_export)
    if export["schema"] != "alignment.decoder_export.v1" or export["status"] != "complete":
        raise ValueError("D must provide a completed immutable export, never its active training checkpoint")
    if export["representation"] != c["representation"]:
        raise ValueError("D export and target feature representations differ")
    d = dict(export["decoder"])
    for key in ("weights", "source"):
        d[key] = {**d[key], "path": str(locked_file(d[key], args.decoder_export.parent))}
    d["dependency_sources"] = [{**s, "path": str(locked_file(s, args.decoder_export.parent))}
                               for s in d["dependency_sources"]]
    n = dict(export["normalization"])
    if n["kind"] == "channel_standardize":
        n["statistics"] = {**n["statistics"], "path": str(locked_file(n["statistics"], args.decoder_export.parent))}
        if n["statistics_split_sha256"] != data["split_sha256"] or n["statistics_split"] != "train":
            raise ValueError("D training-statistics split mismatch")
    d["representation_sha256"] = digest(c["representation"])
    d["normalization_sha256"] = digest(n)
    if d["input_space"] != n["decoder_space"]:
        raise ValueError("D export numeric-space mismatch")
    c["decoder"], c["normalization"] = d, n
    c["decoder_export"] = {"path": str(args.decoder_export.resolve()), "sha256": sha(args.decoder_export)}
    c["encoder"]["representation_sha256"] = digest(c["representation"])
    c["encoder"]["weights"] = None if args.encoder_weights is None else {
        "path": str(args.encoder_weights.resolve()), "sha256": sha(args.encoder_weights)}
    for name, path in (("snapshot", args.snapshot),):
        c["data"][name] = {"path": str(path.resolve()), "sha256": sha(path)}
    for name, path in (("train", args.replay_train), ("eval", args.replay_eval)):
        c["replay"][name] = {"path": str(path.resolve()), "sha256": sha(path)}
    c["local_python_paths"] = [str(p.resolve()) for p in args.python_path]
    c["training"]["scope"] = args.scope
    c["resource_lock"] = {"executor": "A", "status": "locked", "scope": args.scope,
                          "note": "Same scope, original M, data, normalization and optimizer schedule for both heads"}
    write_json(args.output, c)
    return {"locked_manifest": str(args.output), "sha256": sha(args.output),
            "scope": args.scope, "models_loaded": 0, "training_executed": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executor", required=True, choices=["A"])
    p.add_argument("--scope", required=True, choices=["full", "last4"])
    for name in ("model", "snapshot", "decoder-export", "replay-train", "replay-eval", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--template", type=Path, default=Path(__file__).parent / "configs" / "paired_short.example.json")
    p.add_argument("--encoder-weights", type=Path)
    p.add_argument("--python-path", type=Path, action="append", default=[])
    p.add_argument("--attest-original", action="store_true")
    args = p.parse_args()
    print(lock(args))


if __name__ == "__main__":
    main()
