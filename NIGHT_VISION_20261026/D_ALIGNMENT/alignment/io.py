"""Hash-locked files and atomic commit records; no device or network operations."""
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def locked_file(spec, root):
    path = (Path(root) / spec["path"]).resolve()
    expected = spec["sha256"]
    if len(expected) != 64 or sha(path) != expected:
        raise ValueError(f"Missing or changed immutable artifact: {path}")
    return path


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


class Budget:
    def __init__(self, seconds):
        if not 0 < seconds <= 3600:
            raise ValueError("Run wall-clock limit must be 1..3600 seconds")
        self.start, self.seconds = time.monotonic(), seconds

    def check(self):
        if time.monotonic() - self.start >= self.seconds:
            raise TimeoutError("Run limit reached; partial evidence cannot be called a matched pair")


def save_checkpoint(engine, optimizer, path, step, run_identity, extra=None):
    import torch
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    # Store ONLY updated parameters. Frozen backbone blocks, E and D are reloaded
    # from pinned inputs. Includes tied embedding/head once via named_parameters.
    state = {name: p.detach().cpu().contiguous().clone()
             for name, p in engine.named_parameters() if p.requires_grad}
    save_file(state, str(path / "trainable.safetensors"))
    rng = {"cpu": torch.get_rng_state()}
    if next(engine.parameters()).device.type == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save({"optimizer": optimizer.state_dict(), "rng": rng}, path / "optimizer.pt")
    write_json(path / "checkpoint.json", {"version": "0.1.0", "step": step,
               "run_identity": run_identity, "extra": extra or {},
               "trainable_sha256": sha(path / "trainable.safetensors"),
               "optimizer_sha256": sha(path / "optimizer.pt")})


def restore_checkpoint(engine, optimizer, path, run_identity, restore_rng=True):
    import torch
    from safetensors.torch import load_file
    path = Path(path)
    meta = read_json(path / "checkpoint.json")
    if meta["version"] != "0.1.0" or meta["run_identity"] != run_identity:
        raise ValueError("Checkpoint version or run identity mismatch")
    if sha(path / "trainable.safetensors") != meta["trainable_sha256"] or sha(path / "optimizer.pt") != meta["optimizer_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    state = load_file(str(path / "trainable.safetensors"))
    expected = {n for n, p in engine.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Checkpoint trainable set changed")
    params = dict(engine.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)
    saved = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    if optimizer is not None:
        optimizer.load_state_dict(saved["optimizer"])
    if restore_rng:
        torch.set_rng_state(saved["rng"]["cpu"])
        if "cuda" in saved["rng"]:
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
    return meta
