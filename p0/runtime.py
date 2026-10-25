"""Shared runtime/serialization, with no access to training data or renderer."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import torch
from model import Config, InverseViT


def read_config(path):
    return Config(**json.loads(Path(path).read_text()))


def device_for(name):
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("CPU fallback must be disabled for these experiments")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; no silent CPU fallback")
    if name not in ("cpu", "mps"):
        raise ValueError("This project is authorized for local CPU/MPS only")
    return torch.device(name)


def synchronize(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def atomic_save(obj, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def save_weights(model, path):
    atomic_save({"config": asdict(model.config), "model": model.state_dict()}, path)


def load_model(path, device="cpu"):
    target = device_for(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = InverseViT(Config(**state["config"]))
    model.load_state_dict(state["model"], strict=True)
    return model.to(target).eval()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    return {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(), "mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
            "cpu_threads": torch.get_num_threads()}
