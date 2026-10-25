"""Independent inference CLI: weights + conditions ONLY; no data module import."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from runtime import load_model, sha256


@torch.inference_mode()
def infer(model, conditions, batch_size=16):
    device = next(model.parameters()).device
    values = torch.as_tensor(conditions, dtype=torch.float32)
    if values.ndim != 2 or values.shape[1] != model.config.condition_dim or not torch.isfinite(values).all():
        raise ValueError("Expected finite B x condition_dim array")
    return torch.cat([model(v.to(device)).cpu() for v in values.split(batch_size)]).numpy()


def save_images(values, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "outputs.npy", values)
    for i, value in enumerate(values):
        Image.fromarray((np.clip(value.transpose(1, 2, 0), 0, 1)*255).round().astype("uint8")).save(out / f"{i:03d}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--conditions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    a = p.parse_args()
    model = load_model(a.weights, a.device)
    outputs = infer(model, np.load(a.conditions, allow_pickle=False))
    save_images(outputs, a.out)
    receipt = {"weights_sha256": sha256(a.weights), "conditions_sha256": sha256(a.conditions),
               "shape": list(outputs.shape), "device": a.device, "target_access": False,
               "range": [float(outputs.min()), float(outputs.max())],
               "out_of_range_fraction": float(((outputs < 0) | (outputs > 1)).mean())}
    (Path(a.out) / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))
