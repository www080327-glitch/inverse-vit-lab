"""Offline CPU smoke checks for the three model paths; no training or downloads."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "p0"))
import torch
from model import Config as P0Config, InverseViT, patchify as p0_patchify, unpatchify as p0_unpatchify

sys.path.insert(0, str(ROOT))
from p1.model import Config as P1Config, PixelAR, patchify as p1_patchify, unpatchify as p1_unpatchify
from p2_qwen.probability import patchify as p2_patchify, unpatchify as p2_unpatchify, nll


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    image48 = torch.randn(1, 3, 48, 48)
    image32 = torch.randn(1, 3, 32, 32)
    check("P0 patch round trip", torch.equal(p0_unpatchify(p0_patchify(image48)), image48))
    check("P1 patch round trip", torch.equal(p1_unpatchify(p1_patchify(image32)), image32))
    check("P2 patch round trip", torch.equal(p2_unpatchify(p2_patchify(image32)), image32))

    p0 = InverseViT(P0Config(name="cpu-smoke", condition_dim=64, image_size=48, patch_size=12,
                              dim=64, depth=1, heads=4, ff_dim=128)).eval()
    with torch.inference_mode():
        y0 = p0(torch.randn(1, 64))
    check("P0 tiny model output finite", y0.shape == (1, 3, 48, 48) and torch.isfinite(y0).all().item())

    p1 = PixelAR(P1Config(dim=32, depth=1, heads=4, ff_dim=64, dropout=0.0)).eval()
    with torch.inference_mode():
        params = p1(torch.tensor([2]), torch.randn(1, 5, 48))
    check("P1 causal model head finite", len(params) == 3 and all(torch.isfinite(v).all().item() for v in params))

    logits = torch.zeros(1, 2, 2)
    means = torch.zeros(1, 2, 2, 48)
    log_scales = torch.full_like(means, -0.7)
    loss = nll((logits, means, log_scales), torch.zeros(1, 2, 48))
    check("P2 continuous probability loss finite", torch.isfinite(loss).item())
    print("Offline CPU smoke checks passed; this does not reproduce any trained result.")


if __name__ == "__main__":
    main()
