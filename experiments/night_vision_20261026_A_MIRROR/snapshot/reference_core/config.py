from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MirrorConfig:
    name: str
    input_dim: int = 1024
    dim: int = 256
    depth: int = 2
    heads: int = 4
    ff_dim: int = 1024
    patch_size: int = 16
    merge_size: int = 2
    channels: int = 3
    position: str = "fixed_2d_sincos_xy"
    initialization: str = "random_pytorch_default_no_encoder_copy"

    def __post_init__(self):
        if min(self.input_dim, self.dim, self.depth, self.heads, self.ff_dim) <= 0:
            raise ValueError("Dimensions must be positive")
        if self.dim % self.heads or self.dim % 4:
            raise ValueError("Dimension must be divisible by heads and four")
        if (self.input_dim, self.patch_size, self.merge_size, self.channels) != (1024, 16, 2, 3):
            raise ValueError("This prototype requires the audited official image geometry")
        if self.position != "fixed_2d_sincos_xy" or self.initialization != "random_pytorch_default_no_encoder_copy":
            raise ValueError("Unsupported position or initialization contract")

    def to_dict(self):
        return asdict(self)


def mirror_full():
    """Configuration only. No parameter storage is allocated here."""
    return MirrorConfig(name="mirror_full", dim=768, depth=12, heads=12, ff_dim=3072)


def mirror_tiny():
    return MirrorConfig(name="mirror_tiny")


def parameter_estimate(c):
    """Exact algebra for our specified Linear/LayerNorm/SDPA architecture."""
    d, f = c.dim, c.ff_dim
    parts = {
        "learned_unmerge": (c.input_dim + 1) * c.merge_size**2 * d,
        "transformer_blocks": c.depth * (4*d*d + 2*d*f + 9*d + f),
        "final_norm": 2*d,
        "pixel_projection": (d + 1) * c.patch_size**2 * c.channels,
        "fixed_position": 0,
    }
    total = sum(parts.values())
    return {"components": parts, "total": total, "fp32_parameter_bytes": total*4,
            "scope": "parameter storage only; excludes activations, gradients and optimizer"}
