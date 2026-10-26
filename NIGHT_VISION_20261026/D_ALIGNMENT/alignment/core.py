"""No model downloads or device selection. All feature predictions use one contract."""
from dataclasses import dataclass
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F

TOKENS, WIDTH = 64, 1024
MODES = ("redraw", "description", "class_only")
SEQUENCE_PROTOCOL = {"start": "<|vision_start|>", "end": "<|vision_end|>",
    "end_emission": "external_frame_after_exactly_64_features_not_learned_stop",
    "positions": "8x8 merged row-major; fixed 2D sincos added to each output input",
    "teacher_forcing": "start->z0; z_i->z_(i+1); input excludes z63"}


@dataclass(frozen=True)
class Condition:
    mode: str
    text: str
    image: str | None = None

    def __post_init__(self):
        if self.mode not in MODES or not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Explicit mode and nonempty text required")
        if (self.mode == "redraw") != (self.image is not None):
            raise ValueError("Only redraw accepts an image; redraw requires a source image")


class FeatureContract(nn.Module):
    """z=(raw-mean)/std for BOTH heads, histories, losses and exported features.

    D can explicitly expect z or raw; the boundary converts exactly once.
    Statistics are supplied from D's immutable training-only artifact, never fitted here.
    """
    def __init__(self, mean=None, std=None, decoder_space="raw"):
        super().__init__()
        mean = torch.zeros(WIDTH) if mean is None else torch.as_tensor(mean).float()
        std = torch.ones(WIDTH) if std is None else torch.as_tensor(std).float()
        if mean.shape != (WIDTH,) or std.shape != (WIDTH,):
            raise ValueError("Expected channel statistics [1024]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Finite means and strictly positive stored effective std required")
        if decoder_space not in ("raw", "model"):
            raise ValueError("decoder_space must be raw or model")
        self.decoder_space = decoder_space
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def normalize(self, raw):
        return (raw.float() - self.mean) / self.std

    def raw(self, z):
        return z.float() * self.std + self.mean

    def for_decoder(self, z):
        return self.raw(z) if self.decoder_space == "raw" else z.float()


def coordinates(width=WIDTH):
    # Fixed, shared 8x8 position code. No new vocabulary or trainable input adapter.
    y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    f = torch.exp(-math.log(10000) * torch.arange(width // 4) / (width // 4))
    xx, yy = x.flatten()[:, None] * f, y.flatten()[:, None] * f
    return torch.cat([xx.sin(), xx.cos(), yy.sin(), yy.cos()], dim=-1)


def freeze(module):
    module.requires_grad_(False)
    module.eval()
    return module


def fingerprint(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        t = value.detach().cpu().contiguous()
        h.update(str((tuple(t.shape), t.dtype)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Alignment(nn.Module):
    """Adapter protocol: prepare(Condition)->prefix; embed(ids); hidden(prefix,suffix).

    Prefix contains only authorized source conditioning. Target z is supplied solely
    to teacher_forced() and loss. generate() has no target argument or dataset access.
    """
    def __init__(self, backbone, decoder, contract, interface, mlp_width=256):
        super().__init__()
        if backbone.hidden_size != WIDTH:
            raise ValueError("direct comparison requires official hidden_size=1024")
        self.backbone = backbone
        self.decoder = freeze(decoder)
        self.contract = contract
        self.interface = interface
        if interface == "direct":
            self.head = nn.Identity()
        elif interface == "mlp":
            self.head = nn.Sequential(nn.Linear(WIDTH, mlp_width), nn.GELU(),
                                      nn.Linear(mlp_width, WIDTH))
        else:
            raise ValueError("Only direct and mlp are supported")
        self.register_buffer("output_coordinates", coordinates())

    def train(self, mode=True):
        super().train(mode)
        self.decoder.eval()
        self.backbone.keep_encoder_frozen()
        return self

    def suffix(self, history):
        """[vision_start+coord0, z0+coord1, ..., z62+coord63].

        Hidden at start predicts z0. Hidden at zi predicts z(i+1).
        z63 is NEVER a training input. End marker is recorded after all 64 outputs.
        """
        if history.ndim != 3 or history.shape[1] > 63 or history.shape[2] != WIDTH:
            raise ValueError("History must be [B,0..63,1024]")
        start = self.backbone.output_start(history.shape[0])
        history = history.to(device=start.device, dtype=start.dtype)
        suffix = torch.cat([start, history], dim=1)
        return suffix + self.output_coordinates[:suffix.shape[1]].to(suffix)

    def teacher_forced(self, condition, raw_targets):
        if raw_targets.shape != (1, TOKENS, WIDTH) or not torch.isfinite(raw_targets).all():
            raise ValueError("v1 uses batch size 1 and finite [1,64,1024] raw targets")
        target = self.contract.normalize(raw_targets.detach())
        prefix = self.backbone.prepare(condition)
        suffix = self.suffix(target[:, :-1])
        hidden = self.backbone.hidden(prefix, suffix)
        return self.head(hidden).float(), target

    @torch.no_grad()
    def generate(self, condition, check=lambda: None):
        self.eval()
        prefix = self.backbone.prepare(condition)
        generated = self.output_coordinates.new_empty((1, 0, WIDTH))
        for _ in range(TOKENS):
            check()
            hidden = self.backbone.hidden(prefix, self.suffix(generated))
            next_z = self.head(hidden[:, -1:]).float()
            if not torch.isfinite(next_z).all():
                raise ValueError("Nonfinite free-generation output")
            generated = torch.cat([generated, next_z], 1)
        return generated

    def decode(self, z):
        if z.ndim != 3 or z.shape[1:] != (TOKENS, WIDTH) or not torch.isfinite(z).all():
            raise ValueError("Decode requires complete finite [B,64,1024]")
        # Deliberately NO no_grad here: frozen D still differentiates its input.
        inputs = self.contract.for_decoder(z)
        parameter = next(self.decoder.parameters(), None)
        if parameter is not None:
            inputs = inputs.to(dtype=parameter.dtype, device=parameter.device)
        return self.decoder(inputs)

    def losses(self, condition, raw_targets, rgb_target=None, pixel_weight=0.):
        pred, target = self.teacher_forced(condition, raw_targets)
        feature = F.mse_loss(pred, target)
        pixel = feature.new_zeros(())
        if pixel_weight:
            if rgb_target is None:
                raise ValueError("Pixel loss requires an explicit scoring/training target")
            image = self.decode(pred)
            if image.shape != rgb_target.shape:
                raise ValueError("D output and RGB target shapes must agree exactly")
            pixel = F.mse_loss(image.float(), rgb_target.float())
        return feature + pixel_weight * pixel, {"feature_mse_teacher_forced": feature,
                                               "pixel_mse_teacher_forced": pixel}
