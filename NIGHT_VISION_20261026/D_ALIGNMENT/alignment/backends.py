"""Official Qwen adapter is lazy and local-files-only. Tiny backend is fixture-only."""
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .core import WIDTH, freeze, FeatureContract, fingerprint
from .io import locked_file, digest, sha


@dataclass
class Prefix:
    embeds: torch.Tensor
    positions: torch.Tensor | None = None


def verify_model_files(spec, root):
    model_dir = (Path(root) / spec["path"]).resolve()
    if spec["repository"] != "Qwen/Qwen3.5-0.8B" or len(spec["revision"]) != 40:
        raise ValueError("Official Qwen immutable revision required; no P2 initialization")
    if spec["origin"] != "official_original_no_p2" or not spec["files"]:
        raise ValueError("Official original model provenance and file inventory required")
    files = {p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*") if p.is_file()
             and (p.suffix in (".safetensors", ".json", ".jinja", ".txt", ".model"))}
    if set(spec["files"]) != files or not {"config.json", "preprocessor_config.json"} <= files or not any(p.endswith(".safetensors") for p in files):
        raise ValueError("Pin every model weight, configuration and tokenizer/processor file")
    for filename, checksum in spec["files"].items():
        locked_file({"path": filename, "sha256": checksum}, model_dir)
    return model_dir


class QwenBackend(nn.Module):
    def __init__(self, spec, encoder, root, device, dtype, scope, checkpointing, representation):
        super().__init__()
        from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
        model_dir = verify_model_files(spec, root)
        if sha(model_dir / "preprocessor_config.json") != representation["processor_sha256"]:
            raise ValueError("Official processor source configuration differs from E/D target provenance")
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, use_safetensors=True, attn_implementation="eager").to(device)
        self.compute_dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False, use_fast=False)
        processor = self.processor.image_processor
        expected = {"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                    "do_resize": True, "resample": 3, "do_rescale": True, "rescale_factor": 1 / 255,
                    "do_normalize": True, "image_mean": [.5, .5, .5], "image_std": [.5, .5, .5],
                    "do_convert_rgb": True}
        for name, value in expected.items():
            if getattr(processor, name, None) != value:
                raise ValueError(f"Unsupported effective official image processor value: {name}")
        c = self.qwen.config
        if c.model_type != "qwen3_5" or c.text_config.hidden_size != WIDTH or c.vision_config.out_hidden_size != WIDTH:
            raise ValueError("Unsupported Qwen geometry")
        if getattr(c.vision_config, "deepstack_visual_indexes", []):
            raise ValueError("v1 does not implement deepstack visual inputs")
        # Encoder can be the bundled official tower or a separately pinned strict state.
        if encoder.get("weights"):
            from safetensors.torch import load_file
            weights = locked_file(encoder["weights"], root)
            original_encoder = fingerprint(self.qwen.model.visual)
            self.qwen.model.visual.load_state_dict(load_file(str(weights)), strict=True)
            if fingerprint(self.qwen.model.visual) != original_encoder:
                raise ValueError("Separate E weights differ from the pinned official model's vision tower")
        self.hidden_size = c.text_config.hidden_size
        self.separate_text_axis = "text_position_ids" in inspect.getsource(type(self.qwen.model.language_model).forward)
        self.configure_scope(scope)
        if checkpointing:
            self.qwen.model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.keep_encoder_frozen()
        self.runtime = {"transformers_version": __import__("transformers").__version__,
                        "language_source_sha256": sha(inspect.getfile(type(self.qwen.model.language_model))),
                        "torch_version": torch.__version__}

    def configure_scope(self, scope):
        self.qwen.requires_grad_(False)
        if scope == "full":
            self.qwen.model.language_model.requires_grad_(True)
            self.qwen.lm_head.requires_grad_(True)
        elif scope == "last4":
            layers = self.qwen.model.language_model.layers
            if len(layers) < 4:
                raise ValueError("Need at least four language blocks")
            for layer in layers[-4:]:
                layer.requires_grad_(True)
        else:
            raise ValueError("Update scope is full or last4")

    @property
    def encoder(self):
        return self.qwen.model.visual

    def keep_encoder_frozen(self):
        freeze(self.encoder)

    @property
    def device(self):
        return self.qwen.get_input_embeddings().weight.device

    def output_start(self, batch):
        ids = torch.full((batch, 1), self.qwen.config.vision_start_token_id, device=self.device, dtype=torch.long)
        return self.qwen.get_input_embeddings()(ids)

    def prepare(self, condition):
        from PIL import Image
        # Condition has already rejected any image for description/class_only.
        content = [{"type": "text", "text": condition.text}]
        image = None
        if condition.mode == "redraw":
            image = Image.open(condition.image).convert("RGB")
            if image.width != image.height:
                raise ValueError("v1 only supports square source images; no unversioned aspect-ratio change")
            content.insert(0, {"type": "image", "image": image})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        args = {"text": [text], "return_tensors": "pt", "padding": False}
        if image is not None:
            # v1 uses square 256 processed images, exactly 64 post-merger positions.
            image = image.resize((256, 256), Image.Resampling.BICUBIC)
            args.update(images=[image], min_pixels=65536, max_pixels=65536)
        inputs = self.processor(**args).to(self.device)
        ids = inputs["input_ids"]
        if ids.shape[0] != 1 or ids.shape[1] > 2048:
            raise ValueError("v1 supports batch 1 and <=2048 source tokens without truncation")
        embeds = self.qwen.get_input_embeddings()(ids)
        if image is not None:
            grid = inputs["image_grid_thw"]
            if not torch.equal(grid, grid.new_tensor([[1, 16, 16]])):
                raise ValueError("Official source image must have grid [1,16,16]")
            with torch.no_grad():
                out = self.qwen.model.get_image_features(inputs["pixel_values"], grid, return_dict=True)
                pooled = out.pooler_output
                feature = torch.cat(pooled, 0) if isinstance(pooled, (tuple, list)) else pooled
            if feature.shape != (64, WIDTH):
                raise ValueError("Unexpected official post-merger source shape")
            mask = ids == self.qwen.config.image_token_id
            if int(mask.sum()) != 64:
                raise ValueError("Official image token count mismatch")
            embeds = embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds), feature.to(embeds))
        else:
            grid = None
            if (ids == self.qwen.config.image_token_id).any() or "pixel_values" in inputs:
                raise ValueError("Image tokens are forbidden in text-only conditioning")
        # Reuse official source-image mRoPE; output suffix is a separate linear
        # sequence with an explicit fixed 8x8 additive coordinate code.
        kwargs = dict(input_ids=ids, image_grid_thw=grid, attention_mask=inputs["attention_mask"])
        if "mm_token_type_ids" in inspect.signature(self.qwen.model.get_rope_index).parameters:
            kwargs["mm_token_type_ids"] = inputs.get("mm_token_type_ids", (ids == self.qwen.config.image_token_id).int())
        pos, _ = self.qwen.model.get_rope_index(**kwargs)
        return Prefix(embeds, pos)

    def hidden(self, prefix, suffix):
        embeds = torch.cat([prefix.embeds, suffix], dim=1)
        length = suffix.shape[1]
        begin = int(prefix.positions.max()) + 1
        ext = torch.arange(begin, begin + length, device=embeds.device).view(1, 1, -1).expand(3, 1, -1)
        pos = torch.cat([prefix.positions, ext], -1)
        # Four axes on versions with a separate text-position axis; three on older
        # releases. Passing 4 is supported by the audited 5.20 implementation.
        if self.separate_text_axis:
            text_pos = torch.arange(embeds.shape[1], device=embeds.device).view(1, 1, -1)
            pos = torch.cat([text_pos, pos], 0)
        with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype,
                            enabled=self.compute_dtype == torch.bfloat16):
            out = self.qwen.model.language_model(inputs_embeds=embeds, position_ids=pos,
                          attention_mask=torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long),
                          use_cache=False, return_dict=True)
        return out.last_hidden_state[:, -length:].float()

    def replay_loss(self, row):
        tokenizer = self.processor.tokenizer
        messages = [{"role": "user", "content": row["prompt"]}]
        prefix = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        complete = self.processor.apply_chat_template(messages + [{"role": "assistant", "content": row["response"]}],
                                                      tokenize=False, add_generation_prompt=False, enable_thinking=False)
        a = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(self.device)
        b = tokenizer(complete, return_tensors="pt", add_special_tokens=False).to(self.device)
        n = a["input_ids"].shape[1]
        if b["input_ids"].shape[1] > 2048 or not torch.equal(a["input_ids"], b["input_ids"][:, :n]):
            raise ValueError("Instruction template prefix mismatch or sequence too long")
        labels = b["input_ids"].clone()
        labels[:, :n] = -100
        if not (labels != -100).any():
            raise ValueError("No assistant response tokens")
        with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype,
                            enabled=self.compute_dtype == torch.bfloat16):
            return self.qwen(**b, labels=labels, use_cache=False).loss

    @torch.no_grad()
    def instruction_response(self, row, max_tokens=128):
        text = self.processor.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = self.processor.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        if tokens["input_ids"].shape[1] > 2048:
            raise ValueError("Instruction evaluation prompt too long")
        with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype,
                            enabled=self.compute_dtype == torch.bfloat16):
            ids = self.qwen.generate(**tokens, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
        return self.processor.tokenizer.decode(ids[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True)


class GridDecoder(nn.Module):
    def __init__(self, module, grid):
        super().__init__()
        self.module, self.grid = module, tuple(grid)

    def forward(self, z):
        return self.module(z, self.grid)


def load_decoder(spec, representation, root, device):
    from safetensors.torch import load_file
    if spec["representation_sha256"] != digest(representation):
        raise ValueError("D must be trained on exactly the encoder/processor/order representation")
    weights = locked_file(spec["weights"], root)
    source = locked_file(spec["source"], root)
    for dependency in spec.get("dependency_sources", []):
        locked_file(dependency, root)
    module_name, name = spec["factory"].split(":")
    # Factory import is an A-configured local Python plugin, never remote code.
    factory = getattr(importlib.import_module(module_name), name)
    if Path(inspect.getfile(factory)).resolve() != source:
        raise ValueError("D factory source does not match pinned source file")
    decoder = factory(**spec.get("kwargs", {}))
    decoder.load_state_dict(load_file(str(weights)), strict=True)
    decoder.to(device=device, dtype=torch.float32)
    if spec["call"] == "features_grid":
        decoder = GridDecoder(decoder, (1, 16, 16))
    elif spec["call"] != "features":
        raise ValueError("D call is features or features_grid")
    return freeze(decoder)


def load_contract(spec, root):
    if spec["kind"] == "raw":
        return FeatureContract(decoder_space=spec["decoder_space"])
    if spec["kind"] != "channel_standardize" or spec["statistics_split"] != "train":
        raise ValueError("Only raw or immutable train-only channel standardization")
    from safetensors.torch import load_file
    state = load_file(str(locked_file(spec["statistics"], root)))
    if set(state) != {"mean", "std"}:
        raise ValueError("Statistics must store mean and effective (already floored) std")
    return FeatureContract(state["mean"], state["std"], spec["decoder_space"])


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(16, 16, batch_first=True)

    def forward(self, x):
        return x + self.gru(x)[0]


class TinyBackend(nn.Module):
    """RANDOM fixture. Not Qwen and not model-quality evidence."""
    hidden_size = WIDTH

    def __init__(self, scope="full"):
        super().__init__()
        self.embedding = nn.Embedding(128, WIDTH)
        self.input_projection = nn.Linear(WIDTH, 16)
        self.layers = nn.ModuleList([TinyBlock() for _ in range(6)])
        self.output_projection = nn.Linear(16, WIDTH)
        self.text_head = nn.Linear(WIDTH, 128)
        self.encoder = freeze(nn.Linear(3, WIDTH))
        self.seen = []
        self.requires_grad_(scope == "full")
        if scope == "last4":
            for block in self.layers[-4:]:
                block.requires_grad_(True)
        self.keep_encoder_frozen()

    def keep_encoder_frozen(self):
        freeze(self.encoder)

    def output_start(self, batch):
        return self.embedding(torch.ones((batch, 1), device=self.embedding.weight.device, dtype=torch.long))

    def prepare(self, condition):
        self.seen.append(condition)
        ids = [2 + b % 120 for b in condition.text.encode()[:16]]
        ids = torch.tensor([ids], device=self.embedding.weight.device)
        embeds = self.embedding(ids)
        if condition.image:
            # Synthetic source; deliberately not reading an actual image/model.
            color = embeds.new_tensor([[1., .2, .4]])
            embeds = torch.cat([self.encoder(color).unsqueeze(1), embeds], 1)
        return Prefix(embeds)

    def hidden(self, prefix, suffix):
        x = self.input_projection(torch.cat([prefix.embeds, suffix], 1))
        for block in self.layers:
            x = block(x)
        return self.output_projection(x)[:, -suffix.shape[1]:]

    def replay_loss(self, row):
        ids = torch.tensor([[3, 4, 5]], device=self.embedding.weight.device)
        logits = self.text_head(self.embedding(ids))
        return F.cross_entropy(logits.view(-1, 128), torch.tensor([4, 5, 6], device=ids.device))


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(WIDTH, 3)

    def forward(self, z):
        return self.projection(z).transpose(1, 2).reshape(z.shape[0], 3, 8, 8)
