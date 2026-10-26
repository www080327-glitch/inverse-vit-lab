"""Optional installed-Transformers integration fixture; no pretrained files loaded."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import inspect
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("300 second fixture limit")))
    signal.alarm(300)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from torch import nn
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from alignment.backends import QwenBackend, Prefix, TinyDecoder
    from alignment.core import Alignment, FeatureContract, fingerprint
    from alignment.io import write_json
    start = time.monotonic()
    torch.manual_seed(2710)
    cfg = Qwen3_5Config(text_config={"vocab_size": 128, "hidden_size": 1024,
        "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 2, "head_dim": 16, "layer_types": ["full_attention"] * 2,
        "max_position_embeddings": 4096, "attn_output_gate": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1.,
                            "mrope_section": [3, 3, 2], "mrope_interleaved": True}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 1024, "num_position_embeddings": 256, "deepstack_visual_indexes": []},
        image_token_id=120, vision_start_token_id=121, vision_end_token_id=122, video_token_id=123)
    adapter = QwenBackend.__new__(QwenBackend)
    nn.Module.__init__(adapter)
    adapter.qwen = Qwen3_5ForConditionalGeneration(cfg).cpu()
    adapter.hidden_size = 1024
    adapter.compute_dtype = torch.float32
    adapter.separate_text_axis = "text_position_ids" in inspect.getsource(type(adapter.qwen.model.language_model).forward)
    adapter.configure_scope("full")
    adapter.keep_encoder_frozen()
    engine = Alignment(adapter, TinyDecoder(), FeatureContract(), "direct")
    ids = torch.tensor([[4, 5, 6]])
    prefix = Prefix(adapter.qwen.get_input_embeddings()(ids), torch.arange(3).view(1, 1, 3).expand(3, 1, 3))
    raw = torch.randn(1, 64, 1024)
    a = adapter.hidden(prefix, engine.suffix(raw[:, :-1]))
    perturbed = raw.clone()
    perturbed[:, 30:] += 10
    b = adapter.hidden(prefix, engine.suffix(perturbed[:, :-1]))
    torch.testing.assert_close(a[:, :31], b[:, :31], rtol=0, atol=0)
    before = fingerprint(adapter.encoder), fingerprint(engine.decoder)
    engine.decode(a).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.qwen.model.language_model.parameters())
    assert all(p.grad is None for p in adapter.encoder.parameters())
    assert all(p.grad is None for p in engine.decoder.parameters())
    assert before == (fingerprint(adapter.encoder), fingerprint(engine.decoder))
    result = {"status": "PASS", "fixture_only": True, "official_pretrained_weights_loaded": False,
              "full_0_8B_model_loaded": False, "architecture": "random 2 language blocks / 1 tiny vision block",
              "parameters": sum(p.numel() for p in adapter.parameters()),
              "hidden_width": 1024, "output_shape": list(a.shape),
              "checks": ["official_language_forward_with_extended_mrope", "causal_future_target_perturbation",
                         "pixel_only_D_to_M_gradient", "frozen_E_D_unchanged"],
              "device": "cpu", "threads": torch.get_num_threads(),
              "transformers_version": __import__("transformers").__version__,
              "elapsed_seconds": time.monotonic() - start}
    write_json(Path(__file__).parents[1] / "evidence" / "hf_random_check.json", result)
    signal.alarm(0)
    print(result)


if __name__ == "__main__":
    main()
