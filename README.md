[中文](README.zh-CN.md) | **English**

# Inverse-ViT Lab

An open research record of three experiments on conditional and autoregressive pixel generation. The code and reports are published for review and collaboration; the results include clear failures.

## Results at a glance

| Stage | Question | Observed result | Status |
|---|---|---|---|
| P0 | Can a ViT map a condition to pixels? | It learned the tested synthetic conditional mappings. The later coordinate holdout experiment did not meet its full geometry gates. | Scoped success; generalization unverified |
| P1 | Can a causal ViT freely generate CIFAR-10 pixels? | Validation likelihood improved, but all 80 fixed free samples were mostly noise and color blocks without reliably recognizable class objects. | Free generation failed |
| P2 | Can a pretrained language backbone add visual generation while retaining text behavior? | All 80 fixed free samples lacked reliably recognizable objects. Text abilities regressed on the fixed task evaluation. | Visual goal failed; text regression observed |

See [stage reports](results/) for measured limits and [methods](docs/METHODS.md) for the architectural differences. There is no SOTA, general image-generation success, or validated architecture claim.

## Repository contents

- `p0/`, `p1/`, `p2_qwen/`: selected model, training, data-preparation, sampling and evaluation source.
- `configs/`: portable examples with relative paths. Review and change hardware, data and output locations before a run.
- `tests/minimal_cpu.py`: no-download CPU smoke checks for all three model paths and the P2 probability head.
- `results/`: sanitized aggregate reports and the complete P0 synthetic coordinate comparison atlas.
- `docs/`: reproducibility, inference, licensing, and collaboration notes.

P1/P2 CIFAR-10 image atlases are not included while publication rights for output images derived from CIFAR-10 remain unclear. No source dataset images are bundled.

## Quick CPU check

With Python and the dependencies in `requirements-minimal.txt` installed:

```sh
python tests/minimal_cpu.py
```

This instantiates small random models and checks tensor shapes, finite outputs, patch round trips, and the P2 continuous probability calculation. It downloads no data or model, does not train, and does not validate research quality. A full dependency environment may take substantial disk space because PyTorch is large.

## Reproducing experiments

Start with [reproduction and inference](docs/REPRODUCE.md). P0 synthetic data can be generated locally. P1 needs the official CIFAR-10 archive; P2 needs separately obtained CIFAR-10 data, WikiText-2, a compatible Qwen3.5-0.8B checkpoint, and suitable compute. Nothing is downloaded by the smoke check. Never place credentials in config files.

The currently published P2 training and recovery code has known control-flow issues from a read-only audit. Do not use it for a long run or resume until the concerns in [P2 results](results/P2.md) are reviewed and fixed.

## Contributing

Please open an issue before a substantial experiment. Share the hypothesis, fixed data split, compute budget, stopping rule, and metrics you will report. Use a topic branch and a pull request against `main`; include a small CPU check when changing code. Keep credentials, personal details, private paths, model weights, optimizer states, raw datasets, and logs out of issues and commits. Follow [CONTRIBUTING.md](CONTRIBUTING.md).

## License and sources

The selected original experiment code is MIT licensed. Third-party software, model, and data have separate terms; see [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). The code license does not license model weights or datasets.
