# Reproduction and independent inference

## CPU smoke test

Install `requirements-minimal.txt` in an isolated environment, then run `python tests/minimal_cpu.py`. It uses random tiny configurations only. It does not access CIFAR-10, Qwen weights, WikiText, the network, or a GPU. The tested P0/P1 dependency lock is `requirements-p0-p1-tested.txt`; the observed P2 CUDA dependency versions are listed in `requirements-p2-cuda-tested.txt` (install the matching CUDA-enabled PyTorch wheel for your platform).

## P0

Use the portable configs `configs/main.json`, `configs/tiny.json`, and `configs/e3.json`. The synthetic data generator writes to a new output directory. For a bounded CPU run, make a copy of a tiny config and set a small step count and an output directory that does not exist. The original experiment was run with MPS FP32; do not treat a tiny smoke run as a reproduction of its result.

## P1

Prepare the official CIFAR-10 Python archive yourself from the [official dataset page](https://cave.cs.toronto.edu/kriz/cifar.html); the code verifies the archive checksum. The dataset is not included. The original run used the fixed stratified 45,000/5,000 training/validation split and did not use augmentation. For exact evaluation, follow the selected config and report the device and package versions. CPU training of the full 20M model is not a practical quick check.

Independent sampling requires only the P1 model code, a compatible weights file, and a class/seed. The weights are not published, so this repository alone cannot reproduce the original sample images.

## P2

P2 sampling and training require a local compatible Qwen3.5-0.8B directory, tokenizer files, CIFAR-10 data, and the separate WikiText-2 train/validation data used by the experiment. The project does not download or include model weights. Qwen's model card identifies its artifacts as Apache-2.0; obtain them from the official [Qwen model repository](https://huggingface.co/Qwen/Qwen3.5-0.8B) and follow its current terms. WikiText-2 has separate attribution/share-alike terms; see [its dataset card](https://huggingface.co/datasets/Salesforce/wikitext). The dataset itself is not included.

The public P2 config is an editable example, not an execution authorization or resource recommendation. The P2 trainer has known recovery and epoch-control concerns; review and fix those before training or resuming. The minimal CPU test covers only probability math, not the transformer integration or training loop.

## Result limits

P0 showed conditional mapping on synthetic data. P1 and P2 free generation failed the visual goal. P2 text instruction behavior regressed on its fixed small evaluation. No result here establishes SOTA, general image-generation ability, or a validated new architecture. Cross-device reproducibility has not been established.
