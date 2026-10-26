# Third-party sources and terms

This repository contains selected original experiment code under the MIT License. The following dependencies and research inputs are separate from that code license:

- **PyTorch**: BSD-style license; installed dependency, not vendored. https://github.com/pytorch/pytorch
- **Hugging Face Transformers**: Apache-2.0; installed dependency/API used by P2, not vendored. https://github.com/huggingface/transformers
- **Qwen3.5-0.8B**: official model repository declares Apache-2.0 for its artifacts. No weights or tokenizer files are included. Check current terms: https://huggingface.co/Qwen/Qwen3.5-0.8B
- **CIFAR-10**: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton; cite *Learning Multiple Layers of Features from Tiny Images* (2009), as requested by the official dataset page: https://cave.cs.toronto.edu/kriz/cifar.html. The official page does not state an explicit dataset license. No full raw archive is included. P1/P2 output atlases remain omitted; the independent frozen-vision experiment includes its fixed research comparisons and small paired interface samples as described below.
- **WikiText-2**: Salesforce WikiText dataset. Its card lists CC-BY-SA-3.0 and GFDL metadata and describes CC BY-SA terms; no text data or prompt/output transcripts are included. Review current terms: https://huggingface.co/datasets/Salesforce/wikitext
- **P0 figures** are generated from the project's synthetic circle task, not CIFAR-10.

No third-party model implementation source, paper full text, abstract collection, weights, or datasets are redistributed here. A dependency's license does not relicense its model artifacts or training data.

## Frozen vision reconstruction experiment

The independent mirror-decoder experiment references [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) at revision `2fc06364715b967f1860aea9cf38778875588b17`, [Transformers v5.2.0](https://github.com/huggingface/transformers/tree/v5.2.0), and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). Official model weights and the full dataset archive are not redistributed. Release weights are the experiment's randomly initialized, trained decoder parameters. The experiment includes its fixed, CIFAR-derived source/reconstruction contact sheets for comparison; upstream imagery and model terms remain with their respective sources.
