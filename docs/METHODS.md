# Three-stage architecture

| Stage | Conditioning and sequence | Output and objective | What the evidence says |
|---|---|---|---|
| P0 | A 64-dimensional condition token plus 16 learned 2D position queries; bidirectional Transformer. E3 uses a fixed random projection of circle-center coordinates. | One shared linear head predicts 12×12 RGB patches in parallel; deterministic pixel MSE. | Verified conditional mapping on the tested synthetic task. E3 partial coordinate control did not pass all interpolation/extrapolation geometry gates. No general image generation claim. |
| P1 | CIFAR-10 class embedding and BOS, followed by up to 63 previous 4×4 RGB patches; causal Transformer over 64 patch positions. | Per-patch K=8 shared-component diagonal Gaussian mixture over 48 values; continuous NLL. Sampled raw patches are fed back autoregressively. | Validation NLL improved, while the fixed 80-image free-generation set failed visual recognition. |
| P2 | Class text prompt and image BOS enter the Qwen language backbone; each prior 4×4 RGB patch is projected to the language hidden size; causal next-patch prediction. | Tanh-bounded K=8 shared-component Gaussian mixture over pre-tanh patch values; image and text language-model losses alternate at a 4:1 step ratio. | Fixed free-generation samples failed; the fixed text instruction evaluation regressed. P2 changes both backbone scale/pretraining and output distribution, so it is not a controlled architecture-only comparison. |

All reported free samples start from a prompt/condition and empty image prefix. Teacher-forced next-patch diagnostics are not free generation. Training uses fixed splits and checkpoint criteria documented in the sanitized stage reports. P0's synthetic experiment is not evidence about natural images.
