# 多分辨率视觉接口与路由机制核查

核查日期为 2026 年 10 月 27 日，Asia/Shanghai。官方源代码只作为文本读取，没有导入、执行或加载相关模型。固定提交、完整文件校验和及下载时间保存在 [sources/provenance.json](sources/provenance.json)。

## Qwen 视觉特征怎样进入文字主干

Qwen2-VL 的参考版本是 Transformers v4.49.0，固定提交 `a22a4378d97d06b7a1d9abad6e0086d30fdea199`。实际路径是像素预处理、patch embedding、视觉 Transformer、相邻空间 token 的 merger，随后把视觉向量填入文字 embedding 序列的图像占位符，再将整个 embedding 序列交给语言模型。

可定位的实现是 [PatchMerger 第 269 行](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L269)、[图像特征写入第 1672 行](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1672)、[主干调用第 1733 行](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1733)。本地同名快照位于 `sources/qwen2_vl_modeling.py`。

还核对了 Qwen3.5 的官方 Transformers 实现，固定提交 `a672eb5f474b8b30173e9f96eab66adf292da635`：视觉输出经过 [merger 第 968 行](https://github.com/huggingface/transformers/blob/a672eb5f474b8b30173e9f96eab66adf292da635/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L968)，通过 `pooler_output` 取得图像特征，在 [第 1615 行起](https://github.com/huggingface/transformers/blob/a672eb5f474b8b30173e9f96eab66adf292da635/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1615) 写入图像占位符，再于第 1643 行调用 `language_model`。这里核查的是接口，不据此声称 Qwen3.5 与 Qwen2-VL 的全部层、位置方法和 Attention 类型相同。

视觉向量进入文字主干，不代表原始像素直接、无损地变成文字 token；merger 也不是可逆打包。本原型的原像素库与这个接口不同。

## 动态分辨率与位置编码

[Qwen2-VL 论文第 2.1 节](https://arxiv.org/html/2409.12191v2#S2.SS1) 描述随图像大小变化的视觉 token 数、视觉编码器的二维 RoPE，以及语言模型中的时间、高度、宽度 M-RoPE。它还使用相邻 2×2 视觉 token 合并。处理器的 `smart_resize` 与 [第 226 至 270 行](https://github.com/huggingface/transformers/blob/a22a4378d97d06b7a1d9abad6e0086d30fdea199/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L226) 显示 resize 后再 reshape/transpose 成 patch。

其中 reshape/transpose 本身可逆，前面的 resize 与后面的神经特征合并不能一起称为原像素无损。动态 token 数和多维位置编码都不自动构成按问题选择 top-k 空间区域的路由。

## 放大与内部路由的区别

| 参考 | 原文机制 | 层级归属与本原型区别 |
|---|---|---|
| [ZoomEye 2411.16044 第 3 节及附录 A](https://arxiv.org/html/2411.16044v1#S3) | 将图像组织为树，利用 MLLM 的存在、潜在可见及可回答置信度决定搜索和停止；局部裁剪或全局加局部输入。 | 模型外推理与工具级放大，模型无关且无需额外训练。它不是在同一层内部对保留的原像素 KV 做稀疏 gather。 |
| [LLaVA-UHD 2403.11703 第 3 节](https://arxiv.org/html/2403.11703v1#S3) | 按分辨率与长宽比分片，分别视觉编码；perceiver resampler 用可学习 query 压缩每片 token，再用空间分隔组织进入 LLM 的表示，另有全局概览。 | 属于模型视觉接口内的多分辨率编码与压缩；resampler 的可学习 query 不等于用户问题条件的 top-k 路由。不能把分片加压缩直接当成只计算问题相关区域。 |
| Qwen2-VL | 动态分辨率、视觉编码、空间 merger、M-RoPE 和 embedding 注入。 | 模型内部视觉与语言接口；所查路径不展示本原型这样的问题条件细位置选择。 |
| 当前原型 | 粗 MLP 给每个区域生成 4 维 key，任务向量评分、选 top-k，然后仅为选中区域构造细 K/V 并实际执行 Attention。 | 内部小模型的条件计算原语；不调用外部放大工具、不复现上述完整模型。 |

本原型保留的是形状可逆的 raw RGB；粗摘要、类型特征、Attention 输出本身都可能有损。它没有声称是上述论文的创新替代方案或完整复现。

## 既有 CPU 机制实验室的边界

既有 `CPU_MECHANISM_LAB_20261026` 的 `kv_demo.py` 中，`unique_storage_bytes` 按真实 storage 指针去重，`structural_csa_ledger` 用独立缓存反例说明：共享 Top-K 选择结果，并不会减少各层独立 main KV 的容量。那一实验里的层级索引只有结构说明，没有运行生产层级索引器；本次原型新增的是小型空间路由和真实像素 Attention。

相关代码摘录与完整源文件 SHA256 存在 [sources/prior_lab_boundary.txt](sources/prior_lab_boundary.txt)。本次延续三个边界：容量、读取与 FLOPs 分列；首次全局索引有成本；只有相同数学定义的两条路径才要求相等。这里没有实现 DeepSeek 的压缩器、跨层 Full/Reindex/Reuse、SWA 联合注意力或真实 CSA2 forward，不能称为 CSA2 复现。
