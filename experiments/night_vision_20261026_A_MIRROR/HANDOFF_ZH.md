# A_MIRROR 交接

## 当前状态

主 tiny/full 与匹配 pre/post 对照均已完成。完整权重和原始产物在本目录索引的 Release 中，未改动 P0/P1/P2、旧 DCU 或旧 tiny 原型。没有后台训练、语言接口训练或自动续跑。

## 先读取这些文件

1. `README.md`：五个核心问题、实际数据规模和结论边界。
2. `ARTIFACTS.json`：所有附件的大小、SHA-256、逻辑恢复路径。
3. `snapshot/protocol.json`、`snapshot/training_locked.json`：固定目标、共同步数、精度策略、开发选优规则。
4. `snapshot/indices.json`：完整 train/dev/final 原始训练池索引，互不交叠；官方 test 未用。
5. `VISION_FEATURE_CONTRACT.json`：实际 pre hook、post 输出、Processor、位置与空间顺序。
6. `DECODER_EXPORT_FOR_ALIGNMENT.json`：A 侧不可变导出描述；它不是 D_ALIGNMENT 联训已验收的证明。

`fetch_artifacts.py` 会核验所有附件，填回大权重，再以一次目录重命名发布完整本地快照。不要把正在写入的 `.incoming`、`.uploading` 或 `.download` 文件当作 checkpoint。

## checkpoint 与恢复状态

- 主模型 best 均由 dev 选出，实际为 step 2037；last 均为 step 16300。
- `model.safetensors`：F32 参数。训练/推理前向使用已记录的 BF16 autocast；保留原生未裁剪预测。
- `normalization.safetensors`：`mean`、`std`，主 tiny/full 均来自同一 2,000 训练图的通道统计，不能按开发/最终集重拟合。
- `last/restore.pt`：AdamW 状态、CPU/CUDA RNG、已完成 step 与数据顺序种子。
- `best/READY.json` 明确为 `COMPLETE_BEST_MODEL_ONLY`，优化器/RNG 恢复状态在 last；不能把 best 当作完整优化器恢复点。
- 原始运行代码支持有界故障恢复，保留旧时间起点、共同步数与缓存。旧课堂 deadline 已闭合；后续继续训练需要新预算，并按原索引重建可再生缓存。

发生过日志字段重名和均值图广播维度两处普通实现错误，均从原检查点恢复；没有重置训练步数、数据、时间账本或改损失。原始失败记录和源版本都保留，不把失败记录误当最终状态。

## 特征接口

官方模型 revision `2fc06364715b967f1860aea9cf38778875588b17`，Transformers 5.2.0，实际类 `Qwen3_5VisionModel`。

静态图实际处理尺寸 256×256，grid `[1,16,16]`；patch16、temporal2（单图重复最后帧）、merge2，RGB `/255` 后按 mean/std 0.5 归一化。pre 是真实 merger forward-pre-hook 取得的 `[256,768]`，post 是最终 pooler `[64,1024]`。

组内顺序 TL/TR/BL/BR；先按合并组行优先排列，unmerge 后必须恢复全局 raster，才加入二维位置和投影像素。像素头使用 patch 内 HWC，最终 BCHW。视觉全局注意力使坐标只是位置锚点，不能等同于 P2 原始 Patch 语义。

配对接口样本仅交付固定 16 张（8 train、8 dev、0 final）的 raw F32 pre/post 和实际 RGB；训练中实际使用过 1,000/200 的匹配缓存。完整可再生激活缓存、CIFAR 压缩包和官方模型权重未重复分发。

## 防止重复标准化

`NightMirror.forward` 接受 **model-space 标准化后的特征**。原始特征先使用随权重保存的训练统计做一次 `(raw-mean)/std`，再进入工厂；不能在外部和模型包装器里各做一次。

导出工厂为 `night_model:NightMirror`，参数 `{"name":"mirror_full"}`，调用是 features + grid。将快照代码根目录加入 Python import path。对齐接口还必须使用自身 locker 校验 representation/normalization hash、数据 join 和资源预算。

## 语言接口状态

读取到的对接方 READY 是 `READY_CPU_IMPLEMENTATION_ONLY`。其代码/CPU 检查就绪，不等于官方 0.8B 语言权重、DCU 算子、配对数据和指令回放已就绪。本轮未执行 direct/MLP 语言配对，未判断自由生成或指令保持能力。

只有以后具备全部输入和独立授权预算时再接入。不要把重建图称作从文字画图，也不要把 pre 更易重建的结果解释成 merger 细节已经被精确证明全部丢失。

## 最有价值的下一步问题（尚未执行）

充分预热后重新评估 10,000 图数据准备；本轮保守回退不能替代这个实测。另可独立控制 unmerge 容量/预特征监督与共享主干初始化，区分输入信息、适配难度与过拟合。以上都是后续研究方向，不是当前交付已经跑出的结果。
