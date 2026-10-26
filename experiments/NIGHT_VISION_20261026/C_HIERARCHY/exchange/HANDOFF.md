# B 会话视觉记忆机制交接

这里交接一个独立、已运行的 CPU 原型。B 可以直接读取此目录和上一级代码，不依赖本机路径或其他会话。当前不要求 B 返回结果；本任务不修改 B 代码。

先看上一级 [README](../README.md)、[结果报告](../REPORT.md)、[结构图](../figures/architecture.png) 与本目录 [protocol.json](protocol.json)。`READY.json` 在最终验收后才创建；它包含交付状态、关键文件校验和与仓库入口。它没有声称任何完整大模型通过验收。

## 可直接读取的数据

| 文件 | 内容 |
|---|---|
| `sample_test.npz` | 普通测试图的原始 RGB、无损像素库、边界与 mask、粗摘要和 key、Q、选择、细 K/V、权重、稀疏及 dense 输出 |
| `sample_test_blind.npz` | 相同字段，但粗摘要看不到目标位置，用于复查漏检 |
| `sample_gold_evaluation_only.json` | 两个示例的独立真值，只供评测；不传入 router 或读器 |
| `protocol.json` | 数组维度、坐标、尺度、层级、去重和 softmax 定义 |
| `../data/inputs` | 全量训练与固定测试输入清单；只有路径和问题编号 |
| `../data/annotations` | 单独的监督标签；测试标签不能用于训练或非 oracle 推理 |
| `../results/per_question.jsonl` | 每个测试问题、预算和策略的选区、漏检、答案及输出误差 |

使用 `np.load(path, allow_pickle=False)` 即可读取两个 NPZ。`pixels` 为 `[16,8,8,3]` uint8；按四行四列区域逆 permutation 回到 `[32,32,3]`。`boxes` 为全局 `[x0,y0,x1,y1)`，细坐标按区域原点加局部坐标计算。`selected_ids` 已 unique 排序；K/V 与 attention weights 使用此顺序。

## 运行与核验

```sh
cd <本原型目录>
sh run.sh demo
sh run.sh test
sh run.sh evaluate
```

模型只有 7,451 个参数，文件 `../results/model.pt`；无需联网或训练。解释器可用 `LAB_PYTHON` 指定。完整依赖版本与设备记录在 `../results/training.json`，只使用 CPU。若只读取交接 NPZ，只需 NumPy。

普通 k=2 路由在固定集命中 384/384 个目标，粗盲组只命中 46/384，漏检 338 个。不要用后者约 55% 的二分类准确率掩盖漏检；漏检时仍会猜中一部分二分类答案。所有区域都选中才在相同 Q/K/V 定义下比较 dense 数值一致，部分区域允许误差。

## 复用边界

原图持续保存不等于主干表示完全无损；粗摘要是有损的。此原型没有跨层共享或跨问题 KV 缓存，只有按问题的选区与按需细 K/V。空间 V 是教学用 one-hot，真实物理字节计入成本。

不能把局部 softmax 结果直接相加，不能把少算细 Attention 写成整个模型 8 倍加速。本机小问题上，分层整条查询实际比跳过路由的直接 dense 读器慢。没有高分辨率语义质量、GPU/DCU/MPS 性能或自然语言能力结论。它不等于 Qwen、ZoomEye、LLaVA-UHD 或 DeepSeek CSA2 的完整复现。
