# 冻结 Qwen 视觉特征：正式镜像解码器实验

**实际数据为 2,000 训练 / 500 开发 / 500 本轮最终评估图，统一目标为官方预处理后的 256×256 RGB。** 两个主模型各训练 16,300 步、有效 batch 32，各 521,600 次样本呈现。它是 `D(E(image))` 重建，**不是文字生图或自由生成**。

1. **full 真正跑了吗？** 跑了。随机初始化的可学习 unmerge → 256×768 → 12 层、12 头、FFN3072 → 16×16 RGB 像素投影，共 **88,795,392 参数**。权重载荷确有 12 个 Transformer block；独立重载通过。
2. **比 tiny 怎样？** 最终集 MSE：full **0.021108**，tiny **0.022460**，full 低约 **6.02%**。参数约多 31.4 倍，训练阶段墙钟约多 8.8 倍，收益有限；图册仍有模糊与明显格纹。
3. **能否泛化？** 两者均优于最终集固定训练均值图的 MSE **0.063081**，但仍有训练—验证差距。只支持这批 CIFAR-10 划分内的重建，不证明任意图像、新分布或历史从未见过的图像泛化。
4. **pre/post 说明什么？** 在相同 1,000 训练 / 200 开发图、相同 2,000 步的浅层对照中，pre 的开发 MSE **0.007980**，post 的为 **0.023019**。支持 pre 中有更容易解码的细节，不能把差值直接称为 merger 不可逆信息损失：post 多一个 3,148,800 参数的输入适配层，初始化抽样轨迹也不完全相同。
5. **语言接口运行了吗？** **NOT_RUN**。主实验结束时原课堂余量不足 60 分钟，未准备远端完整语言权重和算子/数据就绪证据。对接方的 `READY` 是 `READY_CPU_IMPLEMENTATION_ONLY`，并非真实 Qwen/DCU 联训验收；本轮没有临时移植框架。

## 直接查看

- [完整中文报告与全部固定图册](snapshot/REPORT_ZH.md)；[离线 HTML](snapshot/REPORT_ZH.html)
- [补充 pre/post 完整 16 图](posthoc_pre_post_16/README.md)
- [交接与恢复说明](HANDOFF_ZH.md)
- [全部原始指标：正确、同类错配、均值特征、边界/梯度/32×32误差](snapshot/results/metrics.json)
- [实际数据索引](snapshot/indices.json)、[正式训练锁](snapshot/training_locked.json)、[闭合预算账本](snapshot/night_budget.json)
- [实际 hook 的特征契约](VISION_FEATURE_CONTRACT.json)、[独立重载与最终状态](snapshot/results/summary.json)
- [Release：完整权重、AdamW/RNG 恢复状态、原生精度未裁剪预测与图册](https://github.com/www080327-glitch/inverse-vit-lab/releases/tag/night-vision-a-mirror-20261026)

## 相同目标下的主结果

表中 train/dev/final 均来自各自 **dev 选出的 best**；不是最后一步训练 batch 的误差。

| 方法 | train MSE / MAE | dev MSE / MAE | final MSE / MAE |
|---|---:|---:|---:|
| 训练均值图 | 0.059756 / 0.202745 | 0.056375 / 0.196192 | 0.063081 / 0.208525 |
| mirror_tiny | 0.013542 / 0.086544 | 0.021880 / 0.110701 | 0.022460 / 0.111447 |
| mirror_full | 0.008555 / 0.068034 | 0.020566 / 0.106081 | 0.021108 / 0.106977 |

全部未舍入数值以 [metrics.json](snapshot/results/metrics.json) 为准。历史 32×32 线性/MLP 结果只作背景，**不进入本轮公平排名**。

最终集同类错配特征 MSE 为 tiny **0.104908**、full **0.104337**；平均特征输入分别为 **0.068989 / 0.067483**。两者全部 500 张图的正确特征都优于其同类错配，支持具体源图依赖，仍不等于自由生成能力。

full 的整体梯度误差较低，但相对内部的边界超额误差为 **0.043649**，高于 tiny 的 **0.040420**。不能用像素 MSE 的小幅改善声称格纹已经解决。

## 匹配 pre/post 诊断

相同 1,000 训练 / 200 开发图片，各 2,000 步、有效 batch 32；两者共用 2 层、768 宽、12 头、FFN3072 的主干规格，参数随机初始化。

| 输入 | 参数量 / 其中输入适配 | train MSE / MAE | dev MSE / MAE |
|---|---:|---:|---:|
| pre | 14,767,872 / 0 | 0.007364 / 0.062913 | 0.007980 / 0.066283 |
| post + unmerge | 17,916,672 / 3,148,800 | 0.012079 / 0.081339 | 0.023019 / 0.113763 |

本诊断不使用 final，也不与主 2,000 图模型混作公平排名。固定 16 图的补充推理只重现已冻结结果，未追加训练；[逐图复核](posthoc_pre_post_16/REPLAY_CONSISTENCY.json)与原记录一致。

### 必须保留的限制

- 首选 10,000 图候选清单已生成，但最初含冷启动的吞吐探针触发了保守回退；正式训练前锁定 2,000 图，没有根据重建分数改选数据。**这个探针未排除一次性初始化开销，不能证明充分预热后 10,000 图不可行。**
- 官方目标是 32×32 源图插值后的 256×256，不是源图不存在的高分辨率细节。
- 单抽样/初始化种子。开发集用于选 checkpoint，最终集只在主模型配置和选择锁定后打开；“本轮最终集”不声称历史未被其他实验使用。
- full 增大了结构和容量，但没有消除格纹。预设全部 train16、dev32、final32 图都展示，没有挑最好案例。
- pre/post 的输入维度和适配层不同，不能宣称绝对完美控制变量，也不能证明更强解码器或更多训练无效。
- 编码器和 merger 始终冻结、eval；不加载 P2 失败语言权重，不运行 P3、扩散、GAN、外部生图模型或本机 MPS。

## 代码、验证与取回

源入口在 [`snapshot/night_model.py`](snapshot/night_model.py) 和 [`snapshot/night_run.py`](snapshot/night_run.py)。旧 tiny-only 原型保持不变，新入口允许 full；实际训练源版本和两次普通错误的原始记录保存在 Release 报告包中。发布副本只移除了运行位置等平台专用绝对路径，见 [差异清单](PUBLICATION_SANITIZATION.json)。模型核心源码、数值结果和权重不因此改变。

```sh
# 无下载 CPU/meta 检查，不分配真实 full 参数
PYTHONPATH=snapshot python test_model_contract.py

# 下载、核对每个 SHA-256，再原子形成完整快照；不导入模型库
python fetch_artifacts.py --destination downloaded_snapshot
```

已执行的 4 项 [CPU/meta 检查](CPU_CONTRACT_TESTS.json)通过；[实际权重头核对](LOCAL_ARTIFACT_VALIDATION.json)确认参数和层数；两个主模型的独立重载通过。源码中的合成检查不等同于模型质量验收。

best 是权重/配置/归一化的完整推理导出；last 另有完整 AdamW、RNG 和 step 状态。恢复训练仍需要按固定索引重建特征/目标缓存并取得新的明确资源预算，不能改写已闭合账本。

完整本地与持久存储快照已逐文件校验；计算、临时接收通道及课堂资源均已停止。仅补充图册的定型后推理用了约 **9.39 秒**，没有追加训练或编码器前向，记录在 [独立诊断账本](posthoc_pre_post_16/DIAGNOSTIC_LEDGER.json)。
