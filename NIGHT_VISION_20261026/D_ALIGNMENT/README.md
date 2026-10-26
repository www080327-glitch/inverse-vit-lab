# Qwen 固定 64 位置视觉输出对齐（D_ALIGNMENT v0.1.0）

比较语言主干隐藏状态直接输出视觉特征（direct）与 1024→256→1024 小 MLP（mlp）。本交付完成本机 CPU 开发；随机模型检查通过。**真实 Qwen 训练尚未执行，图像重画和文字生图均没有成功结论。**

- [给唯一 DCU 会话 A 的交接文档](HANDOFF_A.md)
- [数据、B 描述快照、归一化及 D 导出合同](DATA_CONTRACT.md)
- [READY 与文件校验和](READY.json)
- [CPU 随机张量检查证据](evidence/cpu_checks.json)
- [缩小随机 Qwen 架构检查](evidence/hf_random_check.json)
- [配对短实验配置](configs/paired_short.example.json)

## 本机 CPU 检查

在已有 PyTorch 环境中执行；不会下载数据或模型。CPU 固定单线程，每次检查设 300 秒截止。

```sh
python verify_delivery.py
python run_alignment.py cpu-test
# 已安装所指定 Transformers 时，可额外检查缩小的随机官方架构：
python tests/hf_random_check.py
```

CPU fixture 的 GRU 小模型、随机缩小 Qwen 架构及假特征，只验证实现。它们不使用官方预训练权重，不验证 0.8B 模型质量、真实视觉输入、DCU 性能或真实训练可行性。

## 实验边界

E 为冻结官方视觉编码器；D 为独立训练并完成不可变导出的镜像解码器；M 从官方原版 Qwen 初始化。原词表、文字输出头和官方文字/图像输入处理保留，视觉输出接口单独实现。

阶段 1 是 D(E(image)) 重建，本包不训练 D。阶段 2 是有源图的预测特征重画；阶段 3 是仅根据原版 Qwen 候选描述预测特征。两阶段分别记录。类别词单独作为基线。描述来自 B 的原版模型候选桥接数据，不能称作已经成功的阶段 2 模型自描述。

模型只预测固定 64 个 post-merger 特征，每个 1024 维；没有 GMM 参数输出、扩散、GAN、强化学习或付费教师。直接同维度不代表分布已对齐；两条路线都需要训练。确定性预测不验证多样性，可能平均化。

## A 的入口

`lock_manifest.py` 根据 A 提供的官方模型、数据、D 完成导出、编码器可选独立权重和文字重放路径创建本地配置；`run_alignment.py run-pair` 按同一配置顺序运行 direct 与 mlp。全参是默认候选，A 必须在开始前明确锁定 `full` 或 `last4`。程序不自动降级、不连接 SSH、不寻找或接管其他会话的训练进程。

完整命令、运行上限、显存估算口径和未验证事项见 [HANDOFF_A.md](HANDOFF_A.md)。本仓库不包含权重、原始图片、特征缓存、候选描述内容、凭据或本机私有路径。示例配置必须由 A 补齐并锁定后才能运行。
