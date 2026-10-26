# 给唯一 DCU 会话 A：Qwen 视觉输出对齐 v0.1.0

## 交付结论

代码已完成，单线程 CPU 随机模型检查通过。真实官方 0.8B 权重、真实视觉 processor、真实 E/D 配对、DCU 训练和生成质量 **均未验证**。B 的真实 640+80 描述快照及 D 的正式完成导出未在本轮消费。READY 表示代码可交接，不表示真实训练可以跳过资源与合同检查。

本包只由 A 在资源允许时执行真实训练。开发会话没有连接 DCU、使用 MPS、加载完整 0.8B、接管进程或修改其他会话目录。GitHub 发布仅包括本轮代码、模板、文档和去除私有路径的 CPU 证据。

## A 开始前锁定

1. 获取官方原版 `Qwen/Qwen3.5-0.8B` 的固定 revision。本例为 `2fc06364715b967f1860aea9cf38778875588b17`；与 E 缓存及 D 导出必须一致，不能使用 P2 退化权重。A 负责核对官方获取来源，文件哈希只能证明本地文件没有变化，不能独立证明其从未被改权重。
2. 取得 D 拥有者的已完成不可变导出；含权重、源码、全部本地依赖、原始表示与实际归一化合同。不要填仍在写入的 checkpoint。
3. 将固定图像/特征快照与 B 已完成的原版 Qwen 候选描述快照转为 [DATA_CONTRACT.md](DATA_CONTRACT.md) 的 schema，保留全部 640 train + 80 val，按原 ID 与原图哈希连接。描述效果不得影响划分。
4. 准备独立的文字指令重放 train 和 held-out eval JSONL。评测题不得训练。前后实际回复和 rubric 必须保留，不能只依据 WikiText CE 宣称保留指令能力。
5. 评估可用资源后，明确选择 `full` 或 `last4`，两组保持一致。默认候选是 full；如果只允许最后四块，配置和报告必须准确标为 last4。禁止一组 full、另一组 last4。

## 命令

下列命令相对于本交付目录。所有 `private/...` 是 A 提供的本地占位路径；不会自动下载、连接远端或查找其他训练进程。先使用 A 已有的 ROCm 兼容 PyTorch 环境，检查依赖；不要盲目重装 A 的 torch。ROCm 在 PyTorch 中的设备名仍为 cuda。

```sh
# 本机/环境检查：随机张量，不加载官方权重。
python verify_delivery.py
python run_alignment.py cpu-test
python tests/hf_random_check.py

# 读取两个 complete 快照，生成新的不可变 joined 快照。
python run_alignment.py join-snapshots \
  --images private/image_features/snapshot.json \
  --descriptions private/B_descriptions/snapshot.json \
  --output private/joined

# A 先按资源编辑 example 的步数/时间等，再明确锁定一种 scope。
# full 默认优先；若只能运行末4层，把 --scope 改为 last4，两组一起改。
python lock_manifest.py --executor A --scope full --attest-original \
  --model private/official_qwen \
  --snapshot private/joined/snapshot.json \
  --decoder-export private/D_export/export.json \
  --replay-train private/instruction_train.jsonl \
  --replay-eval private/instruction_eval.jsonl \
  --python-path private/D_code \
  --output alignment.locked.json

# 若 E 需要独立权重，可增加 --encoder-weights private/official_E.safetensors；
# 不传则使用官方模型内置且冻结的视觉 tower；独立 E 权重还会逐状态指纹核对，必须等于该官方 tower。
python run_alignment.py validate --manifest alignment.locked.json

# 只由 A 执行：同一配置先 direct 后 mlp；不会并发争抢设备。
python run_alignment.py run-pair --executor A --device cuda \
  --manifest alignment.locked.json --output runs/paired_short
```

初始模板为每组 40 个优化步、batch 1、累积 1、AdamW lr=1e-5、每 4 步文字重放一次、每组 900 秒（包括加载、训练、评测），共最多约 1800 秒，额外少量进程收尾时间除外。硬性可配置边界为每组最多 200 步与 3600 秒。源 token 上限 2048，视觉输出始终 64，不截断描述后假装完整输入。默认只训练 redraw；feature MSE 权重 1，pixel loss 权重 0。若 A 要同时测试 D 的可微图像损失，可在锁定前设同一个非零 `pixel_loss_weight`，两组必须一致。

配对比较同时检查 M 初始化哈希、冻结 E/D 哈希、实际步数及图像/重放 ID 序列。只要有一组超时、失败、缺少步骤或评测，不能称为已完成的配对短实验。不会自动补时、重跑、扩展预算、降级更新范围或切换模型。

默认每组完整自由生成固定验证快照前 8 张 × 3 模式的特征，每张 64 步；这是有界短实验，不是全部 80 张验证结果。正式全验证需 A 在锁定前设 `evaluation.val_samples=80` 并保证预算。每组保存每步损失/上游梯度、冻结前后检查、特征、图像、每图自由输出指标、语义评分队列及文字前后回复。

## 保存、恢复与独立推理

checkpoint 仅保存更新参数、优化器、随机状态、步数、训练样本日志、原始文字 before 评测及合同身份。冻结组件从原来 hash-locked 的模型/D 恢复。文件完成后最后写 `checkpoint.json`；不覆盖既有输出。断点恢复继续原定总步数和数据顺序，不额外增加训练。

```sh
# 单独恢复某一组；新输出目录，原 manifest 和 interface 必须完全相同。
python run_alignment.py train --executor A --device cuda \
  --manifest alignment.locked.json --interface direct \
  --resume runs/paired_short/direct/step_0020 \
  --output runs/direct_resumed

# 只给描述。请求 JSON 仅含 mode、text，不能包含图像、目标特征或图像 ID。
python run_alignment.py generate --executor A --device cuda \
  --manifest alignment.locked.json --interface direct \
  --checkpoint runs/paired_short/direct/step_0040 \
  --request configs/request_description.json \
  --output runs/description_features.safetensors

# 可替换 request_class_only.json；有源图仅可用 request_redraw.example.json。
# 独立 CPU decode 只载入 D 和统计，不载入 M/E、图片、ID 或目标数据。
python run_alignment.py decode --manifest alignment.locked.json \
  --features runs/description_features.safetensors \
  --output runs/description_rgb.safetensors

# 评分者填写 attribute_queue 中的 scores（0/1）和 judge 后汇总。
python run_alignment.py score-attributes \
  --queue runs/paired_short/direct/evaluation/attribute_queue.jsonl \
  --judgments private/direct_attribute_judgments.jsonl \
  --output runs/direct_attribute_scores.json
```

恢复后的单组不能仅凭目录名与另一组配对：应重新核对完整 sample schedule、初始化/组件哈希和总步数。before 回复来自原始 checkpoint 保存的基线，不把恢复后模型误标成原始模型。独立 decode 是供外部调用的 CPU 路径；本轮只以小型随机 D 检查它，真实大型 D 的 CPU 运行成本未测量。

## 显存估算口径（规划值，未测量）

当前实现将官方 BF16 存储权重加载为 FP32 master 参数，M 内部可使用 BF16 autocast，D/特征合同/损失为 FP32；AdamW 使用 FP32 状态。不能用“0.8B × 2 bytes”估计本实现的全部训练显存。

以总参数约 0.8B、冻结 E 约 0.1B、语言可更新参数约 0.7B 估算，不含 D：

- 全部模型参数：约 `4×0.8B` bytes，即约 2.98 GiB。
- full 梯度：约 `4×0.7B` bytes；两个 Adam 状态约 `8×0.7B` bytes。参数+梯度+状态约 **10.8 GiB**。
- last4 按最后四块约 0.07–0.10B 参数估算：冻结参数仍驻留，但仅这些块有梯度/状态；该部分合计约 **3.8–4.1 GiB**。
- MLP 约 0.525M 参数，FP32 参数/梯度/Adam 合计约 8 MiB；direct 没有这组参数。
- 另加冻结 D 的 `4×P_D` bytes、D 前向/反传激活、语言激活、原文字表 logits、临时算子工作区和框架保留缓存。冻结 D 不意味着训练像素损失时无需保存 D 输入梯度链。

batch 1、source≤2048、64 输出、启用 checkpointing 的保守初始规划范围为 full **14–28 GiB + D 的额外成本**，last4 **6–18 GiB + D 的额外成本**。这不是设备适配或可运行承诺。实际来源长度、D 大小、replay token 数、attention 实现及 ROCm 算子会显著改变峰值；A 必须按实际余量锁定，不能事后把 last4 写成全参。程序输出实际 peak allocated/reserved（如能运行），OOM/超时原样记录为失败或未完成。

## 如何解读结果

1. D(E(image)) 只能支持阶段 1 重建结论。本包不接管 D 的训练或验收。
2. redraw 的自由生成结果只能支持阶段 2 有图重画结论。
3. description 使用 B 的原版模型候选描述，是独立的阶段 3 桥接路径。默认 redraw 训练之后对其推理，也不意味着已做描述生成训练。程序记录 description_training_performed。
4. class_only 单独作为类别词基线。description/class_only 绝不能接收目标图或目标特征作为条件。
5. 64 个位置必须完整自由生成。teacher-forced feature/pixel loss 只作为训练诊断，不能当成自由输出质量。
6. 主体、颜色、布局仅对明确描述属性评分。像素 MSE 为辅助，未表达的精确像素不应强求恢复。属性未评分时标为 pending；原文字回复未按 rubric 评分时标为 unjudged。不能宣称文字能力保留。
7. 确定性接口不验证多样性，也可能产生平均化；本轮不自动引入其他生成架构或付费教师。

如果 A 余量不足，优先完成 redraw 的配对短实验；描述数据与推理路径已经准备，不需要为凑齐阶段 3 结果伪造训练或质量结论。

## 仍未验证

真实 0.8B 官方权重加载与版本兼容、官方 processor 的真实图像/文字输入、D 工厂及真实归一化导出、B 真实快照、实际 640/80 接合、真实训练稳定性、DCU 算子/显存/速度、真实保存恢复、完整 80 张自由评测、主体/颜色/布局满足率、指令能力前后评分、阶段 2 或 3 成功、多样性，均未由本交付证明。
