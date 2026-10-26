# 数据和数值合同

## 固定表示

第一版限定静态正方形 RGB 图像，PIL bicubic 到 256×256，使用官方 processor 的像素归一化、patch 和 merger 规则；E 以 FP32 执行。pre-merger 网格 `[1,16,16]`，post-merger 网格 8×8，64 个输出按 merged row-major 排列，每个 1024 维。原始目标是官方 `pooler_output`，不是 pre-merger、语言隐藏状态或旧概率头输出。

`representation` 必须逐字段一致，至少包括仓库、不可变 revision、特征位置、顺序、E 精度、预处理规则和官方 preprocessor_config.json 文件哈希和本包固定预处理策略。训练缓存、D 导出和 E 声明必须引用相同对象。任何精度、顺序、分辨率或预处理不一致均需要新的版本与显式适配，禁止仅凭形状相同接通。

官方固定配置来源：[Qwen/Qwen3.5-0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)。适配器开发时检查了本机 Transformers 5.20.2 源码；真实官方权重和真实 processor 没有在本轮加载。

## 不可变快照

只读取 `status=complete` 的提交记录。图像/特征快照和 B 描述快照各用一个 `alignment.snapshot.v1` JSON，引用已封存的 JSONL 路径与 SHA-256；记录文件在提交记录之前写完。读端重新校验哈希，拒绝正在写入的文件。为独立处理本包，只需要把两种上游格式导出为这里的 schema，不需要改动上游训练目录。

图像快照的提交记录：

```json
{
  "schema": "alignment.snapshot.v1",
  "status": "complete",
  "kind": "image_features",
  "records": {"path": "image_records.jsonl", "sha256": "64位哈希"},
  "counts": {"train": 640, "val": 80},
  "split_sha256": "固定划分哈希",
  "representation": {"与配置中的完整表示合同相同": "..."}
}
```

每条图像记录包含：

```json
{
  "image_id": "固定图像ID",
  "split": "train",
  "image": {"path": "images/ID.png", "sha256": "原图文件哈希"},
  "source_features": {"path": "features/ID.safetensors", "sha256": "文件哈希", "key": "features"},
  "rgb_target": {"path": "rgb/ID.safetensors", "sha256": "文件哈希", "key": "rgb"},
  "class_word": "cat"
}
```

`features` 为原始 `[64,1024]` 或 `[1,64,1024]` FP 张量；`rgb` 为与 E 处理后几何完全相同的 `[3,256,256]` 或 `[1,3,256,256]`、范围 [0,1] 的 RGB。禁止把原始 32×32 像素直接当 256×256 损失目标。v1 每记录一个文件和一个样本，不隐式猜测大张量中的索引。

固定划分哈希是保持权威记录顺序的列表的 canonical JSON SHA-256：每项仅含 `image_id`、`image_sha256`、`split`；canonical 方式为 Python `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` 的 UTF-8。B 复用该哈希，不按描述质量重新排序、划分或筛选。

B 描述快照的提交记录同上，但 `kind=original_qwen_descriptions`，增加 `source_model`（repository、不可变 revision 等来源信息）。每行：

```json
{
  "image_id": "固定图像ID",
  "image_sha256": "与图像快照相同的原图文件哈希",
  "split": "train",
  "raw_description": "A small red cat on the left.",
  "structure": {
    "subject": {"value": "cat", "explicit_in_description": true},
    "color": {"value": "red", "explicit_in_description": true},
    "layout": {"value": "on the left", "explicit_in_description": true}
  },
  "uncertainty": ["背景没有明确表达"],
  "source_model": {"repository": "Qwen/Qwen3.5-0.8B", "revision": "固定revision"}
}
```

保留原始描述、完整结构字段、不确定性与来源，不对描述进行润色后替换原始文本。低质量、含糊或表达不足的描述仍保留；空描述或缺行应由 B 重新交付完整快照，读端不能删掉样本。描述中没有明确表达的属性不设为评分目标。

`join-snapshots` 精确连接图像 ID、原图文件哈希和 split，要求 720 条一一对应。结果同时保留原图、源特征、RGB 目标及原始 B 记录。记录和引用文件的哈希在 A 运行前检查，数据加载还会拒绝跨划分重复原图内容。

## 三个条件入口

`Condition(mode,text,image)` 是惟一视觉生成条件对象：

| mode | 模型允许接收 | 评分端允许持有 |
|---|---|---|
| redraw | 官方图像输入与固定重画指令 | 原图/原始特征/RGB 参考 |
| description | 原始描述文本 | 参考图、特征、显式描述属性 |
| class_only | 类别词 | 参考图/特征与给定类别 |

description/class_only 的 `image` 字段必须缺省；额外的 target、feature、ID、检索参数会被拒绝。自由生成函数没有 target 参数，也不读取数据集。评测先完成 64 个预测，再读取评分参考。训练 target 仅进入右移 teacher-forcing 历史与损失。描述不能偷偷加入未表达的结构字段或类别标签。

## 右移和图像控制符

视觉输出开始使用原词表中的 `<|vision_start|>` embedding，后接最多 63 个历史特征。每个预测位置叠加同一固定 8×8 二维 sin/cos 坐标；来源图像保留官方 mRoPE，输出段采用从来源段最大位置继续的线性 mRoPE，另加该二维坐标。

| 预测序号 | 当前位置输入 | 监督目标 |
|---|---|---|
| 0 | vision_start + coord(0,0) | z0 |
| 1 | z0 + coord(0,1) | z1 |
| … | … | … |
| 63 | z62 + coord(7,7) | z63 |

固定输出 64 项后，由协议发出 `<|vision_end|>`，它是确定的外部帧结束符，不是新增词表项或训练得到的停止判断。不把终点放在它预测的同一目标位置。自由生成从空历史开始，每次只加入先前预测，不把真实历史逐点拼接冒充自由输出。D 在完整序列之后调用，不声称前缀像素永久确定。

## 标准化与 D 边界

两组共用 `z=(raw-mean)/std`。raw 模式使用零均值、单位 std；标准化模式使用 **D 实际训练时封存的仅训练集逐通道统计**，包含原来 std floor 后的有效 std。绝不在本实验重新拟合另一份统计，也不在验证集拟合。

两组的目标、teacher-forcing 历史、自由预测与 feature loss 都在同一 z 空间。direct 是语言输出隐藏状态本身作为 z；mlp 是同一隐藏状态经 MLP 得到 z。来源图的官方输入特征仍按官方接口使用 raw，和输出历史的 z 有明确分工。

D 的输入合同必须声明：

- `input_space=raw`：恰好做一次 `raw_hat=z_hat*std+mean` 后交给 D。
- `input_space=model`：D 已训练为接收标准化 z，直接交给 D；不能再次反标准化或再次标准化。

导出特征同时保存 `features_model_space` 和 `features_raw`，独立 decode 会检查两者一致。所有转换都是可微的。冻结 D 只冻结参数与随机/训练状态，训练像素损失时不对整个 D 使用 `no_grad`。

## D 的完成导出

D 拥有者应制作一个新的、已完成的不可变导出目录。参考 [decoder_export.example.json](configs/decoder_export.example.json)，把 status 改为 complete，并填写实际权重、factory 定义源码及它使用的本地依赖源码哈希、kwargs、调用方式、表示合同和归一化合同。`features_grid` 调用 `D(features,(1,16,16))`；`features` 调用 `D(features)`。

D 不能指向仍在更新的训练 checkpoint。本包不复制另一个会话正在写入的 checkpoint，也不猜测其类名或归一化状态。`lock_manifest.py` 会解析 D 完成导出里的相对路径，校验文件并锁入 A 的本地配置。标准化示例见 [standardized_normalization.example.json](configs/standardized_normalization.example.json)。

## 指令重放与语义评分

重放 train JSONL 每行含 `id,split="train",prompt,response`；独立 eval JSONL 含 `id,split="eval",prompt,rubric`。ID 和折叠空白/大小写后的 prompt 必须互不重叠。禁止固定评测题进入重放训练；更深层语义近重复还需 A 审计数据。训练只监督 assistant 回复段，每第 4 个视觉优化步加入重放 CE。前后评测通过原文字头贪心输出，保存 prompt、response、rubric，判分初始为 PENDING_REVIEW。

自由生成会导出 `attribute_queue.jsonl`，按每张预测的哈希准备主体、颜色、布局评分。仅对描述明确支持的属性评分；类别词基线只评分给定主体。人工/经授权的独立评分者为每项填 0 或 1，并填写 judge，使用 `score-attributes` 汇总。未评分时结果明确为 PENDING_JUDGMENTS，不能以像素 MSE 替代语义结论。短描述不承担恢复未描述像素的义务。这个评分接口没有自动调用任何付费教师。
