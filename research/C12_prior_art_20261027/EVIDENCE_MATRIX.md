# C12 证据矩阵：12 篇方法级核查

日期：2026-10-27。以下均为所列固定版本的方法阅读，不是全文每行、每张实验表或全部代码的复现。论文性能结果归作者报告；本轮没有运行任何论文模型。

## A. 视觉表示、解码与语言生成

| ID / 原文 | 实际机制、训练与依赖 | 与当前想法重合 | 不能等同之处 / 原文定位 |
|---|---|---|---|
| A1 [RAE — Diffusion Transformers with Representation Autoencoders，2510.11690v1](https://arxiv.org/html/2510.11690v1) | 冻结预训练表示编码器；训练 ViT 解码器，包含 L1/LPIPS/GAN；连续语义潜空间；后续为扩散生成 | 冻结语义视觉塔＋学习到的反向图像通路；表示扰动训练 | 不逆用原权重，不研究原 Qwen 输入接口的轻量输出适配。§3、§4.3、附录 C 为相关位置；§4.3 已有噪声增强 |
| A2 [Argus-Unified，2607.25527v1](https://arxiv.org/html/2607.25527v1) | Stage1 冻结 VLM 视觉编码器，训练量化器和图像解码器；Stage2 训练预训练语言主干及两个接口；理解连续、生成离散 | 两阶段将已有 VLM 扩展为理解/生成模型的组织方式高度相近 | 生成依赖 VQ 与离散预测，不是我们的无量化连续输出。§3.1–3.2、图3、附录 A.1；未将它的成本与我们校园卡价直接比较 |
| A3 [UniFluid — Unified Autoregressive Visual Generation and Understanding with Continuous Tokens，2503.13436v1](https://arxiv.org/html/2503.13436v1) | 统一自回归主干；理解前缀使用 SigLIP；生成使用连续 VAE 表示及逐 Token 扩散头；文字分类损失和视觉损失联合 | 连续图像与离散文字在统一序列中预测 | 自回归不排除内部扩散；理解/生成不是同一个原始编码器输出。§3.1–3.2、§4.1–4.2 |
| A4 [STARFlow2，2605.08029v1](https://arxiv.org/pdf/2605.08029v1) | 冻结 Qwen2.5-VL 通路与可训练 TARFlow 通路交错；FAE/DINO 表示；连续归一化流，似然训练；分阶段激活接口 | 连续、非 VQ、非扩散的因果统一生成与预训练能力保持 | 增加流通路，并非原 Qwen 视觉塔的直接镜像。normalizing flow 不等于 ODE flow matching；额外浅流与主干计算需计费。§2–3，PDF图2/3（页3/5） |
| A5 [GIVT，2312.02116v4](https://arxiv.org/html/2312.02116v4) | 线性连续向量输入；高斯混合输出、似然训练；β-VAE 编解码；因果或 masked 生成 | 连续向量 Transformer 以及不使用视觉离散词表的输出头 | 主要不是直接 RGB，也不是对现有 VLM 的原视觉通路做轻量适配。§3.1–3.2、混合头相关附录 |
| A6 [EAR — Continuous Visual Autoregressive Generation via Score Maximization，2505.07812v1](https://arxiv.org/html/2505.07812v1) | 连续视觉潜表示；条件＋随机输入的神经生成头；能量评分；官方训练基于 masked 框架与预训练连续 VAE | 不必限定 GMM，也不必显式求输出密度的连续生成 | 不是将一个确定性线性像素头改名；未证明既有 VLM 的原 post-merger 空间天然适合此目标。§4，官方 scoreloss.py |

## B. 高分辨率、问题相关细节和动态视觉计算

| ID / 原文 | 实际机制 | 选择发生在哪里 | 保留与缺失的边界 / 阅读位置 |
|---|---|---|---|
| B1 [VISOR — VISion On Request，2603.23495v1](https://arxiv.org/html/2603.23495v1) | 基础版保留视觉特征，用稀疏 cross-attention 读取，并选择少量 self-attention 层更新；策略网络选计算配置 | **LLM 内部的层/计算配置**；cross-attention 通常持续保留，策略重点调整 SA 配置 | 不保证原始像素无损，不是只对选中空间块才首次编码；可与 token reduction 组合的变体单列。§4.2–4.4；PDF页5图5已查看 |
| B2 [FocusLLaVA，2411.14228v1](https://arxiv.org/html/2411.14228v1) | 视觉引导的局部尺度选择＋文字引导 Token 筛选；区域使用多尺度 max-pooling | projector 先做视觉选择，LLM 中间层再做文本选择 | 已有模型内问题相关筛选；早期压缩不等于保留原始细节供后续再读。§3.1–3.3 |
| B3 [Hierarchical and Query-Aware Visual Token Pruning for High-Resolution LVLMs，2610.24102v1](https://arxiv.org/html/2610.24102v1) | training-free；实例预算→裁剪预算→细 Token 分数；查询与视觉显著性共同选择 | 全部发生在 LLM blocks **之前**，建立在已有视觉编码之上 | query-aware 不自动等于原生推理中反复读图；所检版本未提供后续取回被剪内容机制。§III-A–E |
| B4 [LLaVA-UHD，2403.11703v1](https://arxiv.org/html/2403.11703v1) | 按原始分辨率/宽高比切片，压缩切片特征，用空间 schema 表示布局 | 主要在视觉输入预处理与表示组织 | 能作为高分辨率强基线，但不是随问题动态搜索或无损像素库。§3.1–3.3 |
| B5 [ZoomEye，2411.16044v3](https://arxiv.org/html/2411.16044v3) | 围绕原始高分辨率图片建树；依问题、存在/分辨率置信与停止条件多轮探索裁剪 | **外部搜索/多次模型调用** | 可以回退探索其他区域，但不是改造内部 Attention。原图读取与全部调用应计成本。§3.1–3.5；官方README已核查 |
| B6 [AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition，2512.03794v1](https://arxiv.org/html/2512.03794v1) | 先看低分辨率全图，必要时调用 bounding-box 工具获取高分辨率局部；DTPO 区分工具与答案训练信号 | **模型决策驱动的外部工具请求** | 自适应视觉读取已有先例；不等于内部层级K/V。§3方法；与另一个同名2024 AdaptVision 不混用 |

## 官方代码核查与限制

以下记录的是实际读到的文件，不表示安装或执行成功。blob SHA 固定该次文件内容；链接中的 main 会随作者更新，应先核对 SHA。不存在可靠入口的条目写“未核实”，不是写“没有代码”。

| 工作 | 本次核查入口 | 内容标识 / 状态 |
|---|---|---|
| RAE | [README](https://github.com/bytetriper/RAE/blob/main/README.md)、[src/stage1/rae.py](https://github.com/bytetriper/RAE/blob/main/src/stage1/rae.py) | README `ffd68279d0944c49aeb09ea30fef77716c9a3c8f`；rae.py `add53bac5d5862e4cf02478300a5ff9b7bb2086d`。冻结 encode、可微 decode、可选特征标准化和噪声路径已读；权重未下载 |
| STARFlow2 | [官方 STARFlow README](https://github.com/apple-aiml-research/ml-starflow/blob/main/README.md) | `ff0b36be6dc99a0d14ea9eff16ea3a7bf6c64084`；明确 Code TBD。已发布 STARFlow/V 不等于2 |
| ZoomEye | [官方README](https://github.com/om-ai-lab/ZoomEye/blob/main/README.md) | `8fc37539f99a216f13270595eb5728bb695c3530`；Qwen2.5-VL/InternVL 等适配入口已列；没有运行 |
| AdaptVision | [官方README](https://github.com/AdaptVision/AdaptVision/blob/main/README.md) | `d7bdcf2d64f82010f098758e6a86175e80df2787`；工具调用与评测入口存在，训练例子涉及外部奖励API，不自动照搬运行 |
| GIVT | [big_vision 官方目录](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/givt) | 本项目此前阅读的官方配置与发布页提供配套生成器/VAE；本轮方法对照不等于再次运行验收 |
| EAR | [官方仓库](https://github.com/shaochenze/EAR)、[scoreloss.py](https://github.com/shaochenze/EAR/blob/main/models/scoreloss.py) | 本项目此前阅读的 `scoreloss.py` blob `b72488427e440fe4eb199c4250f59bbf8e13fa73`；本轮没有重跑。明确区分原文方法和发布实现 |
| LLaVA-UHD | [作者仓库](https://github.com/thunlp/LLaVA-UHD) | 有公开入口，本轮重点为原文方法，未逐文件审查整套实现 |
| Argus-Unified / UniFluid / VISOR / FocusLLaVA / B3层级查询剪枝 | 原文、作者关联页面与定向检索 | 本次未确认支持完整复现的作者代码/权重，不因第三方同名仓库就补写为官方实现 |

## 版本与命名陷阱

1. AdaptVision 本表指 `2512.03794` 与 `AdaptVision/AdaptVision`；不是 `2408.16986` / `harrytea/AdaptVision`。
2. ZoomEye 本表指 `om-ai-lab/ZoomEye` 的视觉探索论文，不是其他同名网络检索产品。
3. STARFlow、STARFlow-V 和 STARFlow2 是不同版本，不能将一个的代码状态或参数套到另一个上。
4. RAE 当前源码与 v1 原文的噪声幅度分布存在具体差异；选择其中一种复现时明确版本，不自行宣布论文整体无效。
5. FocusLLaVA 的多尺度池化是 max-pooling；不能沿用未经核实的 average-pooling 描述。
6. VISOR 的“完整视觉信息”指保留其编码表示；视觉编码前的缩放、特征抽取仍可能有损，不是所有原始RGB可逆。

## 核查边界

方法核查不产生新的性能结果。关键图片只核对所列页，其他图表不冒充已经目视检查。未排除未被检索到的同类论文、未来版本、未公开工作；不得以此报告写“首次”“唯一”或法律意义上的可专利性保证。
