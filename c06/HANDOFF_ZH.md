# C06交接：源图细节库 + 按需ROI读取

## 给主实验的建议

优先接入 **read_details(handle, directory, rois, expected_rgb_sha256)**，先用完整原始patch库作为最容易审计的基线。返回源图RGB裁块和真实文件read字节统计；不把原图偷偷混进feature-only decoder。

handle独立保存源图RGB的SHA256、源尺寸/dtype、raw_bank相对路径，以及semantic.json句柄。语义摘要以后接真实Qwen特征；本交付中仅用固定种子的4×16 float32合成向量，producer带SYNTHETIC，未运行Qwen。

最小组合：
1. 图像输入时同时保存语义特征和完整原始patch库。
2. 问题驱动定位器先用粗图/语义摘要，输出源坐标ROI；本实验用程序化场景的已知坐标，只验证存储和读取。
3. 调用read_details读取相交块；将RGB裁块另行编码，或用于明确标记的source-assisted重建。
4. 输出has_source_bypass、ROI坐标、源哈希、实际字节；与feature_only的图像误差、问答结果严格分栏。
5. 纯文字生图、缺失源图、错误源哈希、不同视频帧，都应返回不可用或报错；不得生成假的原图残差。

## 文件导航

- detail_bypass.py：pack/unpack、bank文件格式、BankReader、ResidualReader、语义独立输入、read_details。
- scenes.py：全部原创场景与无额外依赖的PNG写入器。
- test_bypass.py：整数往返、边界、重叠、损坏数据、源身份、合成标记测试。
- run_experiment.py：预算限制、固定实验、实际指标、中文报告、ZIP生成。
- outputs/scenes/：4张源图与完整ROI/哈希索引。
- outputs/desktop_*/：raw.bank、coarse/residual两bank及manifest、语义输入和句柄、全部目标的四栏像素对照。
- REPORT_ZH.md：实际字节、误差、ROI与边界证据、结论边界。
- RESULT.json：逐场景/逐ROI结果、read extents、环境和时间。
- PACKAGE.json：交付ZIP实际字节、SHA256与CRC验证结果。

## 不允许混淆的结论

“全部源信息保留”由无损源库保证；“主干Token不减少”与之独立；“每个位置计算量不降低”也独立。
本次未测量主干Token、FLOPs、Qwen速度或端到端性能。原始patch文件不是Qwen token序列。
粗图均值有损，带精确int16残差后才可还原。残差、坐标表、padding、所有头与辅助元数据均计费，不能声称无损压缩。
文件按需读取量是逻辑read返回字节，包含冷启动元数据；不能等同实际磁盘I/O或推断模型加速。

## 后续最小有价值的主实验

固定同一批源图与问题，比较 coarse_only / feature_only / feature_plus_source_ROI / full_source。
ROI策略先统一用oracle，隔离“取回细节是否有用”与“能否选中细节”；再换真实语义定位器。
只允许source-assisted组读bank，feature-only组在API层禁用此能力。
对使用Qwen resize/crop的真实输入，必须保存并验证到源图的坐标映射，本交付仅演示identity坐标。
必要时可增加来源签名、文件校验、加密、多请求缓存或块级熵编码；这些尚未验证，不属于此次成功结论。
