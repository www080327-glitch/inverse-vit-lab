# 第三方来源

自写原型代码和程序化合成数据使用所在仓库的 MIT 许可。以下材料保留各自来源和条件，不被重新标注为本原型的 MIT 材料。

- `sources/qwen2_vl_modeling.py`、`sources/qwen2_vl_processor.py`、`sources/qwen3_5_modeling.py` 是 Qwen Team 与 Hugging Face Transformers 的代码文本快照，Apache 2.0，原头部保留，许可位于 `sources/Transformers_LICENSE`。没有执行这些文件。
- Qwen2-VL、ZoomEye 和 LLaVA-UHD 通过作者论文链接引用；未在交付中重发布整篇论文。
- `data/earth_original.jpg` 与其不插值裁取、可逆分块来自 NASA Goddard Space Flight Center 的 Blue Marble 2002 卫星观测复合图。图像制作：Reto Stöckli；增强与三维合成：Robert Simmon。来源页：[NASA Earth Observatory](https://science.nasa.gov/earth/earth-observatory/the-blue-marble-true-color-global-imagery-at-1km-resolution/)，具体下载地址和校验和见 `sources/provenance.json`。它是观测图像的复合产品，不是单张相机曝光，也不是本实验合成桌面数据。
- `sources/prior_lab_boundary.txt` 是既有本地 CPU 机制实验室中自写代码的边界摘录，不代表引入完整 DeepSeek 实现。
