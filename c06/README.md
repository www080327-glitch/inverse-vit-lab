# C06 source-detail bypass

独立CPU原型：保存完整源RGB，按ROI读取；与语义编码器可逆性分开评估。

- [中文报告](REPORT_ZH.md)
- [中文交接](HANDOFF_ZH.md)
- [实测结果](RESULT.json)
- [交付ZIP](C06_DELIVERY.zip)
- [ZIP大小与SHA256](PACKAGE.json)
- [固定场景索引](outputs/scenes/SCENES.json)
- [单元测试日志](outputs/TEST_LOG.txt)

## 独立运行

仅NumPy和Python标准库，不依赖本仓库其他代码、不下载数据或模型。

~~~sh
python3 -m venv --system-site-packages /tmp/c06-venv
/tmp/c06-venv/bin/python -c 'import numpy' || /tmp/c06-venv/bin/pip install --only-binary=:all: -r c06/requirements.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /tmp/c06-venv/bin/python c06/run_experiment.py
~~~

解压交付ZIP后可在C06_DELIVERY目录运行 python run_experiment.py。
脚本本身限制2个CPU、2个数值线程、1100秒墙钟/进程CPU；Linux为已验证执行平台。
生成RESULT.json、REPORT_ZH.md、outputs、C06_DELIVERY.zip和外置PACKAGE.json。
PACKAGE.json不放入它描述的ZIP，避免zip哈希自引用；ZIP中含RESULT.json。
固定场景没有训练/验证划分，因为本实验没有训练、选模或参数拟合。

## 最小调用

~~~python
from pathlib import Path
from detail_bypass import save_bank, BankReader

# image: integer H,W,3 array; uint8 is used in the measured scenes
save_bank(Path("raw.bank"), image, tile=32)
with BankReader("raw.bank") as reader:
    crops = reader.read_many([(31, 31, 35, 35), (32, 32, 42, 42)])
    stats = reader.stats()
~~~

完整patch库只排列；整数粗图+精确残差是可还原表示，存储通常扩大。
真实Qwen特征实验本次未运行；语义向量为显式SYNTHETIC。
