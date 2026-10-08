# 开始使用

当前阶段：真实社区子集的轻量基线。正式性能结论需要更大、来源核验过的评测集。

## 环境

独立环境位于工作区 `.venvs/mm01`，使用 Python 3.12。当前运行进程检测不到 MPS，自动回退 CPU；不是对硬件支持的结论。

从工作区 `/Users/meiying/Documents/Codex/2026-10-07/zhe` 运行：

```bash
.venvs/mm01/bin/python -m unittest discover -s outputs/career-projects/mm01-content-classification/tests -v
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/prepare_data.py
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/run_baselines.py --backend resnet --output runs/my-resnet-run
```

每次输出目录必须新建，防止覆盖已有实验。依赖记录在 requirements-lock.txt；更换机器可建 Python 3.12 虚拟环境再安装锁定依赖。

## CLIP 阶段

CLIP 特征提取及三路对比已在 CPU 跑通，权重已缓存。将 `--backend resnet` 改为 `--backend clip`，指定新的输出目录。冻结编码器，只训练分类器，不称为 CLIP 全参数训练或 VLM 微调。

## 数据格式

`data/processed/{train,val,test}.jsonl` 每行包含 id、text、image（相对项目目录）、image_sha256、label（0..5）、source_url。

数据和权重保存在 data 下，不进入版本控制；实验评测目录为 runs。原始 TSV 的转换后续补充。

## 单条预测

使用 scripts/predict.py，传入 --artifact runs/clip-v2/fusion-classifier.joblib、--image 图片绝对路径、--text 描述文本。只加载本项目生成的可信模型文件。

## 当前报告

见 reports/phase1-report.md 和 runs/clip-v2/summary.md。

## 第二阶段复现

先复用或获取官方元数据，然后核验并运行新的实验目录：

```bash
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/download_official_metadata.py
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/verify_provenance.py
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/run_phase2.py --output runs/my-verified-run --require-verified
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/summarize_phase2.py --run runs/my-verified-run
```

代码自动将数据清单写入 `data/processed/my-verified-run`。当前正式记录是 `runs/phase2-verified`，融合模型为其中的 `fusion-classifier.joblib`，可通过 predict.py 加载。完整来源核验报告见 reports/phase2-report.md。

## PyTorch 训练

从工作区根目录运行，输出目录必须尚不存在：

```bash
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/train_torch.py --output runs/my-torch-run
```

此入口使用来源核验后的清单和缓存 CLIP 特征，训练 text/image/fusion 线性头及 fusion MLP；代码使用 torch.nn、CrossEntropyLoss 和 AdamW。默认最多 60 轮，验证集早停。每次开始时自动保存当前源代码快照，权重用 PyTorch 的 .pt 格式保存。
