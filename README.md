# 图文内容分类与少样本适配

已完成前两阶段的 Python/PyTorch 重跑，当前所有模型训练、优化、TF-IDF/SVD、标准化与指标使用 Python/PyTorch；源码保存为 .py，模型与特征保存为 .pt。旧实验模型、评测与派生缓存已清除。

## 当前结果

- 第一阶段：runs/pytorch-phase1-resnet、runs/pytorch-phase1-clip。
- 第二阶段：runs/pytorch-phase2-exploratory、runs/pytorch-phase2-verified。
- 详细报告：[重跑报告](reports/pytorch-rerun-report.md)。

每组保存 source_snapshot、experiment-config.json、source-manifest.json、environment.json、input_manifests、training-log.jsonl、采样 ID、.pt 权重、预测、图表与指标。所有当次源码快照不可修改。

当前输入：data/processed/phase1、phase2-exploratory、phase2-verified；原始数据、图片和预训练编码器保留。source_history 仅保留替换前的 Python 源码，不作当前训练入口。

## 范围

公开数据集 6 类图文分类，覆盖内容治理 JD 的分类、数据质量、少样本与模型评测要求；不能直接描述为企业真实违规审核系统。当前编码器冻结，尚未进行 VLM SFT、LoRA 或端到端 CNN 调优。

## 使用

见 [运行说明](QUICKSTART.md)、[学习说明](LEARNING.md) 与上层 EXPERIMENT_RULES.md。源码入口：scripts/train_torch.py、scripts/run_baselines.py、scripts/run_phase2.py；模型实现：src/torch_models.py；实验实现：src/torch_runner.py。
