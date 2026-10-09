# 图文内容分类与少样本适配

**状态：按用户要求于 2026-10-08 搁置，停止新增实验。后续多模态项目集中于 LLaVA 与 Qwen3-VL。已有源码、数据和结果保留。**

**项目主线：双模态特征适配、少样本评估和融合分析。现有 CLIP/ResNet 实验是 baseline；SigLIP 2 冻结特征是新增 baseline，模型更新本身不是主要贡献。**

见 [Baseline 与主要工作边界](PROJECT_SCOPE.md)。新增特征 Adapter 位于 src/feature_adapter.py，尚不能描述为编码器内部微调。

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

## VS Code 与系统 Python

当前开发使用 `/usr/bin/python3`（3.9.6），依赖安装在工作区 `.system-python-packages`，VS Code 自动设置 PYTHONPATH。系统全局包目录未获得写权限。详见 [启动说明](VSCODE_START.md)。


## 当前 VS Code / MPS 入口

以后运行统一在 VS Code。打开运行和调试，选择“07 MPS 图文训练（四个模型）”，按 F5，输入新的实验目录名。默认 train_torch.py 已改为 Float32/AdamW，编码器与分类器均使用可用 GPU；旧实验复现必须指定 --optimizer lbfgs，仍使用 CPU。详情见 VSCODE_START.md。

最新实际运行：runs/vscode-mps-v1。四组测试 Macro-F1：文本 0.5191、图像 0.5082、图文线性融合 0.6435、融合 MLP 0.5904。源代码、VS Code 配置、输入哈希、日志与四份 .pt 权重均保存。来源为既有社区子集，单种子且重复使用测试集；这是探索结果。

## Baseline 与主要方法的验证结果

CLIP 与 SigLIP2 的全量、严格 2/4-shot 对比均已从 VS Code/MPS 完成，固定种子 42/43/44。详细结果和限制见 [实验报告](reports/baseline-main-progress.md)。

| 编码器 | 全量线性融合 | 全量 MLP | 全量特征 Adapter |
|---|---:|---:|---:|
| CLIP | 0.6016±0.0363 | 0.5713±0.0165 | 0.5848±0.0263 |
| SigLIP2 | 0.6467±0.0200 | 0.6682±0.0112 | 0.6321±0.0418 |

指标为测试 Macro-F1，均值±样本标准差。SigLIP2 的对照模型在这些设置下均值更高；初始特征 Adapter 未超过相同编码器的 MLP，不支持将它描述为有效提升或创新成果。编码器冻结，特征适配不是编码器内部微调。测试集已反复使用，结果仅供探索，需独立数据验证。

模型官方权重 SHA256 已验证；所有实验保存 .py 源码快照、配置、环境、输入哈希、训练 ID、日志、.pt 权重和预测。项目测试 19 项已通过（无跳过），包含 CPU/MPS 训练和重载。

## v0.6.0：可学习模态门控与消融

新增 `src/gated_fusion.py`：在冻结的 SigLIP2 图文特征上学习样本级两模态权重，再交给与 baseline 相同的 128 维 MLP。初始权重为 0.5/0.5，分类器初始化与随机数状态匹配 baseline；新增 49,250 参数，因此不是参数量完全相同的对照。

| 训练条件 | MLP baseline | 门控 MLP |
|---|---:|---:|
| 全量 1007 条 | 0.6682±0.0112 | 0.6378±0.0232 |
| 每类 2 条（12 条） | 0.3459±0.0112 | 0.3487±0.0150 |
| 每类 4 条（24 条） | 0.3934±0.0183 | 0.3930±0.0107 |

测试 Macro-F1，三个预设种子 42/43/44 的均值±样本标准差。门控全量退步，少样本差异很小，不能声称稳定提升。固定门控消融在全量时为 0.6566±0.0134，高于动态门控，提示当前学到的权重可能损害泛化；这是探索性解释，需要独立数据验证。

本轮完成 9 组实验、54 个分类头，保存逐样本预测、门控诊断和缺失文本/图片、打乱图片的消融。编码器保持冻结，测试集已反复使用。完整表与逐种子数据见 [门控实验报告](reports/gated-fusion-v0.6.0/summary.md)。下一步优先复核数据来源偏差和验证协议，暂不把门控作为简历中的性能提升点。

版本见 [更新日志](CHANGELOG.md) 与 [历史说明](VERSION_HISTORY.md)。以后每次完成更新并通过检查后，提交 main、创建带说明的版本标签并推送 GitHub。
