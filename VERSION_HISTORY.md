# PyTorch 版本历史

main 按顺序保留四个版本，均为 Python/PyTorch：

1. 单模态与图文融合基线（pytorch-phase1-clip）
2. 来源核验与严格少样本实验（pytorch-phase2-verified）
3. 系统 Python 3.9 兼容版（system-python-phase1-clip）
4. VS Code/MPS、Float32/AdamW 四模型版（vscode-mps-v1）

前两阶段曾存在非 PyTorch 实现，但已经完成 PyTorch 重跑。本仓库使用重跑时保存的 PyTorch 源码快照；非 PyTorch 源码仅保留在本机，不加入任何提交。提交时间为恢复时间，不冒充原始开发时间。

历史源码保存在各提交的 src、scripts、tests、configs 中。前两版包含 torch.optim.LBFGS 分类器，最新版使用 torch.optim.AdamW。旧版 Float64 分类头使用 CPU；最新 Float32 分类头支持 MPS。

版本源码通过 AST 核验没有 sklearn/joblib 导入，并含真实 PyTorch 优化器与权重保存。实际重跑验证见 reports/pytorch-migration-verification.json。旧源码快照与训练产物保留本机；Git 中的版本为可公开源码及报告。
