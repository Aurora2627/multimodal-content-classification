# PyTorch 版本历史

main 保留以下前四个版本，均为 Python/PyTorch：

1. 单模态与图文融合基线（pytorch-phase1-clip）
2. 来源核验与严格少样本实验（pytorch-phase2-verified）
3. 系统 Python 3.9 兼容版（system-python-phase1-clip）
4. VS Code/MPS、Float32/AdamW 四模型版（vscode-mps-v1）

前两阶段曾存在非 PyTorch 实现，但已经完成 PyTorch 重跑。本仓库使用重跑时保存的 PyTorch 源码快照；非 PyTorch 源码仅保留在本机，不加入任何提交。提交时间为恢复时间，不冒充原始开发时间。

历史源码保存在各提交的 src、scripts、tests、configs 中。前两版包含 torch.optim.LBFGS 分类器，最新版使用 torch.optim.AdamW。旧版 Float64 分类头使用 CPU；最新 Float32 分类头支持 MPS。

版本源码通过 AST 核验没有 sklearn/joblib 导入，并含真实 PyTorch 优化器与权重保存。实际重跑验证见 reports/pytorch-migration-verification.json。旧源码快照与训练产物保留本机；Git 中的版本为可公开源码及报告。

## 后续更新与标签

5. v0.5.0：SigLIP2 冻结特征 baseline、初始 Adapter、两种编码器三种子及严格少样本对照。对应 a8ff31631a731d6826493da6c8584fc716defbaf；标签是在第六轮更新时补充，用于定位既有提交。
6. v0.6.0：样本级模态门控 MLP、匹配初始化及消融。结果未呈现稳定优势，详见 CHANGELOG.md 和 reports/gated-fusion-v0.6.0。

当前版本由 VERSION 记录。每轮完成后更新说明，提交 main，创建带说明的 Git 标签并同步 main/标签到 GitHub。原始实验快照不可修改；版本标签只标记真实已完成源码，不伪造早期发布时间。

7. v0.6.1：按用户要求搁置项目，仅更新状态，不新增模型实验。
