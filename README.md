# 图文内容分类与少样本适配

状态：已在 Mac CPU 完成来源核验、精确/感知哈希过滤、扩大基线评测、5 种子严格少样本实验与模型保存；尚未进行 VLM 微调。

方向：多模态/CV

对应需求：小红书两份内容治理 JD；OPPO CNN 训练 JD

## 任务

输入图像与文本，预测公开数据集定义的内容类别；不把研究数据标签称为企业真实违规标准。

## 实现阶段

1. 文本单模态基线
2. ResNet 图像分类基线
3. 冻结 CLIP 编码器并训练融合分类头
4. 少样本与困难样本实验
5. 可选：小型 VLM Few-shot 和 LoRA 对比

## 第一阶段起点

先检查 Fakeddit 的数据获取与标签定义，只使用同时具有图像和文本的样本。抽取按类别分层的小子集并保留原划分；去重后再次检查划分。

## 实验与验收

- Macro-F1、各类 Precision/Recall、PR 曲线、混淆矩阵
- 文本/图像/融合消融，标注数量学习曲线
- 重复图像及近似文本泄漏检查，固定测试集
- 错误分布、随机种子、配置和运行日志

## 运行环境

本地先提取小批量特征并训练分类头；小型 ResNet 使用 MPS/CPU。完整 VLM SFT 默认走外部 GPU，具体规模待硬件确认。

不预填效果提升数据；只有可复现的真实结果才进入简历。

## 参考

- https://github.com/entitize/Fakeddit
- https://github.com/mlfoundations/open_clip

## 当前可运行内容

见 QUICKSTART.md。数据限制见 reports/data-source.md；学习重点见 LEARNING.md。当前正式记录使用 runs/baseline-v2 与 runs/clip-v2；第一阶段报告见 reports/phase1-report.md。编码器冻结，分类头训练；不能称作 CNN 端到端调优成果。

## 第二阶段

当前评测版本：`runs/phase2-verified`，数据清单：`data/processed/phase2-verified`。结果与限制见 [第二阶段报告](reports/phase2-report.md)。文本/图像/融合测试 Macro-F1 为 0.5208/0.4992/0.5896。

## 实验实现与源码保存

后续实验按用户要求使用 Python/PyTorch，详见工作区 EXPERIMENT_RULES.md。新的训练入口为 `scripts/train_torch.py`。每次运行在独立目录保存 `source_snapshot/`、`experiment-config.json`、`source-manifest.json`、`environment.json`、`training-log.jsonl`、`.pt` 权重、指标和预测结果。原 scikit-learn 基线保留为历史对照，不能将后补源码快照当作历史精确版本。
