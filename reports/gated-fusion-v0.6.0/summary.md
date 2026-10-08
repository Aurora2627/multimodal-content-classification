# SigLIP2 可学习模态门控实验

全量/每类 2/4 条训练样本，种子 42/43/44，均值±样本标准差。相同训练 ID、分类器初始化、优化器和早停协议。门控额外参数单列；替换模态为训练均值。打乱图片只测配对敏感性，不能证明事实核验能力。固定门控消融保留训练后的分类器，不能替代独立训练的 MLP 基线。

| 每类样本（0=全量） | 模型 | 评测条件 | Macro-F1 | 参数 |
|---:|---|---|---:|---:|
| 0 | text-linear | intact | 0.5065 ± 0.0072 | 4614 |
| 0 | image-linear | intact | 0.5345 ± 0.0199 | 4614 |
| 0 | fusion-linear | intact | 0.6467 ± 0.0200 | 9222 |
| 0 | fusion-linear | missing-text | 0.5324 ± 0.0165 | 9222 |
| 0 | fusion-linear | missing-image | 0.4807 ± 0.0116 | 9222 |
| 0 | fusion-linear | shuffled-image | 0.3698 ± 0.0060 | 9222 |
| 0 | fusion-mlp | intact | 0.6682 ± 0.0112 | 197510 |
| 0 | fusion-mlp | missing-text | 0.5537 ± 0.0122 | 197510 |
| 0 | fusion-mlp | missing-image | 0.4737 ± 0.0126 | 197510 |
| 0 | fusion-mlp | shuffled-image | 0.3736 ± 0.0117 | 197510 |
| 0 | fusion-feature-adapter | intact | 0.6321 ± 0.0418 | 207494 |
| 0 | fusion-feature-adapter | missing-text | 0.4984 ± 0.0485 | 207494 |
| 0 | fusion-feature-adapter | missing-image | 0.4673 ± 0.0293 | 207494 |
| 0 | fusion-feature-adapter | shuffled-image | 0.3620 ± 0.0101 | 207494 |
| 0 | fusion-gated-mlp | intact | 0.6378 ± 0.0232 | 246760 |
| 0 | fusion-gated-mlp | missing-text | 0.5260 ± 0.0086 | 246760 |
| 0 | fusion-gated-mlp | missing-image | 0.4678 ± 0.0119 | 246760 |
| 0 | fusion-gated-mlp | shuffled-image | 0.3866 ± 0.0055 | 246760 |
| 0 | fusion-gated-mlp | fixed-neutral-gate | 0.6566 ± 0.0134 | 246760 |
| 2 | text-linear | intact | 0.2506 ± 0.0197 | 4614 |
| 2 | image-linear | intact | 0.2549 ± 0.0333 | 4614 |
| 2 | fusion-linear | intact | 0.3183 ± 0.0140 | 9222 |
| 2 | fusion-linear | missing-text | 0.2595 ± 0.0105 | 9222 |
| 2 | fusion-linear | missing-image | 0.2372 ± 0.0452 | 9222 |
| 2 | fusion-linear | shuffled-image | 0.2198 ± 0.0052 | 9222 |
| 2 | fusion-mlp | intact | 0.3459 ± 0.0112 | 197510 |
| 2 | fusion-mlp | missing-text | 0.3102 ± 0.0136 | 197510 |
| 2 | fusion-mlp | missing-image | 0.2941 ± 0.0135 | 197510 |
| 2 | fusion-mlp | shuffled-image | 0.2625 ± 0.0211 | 197510 |
| 2 | fusion-feature-adapter | intact | 0.3072 ± 0.0093 | 207494 |
| 2 | fusion-feature-adapter | missing-text | 0.2873 ± 0.0171 | 207494 |
| 2 | fusion-feature-adapter | missing-image | 0.2538 ± 0.0358 | 207494 |
| 2 | fusion-feature-adapter | shuffled-image | 0.2298 ± 0.0434 | 207494 |
| 2 | fusion-gated-mlp | intact | 0.3487 ± 0.0150 | 246760 |
| 2 | fusion-gated-mlp | missing-text | 0.3060 ± 0.0211 | 246760 |
| 2 | fusion-gated-mlp | missing-image | 0.2975 ± 0.0138 | 246760 |
| 2 | fusion-gated-mlp | shuffled-image | 0.2532 ± 0.0413 | 246760 |
| 2 | fusion-gated-mlp | fixed-neutral-gate | 0.3511 ± 0.0150 | 246760 |
| 4 | text-linear | intact | 0.2731 ± 0.0169 | 4614 |
| 4 | image-linear | intact | 0.3056 ± 0.0127 | 4614 |
| 4 | fusion-linear | intact | 0.3569 ± 0.0215 | 9222 |
| 4 | fusion-linear | missing-text | 0.3060 ± 0.0218 | 9222 |
| 4 | fusion-linear | missing-image | 0.2729 ± 0.0202 | 9222 |
| 4 | fusion-linear | shuffled-image | 0.2550 ± 0.0217 | 9222 |
| 4 | fusion-mlp | intact | 0.3934 ± 0.0183 | 197510 |
| 4 | fusion-mlp | missing-text | 0.3242 ± 0.0365 | 197510 |
| 4 | fusion-mlp | missing-image | 0.3020 ± 0.0306 | 197510 |
| 4 | fusion-mlp | shuffled-image | 0.2560 ± 0.0101 | 197510 |
| 4 | fusion-feature-adapter | intact | 0.3500 ± 0.0189 | 207494 |
| 4 | fusion-feature-adapter | missing-text | 0.3261 ± 0.0247 | 207494 |
| 4 | fusion-feature-adapter | missing-image | 0.2839 ± 0.0190 | 207494 |
| 4 | fusion-feature-adapter | shuffled-image | 0.2625 ± 0.0102 | 207494 |
| 4 | fusion-gated-mlp | intact | 0.3930 ± 0.0107 | 246760 |
| 4 | fusion-gated-mlp | missing-text | 0.3254 ± 0.0354 | 246760 |
| 4 | fusion-gated-mlp | missing-image | 0.3012 ± 0.0255 | 246760 |
| 4 | fusion-gated-mlp | shuffled-image | 0.2553 ± 0.0112 | 246760 |
| 4 | fusion-gated-mlp | fixed-neutral-gate | 0.3913 ± 0.0167 | 246760 |

编码器冻结，测试集已用于多轮探索。三种子不足以断言统计显著性；305 条验证集对 12/24 条训练样本的选择影响较大。
