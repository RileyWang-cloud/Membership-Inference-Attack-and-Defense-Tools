# 分类模型成员推断攻击与防御 Benchmark —— 完整实验报告

> 依据《分类模型成员推断攻击与防御 Benchmark 实验方案（v2）》执行。
> 双机完全独立并行协议（`benchmark/WORK_DIVISION.md`）：3090 侧承担
> Purchase / Texas / MNIST，4090 侧承担 CIFAR-10，本报告为双机交付的合并汇总。
> 代码基线：GitHub `8941d7f`（3090 侧同步至 `0f3e83e`，仅新增 LabelSmoothing 与
> metric_based 设备修复，无协议影响）。3090 侧交付分支即本分支；4090 侧交付分支
> `benchmark/cifar10-resnet18-4090`（commit `81f2736`）。

---

## 1. 执行概要

| 项目 | 3090 侧（本机） | 4090 侧 |
| --- | --- | --- |
| 数据集 | Purchase/MLP、Texas/MLP、MNIST/CNN | CIFAR-10/ResNet18 |
| seeds | 0, 1, 2（派生种子 shadow=100s+50、ref_i=100s+i） | 同左 |
| 攻击 | 10 种（A01–A10，方案 §8.1） | 同左 |
| 防御 | D01–D08（方案 §10.1） | 同左 |
| 重攻击 | Loss / Shadow / LiRA / RMIA / QMIA | 同左 |
| 环境 | Python 3.13.12 / torch 2.6.0+cu124 / RTX 3090 24G | Python 3.12.3 / torch 2.11.0+cu128 / RTX 4090 24G |
| 公平性校验 | §21 自动校验全部通过 | 每 seed 21 项全部通过 |
| 结果 | `results/summary/`（本机）+ `results/summary_combined/`（合并） | 分支内 `results/summary/` |

数据规模（实测）：Purchase 19,720×600、Texas 10,669×6,169（均按 §4.3 缩减划分）、
MNIST 60k+10k、CIFAR-10 50k+10k（§4.2 划分）。攻击评估集 member:non-member =
1:1（图像 10k+10k；Purchase 3k+3k；Texas 1.7k+1.7k）。Reference 协议：4 refs ×
pool 50% 无偏采样，纯 offline（RMIA 论文低成本设定；γ=2，a=0.2 表格 / 0.3 图像）。

## 2. Table 1 · Target Utility（3 seeds mean±std）

| Dataset | Model | Clean Test Acc | 备注 |
| --- | --- | ---: | --- |
| MNIST | CNN | 0.9882±0.0009 | train/val gap ≈1.5%，几乎不过拟合 |
| Purchase | MLP | 0.6838±0.0148 | train acc 100%，过拟合充分 |
| Texas | MLP | 0.3612±0.0142 | 1.7k 训练样本 / 6169 维，天然欠拟合 |
| CIFAR-10 | ResNet18 | 0.6577±0.0343 | 无增广 100ep 配方（方案 §5.3） |

## 3. Table 2 · Clean Attack Benchmark（AUROC，mean±std）

完整表见 `results/summary_combined/table2_clean_attack.csv`（40 行）。要点：

| Dataset | Loss | Confidence | Shadow | LiRA(offline) | RMIA | QMIA | RAPID |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Texas | 0.890±0.016 | 0.890±0.016 | 0.858±0.003 | 0.857±0.011 | 0.910±0.007 | 0.814±0.010 | **0.930±0.007** |
| Purchase | 0.769±0.010 | 0.770±0.010 | 0.720±0.008 | 0.764±0.008 | 0.756±0.012 | 0.709±0.011 | **0.838±0.009** |
| CIFAR-10 | 0.728±0.028 | 0.740±0.019 | 0.778±0.015 | 0.646±0.016 | 0.784±0.015 | 0.641±0.021 | **0.833±0.012** |
| MNIST | 0.504±0.003 | 0.510±0.002 | 0.514±0.000 | 0.535±0.003 | 0.513±0.001 | 0.517±0.002 | **0.545±0.003** |

结论：

1. **泄露梯度 Texas > CIFAR-10 ≈ Purchase ≫ MNIST**，与各 Target 的过拟合程度一致。
2. **RAPID 在全部数据集上最强**（reference + 校准特征在等预算 4 refs 下依然领先）；
   RMIA 紧随其后，且其 TPR@1%FPR 在低 FPR 段有优势（如 Purchase 0.076 vs Loss 0.009）。
3. MNIST 全部攻击 ≈ 随机：clean CNN 泄露极小，该数据集用于检验协议而非展示攻防差异。
4. 低预算 LiRA（4 OUT refs）弱于其 128-shadow 全预算设定（RMIA 论文中的已知现象），
   AUROC 仍有效度量 reference 族泄露。

## 4. Table 3 · Defense Utility（完整表见 summary_combined/table3）

| Defense | MNIST | Purchase | Texas | CIFAR-10 |
| --- | --- | --- | --- | --- |
| D01 DP-SGD | +0.018 | +0.115 | +0.144 | +0.187 |
| D02 RelaxLoss | +0.065 | +0.082 | −0.040 | −0.111 |
| D03 HAMP | −0.001 | +0.049 | −0.157 | −0.111 |
| D04 EarlyStop | +0.005 | +0.038 | +0.029 | −0.025 |
| D05 AdvReg | +0.005 | +0.034 | −0.001 | −0.059 |
| D06 MemGuard | 0.000 | 0.000 | 0.000 | 0.000 |
| D07 Mixup | +0.002 | +0.183 | −0.098 | −0.132 |
| D08 LabelSmoothing | −0.002 | +0.109 | −0.040 | −0.135 |

（正数 = 掉点。）无增广低精度的 Texas/CIFAR 上，多数训练期正则反而提升精度；
MemGuard 因构造保证（argmax 不变）在所有数据集零损失。

## 5. Table 4 · Defense Privacy（PrivacyGain = clean AUROC − defended AUROC）

完整表见 `summary_combined/table4_defense_privacy.csv`（160 行 = 4 数据集 × 8 防御 × 5 攻击）。

**防御排名（按跨数据集一致性）：**

1. **D01 DP-SGD —— 最强且最全面**。PrivacyGain：CIFAR +0.22~0.28、Texas +0.24~0.33、
   Purchase +0.14~0.19；代价 UtilityDrop 0.19/0.14/0.12（最大）。配置：3090 侧
   σ=0.1/C=1.0（target_val 选定）；4090 侧 σ=1.0/C=1.0 + GroupNorm 变体（各自留档）。
2. **D02 RelaxLoss / D03 HAMP / D04 EarlyStop —— 高性价比区间**。RelaxLoss 对
   reference 族最强（Texas 的 LiRA +0.29、CIFAR 的 Shadow +0.24），MNIST 上掉点 0.065；
   EarlyStop 几乎零成本（Texas +0.18~0.31，UtilityDrop 仅 0.03）；HAMP 稳定居中
   （Texas +0.12~0.17），并伴随精度提升。
3. **D06 MemGuard —— 零 utility 代价、小幅稳定收益**（0~0.06），对学习型攻击
   （ShadowBased）最有效，符合其攻击器模拟的设计目标。
4. **D07 Mixup / D08 LabelSmoothing —— 表格与图像上的"隐私催化剂"**：
   Purchase 的 metric 族攻击 AUROC 不降反升（Loss 0.77→0.94 / 0.93，gain −0.16~−0.19）、
   CIFAR 同向（Loss −0.10 / −0.15）、MNIST 同向（−0.07 / −0.04）；仅 Texas 部分缓解。
   与 "Be Careful What You Smooth For"（ICLR 2024）结论一致：置信度平滑可作
   privacy shield 亦可作 catalyst，取决于攻击特征空间。Mixup 在二值表格上另有
   utility 崩溃（Purchase 0.68→0.50，凸组合失效的已知模式）。
5. **MNIST 无泄露可防**：clean 攻击 ≈ 随机，全部防御 gain ≈ 0（±0.03 内）。

## 6. 偏差（deviation）登记

| 项 | 侧 | 内容 |
| --- | --- | --- |
| DP-SGD σ | 双方 | 3090：σ=0.1（val 扫描 0.1/0.5/1.0 → 0.60/0.29/0.18）；4090：σ=1.0 + GroupNorm(8) ResNet18 |
| DP-SGD 实现 | 双方 | 仓库逐样本 autograd，无隐私会计（方案 §10.4 既定取舍） |
| MemGuard 设备 | 双方 | 仓库 perturb 路径设备错配，3090 全程 CPU / 4090 修正 stats 块后 GPU |
| LiRA 口径 | 双方 | offline 单侧 OUT 高斯；3090 对 z 裁 ±5 稳定阈值校准；4090 未裁剪（AttackAcc≈0.50，以 AUROC 为准） |
| LiRA AttackAcc 跨机比较 | — | 两机 shadow 校准数值路径不同，该列不作跨机比较，AUROC 可比 |
| CIFAR NaN 发散 | 4090 | SGD lr=0.1 无 warmup 偶发发散，重试机制（seed+1000k）留档，最终模型全部收敛 |
| MNIST 弱泄露 | 3090 | 属数据/模型性质，非缺陷；协议原样保留 |

## 7. 公平性与可复现

- 双机各自 §21 清单校验全部通过（manifest 不变量、派生种子、ref 50% 无偏采样、
  单一 target checkpoint、缓存-清单尺寸一致、阈值仅 shadow 校准、TPR@0.1% 口径）。
- 划分唯一来源 = `splits/<dataset>/seed<k>/manifest.json`（含数据 sha256 与 git commit）；
  Shadow/Reference Bundle 在 Clean 与全部防御间复用（非自适应威胁模型）；
  防御与 Clean 严格 paired（同 target_train/val、同评估集）。
- 复现：`pip install -r requirements.txt` →
  `python benchmark/scripts/run_all.py --datasets purchase,texas,mnist`（断点续跑）；
  CIFAR 侧见 4090 分支同名脚本。合并表：`python benchmark/scripts/combine_two_sides.py`
  （需 4090 分支存在于本地 remote ref）。

## 8. 结论与建议

1. 等预算（4 refs）审计场景下，**RAPID/RMIA 是当前最强攻击组合**，部署审计应优先采用。
2. **防御选型**：可用性敏感 → EarlyStop / MemGuard；隐私敏感且容忍掉点 → DP-SGD；
   折中 → RelaxLoss / HAMP。**对表格数据慎用 Mixup/LabelSmoothing**——不仅不防护，
   反而放大 metric 族泄露。
3. 后续可扩展：full-budget LiRA（128 OUT，§4.4 附录）、{1,2,4} ref 免费消融
   （Bundle 已含逐 ref 输出）、adaptive（defense-aware）攻击、online reference 协议。
