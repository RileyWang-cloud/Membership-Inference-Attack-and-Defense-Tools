# MIA Benchmark 编排层实现说明（本机 = 3090 侧）

> 依据《分类模型成员推断攻击与防御 Benchmark 实验方案（v2）》与
> `benchmark/WORK_DIVISION.md`（双机完全独立并行版）实现。
> 代码基线：GitHub commit `0f3e83e`（含 LabelSmoothing 提交）。

## 复现入口

```bash
pip install -r requirements.txt

# 数据准备（一次性）
#   Purchase/Texas tgz → benchmark/data/raw/（本机已从 privacytrustlab/datasets 镜像下载）
#   MNIST/CIFAR 由 torchvision 自动下载

python benchmark/scripts/build_manifest.py --dataset purchase --seeds 0,1,2   # W1
python benchmark/scripts/run_all.py --datasets purchase,texas,mnist          # 全流程（可断点续跑）
python benchmark/scripts/check_fairness.py                                   # §21 自动校验
```

单步入口：`train_target.py`（W4）→ `build_bundles.py`（W7/W8）→ `run_attacks.py`
（STEP 6）→ `train_defenses.py`（W9）→ `reattack.py`（STEP 10）→ `aggregate.py`（四张主表）。

## 模块地图

| 模块 | 对应工作项 | 说明 |
| --- | --- | --- |
| `miab/data.py` | W2 | Purchase/Texas 官方 tgz 解析 + 固定种子 universe 采样（替代遗留外部置换文件）；MNIST/CIFAR torchvision |
| `miab/models.py` | W3 | `mlp_512_256_128_64`（RMIA 论文 A 附）、`mnist_cnn`、`resnet18_cifar`（收编）+ GroupNorm 变体（DP-SGD 用，本机不运行） |
| `miab/manifest.py` | W1 | 类分层划分、全局索引、派生种子、sha256、全部 §4 不变量断言 |
| `miab/training.py` / `caching.py` | W4 | fp32 训练循环（§23.1 口径）+ 全分区输出缓存（§7） |
| `miab/eval_common.py` | W5 | 唯一指标出口：AUROC / TPR@FPR（预算内最大 TPR）/ shadow 标准化阈值 Acc（§14.1 逐字实现，NaN 鲁棒） |
| `miab/bundles.py` | W7/W8 | Shadow Bundle（§9.1）/ Reference Bundle（§9.2：4 refs、pool 50% 无偏采样、逐 ref 概率矩阵、in/out 矩阵） |
| `miab/attack_runner.py` | STEP 6/10 | 10 攻击统一编排；metric 族直接算分，QMIA/ShadowBased/RAPID 走仓库类，LiRA/RMIA 按论文公式（见下） |
| `miab/defense_runner.py` | W9 | D01–D08 注册表；超参一律沿用 Clean 配方（§10.2），偏离记 `deviation` |
| `Defense/mixup.py` | W10 | 新写（唯一需新写的防御；MemGuard/LabelSmoothing 已在仓库） |

## 关键协议决策（评审要点）

1. **LiRA offline 变体**：仓库 `compute_lira_scores` 是 IN−OUT 双侧 LLR，纯 offline
   协议下（评估样本对所有 ref 均 OUT）退化为 0。编排层按 LiRA 论文 offline 形式实现：
   logit 空间 `score = (x − μ_out)/σ_out`（单侧 z 分数），复用仓库的
   `extract_true_label_confidences` / `logit_transform`。σ≈0（OUT ref 置信度同时触裁剪
   边界）会使 z 爆炸，裁剪到 ±5（评估侧排序不变，仅稳定 shadow 均值校准）。
2. **RMIA**：逐字复刻 `utils_rmia.compute_rmia_scores` 公式（pr 修正 a、γ 阈值、
   population 比例计数），γ=2、a=0.2/0.3 按 §4.4 配置写入 configs。
3. **RAPID**：完全 Bundle 驱动（orig = −CE(target)，ref_mean = 4 refs 的 −CE 均值，
   calibrated = orig − ref_mean）；仓库 `RAPIDAttack` 的 signals 路径不触碰
   reference_manager，传未训练 stub 满足其 fit 的接口要求。
4. **Attack Accuracy 阈值**：全部攻击（含 LiRA/RMIA/QMIA/RAPID）都在 shadow model
   上重算同型攻击分数做 §14.1 标准化；ShadowBased/RAPID 的 shadow 侧分数取自其
   自身训练数据（有轻微乐观偏差，AUROC 不受影响，已在结果中注明）。
5. **HAMP**：hybrid 防御完整接线——defended target 取 `protected_predictor`
   （rank-preserving replacement），reference logits 来自 Shadow Bundle 的
   shadow_test 输出；超参对齐 Clean 配方，记 deviation。
6. **MemGuard**：按 §10.3 只用 Shadow Bundle（shadow_train/shadow_test 后验）训练
   噪声生成器；仓库实现的 perturb 路径有设备错配 bug，本 benchmark 全程 CPU 运行
   该防御（已记录 deviation）。
7. **DP-SGD**：σ/C 配置在 target_val 上选择（不触碰评估集），最终值记录于
   `defense_DPSGD.json` 的 recipe/deviation；逐样本 autograd 实现（无隐私会计，
   §10.4 的既定取舍）。
8. **Mixup 在表格上的 utility 崩溃**（nonmember acc ≈ 随机）是二值特征凸组合的
   已知失效模式，按协议保留为 benchmark 结果而非 bug。

## 已知结果注记（seed 0 初跑）

- Texas 泄露最重（RAPID AUROC 0.92、RMIA 0.90），Purchase 中等（RAPID 0.83），
  MNIST 接近随机（30ep CNN 几乎不过拟合，符合文献）。
- EarlyStop / RelaxLoss / HAMP(修正接线后) 有效降 AUROC；
  LabelSmoothing / Mixup 在 Purchase 上反而放大泄露（"privacy catalyst"效应，
  与 ICLR 2024 相关研究一致）。

## 目录

严格对齐方案 §16：`configs/ splits/ checkpoints/ bundles/ cache/ results/ logs/`
+ `data/`（raw/processed/torchvision）。
