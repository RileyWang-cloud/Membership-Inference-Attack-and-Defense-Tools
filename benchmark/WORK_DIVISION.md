# MIA Benchmark 双机分工表（完全独立并行版）

> 依据《分类模型成员推断攻击与防御 Benchmark 实验方案（v2）》制定。
> 前提：两机之间**无网络互联**——不共享 git、不交换代码 / manifest / Bundle / 中间产物、无过程协同；双方从同一仓库快照（GitHub commit `8941d7f`）出发，按同一份方案 v2 **独立实现、独立运行、独立交付**。

---

## 0. 总则

1. **唯一共同契约 = 方案 v2 全文**：划分尺寸（§4）、种子规则（§3）、模型与训练配方（§5）、攻击集合与超参（§8）、防御集合与超参对齐规则（§10）、指标口径（§14）、结果表格式（§16–§20）均以方案为准。双方实现细节可以不同，协议字段不得偏离。
2. **数据集归属即全流程归属**：每个数据集的 manifest → Clean Target → 输出缓存 → Shadow/Reference Bundle → 10 种攻击 → D01–D08 → 重攻击 → 聚合，全部在归属机器上完成，无任何跨机依赖。
3. **可比性来源**：同一 repo 快照 + 同一方案 + 同一 seeds [0,1,2] + 同一指标公式（§14.1 逐字实现）。manifest 因数据集单一主人，**无需跨机一致**。
4. **最终 Benchmark = 两机交付行的拼接**：四张主表（§20）按行拼接即得，纯离线人工合并，无计算依赖。

---

## 1. 数据集归属与计算量

| 机器 | 硬件 | 承担 Benchmark | 预计纯计算（方案 §23.3） |
| --- | --- | --- | --- |
| 本机 | RTX 3090 24GB | B-C1-A Purchase/MLP、B-C1-B MNIST/CNN、B-C2-A Texas/MLP | ≈ 5.5 h 训练 + < 3 h 攻击侧 + 附录缓冲 |
| 远端 | RTX 4090 24GB | B-C2-B CIFAR-10/ResNet18（含 DP-SGD GroupNorm 变体大头） | ≈ 7 h 训练 + < 1.5 h 攻击侧 |

---

## 2. 各机全栈工作项（从同一 repo 快照独立开发，无代码交换）

### 3090 侧（本机）

| # | 工作项 | 备注 |
| - | ------ | ---- |
| W1 | manifest 生成器 | 覆盖本机 3 数据集；类分层、全局索引、派生种子、sha256（§4） |
| W2 | 加载器：Purchase/Texas 新写（官方 tgz + sha256）、MNIST torchvision | |
| W3 | 模型：`mlp_512_256_128_64`（RMIA 论文 A 附规格）、`mnist_cnn` 新写 | ResNet18 收编也在本机完成（通用定义，本机不运行） |
| W4 | Target 训练循环 + 全分区输出缓存（§7） | |
| W5 | `eval_common.py` 评估口径统一（§14.1 shadow 标准化阈值） | |
| W6 | QMIA 图像输入修复（本机 MNIST 上验证） | |
| W7 | Shadow Bundle 序列化（§9.1） | |
| W8 | Reference Bundle + utils_lira/rmia/rapid 三 manager hydrator（§9.2） | 4 refs、pool 50% 采样、纯 offline |
| W9 | 防御配置驱动训练器 D01–D08 | DP-SGD 用现有逐样本实现（表格/MNIST 适用，§10.4） |
| W10 | `Defense/mixup.py` 新写（§10.5） | **本机自行完成**（唯一需新写的防御） |
| W11 | MemGuard 接入（`Defense/memguard.py` 已存在，commit 25117d4 CCS 2019 重写版） | 按 §10.3 协议接入：噪声生成器只用 Shadow Bundle，禁止触碰 target 分区 |
| — | LabelSmoothing 接入（`Defense/label_smoothing.py` 已存在，commit 3289a34） | D08 直接接入，其闭环 demo 可作 runner 模板 |
| W12 | `reattack.py` + `aggregate.py` | 产出本机 3 数据集的四张表行 |
| W13/W14 | requirements.txt、`check_fairness.py`（§21 清单自动校验） | |

### 4090 侧（远端，另一人按方案独立实施）

| # | 工作项 | 备注 |
| - | ------ | ---- |
| W1 | manifest 生成器（CIFAR-10） | 按 §4.2 尺寸与 §3 派生种子规则独立实现 |
| W2 | CIFAR-10 torchvision 加载器 | |
| W3 | ResNet18 收编（CIFAR 风格 3×3 stem）+ **GroupNorm 变体**（DP-SGD 用） | GN 替换 BN 记录 deviation（§10.4） |
| W4–W6 | 训练缓存 / 评估口径 / QMIA 修复 | 与 3090 侧同源不同实现，公式以 §14.1 为准 |
| W7–W9 | Bundle / 防御训练器 | DP-SGD = GroupNorm 变体 + 现有逐样本反传；实测 > 2 h/次再考虑 opacus |
| W10/W11 | Mixup / MemGuard | 远端自行实现或裁剪，协议按 §10 |
| W12 | reattack + aggregate | 产出 CIFAR-10 的四张表行 |
| W13/W14 | requirements、fairness check | 独立实现，§21 清单逐项留档 |

---

## 3. 并行排期（无同步点，各自推进）

| Day | 3090（本机） | 4090（远端） |
| --- | --- | --- |
| 1 | 环境 + P0（W1–W6）+ 表格/MNIST smoke test，回填实测耗时 | 环境 + P0 + CIFAR smoke test，回填实测耗时 |
| 2 | MNIST → Purchase → Texas 各 seed 0 全流程（Stage I–II，10 攻击 clean baseline） | CIFAR-10 seed 0 全流程 |
| 3 | seeds 1、2 + D01–D08（每防御重跑输出缓存） | CIFAR-10 seeds 1、2 + D01–D08（DP-SGD 大头） |
| 4 | 本机重攻击（Loss/Shadow/LiRA/RMIA/QMIA）+ aggregate 本机四张表行；缓冲跑附录（表格/MNIST full-budget LiRA、{1,2,4} ref 消融） | CIFAR-10 重攻击 + aggregate；full-budget LiRA 默认跳过（仅主表完成且剩 > 1 天才跑） |

双方 Day 2–3 纯计算每天仅数小时，其余为各自返工缓冲；不设任何跨机同步节点。

---

## 4. 交付物（各机独立交付，按方案格式）

每机最终交付一套，格式严格对齐方案 §16 目录与 §18/§19/§20 表格：

```text
1. results/          # clean_attack / defense_attack / summary（JSON+CSV）
                     # 每条记录含：dataset, seed, defense, attack,
                     # AUROC / Attack Acc / TPR@1%FPR（图像加 TPR@0.1%、TPR@0%FPR）,
                     # Utility Drop, Privacy Gain, deviation, git commit, torch/cuda 版本
2. splits/           # 本机数据集全部 manifest.json（划分可复现依据）
3. configs/          # 本机数据集全部 yaml
4. logs/             # 训练与攻击日志（含 smoke test 实测耗时回填）
5. 四张主表的本机行  # mean ± std（3 seeds）
```

最终 Benchmark 报告 = 两机交付物按数据集行拼接（离线完成）。

---

## 5. 独立实现的一致性风险与对策

```text
1. 指标口径分叉：TPR@FPR = "FPR ≤ 预算内最大 TPR"；Attack Accuracy 阈值必须走
   shadow 标准化 s' = (s − μ_nm)/(μ_m − μ_nm) 后取 0.5（§14.1 公式逐字实现）
2. 划分分叉：严格按 §4 分区尺寸 + §3 派生种子（shadow=100*seed+50, ref=100*seed+i）
3. 防御超参分叉：一律沿用各数据集 Clean 配方（§5），偏离仅限算法必需并记 deviation
4. 环境分叉：双方记录 Python/torch/CUDA 版本进结果；计时口径 fp32 不开 AMP（§23.1）
5. 起点分叉：双方从同一 GitHub commit（8941d7f）克隆，结果中记录各自实际 commit hash
6. §21 公平性清单双方各自逐项勾选并留档，随交付物提交
```
