# 推荐系统 MIA 统一接口说明

## 1. 目标

本文档说明以下三种推荐系统成员推理攻击如何通过当前仓库的统一接口调用：

- `ME-MIA`
- `Biased-MIA`
- `DL-MIA`

三者都兼容项目中的统一攻击接口：

- 输入：`AttackInput`
- 输出：`AttackOutput`
- 统一入口：`attack.run(attack_input)`
- 标签约定：`1 = member`，`0 = non-member`
- 分数方向：`membership_scores` 越大，越像 member

对应实现文件：

- [me_mia.py](./me_mia.py)
- [biased_mia.py](./biased_mia.py)
- [dl_mia.py](./dl_mia.py)

---

## 2. 运行依赖

最小依赖：

- `numpy`
- `torch`
- `scikit-learn`

安装示例：

```bash
pip install numpy torch scikit-learn
```

如果你要让 `ME-MIA` 直接调用原始 `Attack/utils_memia/RecStudio/utils.py` 的 `score()`，还需要满足 `Attack/utils_memia/` 目录对应的额外依赖。

---

## 3. 统一调用方式

```python
from Attack.shadow_based import AttackInput

attack = SomeAttack(...)
attack_input = AttackInput(...)
attack_output = attack.run(attack_input)
```

最常用的输出字段：

- `attack_output.membership_scores`
- `attack_output.membership_preds`
- `attack_output.evaluation`

其中：

- `membership_scores` 是主结果
- `membership_preds` 是阈值化后的二值预测
- 只要 `AttackInput.membership_labels` 存在，`run()` 会自动计算 `accuracy / auroc / tpr_at_fpr`

---

## 4. AttackInput 字段在推荐系统中的含义

- `target_model`
  - 待攻击的目标推荐模型
  - 如果攻击直接吃预计算特征，可以为 `None`

- `samples`
  - 推理阶段要攻击的数据
  - 对推荐系统方法，通常不是普通 `X`，而是下面几类之一：
    - `score_dict`
    - 差分向量
    - 三路分支特征
    - 用户交互序列与推荐结果

- `membership_labels`
  - 只用于评估，不是推理必须输入

- `signals`
  - 已预计算的攻击信号
  - 例如：`score_dict`、`vectors`、`features`

- `shadow_data`
  - 训练攻击器所需的影子数据
  - 对 `ME-MIA / Biased-MIA / DL-MIA` 都是必须的

- `config`
  - 方法私有超参数和回调函数

---

## 5. ME-MIA

### 5.1 当前实现方式

`ME-MIA` 是“接口层重写 + 原始打分逻辑复用”的模式：

- 统一接口层在 [me_mia.py](./me_mia.py)
- 如果输入已经是 `score_dict`，直接在接口层训练/推理
- 如果输入是 `target_model/shadow_model + datasets`，内部会调用原始 `Attack/utils_memia/RecStudio/utils.py` 的 `score()`

### 5.2 训练输入

`shadow_data` 支持两种形式。

形式 1：直接提供影子分数字典

```python
shadow_data = {
    "member_scores": member_shadow_scores,
    "nonmember_scores": nonmember_shadow_scores,
}
```

形式 2：提供影子模型和数据集

```python
shadow_data = {
    "shadow_model": shadow_model,
    "member_dataset": member_shadow_datasets,
    "nonmember_dataset": nonmember_shadow_datasets,
    "score_split": "train",
}
```

### 5.3 推理输入

支持三种常见形式。

形式 1：未标注 `score_dict`

```python
signals = {
    "score_dict": target_score_dict,
}
```

形式 2：显式 member / non-member 分组

```python
signals = {
    "member_scores": member_target_scores,
    "nonmember_scores": nonmember_target_scores,
}
```

形式 3：目标模型 + 数据集

```python
samples = {
    "member_dataset": member_target_datasets,
    "nonmember_dataset": nonmember_target_datasets,
    "score_split": "train",
}
```

### 5.4 最小示例

```python
from Attack.me_mia import MEMIAAttack
from Attack.shadow_based import AttackInput

attack = MEMIAAttack()

attack_input = AttackInput(
    target_model=target_model,
    samples={
        "member_dataset": member_target_datasets,
        "nonmember_dataset": nonmember_target_datasets,
        "score_split": "train",
    },
    membership_labels=[1] * num_member + [0] * num_nonmember,
    shadow_data={
        "shadow_model": shadow_model,
        "member_dataset": member_shadow_datasets,
        "nonmember_dataset": nonmember_shadow_datasets,
        "score_split": "train",
    },
    config={
        "mia_data_mode": "mean",
        "classifier_mode": "mean",
        "batch_size": 1024,
        "epochs": 300,
        "lr": 1e-3,
        "feature_start": 64,
    },
)

output = attack.run(attack_input)
```

---

## 6. Biased-MIA

### 6.1 当前实现方式

`Biased-MIA` 的统一包装器在 [biased_mia.py](./biased_mia.py)。

它内部实现的是原方法最稳定的核心链路：

1. 计算用户级差分向量
   - 交互物品均值向量
   - 减去推荐物品均值向量
2. 用差分向量训练二分类 MLP
3. 输出成员概率

### 6.2 可接受输入

支持三条路径。

路径 1：直接给差分向量

```python
shadow_data = {
    "member_vectors": member_shadow_vectors,
    "nonmember_vectors": nonmember_shadow_vectors,
}
```

路径 2：给原始交互、推荐结果和 item embedding

```python
shadow_data = {
    "member_interactions": member_shadow_interactions,
    "nonmember_interactions": nonmember_shadow_interactions,
    "member_recommendations": member_shadow_recommendations,
    "nonmember_recommendations": nonmember_shadow_recommendations,
    "item_embeddings": item_embeddings,
}
```

路径 3：只给交互，推荐和 embedding 由回调或模型补齐

```python
attack_input = AttackInput(
    target_model=target_model,
    samples={
        "member_interactions": member_target_interactions,
        "nonmember_interactions": nonmember_target_interactions,
    },
    shadow_data={
        "shadow_model": shadow_model,
        "member_interactions": member_shadow_interactions,
        "nonmember_interactions": nonmember_shadow_interactions,
    },
    config={
        "recommend_fn": recommend_fn,
        "item_embedding_fn": item_embedding_fn,
    },
)
```

其中：

- `recommend_fn` 负责从模型和交互序列生成推荐结果
- `item_embedding_fn` 负责从模型中导出 item embedding
- 如果模型本身暴露了常见字段，`item_embedding_fn` 可以省略
  - 例如 `item_embeddings.weight`
  - 或 `embeddings_item.weight`

### 6.3 最小示例

```python
from Attack.biased_mia import BiasedMIAAttack
from Attack.shadow_based import AttackInput

attack = BiasedMIAAttack(
    hidden_dims=(32, 8),
    batch_size=256,
    lr=1e-2,
    momentum=0.7,
    epochs=15,
)

attack_input = AttackInput(
    target_model=None,
    samples={
        "member_interactions": member_target_interactions,
        "nonmember_interactions": nonmember_target_interactions,
        "member_recommendations": member_target_recommendations,
        "nonmember_recommendations": nonmember_target_recommendations,
        "item_embeddings": item_embeddings,
    },
    membership_labels=[1] * num_member + [0] * num_nonmember,
    shadow_data={
        "member_interactions": member_shadow_interactions,
        "nonmember_interactions": nonmember_shadow_interactions,
        "member_recommendations": member_shadow_recommendations,
        "nonmember_recommendations": nonmember_shadow_recommendations,
        "item_embeddings": item_embeddings,
    },
)

output = attack.run(attack_input)
```

---

## 7. DL-MIA

### 7.1 当前实现方式

`DL-MIA` 的统一包装器在 [dl_mia.py](./dl_mia.py)。

它支持两种层级：

- 完整特征层：直接提供 `vector / semantic / syntax`
- 向量层：只提供差分向量或原始交互，接口内部补齐三路输入

### 7.2 三路特征的来源

优先级如下。

1. 你直接提供三路特征
2. 你提供 `feature_builder` / `feature_extractor` 回调
3. 你只提供差分向量，接口内部用一个轻量投影器构造 `semantic / syntax`

说明：

- 第 1 种最接近论文原实现
- 第 2 种适合你已经有 joint model 或原始特征抽取代码
- 第 3 种是可运行兜底方案，能让统一接口完整闭环，但不等价于原论文 joint-training

### 7.3 可接受输入

路径 1：直接给三路特征

```python
shadow_data = {
    "member_features": {
        "vector": vector_member,
        "semantic": semantic_member,
        "syntax": syntax_member,
    },
    "nonmember_features": {
        "vector": vector_nonmember,
        "semantic": semantic_nonmember,
        "syntax": syntax_nonmember,
    },
}
```

路径 2：给差分向量

```python
shadow_data = {
    "member_vectors": member_shadow_vectors,
    "nonmember_vectors": nonmember_shadow_vectors,
}
```

路径 3：给原始交互、推荐和 embedding

```python
shadow_data = {
    "member_interactions": member_shadow_interactions,
    "nonmember_interactions": nonmember_shadow_interactions,
    "member_recommendations": member_shadow_recommendations,
    "nonmember_recommendations": nonmember_shadow_recommendations,
    "item_embeddings": item_embeddings,
}
```

路径 4：交互 + 回调

```python
config = {
    "recommend_fn": recommend_fn,
    "item_embedding_fn": item_embedding_fn,
    "feature_builder": feature_builder,
}
```

其中 `feature_builder` 可以直接返回：

```python
{
    "vector": ...,
    "semantic": ...,
    "syntax": ...,
}
```

### 7.4 最小示例

```python
from Attack.dl_mia import DLMIAAttack
from Attack.shadow_based import AttackInput

attack = DLMIAAttack(
    hidden_dims=(32, 8),
    batch_size=256,
    lr=1e-3,
    epochs=50,
    alpha=0.1,
    beta=0.0,
)

attack_input = AttackInput(
    target_model=None,
    samples={
        "member_interactions": member_target_interactions,
        "nonmember_interactions": nonmember_target_interactions,
        "member_recommendations": member_target_recommendations,
        "nonmember_recommendations": nonmember_target_recommendations,
        "item_embeddings": item_embeddings,
    },
    membership_labels=[1] * num_member + [0] * num_nonmember,
    shadow_data={
        "member_interactions": member_shadow_interactions,
        "nonmember_interactions": nonmember_shadow_interactions,
        "member_recommendations": member_shadow_recommendations,
        "nonmember_recommendations": nonmember_shadow_recommendations,
        "item_embeddings": item_embeddings,
    },
)

output = attack.run(attack_input)
```

---

## 8. 推荐的主程序接入方式

如果你的主程序想保持完全统一，建议按下面的优先级接入：

1. `ME-MIA`
   - 优先传 `score_dict`
   - 如果没有，就传 `model + datasets`

2. `Biased-MIA`
   - 优先传 `interactions + recommendations + item_embeddings`
   - 如果没有显式 `item_embeddings`，就提供 `item_embedding_fn` 或可导出 embedding 的模型

3. `DL-MIA`
   - 如果你已有 joint model 输出，优先直接传 `vector / semantic / syntax`
   - 如果你只有原始交互，先用 `feature_builder` 接入你自己的特征抽取逻辑
   - 如果只是想先跑通统一接口，可以只传差分向量或原始交互，让接口走内部兜底投影

---

## 9. 可直接运行的示例脚本

统一示例见：

- [rec_mia_demo.py](./rec_mia_demo.py)

运行方式：

```bash
python Attack/rec_mia_demo.py
python Attack/rec_mia_demo.py --method me
python Attack/rec_mia_demo.py --method biased
python Attack/rec_mia_demo.py --method dl
python Attack/rec_mia_demo.py --method compare
```

---

## 10. COMPARE

`COMPARE` is the proposed recommender-system MIA method. Its unified adapter is
implemented natively in this repository:

- [compare_mia.py](./compare_mia.py)

The method trains an attack classifier on shadow-side COMPARE features and
infers target-user membership from target-side COMPARE features.

Default feature definition:

```text
target_ndcg_10
ndcg_diff_alpha_i = target_ndcg_10 - surrogate_ndcg_10_alpha_i
alpha_i in {0.0, 0.1, ..., 1.0}
```

With 11 alpha values, the default feature vector is 12-dimensional.

Minimal usage with precomputed features:

```python
from Attack.compare_mia import CompareMIAAttack
from Attack.shadow_based import AttackInput

attack = CompareMIAAttack()

attack_input = AttackInput(
    target_model=None,
    samples={
        "member_features": target_member_compare_features,
        "nonmember_features": target_nonmember_compare_features,
    },
    membership_labels=[1] * num_target_member + [0] * num_target_nonmember,
    shadow_data={
        "member_features": shadow_member_compare_features,
        "nonmember_features": shadow_nonmember_compare_features,
    },
)

output = attack.run(attack_input)
```

Minimal usage with raw COMPARE metric rows:

```python
attack_input = AttackInput(
    target_model=None,
    samples={
        "member_metrics": target_member_rows,
        "nonmember_metrics": target_nonmember_rows,
        "alpha_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    shadow_data={
        "member_metrics": shadow_member_rows,
        "nonmember_metrics": shadow_nonmember_rows,
        "alpha_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
)
```
