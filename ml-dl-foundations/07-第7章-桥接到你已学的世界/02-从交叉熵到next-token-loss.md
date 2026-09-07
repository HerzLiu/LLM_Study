---
tags: [ml-dl-foundations, 第7章, 桥接, 交叉熵, next-token, SFT, RM]
chapter: 7
section: 7.2
---

# 7.2 从交叉熵到 next-token loss

⬅ [01-从MLP到Transformer的桥梁](01-%E4%BB%8EMLP%E5%88%B0Transformer%E7%9A%84%E6%A1%A5%E6%A2%81.md) | ➡ [03-从SGD到PPO-GRPO](03-%E4%BB%8ESGD%E5%88%B0PPO-GRPO.md)

---

## 🎬 故事比喻：1000 类分类放大版

```
你前面学的 (§3.2 / §4.4):
  多分类 → softmax + CE loss
  例: 手写数字识别, 10 类

LLM next-token prediction:
  多分类 → softmax + CE loss
  例: 预测下一个 token, 50000 类（词表大小）

数学上一模一样, 只是类别数从 10 → 50000。
```

> 这一节让你看 Happy-LLM 训练代码里 `loss = F.cross_entropy(logits, labels)` 时**完全懂**。

---

## 📐 7.2.1 第 4 章 CE 速回顾

多分类交叉熵：

$$\mathcal{L}_{\text{CE}} = -\log \hat{p}_y = -\log \frac{e^{z_y}}{\sum_j e^{z_j}}$$

其中 $\mathbf{z} \in \mathbb{R}^C$ 是模型输出 logits，$y$ 是真实类别。

→ 详见 [§4.4.5](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/04-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%A4%A7%E5%85%A8.md)。

---

## 🌐 7.2.2 LLM 的 next-token loss

**LLM 在做什么？**

```
输入: "The cat sat on the"   (5 个 token)
输出: 预测下一个 token       (从词表 50000 个中选)

模型架构:
  Token IDs [B, 5]
    ↓ Embedding → Transformer × N
  Hidden state [B, 5, dim]
    ↓ LM Head (nn.Linear(dim, vocab_size))
  Logits [B, 5, vocab_size=50000]
    ↓ softmax + 取 argmax 或 sample
  Next token ID
```

**Loss 是什么？**

对每个位置 t，模型预测第 t+1 个 token 的概率分布 $\hat{p}^{(t)} \in \mathbb{R}^V$，真实第 t+1 个 token 是 $y^{(t)}$：

$$\mathcal{L}_t = -\log \hat{p}^{(t)}_{y^{(t)}}$$

整个序列的 loss：

$$\mathcal{L} = \frac{1}{T-1} \sum_{t=1}^{T-1} \mathcal{L}_t = -\frac{1}{T-1} \sum_t \log P(y^{(t+1)} \mid y^{(1)}, \dots, y^{(t)})$$

→ **每个位置独立做一次 50000 类的多分类**。
→ 数学上**和你学过的多分类 CE 完全相同**，只是规模大。

---

## 🔢 7.2.3 数值小例子

设词表 V = {"cat", "dog", "sat", "the", "on"}（5 个 token）。

句子 "the cat sat":
- t=0 输入 "the" → 应该预测 "cat"（id=0）
- t=1 输入 "the cat" → 应该预测 "sat"（id=2）

模型输出 logits（举例）：

```
位置 0:  logits = [3.0, 1.0, 0.5, 2.0, 0.2]
         softmax = [0.60, 0.08, 0.05, 0.22, 0.04]
         真实 = 0 ("cat")
         loss_0 = -log(0.60) = 0.51

位置 1:  logits = [0.5, 0.3, 4.0, 1.2, 0.8]
         softmax = [0.03, 0.02, 0.85, 0.07, 0.04]
         真实 = 2 ("sat")
         loss_1 = -log(0.85) = 0.16

总 loss = (0.51 + 0.16) / 2 = 0.335
```

→ 模型对位置 1 更确定（loss 小），位置 0 较模糊。

---

## 💻 7.2.4 PyTorch 代码

### 朴素实现

```python
import torch
import torch.nn.functional as F

# 模型输出
logits = model(input_ids)         # [B, T, V]

# 错位: 用前 t 个 token 预测第 t+1 个
shift_logits = logits[:, :-1, :].contiguous()     # [B, T-1, V]
shift_labels = input_ids[:, 1:].contiguous()      # [B, T-1]

# Reshape 成 (B*T-1, V) 和 (B*T-1,)
loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
)
```

→ 这就是 Happy-LLM 第 5 章 SFT 训练里的 loss。

### SFT 特殊：只对 assistant 部分算 loss

```python
# labels 里把 system / user 部分设为 -100
labels = [-100, -100, ..., -100,   # system
          -100, -100, ..., -100,   # user
          token_a1, token_a2, ...] # assistant 真实 token

# CrossEntropyLoss 默认 ignore_index=-100
loss = F.cross_entropy(logits, labels, ignore_index=-100)
# 只对 labels != -100 的位置算 loss
```

→ Happy-LLM §6.3 SFT 实战要点详讲。

---

## 🎓 7.2.5 SFT loss 和"多分类 CE"完全一回事

| 维度 | 多分类 CE（§4.4） | LLM SFT loss |
|---|---|---|
| 输入 | 特征向量 | token 序列 |
| 类别数 | 10（如 MNIST） | **50000**（词表） |
| 模型 | MLP | Transformer |
| 输出 | logits [B, 10] | logits [B, T, V] |
| Loss | $-\log \hat{p}_y$ | $\frac{1}{T} \sum_t -\log \hat{p}^{(t)}_{y^{(t)}}$ |
| 求和 | 1 个位置 | T 个位置求平均 |

→ **唯一区别**：LLM 是 T 个独立多分类问题串起来。

---

## 🎯 7.2.6 Perplexity（PPL）= e^(avg loss)

$$\text{PPL} = \exp(\text{avg NLL}) = \exp(\mathcal{L})$$

如果 loss = 2.0，PPL = $e^{2.0} \approx 7.4$ → "模型平均在 7.4 个候选 token 中犹豫"。

| Loss | PPL | 含义 |
|---|---|---|
| 0 | 1 | 完美 |
| 1 | 2.7 | 良好 |
| 3 | 20 | 中等 |
| 5 | 148 | 一般 |
| 10 | 22026 | 随机 |

→ 看到 paper 报 PPL = 3.5 → $e^{3.5} \approx 33$ → "33 个候选" → 不错。
→ 详见 [§1.5.5](../01-%E7%AC%AC1%E7%AB%A0-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/05-%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87%E5%85%A8%E5%AE%B6%E6%A1%B6.md)。

---

## 🌳 7.2.7 LLM 训练三阶段 loss 对比

| 阶段 | Loss | 区别 |
|---|---|---|
| **Pretrain** | next-token CE（**所有 token**算 loss） | 海量数据，预测下一个词 |
| **SFT** | next-token CE（**只 assistant** 部分算 loss） | 指令对，CE + label_mask=-100 |
| **RM 训练** | Bradley-Terry: $-\log \sigma(r_w - r_l)$ | **不是 next-token CE**，是 pairwise BCE |
| **PPO** | PG clip + KL | 不直接用 CE，但 reward 用 RM |
| **DPO** | 隐式 reward 差 + sigmoid + NLL（完整公式见下） | **BCE 的变形** |
| **GRPO** | 组内相对 advantage + clip + KL | 类 PPO |

DPO 完整 loss：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)$$

→ **所有 SFT / 预训练 loss 都是 CE 的变形**。
→ **所有 RLHF / DPO loss 都是 BCE 的变形**（基于 Bradley-Terry）。

---

## 🔑 7.2.8 你应该带走的 3 件事

1. **LLM next-token loss = 大词表的多分类 CE**——你已学过的，规模大而已
2. **SFT 的 -100 mask** = "只算 assistant 部分的 CE"
3. **Perplexity = exp(loss)** = "模型平均候选数"，直观指标

→ 看 Happy-LLM 训练代码里 `F.cross_entropy(logits, labels, ignore_index=-100)` 这一行，你现在完全懂。

---

## ⚠️ 7.2.9 常见误解

| 误解 | 真相 |
|---|---|
| "LLM 训练用特殊 loss" | **不！** 就是大词表的 CE |
| "SFT 和 Pretrain 用不同 loss" | **同样 CE**，区别在**mask**（SFT 只 assistant 算） |
| "PPL 越低越好" | 训练集 PPL 极低可能 overfit |
| "CE loss 越小，输出越对" | 看任务——可能 reward hacking |
| "LM Head 大头" | 是。LLaMA-7B 词表 32K，LM Head 占 32K × 4096 ≈ 130M 参数 |

---

## 📌 7.2 节要点

| 来源 | 桥接到 |
|---|---|
| §4.4 多分类 CE | → LLM next-token loss |
| §4.4 BCE | → RM Bradley-Terry / DPO loss |
| §1.5 评估指标 PPL | → LLM perplexity |
| §3.2 逻辑回归（softmax 回归） | → next-token prediction（巨型逻辑回归） |

**最重要的一句**：
> **LLM 训练的 loss 全是 CE / BCE 的变形**。
> 你**学透 §4.4**，就懂了 LLM 训练 90%。

---

## 🔗 延伸阅读

- 上一节：[01-从MLP到Transformer的桥梁](01-%E4%BB%8EMLP%E5%88%B0Transformer%E7%9A%84%E6%A1%A5%E6%A2%81.md)
- 下一节：[03-从SGD到PPO-GRPO](03-%E4%BB%8ESGD%E5%88%B0PPO-GRPO.md)
- CE Loss 基础：[04-损失函数大全](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/04-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%A4%A7%E5%85%A8.md)
- 已有锚点：[06-SFT有监督微调](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-SFT%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md)
- 已有锚点：[03-SFT实战要点](../../Happy-LLM/06-%E7%AC%AC6%E7%AB%A0-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%AE%9E%E8%B7%B5/03-SFT%E5%AE%9E%E6%88%98%E8%A6%81%E7%82%B9.md)

---

⬅ [01-从MLP到Transformer的桥梁](01-%E4%BB%8EMLP%E5%88%B0Transformer%E7%9A%84%E6%A1%A5%E6%A2%81.md) | ➡ [03-从SGD到PPO-GRPO](03-%E4%BB%8ESGD%E5%88%B0PPO-GRPO.md)
