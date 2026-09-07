---
tags: [ml-dl-foundations, 第7章, 桥接, RM, DPO, RLVR, BCE]
chapter: 7
section: 7.4
---

# 7.4 从损失函数到 RM / DPO / RLVR

⬅ [03-从SGD到PPO-GRPO](03-%E4%BB%8ESGD%E5%88%B0PPO-GRPO.md) | ➡ [05-你现在掌握的全景图](05-%E4%BD%A0%E7%8E%B0%E5%9C%A8%E6%8E%8C%E6%8F%A1%E7%9A%84%E5%85%A8%E6%99%AF%E5%9B%BE.md)

---

## 🎬 故事比喻：BCE 的 5 个化身

```
你已学 BCE（§4.4.4）:
  L = -[y log p̂ + (1-y) log(1-p̂)]
  → 二分类用

LLM 世界里, 这个公式有 5 个化身:

  1. 二分类 BCE       (§4.4)
  2. Pairwise BT (RM) (agentic-rl §2.1)
  3. DPO loss         (agentic-rl §2.2)
  4. KTO loss         (agentic-rl §2.2 变体)
  5. RLVR 也算（间接）

全是 BCE 的变形。
```

> 这一节让你看 agentic-rl 第 2 章的算法（RM/DPO/RLVR）时，不再觉得它们是"全新东西"。

---

## 🪪 7.4.1 RM 训练 = Pairwise BCE

回忆 agentic-rl §2.1：

$$\mathcal{L}_{\text{RM}} = -\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

**用 BCE 的视角看**：

定义"二分类"：
- 正例：$y_w$ 比 $y_l$ 好（label = 1）
- "预测"：$\sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$

→ BCE 的特例：当 y=1 时 $\mathcal{L} = -\log \hat{p}$，即上式。

**直觉**：「**让 reward model 给好答案打更高分**」——本质是个二分类问题。

→ 详见 [agentic-rl §2.1.4](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/01-RLHF%E4%B8%89%E9%98%B6%E6%AE%B5%EF%BC%88SFT-RM-PPO%EF%BC%89.md)。

---

## 🪪 7.4.2 DPO = "把 RM 隐式塞进 policy"的 BCE

DPO loss（agentic-rl §2.2）：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

**和 RM loss 对比**：

| 部分 | RM | DPO |
|---|---|---|
| 形式 | $-\log \sigma(\Delta r)$ | $-\log \sigma(\Delta r_{\text{隐式}})$ |
| $\Delta r$ 来源 | RM 直接打分相减 | 用 $\pi_\theta / \pi_{ref}$ 隐式表达（见下方公式） |
| 训的参数 | $\phi$（RM） | $\theta$（policy） |

DPO 的 $\Delta r$（隐式）展开：

$$\Delta r_{\text{隐式}} = \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}$$

→ **DPO 数学结构 = RM loss 结构**，只是 reward 由 $\pi_\theta$ 隐式表达。

→ 详见 [agentic-rl §2.2.2 推导](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/02-DPO%EF%BC%88%E5%81%8F%E5%A5%BD%E7%9B%B4%E6%8E%A5%E4%BC%98%E5%8C%96%EF%BC%89.md)。

---

## 🪪 7.4.3 RLVR Reward = "可验证的二分类"

回忆 agentic-rl §2.4：

$$R_{\text{RLVR}}(x, y) = V(x, y) = \mathbb{1}[\, y \text{ 通过 } x \text{ 的所有验证规则}\,]$$

**关键**：$V$ 是 0/1（pass/fail），本质是一个**确定性的"二分类验证器"**——但**不需要学**（直接 pytest 跑就行）。

**和 RM 对比**：

| 维度 | RM | RLVR Verifier |
|---|---|---|
| 形式 | 学出来的标量 | 程序定义的 0/1 |
| 参数 | $\phi$ 几亿个 | **0 参数** |
| 数据需求 | 偏好对几万 | $(x, \text{gold})$ 对 + 自动判定 |

→ **RLVR 是 RM 的"无参数化"版本**——把"学习排序"换成"程序验证"。

---

## 🌳 7.4.4 LLM 训练所有 loss 的家族树

```
                  最大似然 MLE (§2.3.5)
                          │
                ┌─────────┴────────────┐
                │                      │
        监督学习路线              强化学习路线
                │                      │
        ┌───────┴────────┐             │
        │                │             │
    Cross-Entropy     Negative          │
    (多分类)         Log-Likelihood     │
        │                │              │
    next-token CE   = CE 的特例         │
    (Pretrain + SFT)                    │
        │                               │
        │                          Policy Gradient
        │                          ∇log π · A
        │                          (REINFORCE)
        │                               │
        │                          + clip + KL
        │                               │
        │                          PPO (RLHF)
        │                          │
        │                          ├── + 组内相对 advantage
        │                          │   → GRPO
        │                          │
        │                          └── reward 改用 verifier
        │                              → RLVR
        │
        └── BCE（二分类）
              │
              ├── Pairwise Bradley-Terry → RM 训练
              │                                  │
              │                                  └── 不在线 rollout
              │                                      → DPO
              │
              └── KTO / IPO / 各种变体
```

→ **所有 LLM 训练 loss 都能追溯到 MLE → CE / BCE 这两条根**。

---

## 🎯 7.4.5 一张表搞清"哪个 loss 用在哪"

| 阶段 | Loss 形式 | 根 |
|---|---|---|
| **Pretrain** | next-token CE（所有 token） | CE / MLE |
| **SFT** | next-token CE（只 assistant，mask=-100） | CE / MLE |
| **RM 训练** | $-\log \sigma(r_w - r_l)$ | BCE / Bradley-Terry |
| **RLHF PPO** | $L^{\text{CLIP}} - \beta \text{KL}$ | PG + 经典 ML 概念 |
| **DPO** | 隐式 reward 差的 BCE（完整公式见下） | BCE（隐式 reward） |
| **GRPO** | 类 PPO，advantage = 组内相对 | PG + 组内统计 |
| **RLVR** | 同 GRPO，reward = verifier(x, y) | PG + 程序验证 |
| **PRM 训练** | 步骤级 BCE / regression | BCE / MSE |

DPO loss 完整形式：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)$$

→ **没有任何 loss 是"全新发明"**——都是经典 ML loss 的组合或变形。

---

## 🔑 7.4.6 你应该带走的 3 件事

1. **RM / DPO / KTO 等偏好优化 loss = BCE 的变形**
2. **PPO / GRPO / RLVR 的更新 = PG（反向传播 + AdamW）+ 不同 reward**
3. **整个 LLM 训练 = CE（监督部分）+ BCE 变形（偏好部分）+ PG（RL 部分）**

→ 看到任何新论文的 loss，你能用本节的家族树**秒定位**。

---

## ⚠️ 7.4.7 常见误解

| 误解 | 真相 |
|---|---|
| "DPO 是新算法" | 数学上和 RM + PPO 等价（推导出来的），**只是实现简化** |
| "RLVR 是 RL 算法" | RLVR 是**reward 范式**（用 verifier），算法常配 GRPO |
| "PRM 是新 loss" | 给每步打个 BCE 分而已 |
| "RM 训练用 MSE" | **错**！用 Bradley-Terry（pairwise BCE） |
| "BCE 只能二分类" | 这些化身证明：**只要建模成"对 vs 错"就能用** |

---

## 📌 7.4 节要点

| 算法 | 本质 | 桥到第 4 章 |
|---|---|---|
| SFT loss | next-token CE | §4.4 CE |
| RM loss | Pairwise BCE | §4.4 BCE |
| DPO loss | BCE 隐式版 | §4.4 BCE |
| PPO loss | PG + clip + KL | §4.5 PG + §2.3 KL |
| GRPO loss | PG + 组内 advantage | 同 PPO |
| RLVR reward | 程序定义 0/1 | （不是 loss，是 reward） |

**最重要的一句**：
> **LLM 训练的所有 loss 都是 CE / BCE / PG 的组合**——读完本节，你看 agentic-rl 任何论文都不会觉得"全新"。

---

## 🔗 延伸阅读

- 上一节：[03-从SGD到PPO-GRPO](03-%E4%BB%8ESGD%E5%88%B0PPO-GRPO.md)
- 下一节：[05-你现在掌握的全景图](05-%E4%BD%A0%E7%8E%B0%E5%9C%A8%E6%8E%8C%E6%8F%A1%E7%9A%84%E5%85%A8%E6%99%AF%E5%9B%BE.md)
- Loss 基础：[04-损失函数大全](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/04-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%A4%A7%E5%85%A8.md)
- RM / DPO 详细：[00-章节总览](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 公式速查：[04-公式速查表](../../agentic-rl-knowledge-base/%E9%99%84%E5%BD%95/04-%E5%85%AC%E5%BC%8F%E9%80%9F%E6%9F%A5%E8%A1%A8.md)

---

⬅ [03-从SGD到PPO-GRPO](03-%E4%BB%8ESGD%E5%88%B0PPO-GRPO.md) | ➡ [05-你现在掌握的全景图](05-%E4%BD%A0%E7%8E%B0%E5%9C%A8%E6%8E%8C%E6%8F%A1%E7%9A%84%E5%85%A8%E6%99%AF%E5%9B%BE.md)
