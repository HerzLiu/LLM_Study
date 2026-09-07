---
tags: [ml-dl-foundations, 第7章, 桥接, SGD, PPO, GRPO, RL]
chapter: 7
section: 7.3
---

# 7.3 从 SGD 到 PPO / GRPO

⬅ [02-从交叉熵到next-token-loss](02-%E4%BB%8E%E4%BA%A4%E5%8F%89%E7%86%B5%E5%88%B0next-token-loss.md) | ➡ [04-从损失函数到RM-DPO-RLVR](04-%E4%BB%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%88%B0RM-DPO-RLVR.md)

---

## 🎬 故事比喻：训练算法的"进化树"

```
SGD (1950s)
   ↓ + 动量
SGD + Momentum (1960s)
   ↓ + 自适应学习率
Adam / AdamW (2014/2017)
   ↓ + 用 reward 代替 supervised label
REINFORCE (1992) - 策略梯度
   ↓ + 减 baseline 降方差
Vanilla Policy Gradient
   ↓ + advantage（V 函数）
Actor-Critic
   ↓ + clip 限制更新幅度
PPO (2017) ⭐
   ↓ + 组内相对 advantage 省 critic
GRPO (2024) ⭐ 当前 SOTA
   ↓
你已学的 agentic-rl-knowledge-base
```

> 这一节把第 4 章的 SGD 链到你已学的 agentic-rl §1-§2 的 PPO/GRPO。

---

## 🎯 7.3.1 从监督学习到强化学习

**监督学习**（第 4 章）：

```
有 (x, y) 标签
→ Loss = CE(model(x), y)
→ ∇Loss → SGD/Adam 更新
```

**强化学习**（agentic-rl §1）：

```
没有 (x, y) 标签
→ Agent 在环境里 rollout 一条 trajectory τ
→ 总 reward R(τ)
→ ∇log π(τ) · R(τ) → SGD/Adam 更新
```

→ **不同 loss / 信号源，但更新引擎仍是 SGD 家族**。

---

## 📐 7.3.2 策略梯度（PG）= 反向传播 + reward

回忆 [agentic-rl §1.3](../../agentic-rl-knowledge-base/01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/03-Policy-Gradient%E4%B8%8EActor-Critic.md)：

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t\right]$$

**分解**：

| 组件 | 来自第几章 |
|---|---|
| $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | §4.5 反向传播（这就是 CE loss 对 logits 的梯度形式） |
| $G_t$（累积奖励） | RL 特有（agentic-rl §1.1） |
| $\mathbb{E}_\tau[\cdot]$ | §2.3 期望（用 N 个 trajectory 平均） |
| 用 SGD 更新 | §4.6 优化器 |

→ **PG = "用 reward 当 label" 的反向传播**。
→ 数学上和 CE loss 反向传播**完全同构**。

---

## 🎬 7.3.3 用监督学习术语类比 RL

| 监督学习 | RL（PG） |
|---|---|
| (x, y) 对 | (state, action, reward) trajectory |
| Loss = CE(model(x), y) | "Loss" = $-\log \pi(a \mid s) \cdot G$ |
| Gradient: ∇CE | $\nabla \log \pi \cdot G$ |
| SGD/Adam 更新 | 同样 SGD/Adam |
| 一次前向+反向 | 一次 rollout + 一次反向（**但 rollout 很贵**！） |

→ **唯一根本区别**：监督有 ground truth label，RL 没有，只有延迟稀疏的 reward。

→ 详见 [agentic-rl §1.3 PG 推导](../../agentic-rl-knowledge-base/01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/03-Policy-Gradient%E4%B8%8EActor-Critic.md)。

---

## 🛡 7.3.4 PPO = PG + Clip + KL

直接看 PPO 的目标函数：

$$L^{\text{PPO}} = \mathbb{E}_t[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{ref})$$

**拆解**：

| 组件 | 来自本书哪 |
|---|---|
| $r_t = \pi_\theta / \pi_{\theta_{old}}$（importance ratio） | §2.3 概率比 |
| $\hat{A}_t$（advantage） | §4.4 损失函数概念（"比平均好多少"） |
| $\min, \text{clip}$ | 限制更新幅度（PPO 特有） |
| KL penalty | §2.3 KL 散度 |
| 整体优化 | §4.6 AdamW |

→ **PPO 仍然是反向传播 + AdamW 更新**——只是 loss 长得复杂点。

→ 详见 [agentic-rl §1.4 PPO](../../agentic-rl-knowledge-base/01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)。

---

## 🆕 7.3.5 GRPO = PPO + 组内相对 advantage（省 critic）

GRPO 的目标函数（agentic-rl §2.3）：

$$L^{\text{GRPO}} = \mathbb{E}\left[\frac{1}{K}\sum_{i=1}^K \frac{1}{|y_i|}\sum_t \min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}) \hat{A}_i)\right] - \beta \cdot \text{KL}$$

其中：

$$\hat{A}_i = \frac{R_i - \mu_g}{\sigma_g}, \quad \mu_g = \frac{1}{K}\sum_j R_j$$

**和 PPO 的区别**：

| 维度 | PPO | GRPO |
|---|---|---|
| Advantage 计算 | $G - V$（需要 critic） | $(R - \mu_g) / \sigma_g$（同 prompt 组内统计） |
| Critic | 需要 | **不需要**（省一半显存） |
| 训练成本 | 中 | 低（但 rollout 贵 K 倍） |

→ GRPO **仅仅是 advantage 的算法不同**，其它（clip + KL + AdamW）和 PPO 一样。

→ 详见 [agentic-rl §2.3 GRPO](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)。

---

## 🔄 7.3.6 整个谱系一张图

```
                          通用优化引擎
                          ──────────
                          SGD / Adam / AdamW
                          (§4.6)
                                 │
        ┌────────────────────────┴────────────────────────┐
        │                                                  │
   监督学习                                            强化学习
        │                                                  │
   ∇CE(model(x), y)                              ∇log π(a|s) · A(s,a)
   (你学过的 §4.4 + §4.5)                       (agentic-rl §1.3 PG)
        │                                                  │
        │                              ┌──────────────────┼────────────┐
        │                              │                  │            │
        │                          REINFORCE      Actor-Critic      PPO
        │                          (G 直接当 A)   (V 做 baseline)   (clip + KL)
        │                              │                              │
        │                              │                       ┌──────┴────────┐
        │                              │                       │               │
        │                              │                  AdaPPO         GRPO
        │                              │                                (组内 advantage)
        │                              │                                       │
        ▼                              ▼                                       ▼
   SFT / Pretrain                  Tabular RL                            DeepSeek-R1
   (Happy-LLM §4-§6)              (经典 RL)                              SWE-RL
                                                                         (你的方向)
```

→ **从 §4.6 SGD/AdamW 出发，监督学习一条线，强化学习一条线，最终汇到 GRPO 训 LLM**。

---

## 🎯 7.3.7 你应该带走的 3 件事

1. **PG / PPO / GRPO 仍然是 SGD 家族的更新**——只是 loss 不同
2. **PG 的 ∇log π · A 和 CE 的 ∇log p 数学上同构**——都是反向传播
3. **GRPO 仅仅省了 critic**，其它和 PPO 一样

→ 你看 agentic-rl 全本，再回这里看 SGD → PPO → GRPO 的演化，**所有"高大上"的 RL 算法都不再神秘**。

---

## ⚠️ 7.3.8 常见误解

| 误解 | 真相 |
|---|---|
| "RL 算法和监督学习完全不同" | **不**——更新引擎仍是 SGD/Adam，只是 loss 不同 |
| "PG 不用反向传播" | **用**！只是 loss 是 `-log π · A` 而非 CE |
| "PPO 太复杂学不会" | **就是 PG + 2 个加项（clip, KL）**——比看上去简单 |
| "GRPO 是全新算法" | **是 PPO 的微调**（省 critic 而已） |
| "RL 需要专门优化器" | **不**——AdamW 几乎所有 LLM RL 都用 |

---

## 📌 7.3 节要点

| 算法 | 一句话 |
|---|---|
| SGD | $\theta -= \eta \nabla L$ |
| AdamW | + 动量 + 自适应 lr + weight decay |
| **REINFORCE** | loss = $-\log \pi \cdot G$ |
| **Actor-Critic** | loss = $-\log \pi \cdot A$，A 用 V 算 |
| **PPO** | + clip + KL |
| **GRPO** | + 组内 advantage 替代 V |

**最重要的一句**：
> **PPO / GRPO 没有发明新的更新引擎**——仍是 AdamW 反向传播。
> 它们发明的是**用什么当 "loss / label"**（reward + clip + KL）。

---

## 🔗 延伸阅读

- 上一节：[02-从交叉熵到next-token-loss](02-%E4%BB%8E%E4%BA%A4%E5%8F%89%E7%86%B5%E5%88%B0next-token-loss.md)
- 下一节：[04-从损失函数到RM-DPO-RLVR](04-%E4%BB%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%88%B0RM-DPO-RLVR.md)
- SGD 基础：[06-优化器进化-SGD到AdamW](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/06-%E4%BC%98%E5%8C%96%E5%99%A8%E8%BF%9B%E5%8C%96-SGD%E5%88%B0AdamW.md)
- 反向传播：[05-梯度下降-反向传播完整推导](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md)
- PG 详细：[03-Policy-Gradient与Actor-Critic](../../agentic-rl-knowledge-base/01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/03-Policy-Gradient%E4%B8%8EActor-Critic.md)
- PPO 详细：[04-PPO与KL约束](../../agentic-rl-knowledge-base/01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)
- GRPO 详细：[03-GRPO（组内相对优势）](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)

---

⬅ [02-从交叉熵到next-token-loss](02-%E4%BB%8E%E4%BA%A4%E5%8F%89%E7%86%B5%E5%88%B0next-token-loss.md) | ➡ [04-从损失函数到RM-DPO-RLVR](04-%E4%BB%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%88%B0RM-DPO-RLVR.md)
