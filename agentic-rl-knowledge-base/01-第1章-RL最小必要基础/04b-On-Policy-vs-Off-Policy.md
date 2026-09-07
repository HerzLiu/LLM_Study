---
tags: [agentic-rl, 第1章, on-policy, off-policy, importance-sampling, PPO, DPO, GRPO]
chapter: 1
section: 1.4b
source: "agentic-rl-learning-map/01-boundaries.md §必学 row 7 (on-policy vs off-policy)"
---

# 1.4b On-Policy vs Off-Policy（必学却最常被忽略的一对概念）

⬅ [04-PPO与KL约束](04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md) | ➡ [05-Sparse-Reward与信用分配](05-Sparse-Reward%E4%B8%8E%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D.md)

---

## 🎬 故事比喻：自己练 vs 看录像练

```
On-policy（自己练）:
   "下棋只能用我现在的策略下，下完一盘根据结果改策略，
    然后必须扔掉这盘棋谱，再用新策略下新的一盘"
   → 慢，但数据和当前策略匹配，更新稳

Off-policy（看录像练）:
   "我可以看任何人（包括过去的我）下的棋谱来学习"
   → 快，可以复用大量历史数据；但要小心"棋谱是别人的策略下的"
```

> **PPO 是 on-policy** → 这就是为什么 LLM RLHF 训练贵
> **DPO 是 off-policy** → 这就是为什么 DPO 像 SFT 一样简单
> **GRPO 是 on-policy（K 倍贵）** → 这就是为什么 DeepSeek-R1 训练需要 GPU 集群
>
> 这对概念是**整个第 2 章算法选择的根本依据**——但原始资料只在 `01-boundaries.md` 列了一句"必学"，本节专门展开。

---

## 📐 1.4b.1 精确定义

设 $\pi_\theta$ 是**当前要训的策略**，$\mu$ 是**产生训练数据的策略**（也叫 behavior policy）。

| 类型 | 关系 | 含义 |
|---|---|---|
| **On-policy** | $\mu = \pi_\theta$ | 训练数据必须来自**当前策略**自己 |
| **Off-policy** | $\mu \ne \pi_\theta$（任意） | 训练数据可来自**任何**策略（旧策略 / 其它策略 / 静态数据集） |

**关键区别在一个词**：**采样分布**。

---

## 🎯 1.4b.2 为什么这个区别决定一切

回忆 policy gradient（[§1.3.1](03-Policy-Gradient%E4%B8%8EActor-Critic.md)）：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]$$

注意**期望是对 $\tau \sim \pi_\theta$** 求的——即**必须用当前 $\pi_\theta$ 采样**才能直接估计梯度。

这就是 **on-policy 的硬要求**：
- 每次更新 $\theta$，旧数据就**严格无效**（因为分布变了）
- 必须**重新 rollout** 才能继续训练

### 那为什么 PPO 还能"多 epoch 更新"？

PPO 用了一个**轻微的 off-policy trick**：

$$\nabla_\theta J \approx \mathbb{E}_{\tau \sim \pi_{\theta_{old}}}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)} \cdot \hat{A}_t\right]$$

→ 用**旧策略 $\pi_{\theta_{old}}$ rollout** 一批数据，更新 $\theta$ 几个 epoch（即"重要性采样换分布"）。
→ 但**只能更新一小步**——超出 $1 \pm \epsilon$ 区间 clip 就触发（详见 [§1.4](04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)）。
→ 所以 PPO 严格说是 "**near on-policy**" / "**多步 off-policy 但加 clip 约束**"。

---

## ⚖️ 1.4b.3 Importance Sampling（核心数学技巧）

**问题**：想算 $\mathbb{E}_{x \sim p}[f(x)]$，但只能从 $q$ 采样，怎么办？

**Importance Sampling 公式**：

$$\mathbb{E}_{x \sim p}[f(x)] = \int p(x) f(x) \, dx = \int q(x) \cdot \frac{p(x)}{q(x)} \cdot f(x) \, dx = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $p(x)$ | 目标分布（"想要"的分布） |
| $q(x)$ | 采样分布（"实际能用"的分布） |
| $\frac{p(x)}{q(x)}$ | **importance weight**（重要性权重） |
| $f(x)$ | 你想算期望的函数 |

**人话**："**用 $q$ 采样 + 给每个样本乘上 $p/q$ 比率，期望不变**"。

### 在 RL 里的应用

| 概念 | 对应 |
|---|---|
| 目标 $p$ | 当前策略 $\pi_\theta$ |
| 采样 $q$ | 行为策略 $\mu$（如 $\pi_{\theta_{old}}$） |
| importance weight | $\rho_t = \pi_\theta(a_t \mid s_t) / \mu(a_t \mid s_t)$ |
| $f$ | $\hat{A}_t$ 或 $G_t$ |

### Importance Sampling 的致命问题：方差爆炸

如果 $\pi_\theta$ 和 $\mu$ 差很多（KL 大），有些 $\rho_t$ 会**极大**或**极小**，造成 estimator 方差爆炸：

```
极端例子:
  π_θ(a|s) = 0.5,  μ(a|s) = 0.001
  → ρ = 500 → 这个样本被放大 500 倍
  → 梯度方差爆炸 → 训练崩
```

**这就是为什么**：

- **PPO 加 clip $\rho \in [1-\epsilon, 1+\epsilon]$**：硬性阻止 $\rho$ 离 1 太远
- **PPO 加 KL penalty**：软性阻止 $\pi_\theta$ 离 $\pi_{ref}$ 太远
- **GRPO 也要 clip**（继承 PPO）：同理

→ off-policy **不是免费午餐**——分布差距越大，方差越大，必须用 clip / KL / IS truncation 等技巧约束。

---

## 🆚 1.4b.4 算法分类对照表（最重要的一张）

| 算法 | On/Off | 数据来源 | 数据可复用？ | 每次更新成本 |
|---|---|---|---|---|
| **REINFORCE** | On | 当前 $\pi_\theta$ rollout | ❌ 一次性 | 1 次 rollout / 1 次 update |
| **Actor-Critic (A2C)** | On | 当前 $\pi_\theta$ rollout | ❌ | 1 次 rollout / 1 次 update |
| **PPO** | "**近 On**"（多 epoch off） | $\pi_{\theta_{old}}$ rollout（最近 1 batch） | ✅ 同 batch 内多 epoch | 1 次 rollout / **K 次** update |
| **GRPO** | On（多 sample on） | $\pi_{\theta_{old}}$ 同 prompt sample K 次 | ✅ 组内 | **K 倍** rollout / 1 次 update |
| **DPO** | **Off** | 静态偏好对数据集 | ✅ **完全复用** | 0 次 rollout / 多 epoch |
| **SFT** | **Off**（不算 RL） | 静态指令对数据集 | ✅ 完全复用 | 0 次 rollout / 多 epoch |
| **Q-learning / DQN** | **Off** | replay buffer（任意历史） | ✅ 完全复用 | 0 次 rollout / 1 次 update |
| **SAC** | **Off** | replay buffer | ✅ | 0 次 rollout / 1 次 update |

### 速记规则

> **看到 "rollout" 字样 = on-policy 痕迹**
> **看到 "replay buffer / 静态数据集" 字样 = off-policy 痕迹**

---

## 💰 1.4b.5 为什么这个区别决定训练成本

### On-policy（PPO / GRPO）的成本结构

```
for iteration in range(N):
    [BOTTLENECK] rollout 一批 prompt → response  # 用 vLLM 也要几分钟
                 计算 reward / KL / advantage
    for ppo_epoch in range(K=1~4):
        gradient update
```

→ **每次 update 都需要 fresh rollout** → LLM RL 训练贵的根本原因。
→ GRPO 更贵：同 prompt 要 sample $K=8 \sim 16$ 次。

### Off-policy（DPO）的成本结构

```
# 一次性准备数据
dataset = load_preference_pairs()  # 静态

for epoch in range(N):
    for batch in dataset:
        gradient update  # 和 SFT 一样
```

→ **完全没有 rollout** → 训练成本和 SFT 同级 → 这就是 DPO "**像 SFT 一样训**" 的根源。

### 数值对比（粗略）

| 方法 | 训 1B 模型 1 epoch（粗估） | 主要瓶颈 |
|---|---|---|
| SFT | 2-4 小时（单卡） | GPU forward/backward |
| DPO | 4-8 小时（单卡） | 同 SFT + 多算 ref logprob |
| PPO | **几天到几周**（多卡） | **rollout 占 60%+ 时间** |
| GRPO | PPO 的 K 倍（$K \approx 8$） | rollout 更贵 |

→ 这就是为什么社区涌向 GRPO + 高效 rollout（vLLM）+ async RL（OpenRLHF）的根本动力。

---

## 🎓 1.4b.6 在 Agentic RL 里的实战影响

### 影响 1：算法选择

| 你的资源 | 推荐 |
|---|---|
| 没 GPU / 只有 API | **DPO**（用 GPT-4 当 judge 构造偏好对，然后离线训） |
| 单卡 GPU（24GB+） | **DPO** 或 **小模型 PPO**（如 Qwen3-0.6B） |
| 多卡 GPU 集群 | **GRPO + RLVR** 或 **PPO** |
| 工业级集群 | **DAPO / OpenRLHF / verl** + async RL |

### 影响 2：数据策略

| 算法 | 数据生命周期 | 备注 |
|---|---|---|
| DPO | 一次构造，反复训 | （静态偏好对） |
| PPO | 用一次就过期 | （on-policy 强制 fresh） |
| GRPO | 同 prompt 一次 sample K 个，K 个数据用一轮 | （半 on-policy） |

### 影响 3：debug 难度

| 现象 | On-policy 排查 | Off-policy 排查 |
|---|---|---|
| Reward 不涨 | KL 失控？rollout 质量？value 不准？ | 数据分布偏？β 太小？ |
| 输出漂移 | KL penalty 不够 | reference 选错？ |
| 训练崩 | importance ratio 爆炸？lr 太大？ | overfit 偏好对 |

→ **On-policy 训练问题面更广**（rollout、reward、value、KL 都可能崩）；off-policy 更简单但**天花板低**。

---

## ⚠️ 1.4b.7 常见误解

| 误解 | 真相 |
|---|---|
| "PPO 是 on-policy 所以不能复用 rollout" | **错**。PPO 允许同 batch rollout 多 epoch 更新（轻微 off-policy） |
| "DPO 是 RL" | **半对**。DPO 数学上和 RLHF 等价，但**没有 rollout、没有 on-policy 更新**，更像偏好优化 |
| "Off-policy 一定比 on-policy 弱" | **错**。Q-learning / SAC 等 off-policy 方法在很多任务上 SOTA |
| "Importance sampling 完美 unbiased" | **错**。理论 unbiased，**实际方差爆炸**——必须 clip / truncate |
| "GRPO 不要 critic 就是 off-policy" | **错**。GRPO 仍是 on-policy（每次更新前必须 fresh rollout K 次） |

---

## 📌 1.4b 节要点

| 概念 | 一句话 |
|---|---|
| **On-policy** | 数据必须来自**当前策略**（PPO/GRPO/REINFORCE） |
| **Off-policy** | 数据可来自任何策略（DPO/Q-learning/SAC） |
| **Importance sampling** | 用 $\rho = p/q$ 换分布，**方差易爆炸** |
| **PPO 真实位置** | "**near on-policy**"——allow 多 epoch 但有 clip 约束 |
| **训练成本根源** | on-policy 必须 rollout → 决定了 LLM RL 贵 |
| **算法选择直觉** | 算力足 → on-policy（PPO/GRPO）；算力少 → off-policy（DPO） |

---

## 🔗 延伸阅读

- 上一节：[04-PPO与KL约束](04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)（PPO 的 clip 和 KL 都是为了控制 IS 方差）
- 下一节：[05-Sparse-Reward与信用分配](05-Sparse-Reward%E4%B8%8E%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D.md)
- DPO 详解：[02-DPO（偏好直接优化）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/02-DPO%EF%BC%88%E5%81%8F%E5%A5%BD%E7%9B%B4%E6%8E%A5%E4%BC%98%E5%8C%96%EF%BC%89.md)
- GRPO 详解：[03-GRPO（组内相对优势）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)
- 算法对比总表：[06-五大算法对比表](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/06-%E4%BA%94%E5%A4%A7%E7%AE%97%E6%B3%95%E5%AF%B9%E6%AF%94%E8%A1%A8.md)
- 原始资料：`01-boundaries.md` §"必须补的 RL 基础" row 7
- 经典参考：Sutton & Barto Ch 5.5 (Off-policy Prediction via Importance Sampling)

---

⬅ [04-PPO与KL约束](04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md) | ➡ [05-Sparse-Reward与信用分配](05-Sparse-Reward%E4%B8%8E%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D.md)
