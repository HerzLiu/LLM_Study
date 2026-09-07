---
tags: [agentic-rl, 第1章, policy-gradient, actor-critic, REINFORCE, RL基础]
chapter: 1
section: 1.3
source: "agentic-rl-learning-map/01-boundaries.md §必须补的 RL 基础, 04-six-week-roadmap.md §第1周"
---

# 1.3 Policy Gradient 与 Actor-Critic

⬅ [02-Policy-Value-Advantage](02-Policy-Value-Advantage.md) | ➡ [04-PPO与KL约束](04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)

---

## 🎬 故事比喻：教练 + 球员

```
你是教练（algorithm）想让一个球员（policy π）变厉害。

【REINFORCE 模式】（最朴素）
   球员打一场比赛 → 输/赢 → 你说"赢的动作多做，输的少做"
   问题：方差极大。可能赢了但其实 90% 动作都是瞎打

【Actor-Critic 模式】（更稳）
   球员（Actor=π）打 + 解说员（Critic=V）实时点评
   每个动作都得"它比平均好多少"的评分
   问题：要训两个模型；评分不准还会带偏球员
```

> **PPO、GRPO 都是 Actor-Critic 的特殊形式**。
> 本节让你看懂"为什么 reward 能反过来训练 π"——这是所有 RL 算法的底层逻辑。

---

## 🧮 1.3.1 策略梯度定理（核心定理 + 完整推导）

**核心问题**：我想让 $\pi_\theta$ 变好。怎么算梯度 $\nabla_\theta J(\theta)$？

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \gamma^t r_t \right] = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

其中 $R(\tau) = \sum_t \gamma^t r_t$ 是整条轨迹的回报。

**难点**：$\tau$ 的分布**依赖** $\pi_\theta$（$\theta$ 一变，"哪些轨迹会出现"也变），不能直接对 $\theta$ 求导。

### 推导（4 步，每步逐项展开）

**Step 1**：把期望展开成对所有可能轨迹的求和（积分）

$$J(\theta) = \int P(\tau \mid \theta) R(\tau) \, d\tau$$

其中 $P(\tau \mid \theta) = P(s_0) \prod_t \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$（见 [§1.1.4](01-MDP%E4%B8%8E%E8%BD%A8%E8%BF%B9%EF%BC%88%E7%94%A8Agent%E8%AF%AD%E8%A8%80%E8%AE%B2RL%EF%BC%89.md)）。

**Step 2**：对 $\theta$ 求梯度（$R(\tau)$ 不依赖 $\theta$，可拿到积分内）

$$\nabla_\theta J(\theta) = \int \nabla_\theta P(\tau \mid \theta) \cdot R(\tau) \, d\tau$$

**Step 3**：用 **log-derivative trick**（关键技巧）

$$\nabla_\theta P(\tau \mid \theta) = P(\tau \mid \theta) \cdot \nabla_\theta \log P(\tau \mid \theta)$$

（这是恒等式 $\nabla \log f = \nabla f / f$ 的变形）

代回：

$$\nabla_\theta J(\theta) = \int P(\tau \mid \theta) \cdot \nabla_\theta \log P(\tau \mid \theta) \cdot R(\tau) \, d\tau$$
$$\quad = \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log P(\tau \mid \theta) \cdot R(\tau)\right]$$

**Step 4**：展开 $\log P(\tau \mid \theta)$，发现**只有策略项含 $\theta$**

$$\log P(\tau \mid \theta) = \log P(s_0) + \sum_t \log \pi_\theta(a_t \mid s_t) + \sum_t \log P(s_{t+1} \mid s_t, a_t)$$

对 $\theta$ 求导，**初始分布和环境转移都是常数**（不含 $\theta$），全部消失：

$$\nabla_\theta \log P(\tau \mid \theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

→ **这就是为什么 RL 不需要知道环境模型**——梯度里没有 $P$！

**最终策略梯度定理**：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau) \right]}$$

### 逐项再解读这个公式

| 符号 | 含义 | 形状 |
|---|---|---|
| $\nabla_\theta$ | 对 $\theta$ 的梯度算子 | 输出和 $\theta$ 同形状的向量 |
| $\log \pi_\theta(a_t \mid s_t)$ | 这步动作的对数概率 | 标量 |
| $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | 对参数求导后的 score function | 和 $\theta$ 同形状 |
| $R(\tau)$ | 整条轨迹的回报 | 标量 |
| $\sum_t$ | 对一条轨迹的所有时间步加和 | 标量乘法 |
| $\mathbb{E}_{\tau \sim \pi_\theta}$ | 对多条轨迹求期望（实际用采样估计） | 平均 |

**人话翻译**：

> **梯度 = "这条轨迹每个动作的 log 概率梯度" × "整条轨迹的总奖励"**
>
> 直觉：**奖励高的轨迹 → 把那条轨迹上所有动作的概率梯度方向放大 → 整体提高这些动作的概率**

### 更精细的 "reward-to-go" 形式（用 $G_t$ 代替 $R(\tau)$）

注意 $R(\tau) = \sum_t \gamma^t r_t$ 是**整条**回报，但**第 $t$ 步的动作只能影响第 $t$ 步及之后的 reward**（不能影响过去）。

可以证明，把 $R(\tau)$ 换成 $G_t$（从 $t$ 开始的 reward-to-go）梯度期望**不变**，方差**更小**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]$$

→ 这是**实际工程**用的形式。
→ $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ 见 [§1.1.4](01-MDP%E4%B8%8E%E8%BD%A8%E8%BF%B9%EF%BC%88%E7%94%A8Agent%E8%AF%AD%E8%A8%80%E8%AE%B2RL%EF%BC%89.md)。

### 一图看懂

```
              ┌──────────────────────────────────┐
              │  采样一条 trajectory τ            │
              │  (s_0, a_0, r_0, ..., s_T, a_T)   │
              └──────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │                               │
              ▼                               ▼
       计算每步 log π(a|s)              计算每步 return G_t
              │                               │
              └───────────┬───────────────────┘
                          │ 相乘 + 加和
                          ▼
                  ▽_θ J ≈ Σ ▽log π(a_t|s_t) · G_t
                          │
                          ▼
                  θ ← θ + α · ▽_θ J
```

→ 这就是 **REINFORCE** 算法（最朴素的 policy gradient）。

---

## 😢 1.3.2 REINFORCE 的痛点：方差极大

```
Run 1: τ_1 → G = +10  → 所有动作都被强化
Run 2: τ_2 → G = -5   → 所有动作都被削弱
Run 3: τ_3 → G = +100 → 所有动作被疯狂强化
```

**问题**：

1. **G 数值波动大** → 梯度方向忽左忽右 → 训练不稳
2. **所有动作"共担"奖惩** → 即使 τ 里有"对的动作"和"错的动作"，**都被同等对待**

→ 这就是 **credit assignment 问题**（见 [05-Sparse-Reward与信用分配](05-Sparse-Reward%E4%B8%8E%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D.md)）。

---

## 🎬 1.3.3 Baseline：减一个基准（方差缩减的核心技巧）

**改进**：用 $G_t$ **减去一个只依赖 $s_t$ 的基线** $b(s_t)$：

$$\nabla_\theta J = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \big(G_t - b(s_t)\big)\right]$$

### 关键定理：Baseline 不引入偏差

**断言**：减去任意只依赖 $s_t$ 的 $b(s_t)$，梯度的**期望保持不变**。

**为什么？** 因为额外项

$$\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot b(s_t)\right]$$

在对 $a_t$ 求期望时为 0：

$$\mathbb{E}_{a_t \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\right] = \sum_{a_t} \pi_\theta(a_t \mid s_t) \cdot \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}$$
$$\quad = \sum_{a_t} \nabla_\theta \pi_\theta(a_t \mid s_t) = \nabla_\theta \underbrace{\sum_{a_t} \pi_\theta(a_t \mid s_t)}_{=1} = \nabla_\theta 1 = 0$$

→ 因为 $\pi$ 对所有 $a$ 的概率和恒为 1，对它求梯度 = 0。
→ $b(s_t)$ **不依赖 $a_t$**，可提到求和外面，所以整体期望也 = 0。

**重要前提**：$b$ 只能依赖 $s_t$，**不能依赖 $a_t$**——否则上面的推导不成立。

### 为什么要减 baseline？（直观上：方差缩减）

| 选法 | 方差 | 效果 |
|---|---|---|
| $b = 0$（REINFORCE） | $\text{Var}[G_t]$ 全部 | 高方差，收敛慢 |
| $b = \text{const}$（如平均 return） | 略小 | 没怎么变 |
| $b = V^\pi(s_t)$（**最优 baseline**） | 通常**显著**减小 | 信号"零均值化"，干净 |
| $b = \bar{r}_{\text{group}}$（GRPO） | 类似 V | 不要 critic 时的替代 |

### 当 $b = V^\pi(s_t)$：advantage 出现

代回梯度公式：

$$\nabla_\theta J \approx \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \underbrace{(G_t - V^\pi(s_t))}_{\approx \, A^\pi(s_t, a_t)}\right]$$

**注意** $G_t - V(s_t) \approx Q(s_t, a_t) - V(s_t) = A(s_t, a_t)$（因为 $G_t$ 是 $Q$ 的无偏样本估计）。

所以：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^\pi(s_t, a_t)\right]}$$

→ 这就是 **Actor-Critic 的核心更新公式**。
→ 实战里 $A$ 用 **GAE** 估计（见 [§1.2.4](02-Policy-Value-Advantage.md)）。
→ **PPO** 在这个基础上再加 **clip + KL**（见 [§1.4](04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)）。
→ **GRPO** 把 $V(s_t)$ 换成"**同 prompt 组内 reward 均值**"（见 [§2.3](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)）。

---

## 🎭 1.3.4 Actor-Critic 架构

```
              ┌──────────────────────────────┐
              │       状态 s_t                │
              └──────────────────────────────┘
                       │              │
                       ▼              ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  Actor: π(a|s)   │  │  Critic: V(s)    │
        │  (LLM policy)    │  │  (Value head)    │
        └──────────────────┘  └──────────────────┘
                       │              │
                       │              │
                       ▼              ▼
                  采样 a_t       预测 V(s_t)
                       │              │
                       ▼              │
              env.step(a_t) → r_t     │
                       │              │
                       └──────┬───────┘
                              ▼
                   计算 advantage A_t
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
        更新 Actor:                    更新 Critic:
        gradient ∝ A_t · ▽log π        minimize (V(s_t) - G_t)²
```

**两套损失同时优化**：

| 角色 | Loss | 含义 |
|---|---|---|
| Actor (π) | $-\mathbb{E}[\log \pi(a\|s) \cdot A]$ | 让"好动作"概率上升 |
| Critic (V) | $\mathbb{E}[(V(s) - G)^2]$ | 让 V 预测越来越准 |

→ 这就是 **PPO** 的双 loss 结构（PPO 加 clip 防止 actor 跑太快，见 [04-PPO与KL约束](04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)）。

---

## 🦴 1.3.5 在 LLM RL 中的具体实现

### 共享 backbone + 双头

```
                    Prompt + context
                          │
                          ▼
              ┌──────────────────────┐
              │   LLM Transformer    │  ← 共享 backbone
              │   (Qwen / Llama)     │
              └──────────────────────┘
                  ↓               ↓
        ┌─────────────┐    ┌──────────────┐
        │  LM Head    │    │  Value Head  │
        │ (输出 token │    │ (输出标量)    │
        │  分布 = π)   │    │   = V(s)     │
        └─────────────┘    └──────────────┘
```

→ 这就是 TRL / OpenRLHF / verl 等框架里 `AutoModelForCausalLMWithValueHead` 的结构。

### LLM 上的 policy gradient 具体形式

每个 token 都是一个 action：

$$\nabla_\theta J = \mathbb{E}\left[\sum_{t} \nabla \log \pi_\theta(token_t | context_{<t}) \cdot A_t\right]$$

→ **如果一个 trajectory 有 1000 个 token，就有 1000 个梯度项**。
→ 这就是为什么 LLM RL 很贵——每个 token 都参与梯度计算。

---

## 🪞 1.3.6 几个变体（一句话）

| 算法 | 一句话 |
|---|---|
| **REINFORCE** | 最朴素：$\nabla \log \pi \cdot G$，方差大 |
| **REINFORCE + baseline** | $\nabla \log \pi \cdot (G - b)$，方差小一些 |
| **Actor-Critic** | baseline 用 V(s)，**两网络协同训** |
| **A2C / A3C** | Actor-Critic 的同步/异步实现 |
| **PPO** | Actor-Critic + **clipping** 防止 π 跑太快（见 [04-PPO与KL约束](04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)） |
| **GRPO** | Actor-Critic + **组内均值代替 V**（见 [03-GRPO（组内相对优势）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)） |
| **REINFORCE++** | OpenRLHF 实现的轻量版本 |

→ **PPO 和 GRPO 是 Agentic RL 的两个主战马**。

---

## 💻 1.3.7 概念伪代码（policy gradient 训练循环）

```python
# 概念示意，不是可跑代码
for iteration in range(N):
    # 1. Rollout: 用当前 π 采样一批 trajectory
    trajectories = []
    for _ in range(batch_size):
        τ = rollout(env, policy)
        trajectories.append(τ)

    # 2. 计算每步 advantage
    for τ in trajectories:
        for t in range(len(τ)):
            τ[t].G = sum(γ**k * τ[t+k].r for k in range(len(τ)-t))
            τ[t].V = critic(τ[t].s)
            τ[t].A = τ[t].G - τ[t].V

    # 3. 算 actor loss
    actor_loss = -mean(τ[t].A * log_prob(policy, τ[t].s, τ[t].a)
                       for τ in trajectories for t in range(len(τ)))

    # 4. 算 critic loss
    critic_loss = mean((τ[t].V - τ[t].G)**2
                       for τ in trajectories for t in range(len(τ)))

    # 5. 更新
    update(policy, actor_loss)
    update(critic, critic_loss)
```

→ 这就是 PPO 的雏形（PPO 多一个 clip 限制 actor 更新幅度）。

---

## ⚠️ 常见误解

| 误解 | 真相 |
|---|---|
| "policy gradient 需要环境可导" | **不需要**！这是 RL 比 supervised learning 灵活的关键 |
| "Actor-Critic 一定比 REINFORCE 强" | 不一定。critic 不准时反而更糟。GRPO 干脆不要 critic |
| "policy gradient 一定 on-policy" | **是的**。这就是为什么 PPO 训练贵——每次更新要重新 rollout |
| "训完一定能收敛" | RL 不保证收敛。reward shaping、hyperparam、初始化都关键 |

---

## 📌 1.3 节要点

| 概念 | 一句话 |
|---|---|
| **策略梯度** | $\nabla \log \pi \cdot G$，奖励高的动作概率被放大 |
| **Baseline** | 减一个基准（V 或组均值），方差小很多 |
| **Actor-Critic** | actor=π，critic=V，**advantage 是训练信号** |
| **REINFORCE** | 最朴素 PG，无 baseline，方差大 |
| **PPO/GRPO** | 都是 Actor-Critic 改进版 |
| **LLM 实现** | Backbone + LM Head (actor) + Value Head (critic) |

---

## 🔗 延伸阅读

- 下一节：[04-PPO与KL约束](04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)
- 应用：[03-GRPO（组内相对优势）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)
- 原始资料：`01-boundaries.md` §"policy gradient、actor-critic"
- CleanRL PPO 实现：https://github.com/vwxyzjn/cleanrl
- 跨链：[04-GRPO训练实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)

---

⬅ [02-Policy-Value-Advantage](02-Policy-Value-Advantage.md) | ➡ [04-PPO与KL约束](04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)
