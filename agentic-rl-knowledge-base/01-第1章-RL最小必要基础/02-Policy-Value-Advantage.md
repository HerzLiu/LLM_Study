---
tags: [agentic-rl, 第1章, policy, value, Q, advantage, RL基础]
chapter: 1
section: 1.2
source: "agentic-rl-learning-map/01-boundaries.md §必须补的 RL 基础, 05-glossary.md"
---

# 1.2 Policy / Value / Advantage（四件套）

⬅ [01-MDP与轨迹（用Agent语言讲RL）](01-MDP%E4%B8%8E%E8%BD%A8%E8%BF%B9%EF%BC%88%E7%94%A8Agent%E8%AF%AD%E8%A8%80%E8%AE%B2RL%EF%BC%89.md) | ➡ [03-Policy-Gradient与Actor-Critic](03-Policy-Gradient%E4%B8%8EActor-Critic.md)

---

## 🎬 故事比喻：考试预测的 4 个角色

```
你是高三学生（agent），下个月高考（episode）：

【π Policy 学生本人】     —— 在每个状态下决定干啥（学/玩/睡）
【V Value 班主任】       —— 看到你现在的状态，预测最终高考多少分
【Q Q-value 学习顾问】   —— 看到你现在的状态 + 你下一步打算干啥，预测最终多少分
【A Advantage 教导主任】 —— 直接告诉你：在这状态下，这个动作比"平均水平"好多少
```

> 这 4 个量是 PPO / GRPO / RLHF / 任何 LLM RL 论文里反复出现的"四大金刚"。
> 看公式时只要记住这个比喻，就不会一脸懵。

---

## 🎯 1.2.1 Policy π（策略）

**定义**：状态到动作的映射。

| 类型 | 数学形式 | 例子 |
|---|---|---|
| **确定性策略** | $a = \pi(s)$，函数 $\pi: S \to A$ | 看到红灯就停 |
| **随机性策略** | $a \sim \pi(\cdot \mid s)$，函数 $\pi: S \times A \to [0, 1]$ | 看到红灯 90% 概率停，10% 闯（实际 RL 几乎都用这种） |

**随机策略的核心性质（必须满足）**：

$$\sum_{a \in A} \pi(a \mid s) = 1, \quad \forall s \in S$$

即对任何 state $s$，$\pi(\cdot \mid s)$ 是 $A$ 上的**合法概率分布**。

### 在 LLM Agent 里

> **LLM 本身 = policy π**。
>
> 输入 prompt + 上下文 ($o_t$) → 输出 token 分布 → sample 一个 token / action ($a_t$)
>
> $$\pi_\theta(a_t \mid o_t) = \text{softmax}\big(z_\theta(o_t) / T\big)_{a_t}$$

**逐项解释**：

- $\theta$：LLM 的**全部参数**（几亿到几千亿个数）
- $z_\theta(o_t) \in \mathbb{R}^{|V|}$：LLM 最后一层 logits，$|V|$ = 词表大小（常见 32k-200k）
- $T$：**温度**（temperature），$T \to 0$ 贪心，$T \to \infty$ 均匀
- $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$：把 logits 归一化为概率分布
- 下标 $a_t$：取出**这个动作（token）**对应的概率

→ **RL 训练就是更新 $\theta$**，让"高 reward 的 $a_t$ 被采样的概率"变大。

### 在 trajectory 上的概率分解

一条完整轨迹 $\tau = (s_0, a_0, \dots, s_T, a_T)$ 在策略 $\pi_\theta$ 下的**对数概率**：

$$\log \pi_\theta(\tau) = \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t)$$

（这里把 $P(s_0)$ 和 $P(s_{t+1} \mid s_t, a_t)$ 当成与 $\theta$ 无关的常数，提取出策略部分）

→ **关键**：trajectory 的对数概率 = 每一步动作对数概率之和。
→ Policy gradient 后面会**对这个量求 $\nabla_\theta$**（[§1.3](03-Policy-Gradient%E4%B8%8EActor-Critic.md)）。

### 关键性质

- **随机性**：LLM 的 temperature > 0 时是随机策略
- **参数化**：$\pi_\theta$ 由神经网络参数化（亿级参数）
- **可微**：$\pi_\theta(a \mid s)$ 对 $\theta$ 处处可导 → 这是 policy gradient 的前提
- **可采样**：给定 $s$，能从 $\pi_\theta(\cdot \mid s)$ **采样**出 $a$（通过 multinomial / nucleus / top-k）

---

## 💰 1.2.2 Value Function V（状态价值）

**定义**：从状态 $s$ 开始，按策略 $\pi$ 走下去，期望能拿到多少 return。

$$V^\pi(s) = \mathbb{E}_{\tau \sim \pi}\left[ \sum_{k=0}^{T} \gamma^k r_k \;\Big|\; s_0 = s \right] = \mathbb{E}_{\tau \sim \pi}[G_0 \mid s_0 = s]$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $V^\pi$ | 一个**函数** $V^\pi: S \to \mathbb{R}$，输入 state、输出标量价值 |
| 上标 $\pi$ | 强调 $V$ **依赖**于使用什么策略——同一个 state 用不同 $\pi$ 价值不同 |
| $\tau \sim \pi$ | 在策略 $\pi$ 下采样轨迹 |
| $\mid s_0 = s$ | **条件**：轨迹起点**固定**在 $s$ |
| $G_0$ | 从 $t=0$ 开始的累积折扣回报 |

**人话**：「**如果我从这个 state 开始，按 π 玩，平均能得多少分**」

### 一个简单例子

```
s = "现在 issue 看完了，repo 还没读"
V^π(s) = 0.4
```

含义：用当前 $\pi$ 从这个 state 跑很多次，**平均拿到 0.4 reward**（假设最终 success = 1.0，则这个 state 的成功概率约 40%）。

### Bellman 期望方程（核心，要会读）

$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)} \mathbb{E}_{s' \sim P(\cdot|s,a)}\Big[ R(s, a) + \gamma V^\pi(s') \Big]$$

**这个等式逐层拆**：

**最外层**：$\mathbb{E}_{a \sim \pi(\cdot \mid s)}[\cdot]$
→ 对当前 state 下**所有可能动作**按策略概率加权

**中间层**：$\mathbb{E}_{s' \sim P(\cdot \mid s, a)}[\cdot]$
→ 对每个 $(s, a)$ 下**所有可能下一状态**按转移概率加权

**最内层**：$R(s, a) + \gamma V^\pi(s')$
→ 即刻 reward + 折扣后的"下一状态价值"

**展开（离散情况）**：

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \Big[R(s, a) + \gamma V^\pi(s')\Big]$$

**从哪里来？** 用 $G_t = r_t + \gamma G_{t+1}$ 的递归（[§1.1.4](01-MDP%E4%B8%8E%E8%BD%A8%E8%BF%B9%EF%BC%88%E7%94%A8Agent%E8%AF%AD%E8%A8%80%E8%AE%B2RL%EF%BC%89.md)）：

$$V^\pi(s) = \mathbb{E}[G_0 \mid s_0 = s] = \mathbb{E}[r_0 + \gamma G_1 \mid s_0 = s]$$
$$\quad = \mathbb{E}[R(s, a) + \gamma \mathbb{E}[G_1 \mid s_1 = s'] ]$$
$$\quad = \mathbb{E}[R(s, a) + \gamma V^\pi(s')]$$

→ "**现在的价值 = 当前 reward + 折扣后下一状态的价值**"

**为什么这个方程重要？**

1. 它把"无限求和的 $G_0$"变成"只看一步 + 递归"——可以**数值迭代**求解
2. 它是 **TD learning**（Temporal Difference，时序差分）的基础
3. PPO 的 critic 训练就是**最小化这个等式两边的差**（[§1.4](04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)）

→ 不要求会推导。**记住"现在 = 即刻 + 折扣未来"** 就够了。

---

## 💎 1.2.3 Q-Function（动作价值）

**定义**：在状态 $s$ **强制选**动作 $a$，然后按 π 走下去，期望能拿多少 return。

$$Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi}\left[ \sum_{k=0}^{T} \gamma^k r_k \;\Big|\; s_0=s, a_0=a \right]$$

**和 V 的关键区别**（划重点）：

| 量 | 条件 | 含义 |
|---|---|---|
| $V^\pi(s)$ | 只固定 $s_0 = s$，$a_0$ 按 $\pi$ 采样 | "**从这状态开始**，按 π 玩平均拿多少" |
| $Q^\pi(s, a)$ | 固定 $s_0 = s$ **和** $a_0 = a$，$a_1, a_2, \dots$ 按 $\pi$ 采样 | "**从这状态选 a**，之后按 π 玩平均拿多少" |

→ $V$ 把 $a_0$ 也按 $\pi$ 平均了；$Q$ **不平均** $a_0$（强制指定）。

### V 和 Q 的精确关系

$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}[Q^\pi(s, a)] = \sum_{a} \pi(a \mid s) \cdot Q^\pi(s, a)$$

**这个等式怎么读**：
- **左边** $V^\pi(s)$：标量
- **右边求和**：对所有动作 $a$，按"$\pi$ 选这个 $a$ 的概率"加权求和"$Q$ 值"
- **等号**：从一个 $s$ 出发，先按 $\pi$ 随机选 $a$（这就是 $V$ 的定义）= 先对每个 $a$ 算 $Q$，再按概率平均

**反方向（Q 的 Bellman）**：

$$Q^\pi(s, a) = R(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a)}\Big[ V^\pi(s') \Big]$$

即"**Q = 即刻 reward + 折扣后的下一状态价值**"。
（注意 $Q$ 的 Bellman 里**不需要**再对 $a_0$ 平均，因为 $a_0$ 已经被固定了）

### 在 LLM Agent 里

Q-function 在 LLM RL 里**很少显式训练**（action space 是整个 token 序列，组合爆炸 $|V|^L$），但概念很有用：

```
Q(o="failing test 在 paths.py", a="read paths.py") = 0.7  ← 这步好
Q(o="failing test 在 paths.py", a="run_tests")    = 0.2  ← 这步糟（你还没读文件就跑测试）
```

→ 实际算法（PPO / GRPO）通常**只训 $V$**（critic），不显式存 $Q$，但用 $Q^\pi(s, a) \approx r_t + \gamma V^\pi(s_{t+1})$ **临时**算。

---

## ⭐ 1.2.4 Advantage A（优势函数）

**定义**：

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

**逐项解读**：

- $Q^\pi(s, a)$：在 $s$ 选 $a$ **这个特定动作**后能拿多少（绝对值）
- $V^\pi(s)$：在 $s$ **按 $\pi$ 平均**能拿多少（基线）
- 相减：这个动作**相对于平均水平**的偏差

**人话**：「**这个动作比"平均水平"好多少**」

| A 值 | 含义 | 训练时怎么用 |
|---|---|---|
| A > 0 | 这个动作**比平均好** | **提高** $\pi(a \mid s)$ 的概率 |
| A < 0 | 这个动作**比平均差** | **降低** $\pi(a \mid s)$ 的概率 |
| A = 0 | 平均水平 | 不动 |

**重要性质**：

$$\mathbb{E}_{a \sim \pi(\cdot \mid s)}[A^\pi(s, a)] = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)] - V^\pi(s) = V^\pi(s) - V^\pi(s) = 0$$

→ **Advantage 在策略下期望为 0**（一个 baseline 项必有的性质）。
→ 这意味着 advantage 是"零均值"的相对量——天然适合做梯度信号。

### 为什么 RL 算法都爱用 A（不用 Q 或 G）

**用绝对 return $G$ 训练的问题（高方差）**：

假设某 episode 全部 reward = 1000 ± 1：

```
Run 1: G = 999  → 所有动作都被"+999"强化
Run 2: G = 1001 → 所有动作都被"+1001"强化
Run 3: G = 1000 → ...
```

**梯度方向几乎不变**，但**幅度巨大且噪声大** → 训练剧烈震荡。

**用 advantage $A$ 训练**：

```
A = G - V(s) ≈ 0 ± 1
→ 只有"真比平均好/差"的动作才有非零梯度
→ 信号干净
```

**数学上**：把 $G$ 换成 $G - b(s)$（任意只依赖 $s$ 的基线 $b$），**梯度期望不变**，但**方差大幅下降**——这就是 §1.3 会详证的"baseline 不引入偏差"定理。

→ **PPO、GRPO、A2C 全部用 advantage（不是 Q）**。

### GAE（Generalized Advantage Estimation，PPO 实战必用）

实际计算 $A$ 时常用 **GAE**（Schulman 2016）：

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$$

其中 **TD 残差**（temporal-difference residual）：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**逐项拆 $\delta_t$**：

| 项 | 含义 |
|---|---|
| $r_t$ | 实际拿到的 reward |
| $\gamma V(s_{t+1})$ | critic 预测的"下一状态价值"（折扣后） |
| $V(s_t)$ | critic 预测的"当前状态价值" |
| $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | "**实际拿到 + 未来预测**" vs "**当前预测**" 的差 |

**意义**：$\delta_t$ 是 **"1 步 advantage 估计"**——如果 $\delta_t > 0$，说明这步比 critic 想的好。

**GAE 是什么？** 把多个不同步长的 advantage 估计**加权求和**：

$$\hat{A}_t^{\text{GAE}} = \delta_t + (\gamma\lambda) \delta_{t+1} + (\gamma\lambda)^2 \delta_{t+2} + \dots$$

**$\lambda$ 控制什么？**（偏差-方差权衡）：

| $\lambda$ | 等价于 | 偏差 | 方差 | 解释 |
|---|---|---|---|---|
| $\lambda = 0$ | $\hat{A}_t = \delta_t$（仅 1 步 TD） | **高**（依赖 critic 估计） | **低** | critic 不准时偏差大，但不依赖未来 |
| $\lambda = 1$ | $\hat{A}_t = G_t - V(s_t)$（完整 Monte Carlo） | **低**（用真实 $G$） | **高** | 不依赖 critic，但需要 episode 全跑完 |
| $\lambda = 0.95$ | 折中（**最常用**） | 中 | 中 | PPO 默认值 |

**直觉**：$\lambda$ 像一个"信任 critic vs 信任真实 return"的旋钮。

**为什么不直接用 $G_t - V(s_t)$？**

- 它就是 $\lambda = 1$ 的特例
- 但 $G_t$ 必须等 episode 跑完才能算
- 而且 $G_t$ 方差比加权 $\delta$ 大很多

**GAE 在 LLM PPO 里的实现**（高层伪代码）：

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    # rewards: [r_0, ..., r_T]
    # values:  [V(s_0), ..., V(s_T), V(s_{T+1})]  (多一个用于 bootstrap)
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae  # 倒序累加
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values[:-1])]  # 用于训 critic
    return advantages, returns
```

→ 这段代码在 TRL、OpenRLHF、verl 里几乎一模一样。
→ 你写 toy gym 时**直接抄**。

---

## 🔄 1.2.5 四件套关系总结

```
        ┌──────────────────────────────────────┐
        │                                      │
        │              Return G                │
        │     (实际跑完一次 episode 的总奖励)    │
        │                                      │
        └──────────────────────────────────────┘
                          ↓ 取期望
                          ↓
        ┌──────────────────────────────────────┐
        │              V(s) = E[G | s]          │
        │              Q(s,a) = E[G | s,a]      │
        │     (期望值，需要估计 / 训练)          │
        └──────────────────────────────────────┘
                          ↓ 相减
                          ↓
        ┌──────────────────────────────────────┐
        │       A(s,a) = Q(s,a) - V(s)          │
        │   (相对量，PPO/GRPO 的训练信号)        │
        └──────────────────────────────────────┘
                          ↓ 用来更新
                          ↓
        ┌──────────────────────────────────────┐
        │                π(a|s)                 │
        │       (策略 = LLM, 我们要训的东西)     │
        └──────────────────────────────────────┘
```

---

## 🧠 1.2.6 在 LLM RL 里的具体对应

| 量 | 在 LLM RL 实现里 |
|---|---|
| **π** | LLM 本身（policy model），输出 token 分布 |
| **V** | **Value head**：在 LLM 顶上加一个 linear 层输出标量 |
| **Q** | 很少显式训，因为 action = 所有可能 token 序列，组合爆炸 |
| **A** | 由 V 计算（GAE）或由 GRPO 的"组内 reward 减去组均值"代替 |

### PPO 用 V

PPO 训练时需要一个 critic（value model），通常和 policy 共享 backbone + 独立 value head。

### GRPO 用"组均值"代替 V

DeepSeekMath 的发现：

> **不需要 critic**。
> 同一 prompt 采样 K 个回答 → 算 K 个 reward → 「单个 reward - K 个 reward 均值」就当 advantage。

这就**省掉了 value head 的训练成本**——是 GRPO 的最大卖点之一。详见 [03-GRPO（组内相对优势）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)。

---

## 🧪 1.2.7 用 Code Agent 例子算一遍

假设一个 toy bug-fix episode：

```
t=0: o=issue,            a=search,    r=0
t=1: o=search result,    a=read,      r=0
t=2: o=file content,     a=edit,      r=0
t=3: o=edit applied,     a=run_tests, r=0
t=4: o=test passed,      a=final,     r=1.0  ← 终局奖励
```

γ = 0.99，T = 4：

```
G_4 = 1.0
G_3 = 0 + 0.99 × 1.0 = 0.99
G_2 = 0 + 0.99 × 0.99 = 0.9801
G_1 = 0 + 0.99 × 0.9801 = 0.9703
G_0 = 0 + 0.99 × 0.9703 = 0.9606
```

每个时间步的 **return G_t** 就是从那一步往后的累积折扣奖励。

**V(s_0) 估计**（假设 critic 给）：

```
V(s_0) ≈ 0.4  ← critic 认为"这个 issue 平均成功率 40%"
```

**Advantage 估计**：

```
A_0 ≈ G_0 - V(s_0) = 0.9606 - 0.4 = 0.56
```

**含义**：这一整条轨迹比"平均水平"好 0.56 → 提高这条 trajectory 上每个动作的概率。

→ 这就是 PPO/GRPO 的训练信号。

---

## ⚠️ 常见误解

| 误解 | 真相 |
|---|---|
| "V 和 Q 是模型给的真值" | **不是**。是估计值，要训练（critic 网络） |
| "advantage 一定正" | 不是。A 可正可负，负就降低这个动作的概率 |
| "GRPO 没用 advantage" | 用了，只是 "advantage ≈ reward - 组均值"，**省掉了 V** |
| "所有 RL 都需要 critic" | DPO / GRPO / REINFORCE 都可以不需要 |
| "value 越大越好" | V 是预测，不是优化目标；要优化的是 π |

---

## 📌 1.2 节要点

| 量 | 一句话 | 在 LLM RL 里 |
|---|---|---|
| **π(a\|s)** | 策略，状态→动作的概率分布 | LLM 本身 |
| **V(s)** | 从 s 开始的期望 return | Value head |
| **Q(s,a)** | 在 s 选 a 后的期望 return | 几乎不显式训 |
| **A(s,a) = Q - V** | 这个动作比平均好多少 | **PPO/GRPO 的核心信号** |
| **G** | 实际跑完的累积奖励 | trajectory return |

---

## 🔗 延伸阅读

- 下一节：[03-Policy-Gradient与Actor-Critic](03-Policy-Gradient%E4%B8%8EActor-Critic.md)
- 应用：[03-GRPO（组内相对优势）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)（GRPO 怎么省掉 V）
- 原始资料：`01-boundaries.md` §"policy、value function、Q/V、advantage"
- 跨链：[04-GRPO训练实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)

---

⬅ [01-MDP与轨迹（用Agent语言讲RL）](01-MDP%E4%B8%8E%E8%BD%A8%E8%BF%B9%EF%BC%88%E7%94%A8Agent%E8%AF%AD%E8%A8%80%E8%AE%B2RL%EF%BC%89.md) | ➡ [03-Policy-Gradient与Actor-Critic](03-Policy-Gradient%E4%B8%8EActor-Critic.md)
