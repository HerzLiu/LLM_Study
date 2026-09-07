---
tags: [agentic-rl, 第1章, PPO, KL散度, clipping, reference-model, RL基础]
chapter: 1
section: 1.4
source: "agentic-rl-learning-map/01-boundaries.md §必须补的 RL 基础, 02-papers.md §PPO, 04-six-week-roadmap.md §第2周"
---

# 1.4 PPO 与 KL 约束（LLM RL 的算法底座）

⬅ [03-Policy-Gradient与Actor-Critic](03-Policy-Gradient%E4%B8%8EActor-Critic.md) | ➡ [04b-On-Policy-vs-Off-Policy](04b-On-Policy-vs-Off-Policy.md)

---

## 🎬 故事比喻：学步车上的安全带

```
你在教小孩走路（policy π）：

【无约束的策略梯度】（REINFORCE）
   "走得快有糖吃" → 小孩一步迈出 5 米 → 摔了 → 再也不敢走
   问题：更新步子太大，π 一下就崩了，再也学不回来

【PPO 的两条安全带】
   ① Clipping: "新走法 vs 旧走法的概率比不能超过 1.2"
       → 一次只能改一点点
   ② KL 惩罚: "不能离原来的你太远"（参考模型 = 你原来的样子）
       → 防止你为了一颗糖把自己变成另一个人
```

> PPO = Proximal Policy Optimization = "**邻近**策略优化"
> "邻近" = 新策略不能离旧策略太远
>
> 这两条安全带是 PPO 能稳定训练 LLM 的关键。
> **没有它们，LLM RL 几乎一训就崩**。

---

## 📜 1.4.1 论文与历史地位

| 字段 | 内容 |
|---|---|
| 论文 | Proximal Policy Optimization Algorithms |
| 作者 / 机构 | John Schulman et al., OpenAI |
| 年份 | 2017 |
| 链接 | https://arxiv.org/abs/1707.06347 |
| 历史地位 | **RLHF 和大部分 LLM RL 框架的算法底座**；InstructGPT 用 PPO；现代 GRPO 也是 PPO 变体 |

→ 本知识库 [3.2 论文卡 #1](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md) 也有它的卡片。

---

## 🧮 1.4.2 PPO 核心公式（逐项拆解）

### 起点：为什么需要"概率比"？

REINFORCE / 朴素 actor-critic 的梯度（[§1.3](03-Policy-Gradient%E4%B8%8EActor-Critic.md)）：

$$\nabla_\theta J = \mathbb{E}_{a \sim \pi_\theta}\left[\nabla \log \pi_\theta(a \mid s) \cdot A(s, a)\right]$$

→ 这是 **on-policy** 期望（要求 $a$ 来自**当前** $\pi_\theta$）。

**问题**：每更新一次 $\theta$，就要**重新 rollout**——LLM 太贵了。
**想法**：用**旧** $\pi_{\theta_{old}}$ rollout 一批数据，**多次**复用更新 $\pi_\theta$。

→ 这需要 **importance sampling**（重要性采样）来"换分布"：

$$\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{\theta_{old}}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} \cdot f(a)\right]$$

引入**概率比**（importance ratio）：

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

| $r_t$ 数值 | 含义 |
|---|---|
| $r_t = 1$ | 新旧策略对这个 $a_t$ **概率一样** |
| $r_t > 1$ | 新策略**更喜欢**这个 $a_t$ |
| $r_t < 1$ | 新策略**更不喜欢**这个 $a_t$ |
| $r_t \gg 1$（如 100） | 新策略**剧烈**偏离 → **危险**！ |

### PPO 的 surrogate objective

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[ \min\Big( r_t(\theta) \hat{A}_t, \; \text{clip}\big(r_t(\theta), 1-\epsilon, 1+\epsilon\big) \hat{A}_t \Big) \right]$$

**逐项翻译每个符号**：

| 符号 | 含义 | 形状 |
|---|---|---|
| $\mathbb{E}_t[\cdot]$ | 对一批 rollout 数据**所有时间步**求平均 | 标量 |
| $r_t(\theta)$ | 上述概率比 | 标量 |
| $\hat{A}_t$ | advantage 估计（用 GAE 算，见 [§1.2.4](02-Policy-Value-Advantage.md)） | 标量 |
| $\epsilon$ | clipping 范围超参，常 **0.1 - 0.2** | 标量 |
| $\text{clip}(x, 1-\epsilon, 1+\epsilon)$ | 把 $x$ 截断到 $[1-\epsilon, 1+\epsilon]$ | 标量 |
| $\min(原始, \text{clipped})$ | 取两者较**小**的（pessimistic bound） | 标量 |

**核心是 $\min$**——它是 PPO "悲观更新" 的精髓。

### $\min$ 在做什么？分 $A > 0$ 和 $A < 0$ 两种情况看

**情况 1：$\hat{A}_t > 0$（这个动作好，想强化）**

$$L = \min(r_t \hat{A}_t, \, \text{clip}(r_t) \hat{A}_t)$$

| $r_t$ 区间 | $r_t \hat{A}_t$ | $\text{clip}(r_t)\hat{A}_t$ | $\min$ 取谁 | 含义 |
|---|---|---|---|---|
| $r_t < 1 - \epsilon$ | 小 | 小（被 clip 到 $1-\epsilon$） | 一样 | 没限制（动作概率本就在下降，符合 $A>0$ 时不想要的方向？反向） |
| $1-\epsilon \le r_t \le 1+\epsilon$ | 同 | 同 | 一样 | 正常更新 |
| $r_t > 1 + \epsilon$ | **大** | clip 到 $(1+\epsilon)\hat{A}_t$（**小**） | **取小的** | **阻止过度强化** |

→ 当动作好 + 新策略想剧烈提高它的概率时，**clip 把梯度截掉**——防止"一步迈太大"。

**情况 2：$\hat{A}_t < 0$（这个动作坏，想削弱）**

$$L = \min(r_t \hat{A}_t, \, \text{clip}(r_t) \hat{A}_t)$$

注意 $\hat{A}_t < 0$，乘以正的 $r_t$ 结果是**负数**——$\min$ 取**更负**的那个。

| $r_t$ 区间 | 哪个更负 | 含义 |
|---|---|---|
| $r_t < 1 - \epsilon$ | $r_t \hat{A}_t$ 更负（因为 $r_t$ 没被截，$r_t < 1-\epsilon$ 更小→乘负数更负？等等需要分析） | 防止过度抑制 |
| $1-\epsilon \le r_t \le 1+\epsilon$ | 相等 | 正常更新 |
| $r_t > 1 + \epsilon$ | $r_t \hat{A}_t$ 更负 | 取它（继续抑制） |

更精确分析：当 $\hat{A} < 0$ 时，clip **保护**新策略不要把这个坏动作的概率压得**太低**（$r_t < 1-\epsilon$ 时 clip 起作用，截到 $(1-\epsilon)\hat{A}$ 比 $r_t \hat{A}$ "**没那么负**"，min 取更负的 $r_t \hat{A}$）——**也就是继续抑制**。

**统一直觉**：

> **PPO 在"更新方向有利"时限速；在"更新方向不利"时不限速**。
> 这是"悲观下界"——拿 $\min$ 永远是更保守的那一边。

### 数值小例子（看明白 clip）

设 $\epsilon = 0.2$，几种情况：

| $\hat{A}_t$ | $r_t$ | $r_t \hat{A}_t$ | $\text{clip}(r_t)\hat{A}_t$ | $\min$ | 含义 |
|---|---|---|---|---|---|
| +1.0 | 1.5 | 1.5 | $1.2 \times 1.0 = 1.2$ | **1.2** | 好动作，新策略激进 → **截断** |
| +1.0 | 1.1 | 1.1 | $1.1 \times 1.0 = 1.1$ | 1.1 | 正常 |
| +1.0 | 0.7 | 0.7 | $0.8 \times 1.0 = 0.8$ | 0.7 | 好动作，新策略反而降低？取原始（信号弱） |
| -1.0 | 1.5 | -1.5 | $1.2 \times -1.0 = -1.2$ | **-1.5** | 坏动作，新策略反而提升它 → 取更负，继续惩罚 |
| -1.0 | 0.5 | -0.5 | $0.8 \times -1.0 = -0.8$ | **-0.8** | 坏动作，新策略想压死 → clip 限速 |

### 一图看懂 clipping（修正版）

```
                        L (loss)
                            │
                            │       A > 0 (好动作)
                            │
                            │      ┌─────── clip 截断（限速）
                            │     /
                            │    /
                            │   /
        ─────────────────── ────── r_t (概率比)
                            │   
                            │   1-ε    1+ε
                            │
                            │  \
                            │   \
                            │    \ ← clip 截断（限速）
                            │     \________ 
                            │       A < 0 (坏动作)
```

→ **关键**：在 $r_t \in [1-\epsilon, 1+\epsilon]$ 之外，PPO **不让梯度继续放大**——这就是"邻近"（proximal）的本质。

### 一图看懂 clipping

```
                  目标函数 L
                       │
   r·A (clipped)       │      r·A (原始)
       ─ ─ ─ ─ ─ ─ ─ ─ │ ─ ─ ─ ─ ─ ─ ─
                       │
A > 0                  │
(动作好)               │
                       │
    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│ ── ── ── ──   ← 取 min 后被截断
                       │
                       ├──────────────────► r (概率比)
                       │
                       │     1-ε      1+ε
                       │
A < 0                  │
(动作差)
                       │
                       │
   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │ ─ ─ ─ ─ ─ ─ ─
       r·A (原始)      │      r·A (clipped)
                       │
                       ↓
            (取较小，让更新更保守)
```

**直觉**：

- 当 **A > 0**（动作好），新策略想拼命提高这个动作 → clip 限制"提高幅度"不超过 1+ε
- 当 **A < 0**（动作差），新策略想拼命降低这个动作 → clip 限制"降低幅度"不超过 1-ε
- **不让一次更新走太远**

→ 这就是 "Proximal"（邻近）的来源。

---

## 🛡 1.4.3 KL 散度约束（LLM RL 的第二条安全带）

PPO 原始论文有 KL penalty 版本，**LLM RL 几乎都用 KL penalty**：

$$L^{LLM-PPO}(\theta) = L^{CLIP}(\theta) - \beta \cdot \mathbb{E}_{s \sim \mathcal{D}}\Big[\text{KL}\big(\pi_\theta(\cdot \mid s) \,\|\, \pi_{ref}(\cdot \mid s)\big)\Big]$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $L^{CLIP}(\theta)$ | 上一节的 clip 目标（要最大化） |
| $\pi_{ref}$ | **Reference model**：通常是 SFT 之后的模型，**冻结不动** |
| $\pi_\theta(\cdot \mid s)$ | 当前 policy 在 $s$ 上的**全 token 分布** |
| $\text{KL}(\pi_\theta \| \pi_{ref})$ | 当前 policy 和 reference 的 KL 散度（标量） |
| $\beta$ | KL 惩罚系数，常 0.01 - 0.1（**LLM 实战**） |
| 负号 | **惩罚**（最大化 $L$ → 想让 KL 小） |
| $\mathbb{E}_{s \sim \mathcal{D}}$ | 对 rollout 数据的所有 state 求平均 |

### KL 散度详细定义

**离散版本**：

$$\text{KL}(P \,\|\, Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

**逐项解读**：

| 部分 | 含义 |
|---|---|
| $P(x), Q(x)$ | 两个概率分布在同一支撑集 $x$ 上的值 |
| $\log \frac{P(x)}{Q(x)}$ | 对每个 $x$，$P$ 比 $Q$ "偏好" 多少（log-ratio） |
| $\mathbb{E}_{x \sim P}$ | 按 $P$ 加权（**注意**不是 $Q$） |

**关键性质**：

| 性质 | 公式 | 含义 |
|---|---|---|
| 非负 | $\text{KL}(P \| Q) \ge 0$ | 永远 ≥ 0 |
| 零当且仅当相等 | $\text{KL}(P \| Q) = 0 \iff P = Q$ | 越大越不同 |
| **非对称** | $\text{KL}(P \| Q) \ne \text{KL}(Q \| P)$ | **写谁在前重要** |
| 不是距离 | 不满足三角不等式 | "散度"≠"距离" |

**LLM RL 里常用的是 $\text{KL}(\pi_\theta \| \pi_{ref})$**（新策略 KL 到 reference）：

→ 当 $\pi_\theta$ 在某个 token 上给了很高概率，但 $\pi_{ref}$ 给得很低 → 这一项**很大** → 强惩罚。
→ 直觉："**新策略不允许在 reference 没想到的地方狂飙**"。

### LLM 上 KL 的具体计算（token-level）

对一条 response $y = (y_1, \dots, y_L)$，逐 token 算 KL：

$$\text{KL}_{token}(s, y_t) = \log \pi_\theta(y_t \mid s, y_{<t}) - \log \pi_{ref}(y_t \mid s, y_{<t})$$

（这是 KL 的**蒙特卡洛估计**——用一个 sample 近似）

整段 response 的 KL：

$$\text{KL}(\pi_\theta \| \pi_{ref})_{response} \approx \sum_{t=1}^{L} \text{KL}_{token}(s, y_t)$$

### KL 的两种用法（写论文要看清是哪种）

**用法 A：作为 loss 惩罚**（PPO 原论文 KL 版本）

$$\mathcal{L} = -L^{CLIP} + \beta \cdot \overline{\text{KL}}$$

（最小化 loss = 最大化 clip + 最小化 KL）

**用法 B：作为 reward shaping**（实践更常见）

每步 reward 改成：

$$\tilde{r}_t = r_t - \beta \cdot \big(\log \pi_\theta(a_t \mid s_t) - \log \pi_{ref}(a_t \mid s_t)\big)$$

→ 把 KL 塞进 reward，advantage 会自动反映它。
→ **OpenAI / Anthropic / 多数 RLHF 代码**用这种。

### 为什么 LLM RL 必须加 KL？

```
没 KL 约束时:
   π 拼命追求高 reward
   → 输出乱写、胡说、变 mode collapse
   → 语言能力被破坏
   → 训练崩

有 KL 约束（β·KL ≥ 0 是惩罚）:
   π 既要追求高 reward
   又要"不要离 SFT 模型太远"
   → 保留语言能力 + 学新行为
```

> 💡 **重要**：KL 不是为了"稳定"，**是为了保护 LLM 的基础能力**。
> 没 KL 约束，模型可能学会"通过输出乱码骗 reward model"。

---

## 🧪 1.4.4 PPO for LLM 完整 pipeline

```
                ┌────────────────────────────┐
                │  1. SFT model (init both)   │
                └────────────────────────────┘
                    │              │
                    ▼              ▼
        ┌──────────────┐      ┌─────────────┐
        │  π_ref       │      │  π_θ        │
        │ (冻结)        │      │ (要训的)     │
        └──────────────┘      └─────────────┘
                                       │
                                       ▼
                                 prompt → 采样
                                       │
                                       ▼
                                 response y
                                       │
                              ┌────────┼────────┐
                              ▼        ▼        ▼
                        reward     log π_ref  log π_θ
                        model       (算 KL)    (算 ratio)
                              │
                              ▼
                         R - β·KL = 最终 reward
                              │
                              ▼
                     advantage A (用 V 算)
                              │
                              ▼
                     PPO clip + actor loss
                              │
                              ▼
                       更新 π_θ + critic V
                                       │
                              loop ────┘
```

→ 整个 RLHF 训练流程见 [01-RLHF三阶段（SFT-RM-PPO）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/01-RLHF%E4%B8%89%E9%98%B6%E6%AE%B5%EF%BC%88SFT-RM-PPO%EF%BC%89.md)。

---

## ⚙️ 1.4.5 PPO 实现细节（看懂术语）

实际 LLM PPO 用了一堆术语：

| 术语 | 含义 |
|---|---|
| **Reference model** $\pi_{ref}$ | 冻结的 SFT 模型，**只用于算 KL** |
| **Policy model** $\pi_\theta$ | 要更新的模型 |
| **Reward model** $r_\phi$ | 已训好的 RM（给整段 response 打分） |
| **Value head** | 加在 policy 上的标量输出层（算 V） |
| **Rollout** | 用当前 π 采样 prompt → response 这一过程 |
| **Mini-batch** | rollout 完一次，分批做多次 epoch 更新 |
| **PPO epoch** | 同一批 rollout 数据用几次（常 1-4 次） |
| **Importance weight** $r_t$ | 新旧策略概率比 |
| **Advantage** $\hat{A}$ | GAE 算的 advantage |

### 几个关键超参

| 参数 | 常用值 | 含义 |
|---|---|---|
| `clip_range ε` | 0.2 | clip 范围 |
| `kl_coef β` | 0.01-0.1 | KL 惩罚强度 |
| `gae_lambda λ` | 0.95 | GAE 偏差-方差权衡 |
| `gamma γ` | 0.99 | 折扣因子 |
| `value_coef` | 0.5-1.0 | critic loss 权重 |
| `ppo_epochs` | 1-4 | 每批数据更新几轮 |

---

## 💻 1.4.6 概念伪代码（LLM PPO）

```python
# 概念示意，来自 PPO 原始论文 + TRL 实现的简化版
# 不是可跑代码，真实实现见 huggingface/trl

# 初始化
policy = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model_path)
ref_policy = freeze(copy(policy))  # 不更新
reward_model = load_reward_model()

for iteration in range(N_iterations):
    # === Phase 1: Rollout ===
    prompts = sample_prompts(batch_size)
    responses = policy.generate(prompts)  # 用当前 π 采样

    # === Phase 2: 算 reward + KL ===
    rewards = reward_model.score(prompts, responses)

    # 计算 token-level KL
    logp_new = policy.log_prob(responses, prompts)
    logp_ref = ref_policy.log_prob(responses, prompts)
    kl = logp_new - logp_ref

    # 用 KL 惩罚后的 reward 算 advantage
    shaped_rewards = rewards - kl_coef * kl
    values = policy.value_head(responses)
    advantages = gae(shaped_rewards, values, gamma=0.99, lam=0.95)

    # === Phase 3: PPO 多轮更新 ===
    logp_old = logp_new.detach()  # 旧策略（这批 rollout 时的）
    for ppo_epoch in range(4):
        logp_curr = policy.log_prob(responses, prompts)  # 当前策略
        ratio = exp(logp_curr - logp_old)

        # PPO clip loss
        loss_unclipped = ratio * advantages
        loss_clipped = clip(ratio, 1-eps, 1+eps) * advantages
        actor_loss = -mean(min(loss_unclipped, loss_clipped))

        # Critic loss
        values_pred = policy.value_head(responses)
        critic_loss = mean((values_pred - returns)**2)

        # 总 loss
        loss = actor_loss + value_coef * critic_loss
        optimizer.step(loss)
```

> ⚠️ 真实 TRL `PPOTrainer` 还有几十个工程细节（micro-batching、grad accumulation、reward normalization 等）。这只是概念骨架。

---

## ⚠️ 1.4.7 PPO 在 LLM 上的"坑"

来自实战经验（多数能在 OpenRLHF、TRL Issue 区看到）：

| 坑 | 表现 | 解决 |
|---|---|---|
| **β 太小** | KL 爆炸，模型胡说 | 加大 β 或加 adaptive KL |
| **β 太大** | π 不动，reward 不涨 | 减小 β |
| **ε 太大** | 单次更新跨度大，崩 | 0.1-0.2 |
| **Rollout 太少** | advantage 方差大 | 加大 batch |
| **Value 不准** | advantage 全错 | 多训几轮 critic |
| **Reward model 漂移** | 模型针对 RM 走捷径 | 用 KL 约束 + reward clipping |
| **Reward hacking** | RM 给高分但实际很烂 | 加 verifier / 人工抽查 |

---

## 🆚 1.4.8 PPO vs GRPO（预告）

| 维度 | PPO | GRPO |
|---|---|---|
| 需要 critic？ | ✅ 是（value head） | ❌ 否 |
| Advantage 来源 | GAE from V | **组内 reward 减组均值** |
| 训练成本 | 中（要训 critic） | 低 |
| 适合任务 | 通用 | **可验证 reward** 任务（数学/代码/工具） |

→ 详见 [03-GRPO（组内相对优势）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)。

---

## 📌 1.4 节要点

| 概念 | 一句话 |
|---|---|
| **PPO** | Proximal Policy Optimization，加 clip 防止 π 跑太远 |
| **Clipping** | 用概率比 $r_t$ 截断在 [1-ε, 1+ε]，限制更新幅度 |
| **KL 约束** | 加 KL penalty 防止 π 偏离 reference（SFT 模型）太远 |
| **Reference model** | 冻结的 SFT 模型，**只用于算 KL** |
| **Value head** | 加在 LLM 上的标量输出层，预测 V |
| **典型超参** | clip_range=0.2, kl_coef=0.01-0.1, γ=0.99 |

---

## 🔗 延伸阅读

- 下一节：[05-Sparse-Reward与信用分配](05-Sparse-Reward%E4%B8%8E%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D.md)
- 应用：[01-RLHF三阶段（SFT-RM-PPO）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/01-RLHF%E4%B8%89%E9%98%B6%E6%AE%B5%EF%BC%88SFT-RM-PPO%EF%BC%89.md)
- 论文卡：[02-必学论文12篇精读卡](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md) #1
- OpenAI Spinning Up PPO：https://spinningup.openai.com/en/latest/algorithms/ppo.html
- CleanRL 单文件 PPO：https://github.com/vwxyzjn/cleanrl
- 跨链：[Happy-LLM RLHF（含 PPO 详解）](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07-RLHF%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.md)

---

⬅ [03-Policy-Gradient与Actor-Critic](03-Policy-Gradient%E4%B8%8EActor-Critic.md) | ➡ [04b-On-Policy-vs-Off-Policy](04b-On-Policy-vs-Off-Policy.md)
