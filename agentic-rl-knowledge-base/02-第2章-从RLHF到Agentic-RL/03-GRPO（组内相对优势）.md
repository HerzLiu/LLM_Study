---
tags: [agentic-rl, 第2章, GRPO, DeepSeekMath, DeepSeek-R1, 组内相对优势, 省critic]
chapter: 2
section: 2.3
source: "agentic-rl-learning-map/02-papers.md §DeepSeekMath, 05-glossary.md §GRPO"
---

# 2.3 GRPO（组内相对优势，省掉 critic）

⬅ [02-DPO（偏好直接优化）](02-DPO%EF%BC%88%E5%81%8F%E5%A5%BD%E7%9B%B4%E6%8E%A5%E4%BC%98%E5%8C%96%EF%BC%89.md) | ➡ [04-RLVR（可验证奖励）](04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md)

---

## 🎬 故事比喻：班级相对排名打分

```
PPO 模式（要 critic 当"老师"）:
   每个学生考完试 → 老师评"绝对分数 V"
   → 学生分数 - 老师评分 = advantage
   → 你需要既培养学生（actor）也培养老师（critic）

GRPO 模式（不要 critic，要"班级排名"）:
   同一道题让 K 个学生答 → 组成一个班级
   → 每个学生的"班级相对分数" = 他的分数 - 全班均值
   → 不需要老师！
```

> **GRPO = Group Relative Policy Optimization**
> DeepSeekMath 提出的"省 critic"算法。
> **DeepSeek-R1** 用它训出长链 reasoning。
> **当下 Code Agentic RL（SWE-RL、open-r1）几乎都用它**。

---

## 📜 2.3.1 论文与定位

| 字段 | 内容 |
|---|---|
| 标题 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
| 作者 / 机构 | Zhihong Shao et al., DeepSeek-AI |
| 年份 | 2024 |
| 状态 | arXiv |
| 链接 | https://arxiv.org/abs/2402.03300 |
| 应用 | DeepSeek-R1 / SWE-RL / open-r1 / 各种 reasoning RL 工作 |

---

## 🎯 2.3.2 核心思想

回顾 PPO（[04-PPO与KL约束](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)）：

$$L^{PPO}(\theta) = \mathbb{E}\big[\min(\rho \hat{A}, \, \text{clip}(\rho) \hat{A})\big] - \beta \cdot \text{KL}, \quad \rho = \frac{\pi_\theta}{\pi_{\theta_{old}}}, \quad \hat{A}_t \approx G_t - V(s_t)$$

PPO 需要 critic $V$ 来算 advantage。

**GRPO 的洞察**：

> 对**同一个 prompt** sample **K 个 response**，每个 response 有自己的 reward $r_i$。
> 用**组内均值** $\bar{r} = \frac{1}{K}\sum_i r_i$ 当 baseline，不要 V！

**Advantage 用组内统计代替 critic**：

$$\hat{A}_i = \frac{R_i - \mu_g}{\sigma_g}, \quad \text{where } \mu_g = \frac{1}{K}\sum_{j=1}^K R_j, \quad \sigma_g = \sqrt{\frac{1}{K}\sum_{j=1}^K (R_j - \mu_g)^2 + \epsilon}$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $i$ | 当前 response 在组里的索引（$i \in [1, K]$） |
| $R_i$ | 第 $i$ 个 response 的**整段 reward**（如 pytest pass=1, fail=0） |
| $\mu_g$ | **组均值**（"这个 prompt 的平均水平"） |
| $\sigma_g$ | **组标准差**（"组内分散度"，加 $\epsilon$ 防 0） |
| $(R_i - \mu_g) / \sigma_g$ | $R_i$ 的 z-score（"这个 response 比组均值好几个标准差"） |
| $\hat{A}_i$ | 这个 response 内**所有 token**共享同一个 advantage |

**关键设计选择**：

- **组内同 $\hat{A}$**：response 内每个 token 都用同一个 $\hat{A}_i$（不分配到 token level）
- **同 prompt 才能组**：不同 prompt 的 $R$ 不可比，因为任务难度不同
- **不要 $V$**：组均值 $\mu_g$ 起到了 $V(s)$ 的 baseline 作用

---

## 📐 2.3.3 GRPO 完整目标函数（逐项展开）

$$L^{GRPO}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{y_i\}_{i=1}^K \sim \pi_{\theta_{old}}(\cdot \mid q)}\Bigg[\frac{1}{K} \sum_{i=1}^K \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\Big(\rho_{i,t} \hat{A}_i, \; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i\Big)\Bigg] - \beta \cdot \mathbb{E}_q\big[\text{KL}(\pi_\theta \| \pi_{ref})\big]$$

其中 token-level 概率比：

$$\rho_{i,t} = \frac{\pi_\theta(y_{i,t} \mid q, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t} \mid q, y_{i,<t})}$$

**逐项拆解**：

| 符号 | 含义 | 维度 |
|---|---|---|
| $q$ | prompt（从数据集 $\mathcal{D}$ 采样） | 1 个 |
| $\{y_i\}_{i=1}^K$ | 同一 $q$ 下 $\pi_{\theta_{old}}$ 采样的 **K 个 response** | K 段 |
| $|y_i|$ | 第 $i$ 个 response 的 token 数 | 标量 |
| $y_{i,t}$ | 第 $i$ 个 response 的第 $t$ 个 token | 1 个 token |
| $\rho_{i,t}$ | **token 级**的 importance ratio | 标量 |
| $\hat{A}_i$ | **response 级**的组内相对 advantage（同一 $i$ 所有 $t$ 共享） | 标量 |
| $\epsilon$ | clip 范围，常 0.2 | 标量 |
| $\beta$ | KL 惩罚系数 | 标量 |
| $\frac{1}{K} \sum_i$ | 对 K 个 response 求平均 | 平均 |
| $\frac{1}{\|y_i\|} \sum_t$ | 对一个 response 的所有 token 求平均（**长度归一化**） | 平均 |

**和 PPO 对比的关键 3 处差异**：

| 维度 | PPO | GRPO |
|---|---|---|
| Advantage | $\hat{A}_t = $ GAE from $V$ | $\hat{A}_i = (R_i - \mu_g) / \sigma_g$，**整 response 共享** |
| Token 级长度归一化 | 没有 | **有** $\frac{1}{\|y_i\|}$ |
| Critic 网络 | 需要 | **不需要** |

→ **长度归一化**避免"长 response 影响过大"——这是 GRPO 在 LLM 上工作良好的关键工程细节之一。

---

## 🔢 2.3.4 数值小例子（一组 K=4 算 advantage 全过程）

对一个 SWE-bench task，$\pi_{\theta_{old}}$ 采样 4 个 trajectory：

```
y_1: 完整修 bug → pytest pass → R_1 = 1.0
y_2: edit 错地方 → pytest fail → R_2 = 0.0
y_3: 修对一半 → 部分 pass → R_3 = 0.5
y_4: 完整修对 → pytest pass → R_4 = 1.0
```

**Step 1**：算组统计

$$\mu_g = \frac{1.0 + 0.0 + 0.5 + 1.0}{4} = 0.625$$

$$\sigma_g = \sqrt{\frac{(1.0-0.625)^2 + (0.0-0.625)^2 + (0.5-0.625)^2 + (1.0-0.625)^2}{4}} \approx 0.415$$

**Step 2**：算每个 advantage

| $i$ | $R_i$ | $R_i - \mu_g$ | $\hat{A}_i = (R_i - \mu_g)/\sigma_g$ | 含义 |
|---|---|---|---|---|
| 1 | 1.0 | +0.375 | **+0.904** | 强化（这条比平均好近 1 个 std） |
| 2 | 0.0 | -0.625 | **-1.506** | 重抑制（这条比平均差近 1.5 个 std） |
| 3 | 0.5 | -0.125 | **-0.301** | 轻抑制 |
| 4 | 1.0 | +0.375 | **+0.904** | 强化 |

**Step 3**：$\hat{A}_i$ 作用到这条 trajectory 的**每个 token 梯度**

→ $y_1$ 和 $y_4$ 的所有 token 概率被**提高**
→ $y_2$ 的所有 token 概率被**压低**最多
→ $y_3$ 的所有 token 概率被**轻微压低**

→ 这就是 GRPO 的训练信号。**没有用到 $V(s)$**。

---

## ⚠️ 2.3.5 当组内 reward 全相同时（边界情况）

若 $\sigma_g = 0$（全 pass 或全 fail），所有 $\hat{A}_i = 0$，**梯度全为 0**——这组**白训了**。

**应对**：

- 加 $\epsilon$ 防除 0（如 $\sigma_g + 10^{-8}$）
- **任务难度分层采样**：避免组里全是 trivial 或 impossible 任务
- DAPO 等改进版用 **dynamic sampling**：filter 掉无效组

---

## 🆚 2.3.4 GRPO vs PPO（核心对比）

| 维度 | PPO | GRPO |
|---|---|---|
| 需要 critic（V）？ | ✅ 是 | ❌ 否 |
| 显存占用 | actor + critic | 只 actor |
| Advantage 计算 | GAE on V | **组内 reward 减均值** |
| 适合 reward 类型 | 任意（连续/稀疏） | **稀疏 reward 尤其好**（如 0/1） |
| 训练成本 | 高 | **低 30-50%** |
| 收敛速度 | 中 | 通常更快 |
| 同 prompt sample 几次？ | 1 次 | **K = 4-16 次** |

> 💡 **GRPO 用同 prompt sample K 次的成本换掉了 critic 训练成本**。
> 对**可验证 reward** 任务（数学、代码、工具），这个 trade-off **非常划算**。

---

## 🧠 2.3.5 为什么 GRPO 在可验证 reward 上好？

对于 reward ∈ {0, 1}（如 pytest 通过 / 不通过）：

```
PPO: 需要 critic 学一个 V，但 V 在 sparse + binary reward 下**很难训准**
     → critic 一不准，advantage 全错，actor 也学崩

GRPO: 直接 sample K 次 → 看到 K 个 0/1
     → 组均值就是"这个 prompt 的难度"（如 0.3 表示 30% 概率能过）
     → reward > 均值 → advantage > 0 → 强化
     → reward < 均值 → advantage < 0 → 抑制
     → 不需要 V，**advantage 完全由 sample 给出**
```

→ 这就是为什么 DeepSeek-R1（reasoning RL）和 SWE-RL（code RL）都用 GRPO。

---

## 🔄 2.3.6 GRPO 训练循环

```
                     ┌──────────────────────────┐
                     │   取一批 prompts          │
                     └──────────────────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │   对每个 prompt           │
                     │   sample K 个 response    │
                     │   (K = 4-16)             │
                     └──────────────────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │   每个 response 算 reward  │
                     │   (verifier / pytest /   │
                     │    答案校验)             │
                     └──────────────────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │   每组算 (r_i - mean)/std  │
                     │   → advantage Â_i         │
                     └──────────────────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │   PPO-style update        │
                     │   (clip + KL penalty)     │
                     │   仅更新 actor π          │
                     └──────────────────────────┘
                                  │
                                  └────── loop ──────
```

---

## 🧪 2.3.7 在 Code Agent 中的具体用法

```
对一个 SWE-bench issue:
   1. agent π_θ 用相同 prompt sample 16 个 trajectory（每个完整修 bug 过程）
   2. 每个 trajectory 跑 pytest → reward ∈ {0, 1}
   3. 假设 4 个 pass, 12 个 fail → mean=0.25, std=0.43
   4. pass 的 advantage = (1-0.25)/0.43 = 1.74  → 大幅强化
   5. fail 的 advantage = (0-0.25)/0.43 = -0.58 → 抑制
   6. PPO-style update + KL 约束
```

**SWE-RL 论文**（[4.6](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/06-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95%EF%BC%88SWE-Gym-SWE-RL%E7%AD%89%EF%BC%89.md)）就是这个套路。

---

## ⚠️ 2.3.8 GRPO 的局限

| 局限 | 说明 |
|---|---|
| **需要 K 倍 rollout 成本** | sample K 次比 PPO 的 1 次贵 K 倍 |
| **K 太小时高方差** | K < 4 时组均值不稳；常用 K = 8-16 |
| **reward 全相同时 advantage = 0** | 一组全 pass 或全 fail → std=0 → 无信号；需要任务难度适配 |
| **不适合 reward 连续且密集** | dense reward 上 PPO + critic 仍更好 |
| **K 越大越贵** | trade-off：质量 vs 成本 |

> 💡 K 取多少？**论文常用 K=8 或 16**；toy 项目可 K=4 起步。

---

## 💻 2.3.9 概念伪代码

```python
# 概念示意，TRL/verl/OpenRLHF 都有 GRPOTrainer
def grpo_step(prompts, policy, ref_policy, reward_fn, K=8):
    all_data = []
    for q in prompts:
        # 1. Sample K responses
        responses = [policy.sample(q) for _ in range(K)]
        # 2. Compute rewards
        rewards = [reward_fn(q, y) for y in responses]
        # 3. Group-relative advantages
        mean_r, std_r = np.mean(rewards), np.std(rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in rewards]

        for y, A in zip(responses, advantages):
            all_data.append((q, y, A))

    # 4. PPO-style update (clip + KL)
    for q, y, A in all_data:
        logp_new = policy.log_prob(y, q)
        logp_old = logp_new.detach()  # 采样时的 logprob
        logp_ref = ref_policy.log_prob(y, q)

        ratio = exp(logp_new - logp_old)
        loss = -mean(min(ratio * A, clip(ratio, 0.8, 1.2) * A))
        kl = mean(logp_new - logp_ref)
        total_loss = loss + beta * kl

        update(policy, total_loss)
```

→ **完整 Python 实战见跨链** [04-GRPO训练实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)。

---

## 📌 2.3 节要点

| 问题 | 一句话答案 |
|---|---|
| GRPO 是什么？ | Group Relative Policy Optimization，PPO 的"省 critic"变体 |
| 怎么省 critic？ | 同 prompt sample K 次，**组内 reward 减均值**当 advantage |
| 适合什么任务？ | **稀疏 + 可验证 reward**（数学、代码、工具） |
| 谁在用？ | DeepSeek-R1、SWE-RL、open-r1、几乎所有 reasoning RL |
| 成本 vs PPO？ | 显存少（无 critic），但 rollout 贵 K 倍 |
| K 取多少？ | 4-16，常用 8 |

---

## 🔗 延伸阅读

- 下一节：[04-RLVR（可验证奖励）](04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md)
- 论文卡：[02-必学论文12篇精读卡](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md) #4 (DeepSeekMath)、#5 (DeepSeek-R1)
- 应用：[06-强推+可选论文索引（SWE-Gym-SWE-RL等）](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/06-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95%EF%BC%88SWE-Gym-SWE-RL%E7%AD%89%EF%BC%89.md) SWE-RL
- 跨链（**强烈推荐看完整训练代码**）：[04-GRPO训练实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)
- 跨链（GRPO 在 reward 设计前置）：[02-数据集与奖励函数](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/02-%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8E%E5%A5%96%E5%8A%B1%E5%87%BD%E6%95%B0.md)
- 原始资料：`02-papers.md` §DeepSeekMath 行

---

⬅ [02-DPO（偏好直接优化）](02-DPO%EF%BC%88%E5%81%8F%E5%A5%BD%E7%9B%B4%E6%8E%A5%E4%BC%98%E5%8C%96%EF%BC%89.md) | ➡ [04-RLVR（可验证奖励）](04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md)
