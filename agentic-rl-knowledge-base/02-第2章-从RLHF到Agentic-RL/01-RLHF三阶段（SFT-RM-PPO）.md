---
tags: [agentic-rl, 第2章, RLHF, InstructGPT, SFT, RM, PPO]
chapter: 2
section: 2.1
source: "agentic-rl-learning-map/02-papers.md §InstructGPT, 04-six-week-roadmap.md §第2周, 05-glossary.md §RLHF"
---

# 2.1 RLHF 三阶段（SFT → RM → PPO）

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-DPO（偏好直接优化）](02-DPO%EF%BC%88%E5%81%8F%E5%A5%BD%E7%9B%B4%E6%8E%A5%E4%BC%98%E5%8C%96%EF%BC%89.md)

---

## 🎬 故事比喻：教 AI 写邮件的三步走

```
Step 1 SFT（监督微调）        ─── "我先抄 1 万封人类写的邮件"
                                  → 学会基本格式 + 礼貌用语
                                  → 但还分不出"哪封写得更好"

Step 2 RM（奖励模型）          ─── "现在给你 5 对邮件 (A 比 B 好)
                                   ×10000 对，你学会判断"
                                  → 学会"打分"

Step 3 PPO（在线 RL）          ─── "你自己写一封 → RM 打分
                                   → 高分动作多做，低分少做"
                                  → 边写边改进
```

> **RLHF = Reinforcement Learning from Human Feedback**
> 这是 ChatGPT、Claude、Gemini 共同的"后训练"骨架。
> 你已经在 [04-PPO与KL约束](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md) 学了 PPO，本节把它放进**完整 pipeline**。

---

## 📜 2.1.1 论文与定位

| 字段 | 内容 |
|---|---|
| 标题 | Training Language Models to Follow Instructions with Human Feedback |
| 作者 / 机构 | Long Ouyang et al., OpenAI |
| 年份 | 2022 |
| 状态 | NeurIPS 2022（CCF-A **需进一步确认**） |
| 链接 | https://papers.nips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf |
| 历史地位 | **ChatGPT 论文的前身**；奠定 SFT + RM + PPO 三阶段范式 |

---

## 🏗 2.1.2 三阶段全图

```
                     【阶段 0: 预训练 LLM】
                     (海量文本 next-token prediction)
                              │
                              ▼
   ┌────────────────────────────────────────────────────────────┐
   │                  【阶段 1: SFT】                            │
   │  数据: (prompt, 高质量 completion) 对（人写或精选）          │
   │  目标: 学会"听指令 + 按格式回话"                             │
   │  损失: -log P_θ(y | x)                                      │
   │  产出: SFT model π_SFT                                       │
   └────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────────┐
   │                  【阶段 2: Reward Model】                   │
   │  数据: (prompt, chosen, rejected) 偏好对                    │
   │  目标: 学会"判断哪个 response 更好"                          │
   │  损失: -log σ(r_φ(x, y_w) - r_φ(x, y_l))                    │
   │  产出: Reward model r_φ                                      │
   └────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────────┐
   │                  【阶段 3: PPO 在线 RL】                    │
   │  初始化: π_θ ← π_SFT, π_ref ← π_SFT (冻结)                  │
   │  循环:                                                       │
   │    1. π_θ 采样 response y                                    │
   │    2. r_φ 给 y 打分 R                                        │
   │    3. 加 KL 惩罚: R' = R - β·KL(π_θ || π_ref)               │
   │    4. PPO 更新 π_θ (见 1.4 节)                              │
   │  产出: 对齐后的 π_θ                                          │
   └────────────────────────────────────────────────────────────┘
```

---

## 📚 2.1.3 阶段 1：SFT 详解

### 数据形态

```jsonl
{"prompt": "解释什么是 RLHF", "completion": "RLHF 是..."}
{"prompt": "写一封感谢信", "completion": "亲爱的..."}
{"prompt": "翻译: hello", "completion": "你好"}
```

### 损失函数

$$\mathcal{L}_{SFT}(\theta) = -\sum_{i=1}^{N} \log P_\theta(y_i \mid x_i) = -\sum_{i=1}^{N} \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid x_i, y_{i,<t})$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $N$ | 训练样本数（几万到几十万） |
| $(x_i, y_i)$ | 第 $i$ 条 (prompt, gold completion) 对 |
| $\|y_i\|$ | 第 $i$ 条 completion 的 token 数 |
| $y_{i,t}$ | 第 $i$ 条 completion 的第 $t$ 个 token |
| $y_{i,<t}$ | 第 $i$ 条 completion 第 $t$ 步**之前**的所有 token |
| $P_\theta(y_{i,t} \mid x_i, y_{i,<t})$ | LLM 在给定 prompt + 已生成 token 条件下，预测第 $t$ 个 token 是 $y_{i,t}$ 的概率 |
| 外层 $\sum_i$ | 对所有样本求和 |
| 内层 $\sum_t$ | 对一个 completion 的所有 token 求和（autoregressive 链式分解） |
| 负号 | 最大化 log 似然 = 最小化负 log 似然（NLL） |

**人话**：「**对每个 token，最大化它在 (prompt + 前面 token) 条件下的对数概率**」

→ 这其实就是**预训练的同款 loss**——只不过数据从"互联网海量文本"变成"高质量指令对"。
→ **本质上 SFT = continued pretraining on instruction data**，没有任何 RL 成分。

### 特点

| 特点 | 说明 |
|---|---|
| **有监督学习** | 知道"标准答案"是什么 |
| **比预训练数据**少很多 | 通常几万到几十万条 |
| **教格式**为主 | 教"听指令的格式"，不教"具体好坏" |

### 在 Code Agent 里

→ **SWE-Gym / SWE-smith 合成的 trajectory 可以做 SFT**：
把"成功修 bug 的轨迹" 当作 (state, action) 对训练，让 agent 模仿成功路径。

---

## 🏅 2.1.4 阶段 2：Reward Model 详解

### 数据形态

人类标注 **偏好对**：

```jsonl
{"prompt": "解释 RLHF", "chosen": "RLHF 是用人类反馈...", "rejected": "RLHF 不知道..."}
```

### 损失函数（Bradley-Terry 模型）

$$\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $\phi$ | RM 的参数（独立于 policy $\theta$） |
| $\mathcal{D}$ | 偏好对数据集 |
| $(x, y_w, y_l)$ | 一对偏好：prompt $x$，chosen $y_w$（winner），rejected $y_l$（loser） |
| $r_\phi(x, y)$ | RM 给 (prompt, response) 的标量打分 |
| $r_\phi(x, y_w) - r_\phi(x, y_l)$ | 偏好分差（**正数 = RM 排序正确**） |
| $\sigma(z) = \frac{1}{1+e^{-z}}$ | sigmoid，把分差映射到 $(0, 1)$ |
| $\sigma(\text{分差})$ | RM 认为 chosen 比 rejected 好的**概率** |
| $-\log \sigma(\cdot)$ | 二分类的 NLL loss |

**为什么用 Bradley-Terry 模型？**

Bradley-Terry 模型假设：

$$P(y_w \succ y_l \mid x) = \frac{e^{r(x, y_w)}}{e^{r(x, y_w)} + e^{r(x, y_l)}} = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

→ 即"chosen 胜出的概率"取决于两个 reward 的差。
→ 这是从体育排名（如 Elo）借来的标准成对比较模型。
→ 我们要最大化"模型认为 chosen 胜出的概率" = 最大化 $\sigma(\Delta r)$ = 最小化 $-\log \sigma(\Delta r)$。

**数值小例子**：

| $r(y_w)$ | $r(y_l)$ | $\Delta r$ | $\sigma(\Delta r)$ | loss $-\log \sigma$ |
|---|---|---|---|---|
| 2.0 | 1.0 | +1.0 | 0.73 | 0.31（RM 学对了，loss 小） |
| 2.0 | 2.0 | 0 | 0.50 | 0.69（RM 没区分） |
| 1.0 | 2.0 | -1.0 | 0.27 | 1.31（**RM 排反了，loss 大**） |
| 5.0 | 0.0 | +5.0 | 0.993 | 0.007（RM 强烈分对，loss 极小） |

**直觉**：「**让 RM 给 chosen 的分数 > rejected 的分数，差距越大越好**」

### RM 架构

```
             prompt + response
                    │
                    ▼
        ┌─────────────────────┐
        │   LLM (通常和 SFT     │
        │   model 同源)         │
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │   Linear Head → 标量  │  ← reward score
        └─────────────────────┘
```

### 在 Code Agent 里

→ **可以用 verifier 代替 RM**（这就是 RLVR 的核心思想）：
```
RM:        "patch A 比 patch B 好" → 学打分
Verifier:  "patch A 跑 pytest 过 + B 不过" → 直接知道好坏，不用学
```

→ 这就是为什么 [04-RLVR（可验证奖励）](04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md) 在代码任务上**几乎完全替代** RM。

---

## 🎯 2.1.5 阶段 3：PPO 详解（回顾 + LLM 特殊化）

PPO 部分已经在 [04-PPO与KL约束](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md) 详讲。这里只说**LLM 特殊化**：

### LLM PPO 完整目标

$$\mathcal{L}_{LLM-PPO}(\theta) = \mathbb{E}_t\Big[\min\big(r_t \hat{A}_t, \, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\big)\Big] - \beta \cdot \mathbb{E}_x\Big[\text{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{ref}(\cdot \mid x)\big)\Big]$$

**3 个组成详细拆解**：

| 项 | 公式 | 作用 |
|---|---|---|
| **Clip 主项** | $\min(r_t \hat{A}_t, \text{clip}(r_t) \hat{A}_t)$ | 限制单步更新幅度（PPO 核心） |
| **KL 惩罚** | $\beta \cdot \text{KL}(\pi_\theta \,\Vert\, \pi_{ref})$ | **防止 π 偏离 SFT 模型太远，保护语言能力** |
| **Advantage** | $\hat{A}_t$ 由 critic + GAE 算 | actor-critic 结构 |

→ Clip 详见 [§1.4](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)，Advantage 详见 [§1.2.4](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/02-Policy-Value-Advantage.md)。

### Reward 实际怎么算（reward shaping 形式）

实践中常**不**把 KL 单独当 loss 项，而是**塞进 reward**：

$$\tilde{r}_t = \begin{cases} -\beta \cdot \big(\log \pi_\theta(a_t \mid s_t) - \log \pi_{ref}(a_t \mid s_t)\big) & \text{if } t < T \\ r_\phi(x, y) - \beta \cdot \big(\log \pi_\theta(a_t \mid s_t) - \log \pi_{ref}(a_t \mid s_t)\big) & \text{if } t = T \end{cases}$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $\tilde{r}_t$ | **shaped reward**（经过 KL 校正） |
| $r_\phi(x, y)$ | RM 对**整段** response 的打分（**只在最后一个 token 给**） |
| $\log \pi_\theta - \log \pi_{ref}$ | token-level KL 估计（详见下方 §1.4.3 跨链） |
| 中间 token | reward 只有 $-\beta \cdot \text{KL}$（"不要乱走"的信号） |
| 末尾 token | reward = RM 分数 - KL（"目标 + 不要乱走"） |

→ token-level KL 详见 [§1.4.3](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)。

**整段总 reward**：

$$R_{total} = r_\phi(x, y) - \beta \cdot \text{KL}\big(\pi_\theta(y \mid x) \,\|\, \pi_{ref}(y \mid x)\big)$$

**重要**：reward **不是**直接用 RM 打分 $r_\phi$，而是 **RM 分数减去 KL 惩罚**。
→ 否则 LLM 会"为了高分变成奇怪输出"。

**$\beta$ 怎么调？**

| $\beta$ | 现象 | 应对 |
|---|---|---|
| 太小（如 0） | KL 暴涨，模型输出胡言乱语 | 调大 |
| 太大（如 1.0） | 模型几乎不动，reward 不涨 | 调小 |
| **0.01-0.1** | 典型范围 | 用 **adaptive KL** 动态调（InstructGPT 用） |

### Adaptive KL（论文里的细节）

InstructGPT 用动态 $\beta$：每步根据**实际 KL** 调整：

$$\beta_{new} = \beta_{old} \cdot \begin{cases} 1.5 & \text{if } \text{KL}_{measured} > 1.5 \cdot \text{KL}_{target} \\ 0.5 & \text{if } \text{KL}_{measured} < 0.5 \cdot \text{KL}_{target} \\ 1.0 & \text{otherwise} \end{cases}$$

→ KL 太大就**增大**惩罚强度，太小就**减小**——自动维持在 target 附近。

### Rollout / Update 循环成本

```
1 次 rollout: 用 π 生成几百到几千 response
            → 每个 response 几十到几千 token
            → 全部要算 logprob
            → 几张 GPU 几小时

1 次 PPO 更新: 用 rollout 数据多 epoch 训
            → 算 ratio、advantage、clip
            → 更新 actor + critic

→ 整个 RLHF 训练: 几天 ~ 几周
```

→ 这就是为什么后来出了 **DPO**（省 PPO）和 **GRPO**（省 critic）。

---

## 🧪 2.1.6 应用于 Code Agentic RL 的样子

把 RLHF 三阶段套到 Code Agent：

| RLHF 三阶段 | Code Agentic RL 对应 |
|---|---|
| **阶段 1 SFT** | 用 SWE-Gym / SWE-smith 的成功 trajectory 训 agent |
| **阶段 2 RM** | **跳过**！用 **pytest 当 verifier** |
| **阶段 3 PPO** | 用 GRPO + RLVR reward 训 agent（pytest 通过率） |

→ Code 任务的最大幸运：**有客观 verifier，不需要训 RM**。
→ 直接跳到 [04-RLVR（可验证奖励）](04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md)。

---

## ⚠️ 2.1.7 RLHF 的局限（推动了 DPO/GRPO/RLVR）

| 局限 | 后续算法的应对 |
|---|---|
| **RM 训练贵 + RM 偏差大** | DPO（绕过 RM）/ RLVR（用客观验证器） |
| **PPO 需要 critic（V head）** | GRPO（组内均值代替 critic） |
| **PPO 是 on-policy，rollout 贵** | DPO（offline，用静态偏好对） |
| **人类反馈贵** | RLAIF（AI 反馈）/ RLVR（验证器反馈） |
| **单轮回答，不适合多步任务** | Agentic RL（多步 trajectory） |

→ 第 2 章后续每一节就是回应这些局限。

---

## 📌 2.1 节要点

| 阶段 | 一句话 | 损失 |
|---|---|---|
| **SFT** | 监督学习教格式 | $-\log P(y\|x)$ |
| **RM** | 学打分（偏好对） | $-\log \sigma(r_w - r_l)$ |
| **PPO** | 在线 RL 优化策略 | clip + KL penalty |

| 关键超参 | 常用值 |
|---|---|
| `clip ε` | 0.2 |
| `kl_coef β` | 0.01-0.1 |
| `gamma γ` | 0.99 |

| Code Agent 中的变化 | 影响 |
|---|---|
| RM → 用 pytest 当 verifier | **直接跳到 RLVR** |
| PPO → GRPO | **省掉 critic** |

---

## 🔗 延伸阅读

- 下一节：[02-DPO（偏好直接优化）](02-DPO%EF%BC%88%E5%81%8F%E5%A5%BD%E7%9B%B4%E6%8E%A5%E4%BC%98%E5%8C%96%EF%BC%89.md)
- PPO 算法详细：[04-PPO与KL约束](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/04-PPO%E4%B8%8EKL%E7%BA%A6%E6%9D%9F.md)
- 论文卡：[02-必学论文12篇精读卡](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md) #2 (InstructGPT)
- 跨链（必读前置）：[Happy-LLM RLHF 章详解](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07-RLHF%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.md)
- 跨链（实战）：[03-SFT训练实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/03-SFT%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)
- 原始资料：`02-papers.md` §InstructGPT 行

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-DPO（偏好直接优化）](02-DPO%EF%BC%88%E5%81%8F%E5%A5%BD%E7%9B%B4%E6%8E%A5%E4%BC%98%E5%8C%96%EF%BC%89.md)
