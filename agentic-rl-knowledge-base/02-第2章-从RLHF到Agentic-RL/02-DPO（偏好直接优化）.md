---
tags: [agentic-rl, 第2章, DPO, 偏好优化, 离线RL]
chapter: 2
section: 2.2
source: "agentic-rl-learning-map/02-papers.md §DPO, 05-glossary.md §DPO"
---

# 2.2 DPO（偏好直接优化）

⬅ [01-RLHF三阶段（SFT-RM-PPO）](01-RLHF%E4%B8%89%E9%98%B6%E6%AE%B5%EF%BC%88SFT-RM-PPO%EF%BC%89.md) | ➡ [03-GRPO（组内相对优势）](03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)

---

## 🎬 故事比喻：RLHF 走捷径

```
RLHF 像点外卖三步:
   1. 找人评价店铺（训 RM）
   2. 训 AI 厨师按 RM 评分做菜（PPO 在线 rollout）
   3. 厨师天天炒菜让 RM 评（贵 + 慢）

DPO 直接说:
   "把全部 (好菜, 坏菜) 对放一起，让厨师直接学'什么是好'"
   → 不要 RM，不要在线炒菜，一锅出
```

> **DPO = Direct Preference Optimization**
> 一句话：**用偏好对直接训练 policy，不要 RM，不要在线 rollout**。
> 工程上把 RLHF 从 3 阶段压成 2 阶段（SFT + DPO）。

---

## 📜 2.2.1 论文与定位

| 字段 | 内容 |
|---|---|
| 标题 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model |
| 作者 / 机构 | Rafael Rafailov et al., Stanford 等 |
| 年份 | 2023 |
| 状态 | NeurIPS 2023（CCF-A **需进一步确认**） |
| 链接 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html |
| 副标题含义 | "**你的 LLM 本身就是个隐式 reward model**"——不需要单独训 RM |

---

## 🎯 2.2.2 核心思想（含闭式解推导）

### Step 1：RLHF + KL 约束的最优解有闭式形式

RLHF 的目标（带 KL 约束）：

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot \mid x)}\big[r(x, y)\big] - \beta \cdot \text{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{ref}(\cdot \mid x)\big)$$

**这个优化问题有解析解**（拉格朗日乘子法推导）：

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{ref}(y \mid x) \cdot \exp\left(\frac{1}{\beta} r(x, y)\right)$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $\pi^*$ | 在给定 reward $r$ 下的**最优策略** |
| $\pi_{ref}(y \mid x)$ | 参考模型给出的概率（"先验"） |
| $\exp\left(\frac{r(x,y)}{\beta}\right)$ | reward 的指数（"reward 越高，权重越大"） |
| $\frac{1}{\beta}$ | KL 越严（$\beta$ 大），reward 影响越小（"贴近 ref"） |
| $Z(x) = \sum_y \pi_{ref}(y\|x) \exp(r(x,y)/\beta)$ | **归一化常数**（partition function），让 $\pi^*$ 概率和=1 |

**直觉**：「**最优策略 = ref × 按 reward 加权再归一化**」（类似 RM 输出的 softmax 加权）

### Step 2：反向工程——从最优策略反推 reward

把闭式解两边取 log，**重排**：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x)$$

→ 这告诉我们：**reward 完全可以用 policy ratio 表示**！
→ 唯一额外的项 $\beta \log Z(x)$ **只依赖 $x$**（不依赖 $y$）——后面会消掉。

### Step 3：把这个 reward 代回 Bradley-Terry

回忆 [RM 训练](01-RLHF%E4%B8%89%E9%98%B6%E6%AE%B5%EF%BC%88SFT-RM-PPO%EF%BC%89.md) 的 Bradley-Terry 模型：

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

代入 Step 2 的 reward 表达：

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{ref}(y_l \mid x)}$$

**注意 $\beta \log Z(x)$ 在相减时消掉了！** 这是 DPO 数学上最优雅的一步。

→ 我们**不需要知道 $r$，也不需要算 $Z$**——直接用 policy ratio 就能算偏好概率。

### Step 4：DPO 损失函数（用 $\pi_\theta$ 代替 $\pi^*$ 来训）

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right) \right]$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $\theta$ | 要训的 policy 参数（**唯一**要更新的） |
| $(x, y_w, y_l)$ | 偏好对：prompt + chosen + rejected |
| $\pi_\theta(y \mid x)$ | 当前 policy 对**整段** response 的概率 = $\prod_t \pi_\theta(y_t \mid x, y_{<t})$ |
| $\pi_{ref}(y \mid x)$ | 冻结的参考模型概率 |
| $\log \frac{\pi_\theta(y\|x)}{\pi_{ref}(y\|x)}$ | **隐式 reward**（"$\pi_\theta$ 相对 ref 给这个 $y$ 多大权重"） |
| $\beta$ | KL 强度（继承自 RLHF 闭式解，**控制 ref 偏离程度**） |
| 减号 $\beta(\cdot)_w - \beta(\cdot)_l$ | 隐式 reward 差（"chosen 相对 ref 应该比 rejected 相对 ref 更被偏好"） |
| $\sigma$ | 把分差映射成"chosen 胜出的概率" |
| $-\log \sigma$ | 标准 binary NLL |

**人话**：「**让 $\pi_\theta$ 给 chosen 的相对提升（vs ref）显著大于 rejected 的相对提升**」

### 实战中怎么算 $\log \pi_\theta(y \mid x)$？

对一段 response $y = (y_1, \dots, y_L)$，**autoregressive 求和**：

$$\log \pi_\theta(y \mid x) = \sum_{t=1}^{L} \log \pi_\theta(y_t \mid x, y_{<t})$$

→ 直接做 forward pass 拿每个位置的 log-prob，加起来即可。
→ 这就是 TRL DPOTrainer 里 `compute_logprobs` 在做的事。

### 数值小例子

设 $\beta = 0.1$，某 prompt 上一对 (chosen, rejected)：

```
log π_θ(y_w | x) = -10.0     log π_ref(y_w | x) = -12.0
log π_θ(y_l | x) = -15.0     log π_ref(y_l | x) = -14.0
```

| 量 | 计算 |
|---|---|
| 隐式 reward (chosen) | $\beta \cdot (-10.0 - (-12.0)) = 0.1 \cdot 2.0 = +0.2$ |
| 隐式 reward (rejected) | $\beta \cdot (-15.0 - (-14.0)) = 0.1 \cdot (-1.0) = -0.1$ |
| 分差 | $+0.2 - (-0.1) = +0.3$ |
| $\sigma(0.3)$ | $\approx 0.574$ |
| $-\log 0.574$ | $\approx 0.555$ |

→ loss = 0.555。如果优化让分差变 +1.0，loss 降到 $-\log \sigma(1) \approx 0.31$。
→ 训练的目标：**让分差越来越正**。

---

## 🆚 2.2.3 DPO vs RLHF（核心对比）

| 维度 | RLHF (PPO) | DPO |
|---|---|---|
| 阶段数 | 3 (SFT + RM + PPO) | 2 (SFT + DPO) |
| 需要 reward model？ | ✅ 是 | ❌ 否（隐式） |
| Online rollout？ | ✅ 是（每次更新前要 sample） | ❌ 否（静态数据集） |
| 训练时显式 KL？ | ✅ 是 | 隐式（在 loss 里通过 β log ratio） |
| 训练稳定性 | 难，需要调 PPO 一堆超参 | 稳定（**像 SFT 一样训**） |
| 训练成本 | 高 | 低 |
| 表达能力 | 强（在线探索） | 弱（受限于静态数据） |
| 适合场景 | 大规模生产（GPT-4 风格） | 中小型 alignment、快速迭代 |

> 📌 **DPO 几乎是"用 SFT 的方式做 RLHF"**——这是它最大的卖点。

---

## 🧪 2.2.4 在 Code Agent 中的用法

DPO 在 Code 任务里**非常自然**——因为可以**自动构造偏好对**：

```
对同一个 SWE-bench issue:
   sample patch A → pytest 通过 ✓
   sample patch B → pytest 失败 ✗

→ (issue, patch_A, patch_B) 就是一对 DPO 数据
```

**优点**：

- 不需要人标
- 一次 rollout 出 16 个 patch 能形成 16×15/2 = 120 对
- 适合 toy bug-fix gym → 入门 RL 训练的最快路径

→ 见 [02-Toy-Bug-Fix-Agent-Gym（核心）](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md) 中的"构造 preference pairs"步骤。

---

## ⚠️ 2.2.5 DPO 的局限

| 局限 | 说明 |
|---|---|
| **数据质量决定上限** | 偏好对必须有信息量；如果 chosen 也很烂，DPO 学不到啥 |
| **不能探索** | 静态数据集，看不到新的 (state, action) 组合 |
| **β 难调** | 太小→偏离 ref；太大→学不动 |
| **会高估 OOD 分布** | DPO 可能让 OOD 概率反而上升（"likelihood inflation"问题） |
| **不太适合长 trajectory** | DPO 假设 (chosen, rejected) 是平等的 response，不擅长多步 |

> 💡 **DPO 不适合 Agentic RL 的"在线探索"需求**。
> 真要做多步 agent RL，回到 GRPO + 在线 rollout 更稳。

---

## 📈 2.2.6 DPO 的变体

| 变体 | 一句话 |
|---|---|
| **IPO** (Identity Preference Optimization) | 用平方损失代替 sigmoid，减少 overfit |
| **KTO** (Kahneman-Tversky Optimization) | 单独对每个样本打分（thumbs up/down），不要成对 |
| **SimPO** | 去掉 reference model，进一步简化 |
| **ORPO** | DPO + SFT 损失联合训练 |

→ 这些都在 TRL 里有实现，**入门只需理解 DPO，其它按需查**。

---

## 💻 2.2.7 概念伪代码

```python
# 概念示意，TRL 的 DPOTrainer 实际接口
from trl import DPOTrainer

# 准备 preference 数据
dataset = load_dataset("preference_pairs")
# 每条: {"prompt": ..., "chosen": ..., "rejected": ...}

# 准备模型
policy = AutoModelForCausalLM.from_pretrained(sft_model)
ref_model = AutoModelForCausalLM.from_pretrained(sft_model)  # 冻结

# 训练
trainer = DPOTrainer(
    model=policy,
    ref_model=ref_model,
    args=TrainingArguments(...),
    train_dataset=dataset,
    beta=0.1,  # KL 强度
)
trainer.train()
```

→ **像 SFT 一样跑**，这就是 DPO 最大的工程优势。

---

## 📌 2.2 节要点

| 问题 | 一句话答案 |
|---|---|
| DPO 解决什么？ | RLHF 太重（要 RM、要 PPO、要 rollout） |
| DPO 怎么做到？ | 数学上 reward 和 policy 有闭式关系，**绕过 RM** |
| 训练时是 on-policy 还是 off-policy？ | **Off-policy**（静态数据集） |
| 需要 reference model 吗？ | 是，通常是 SFT 模型 |
| 适合 Code Agent 吗？ | 适合做 **preference pair 训练**（patch A 过 vs patch B 不过） |
| 不适合什么？ | 长 trajectory 多步 agent；需要在线探索的任务 |

---

## 🔗 延伸阅读

- 下一节：[03-GRPO（组内相对优势）](03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)
- 论文卡：[02-必学论文12篇精读卡](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md) #3
- 应用：[02-Toy-Bug-Fix-Agent-Gym（核心）](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md) § 构造偏好对
- 原始资料：`02-papers.md` §DPO 行

---

⬅ [01-RLHF三阶段（SFT-RM-PPO）](01-RLHF%E4%B8%89%E9%98%B6%E6%AE%B5%EF%BC%88SFT-RM-PPO%EF%BC%89.md) | ➡ [03-GRPO（组内相对优势）](03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)
