---
tags: [agentic-rl, 第2章, PRM, ORM, 过程奖励, 结果奖励, Lets-Verify-Step-by-Step]
chapter: 2
section: 2.5
source: "agentic-rl-learning-map/02-papers.md §Lets Verify Step by Step, 05-glossary.md §ORM/PRM"
---

# 2.5 PRM 与 ORM（过程奖励 vs 结果奖励）

⬅ [04-RLVR（可验证奖励）](04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md) | ➡ [06-五大算法对比表](06-%E4%BA%94%E5%A4%A7%E7%AE%97%E6%B3%95%E5%AF%B9%E6%AF%94%E8%A1%A8.md)

---

## 🎬 故事比喻：考数学的两种判卷

```
ORM 模式（只看答案）:
   "答案 = 42 → 对，给满分"
   "答案 ≠ 42 → 错，0 分"
   → 学生可能瞎写过程，凑对答案

PRM 模式（看每一步）:
   "第 1 步对 +1"
   "第 2 步对 +1"
   "第 3 步用了错的公式 -2"
   "第 4 步推错 -1"
   → 学生学到"严谨的解题方法"
```

> **ORM** = Outcome Reward Model（只评最终答案）
> **PRM** = Process Reward Model（评每一步）
>
> 对于 **Agentic RL 的长程任务**，PRM 是解决 [信用分配](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/05-Sparse-Reward%E4%B8%8E%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D.md) 问题的关键武器。

---

## 📜 2.5.1 论文与定位

| 字段 | 内容 |
|---|---|
| 标题 | Let's Verify Step by Step |
| 作者 / 机构 | Hunter Lightman et al., OpenAI |
| 年份 | 2023 |
| 状态 | arXiv |
| 链接 | https://arxiv.org/abs/2305.20050 |
| 贡献 | 证明 PRM > ORM 在数学推理上；发布 **PRM800K** 数据集（80 万步骤级标注） |
| 配套 repo | https://github.com/openai/prm800k |

---

## 📐 2.5.2 PRM vs ORM 定义

### ORM (Outcome Reward Model) — 数学形式化

$$\text{ORM}_\phi: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}, \quad (x, y) \mapsto r$$

- **输入**：(问题 $x$, **完整**回答 $y$)
- **输出**：**单个**标量 $r$
- **训练**：偏好对 $(x, y_w, y_l)$ + Bradley-Terry loss（同 [§2.1 RM 训练](01-RLHF%E4%B8%89%E9%98%B6%E6%AE%B5%EF%BC%88SFT-RM-PPO%EF%BC%89.md)）

```python
def orm(question, full_response) -> float:
    # 只看最终答案对不对
    return 1.0 if extract_answer(full_response) == gold else 0.0
```

### PRM (Process Reward Model) — 数学形式化

$$\text{PRM}_\phi: \mathcal{X} \times \mathcal{Y}^{*} \to \mathbb{R}, \quad (x, y_{1:t}) \mapsto r_t$$

- **输入**：(问题 $x$, 到第 $t$ 步**为止**的步骤序列 $y_{1:t}$)
- **输出**：第 $t$ 步的**过程分数** $r_t \in [0, 1]$（通常是"这步对的概率"）
- **训练**：每步标注 $(x, y_{1:t}, \text{label}_t \in \{+, -, \text{neutral}\})$ + 二/三分类 loss

```python
def prm(question, steps) -> list[float]:
    # 每一步独立打分
    rewards = []
    for t in range(1, len(steps)+1):
        rewards.append(prm_model.score(question, steps[:t]))  # 注意是前缀
    return rewards
```

**注意**：PRM 输入是**前缀**而非"单步"，因为判定"这步对不对"通常依赖前面的步骤上下文。

### 整段轨迹的 ORM/PRM 总分（用于 reranking）

| 量 | 公式 | 含义 |
|---|---|---|
| ORM 总分 | $S_{\text{ORM}}(x, y) = \text{ORM}_\phi(x, y)$ | 直接给整段 |
| PRM 求和 | $S_{\text{sum}}(x, y) = \sum_{t=1}^{T} \text{PRM}_\phi(x, y_{1:t})$ | 加和所有步骤分 |
| PRM 求积 | $S_{\text{prod}}(x, y) = \prod_{t=1}^{T} \text{PRM}_\phi(x, y_{1:t})$ | 联合概率（任一步错则全错） |
| PRM 最小 | $S_{\text{min}}(x, y) = \min_t \text{PRM}_\phi(x, y_{1:t})$ | "最弱链节"——OpenAI Let's Verify 用 |
| PRM 均值 | $S_{\text{mean}}(x, y) = \frac{1}{T}\sum_t \text{PRM}_\phi(x, y_{1:t})$ | 平均质量 |

> 💡 **聚合方式选择**：
> - `min` 最严（一步错就 0）——适合"长链推理任一步错就崩"的任务
> - `prod` 概率正确（多步独立）——适合数学证明
> - `sum/mean` 容错（一步错不致命）——适合代码 agent（有重试）

### 数据形态对比

| | ORM 数据 | PRM 数据 |
|---|---|---|
| 单条 | (q, y, label) | (q, step_1..k, [label_1, ..., label_k]) |
| 标注成本 | 低（看答案） | **高**（看每一步） |
| 数据量 | 少 → OK | 必须多（PRM800K = 80 万步） |

---

## 🆚 2.5.3 ORM vs PRM 对比

| 维度 | ORM | PRM |
|---|---|---|
| 标注成本 | 低 | **高**（每一步都要标） |
| 信号密度 | **稀疏**（每条 1 个 reward） | **密集**（每步 1 个 reward） |
| Credit assignment | 难 | **容易** |
| 训练稳定性 | 中 | 高 |
| 易受 reward hacking | 容易"凑答案" | "中间步骤合理"才能拿分 |
| Test-time selection | 难 | **容易**（每步剪枝） |
| 用作 RL reward | 配合 GRPO 可用 | **更适合 PPO process reward** |

---

## 🧪 2.5.4 论文核心实验

Let's Verify Step by Step 的关键发现：

| 方法 | MATH 数据集准确率 |
|---|---|
| 基线 (majority voting) | 50.8% |
| ORM-based rerank | 71.4% |
| **PRM-based rerank** | **78.2%** ⭐ |

→ **PRM 显著好于 ORM**，尤其在长链推理任务上。

---

## 🎯 2.5.5 PRM 在 Agentic RL 中的应用

### A. Test-time reranking（Best-of-N selection）

**核心公式**：sample N 个候选 → 选 reranker 打分最高的：

$$y^* = \arg\max_{y_i \in \{y_1, \dots, y_N\}} S(x, y_i)$$

其中 $S$ 可以是 $S_{\text{ORM}}$、$S_{\min}^{\text{PRM}}$、$S_{\text{prod}}^{\text{PRM}}$ 等聚合（见 §2.5.2）。

**伪代码**：

```python
def best_of_n(prompt, policy, scorer, N=16):
    candidates = [policy.sample(prompt) for _ in range(N)]
    scores = [scorer.score(prompt, y) for y in candidates]
    return candidates[argmax(scores)]
```

**OpenAI Let's Verify 论文的关键结果**：

| N | Majority voting | ORM-rerank | **PRM-rerank (min)** |
|---|---|---|---|
| 16 | ~57% | ~68% | **~73%** |
| 256 | ~64% | ~73% | **~78%** |
| 1860 | ~69% | ~76% | **~82%** |

→ **N 越大，PRM 优势越显著**（因为更细的过程信号能从更多候选里挑出真正好的）。

**重要**：这是 "**inference-time scaling**"——**不更新模型参数**，靠多 sample + 好 rerank 提升。
→ DeepSeek-R1、o1 / o3 系列的 test-time 计算扩展都用类似思想。

### B. Training-time process reward

```
agent rollout trajectory
   ↓
PRM 给每一步 r_t
   ↓
直接用 r_t 做 PPO 的步骤级 reward
   ↓
缓解 sparse reward
```

→ 这是 "PRM-based RL"，**更新模型**。

### C. Hybrid: dense PRM + sparse outcome

```
final_reward_t = α · prm_score_t + β · outcome_at_end
```

→ 最常用的实战配方。

---

## 🧬 2.5.6 PRM 在 Code Agent 中的样子

```
trajectory: search → read → edit → run_tests → final

PRM 给每步打分:
   search "normalize_path"     → 0.8（搜对了）
   read "src/paths.py"          → 0.7（读对了文件）
   edit "rstrip('/')"           → 0.3（修改方向不对）
   run_tests                    → 0.6（合理的下一步）
   final                        → 0.4（提交时机太早，还有 fail）

→ 给训练信号: edit 步骤是关键问题
```

### PRM 标注的难点

| 难点 | 说明 |
|---|---|
| **步骤粒度** | 一步 = 一个 tool call？一段思考？ |
| **标注人需要会代码** | 不像看数学答案那么直观 |
| **不同 task 步骤数不同** | 标注规范化难 |
| **标注成本爆炸** | trajectory 30 步，每步都要标 |

→ 因此 Code Agent 实际更多用 **自动 PRM**（执行反馈 / verifier 打分），而不是人工 PRM。
→ 这就是 SWE-Gym 的方向（[06-强推+可选论文索引（SWE-Gym-SWE-RL等）](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/06-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95%EF%BC%88SWE-Gym-SWE-RL%E7%AD%89%EF%BC%89.md)）。

---

## ⚠️ 2.5.7 PRM 的局限

| 局限 | 说明 |
|---|---|
| **标注成本高** | PRM800K 用了几百万美元（OpenAI 内部） |
| **步骤粒度敏感** | 粒度太细→人标不动；太粗→信号弱 |
| **PRM 本身会被攻击** | agent 学会"看起来过程对，结果错"的输出 |
| **OOD 泛化差** | PRM 训练时见过的 step 类型有限 |
| **不适合所有任务** | 创作 / 翻译等没有"步骤对错"的任务 PRM 没意义 |

---

## 🆚 2.5.8 ORM vs PRM 怎么选

| 场景 | 推荐 |
|---|---|
| 数学推理、长链 reasoning | **PRM** |
| 代码生成（单函数） | **ORM** 就够（pytest 是天然 outcome） |
| 代码 agent 多步修 bug | **混合**：ORM 主 + 自动 PRM 辅 |
| 创作 / 对话 | **ORM**（步骤无意义） |
| Toy 项目入门 | **ORM**（PRM 标注太贵） |

---

## 📌 2.5 节要点

| 概念 | 一句话 |
|---|---|
| **ORM** | 只评最终结果（**简单 + 稀疏**） |
| **PRM** | 评每一步（**信号密 + 标注贵**） |
| PRM 解决什么？ | Agentic RL 的 credit assignment 问题 |
| 论文代表？ | Let's Verify Step by Step / PRM800K |
| Code Agent 实践？ | 主用 ORM (pytest)，辅用自动 PRM（执行反馈打分） |
| 警告 | PRM 本身**也会被 reward hacking** |

---

## 🔗 延伸阅读

- 下一节：[06-五大算法对比表](06-%E4%BA%94%E5%A4%A7%E7%AE%97%E6%B3%95%E5%AF%B9%E6%AF%94%E8%A1%A8.md)
- 论文卡：[02-必学论文12篇精读卡](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md) #6 (Let's Verify Step by Step)
- 信用分配前置：[05-Sparse-Reward与信用分配](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/05-Sparse-Reward%E4%B8%8E%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D.md)
- Code Agent 应用：[04-RL如何用在Code-Agent上](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md)
- 原始资料：`02-papers.md` §"Let's Verify Step by Step" 行，`05-glossary.md` §ORM/PRM

---

⬅ [04-RLVR（可验证奖励）](04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md) | ➡ [06-五大算法对比表](06-%E4%BA%94%E5%A4%A7%E7%AE%97%E6%B3%95%E5%AF%B9%E6%AF%94%E8%A1%A8.md)
