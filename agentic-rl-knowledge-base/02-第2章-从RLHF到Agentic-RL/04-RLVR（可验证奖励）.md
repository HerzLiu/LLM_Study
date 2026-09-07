---
tags: [agentic-rl, 第2章, RLVR, verifiable-reward, DeepSeek-R1, 可验证奖励]
chapter: 2
section: 2.4
source: "agentic-rl-learning-map/02-papers.md §DeepSeek-R1, 05-glossary.md §RLVR"
---

# 2.4 RLVR（可验证奖励，Agentic RL 的支柱）

⬅ [03-GRPO（组内相对优势）](03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md) | ➡ [05-PRM与ORM（过程vs结果奖励）](05-PRM%E4%B8%8EORM%EF%BC%88%E8%BF%87%E7%A8%8Bvs%E7%BB%93%E6%9E%9C%E5%A5%96%E5%8A%B1%EF%BC%89.md)

---

## 🎬 故事比喻：考试有标准答案 vs 作文比赛

```
RLHF (RM 路线):
   类似"作文比赛"
   → 文章好不好是主观的
   → 必须请评委（RM）
   → 评委有偏见、训练贵、容易被骗

RLVR (可验证 reward 路线):
   类似"考数学"
   → 答案对错有客观标准（数值/单测/证明）
   → 不需要评委
   → 便宜、客观、可批量
```

> **RLVR = Reinforcement Learning with Verifiable Rewards**
> 不用人类偏好，**用可验证信号**（答案、单测、执行、形式系统）训模型。
>
> **DeepSeek-R1** 是这条路线最有代表性的论文：
> 用"数学题答案对不对"作 reward，**完全跳过 RM**，训出长链推理。
>
> Code Agentic RL **几乎 100% 走 RLVR 路线**——pytest 通过率天生是 verifiable reward。

---

## 📜 2.4.1 论文与定位

### DeepSeek-R1（RLVR + GRPO 代表作）

| 字段 | 内容 |
|---|---|
| 标题 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning |
| 作者 / 机构 | DeepSeek-AI |
| 年份 | 2025 |
| 状态 | arXiv（**Nature 2025 版本已发表**） |
| 链接 | https://arxiv.org/abs/2501.12948 / https://www.nature.com/articles/s41586-025-09422-z |
| 贡献 | 证明 RLVR + GRPO 能让 LLM **自发涌现长链推理 / 反思 / 自验证**行为（"啊哈时刻"） |

---

## 🎯 2.4.2 RLVR 核心定义

资料 `05-glossary.md` 原文：

> **RLVR = Reinforcement Learning with Verifiable Rewards**，使用**可验证奖励**，如数学答案、代码测试、工具执行成功。

### 数学形式化：RLVR reward 函数

记 $V: \mathcal{X} \times \mathcal{Y} \to \{0, 1\}$ 为**verifier**（验证器），$\text{gold}(x)$ 为标准答案，则：

$$R_{\text{RLVR}}(x, y) = V(x, y) = \mathbb{1}\big[\, y \text{ 通过 } x \text{ 的所有验证规则}\,\big]$$

**逐项解读**：

| 符号 | 含义 | 例子 |
|---|---|---|
| $x$ | 任务输入（prompt / issue / question） | "修复 normalize\_path 的 bug" |
| $y$ | 模型输出（response / patch / answer） | 一段 diff |
| $V(x, y)$ | 验证器函数，**确定性 + 自动可计算** | `pytest -q` 是否全过 |
| $\mathbb{1}[\cdot]$ | 指示函数（true=1, false=0） | - |

**和 RLHF reward 的精确对比**：

| 量 | RLHF | RLVR |
|---|---|---|
| Reward 函数 | $r_\phi(x, y)$（**学出来的** RM，连续标量） | $V(x, y)$（**程序定义**的 verifier，常 0/1） |
| 参数 | $\phi$（需要训练） | **无可学参数** |
| 数据需求 | 几万-几十万**人工标注**偏好对 | $(x, \text{gold})$ 对（**自动构造**） |
| 鲁棒性 | RM 可能有偏 | verifier 写错就全错（但写对就稳） |

**RLVR 适用条件（缺一不可）**：

1. 存在**自动**判定 $V(x, y)$ 的方式（不需人）
2. $V$ 在**多次跑**时结果**一致**（避免 flaky）
3. **过 $V$** 强相关于 "**真正完成任务**"（避免 reward hacking）

→ 第 3 条最难——这就是 [reward hacking](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) 反复出现的根因。

### 可验证 reward 的特征

| 特征 | 说明 |
|---|---|
| **客观** | 不依赖人类主观偏好 |
| **自动** | 程序可计算（无需人工） |
| **稳定** | 多次跑结果一致（小心 flaky） |
| **明确** | 通常 0/1 二值（也可连续） |
| **便宜** | 跑一次几毫秒-几秒（对比标注几分钟） |

### 典型可验证 reward 来源

| 任务 | Verifier |
|---|---|
| 数学题 | 答案数值比对（如 GSM8K） |
| 代码生成 | pytest / 编译器 / type checker |
| SQL | execution 结果与 gold 一致 |
| 工具调用 | API 返回成功 + 结果正确 |
| 定理证明 | Lean / Coq 自动检查 |
| 翻译 | BLEU / 回译一致性（弱 verifier） |
| 检索 | top-k 命中（弱 verifier） |

---

## 🧠 2.4.3 RLVR 为什么改变了游戏规则

### 经济性

```
RLHF:
   1 万条偏好对 → 每条 $1-5 标注 → $10k-$50k
   + RM 训练时间

RLVR:
   1 万道数学题（已有 gold answer）→ $0
   verifier 跑 1 次几毫秒 → 几乎免费
```

### 客观性

```
RLHF:
   "回答A 比 B 好"
   → 不同标注员可能不一致
   → RM 学到的是某群人的偏好（有偏见）

RLVR:
   "答案 = 42 ✓ / ≠ 42 ✗"
   → 谁来跑都一样
```

### 可扩展性

```
RLHF:
   想加 10 倍数据 → 标注成本 10 倍

RLVR:
   想加 10 倍数据 → verifier 跑 10 倍时间（线性，无人工）
```

→ 这就是为什么 RLVR 成为 **2024-2025 RL 最主流方向**。

---

## 🆚 2.4.4 RLVR vs RLHF vs DPO

| 维度 | RLHF | DPO | RLVR |
|---|---|---|---|
| Reward 来源 | RM (人标) | 偏好对 (人标) | **客观验证器** |
| 标注成本 | 高 | 中 | **几乎 0** |
| 在线 rollout | ✅ | ❌ | ✅ |
| 适合任务 | 通用对话 | alignment / 偏好 | **数学 / 代码 / 工具 / 推理** |
| 主要算法 | PPO | DPO | **GRPO / PPO + verifier reward** |
| Reward hacking | RM 偏差 | 数据偏差 | **验证器漏洞** |

---

## 🧪 2.4.5 RLVR + GRPO 是 Agentic RL 黄金组合

为什么这两个**配对最佳**？

| 维度 | 单 GRPO | 单 RLVR | GRPO + RLVR |
|---|---|---|---|
| Reward 信号 | 任意 | 客观 | **客观** |
| Critic 需求 | 不要 | 要（PPO 时） | **不要** |
| Advantage 计算 | 组内相对 | GAE | **组内相对** |
| 训练成本 | 中 | 高 | **低** |

→ **DeepSeek-R1**、**SWE-RL**、**open-r1** 都是这个组合。
→ **Code Agentic RL 的标准配方**。

### GRPO + RLVR 完整联合 loss

把 [§2.3](03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md) 的 GRPO loss 和上面的 RLVR reward **合体**：

$$L^{\text{GRPO+RLVR}}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, \{y_i\}_{i=1}^K \sim \pi_{\theta_{old}}(\cdot \mid x)}\!\Bigg[\frac{1}{K}\sum_{i=1}^K \frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\!\min\Big(\rho_{i,t}\hat{A}_i,\, \text{clip}(\rho_{i,t}, 1\!-\!\epsilon, 1\!+\!\epsilon)\hat{A}_i\Big)\Bigg] - \beta\,\text{KL}(\pi_\theta\|\pi_{ref})$$

其中：

$$\hat{A}_i = \frac{V(x, y_i) - \mu_g}{\sigma_g}, \quad \mu_g = \frac{1}{K}\sum_{j=1}^K V(x, y_j), \quad \sigma_g = \sqrt{\frac{1}{K}\sum_j (V(x, y_j) - \mu_g)^2 + \epsilon}$$

**关键变化**：advantage 里的 $R_i$ **直接换成 verifier 输出** $V(x, y_i)$，**整个 pipeline 不需要 RM**。

**和原 RLHF-PPO 的对比表**：

| 组件 | RLHF-PPO | GRPO+RLVR |
|---|---|---|
| Reward 来源 | $r_\phi(x, y)$（RM） | $V(x, y)$（verifier） |
| RM 训练 | ✅ 需要（前置阶段） | ❌ 不要 |
| Critic / V head | ✅ 需要 | ❌ 不要 |
| Advantage | GAE from V | 组内 $(R_i - \mu_g) / \sigma_g$ |
| Rollout | 1 / prompt | **K / prompt** |
| KL penalty | 通常 reward shaping | loss penalty |

→ **核心观察**：RLVR 去掉了"训 RM"，GRPO 去掉了"训 critic"。
→ **训练 pipeline 从 3 个模型简化为 2 个**（policy + ref）。
→ 这就是 DeepSeek-R1 训练 pipeline "**简洁性**" 的根源。

### 数值小例子（GRPO+RLVR 在 Code 任务）

设 $\beta = 0.01$, $\epsilon = 0.2$, K = 4。一个 SWE-bench-like task：

```
sample 4 个 patch:
  y_1: 全测过 → V(x, y_1) = 1
  y_2: 编译失败 → V(x, y_2) = 0
  y_3: 测试 fail → V(x, y_3) = 0
  y_4: 全测过 → V(x, y_4) = 1
```

| $i$ | $V$ | $\hat{A}_i = (V - 0.5)/0.5$ | 训练效果 |
|---|---|---|---|
| 1 | 1 | **+1.0** | 提高 $y_1$ 所有 token 概率 |
| 2 | 0 | **-1.0** | 压低 $y_2$ 所有 token 概率 |
| 3 | 0 | **-1.0** | 同上 |
| 4 | 1 | **+1.0** | 同 $y_1$ |

→ 不需要 RM，不需要 critic，**pytest 一跑就完事**。
→ 完整 Python 实战见 [Hello-Agents 11.4](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)。

---

## 🧬 2.4.6 DeepSeek-R1 训练流程速览

DeepSeek-R1 的训练 pipeline 简化版（**含 R1-Zero 和最终 R1**）：

```
R1-Zero（纯 RLVR，从 base 模型开始）:
   ┌────────────────────────────┐
   │ Base model (no SFT)         │
   └────────────────────────────┘
              │
              ▼
   GRPO + 数学/代码答案 verifier
              │
              ▼
   涌现长链推理 + 反思 + 自验证 ✨
   (但语言混乱，中英混杂)

R1（加 SFT 冷启动 + 多轮 RLVR）:
   ┌────────────────────────────┐
   │ Base model                  │
   └────────────────────────────┘
              │
              ▼
   1. SFT 冷启动（少量高质量长链推理数据）
              │
              ▼
   2. RLVR (推理任务) → 提升推理能力
              │
              ▼
   3. SFT (推理 + 通用任务混合)
              │
              ▼
   4. RLVR + 偏好 (推理 + 安全 + 通用)
              │
              ▼
   最终 R1（推理强 + 语言流畅）
```

→ "RLVR + GRPO 涌现长链推理"是 2025 最重要的发现之一。

---

## 🧱 2.4.7 Code Agentic RL 中的 RLVR 实战

### 简单形式：pass/fail reward

```python
def code_rlvr_reward(patch, task) -> float:
    test_result = pytest_run(apply_patch(task.repo, patch), task.target_tests)
    if test_result.all_pass and not test_result.regression:
        return 1.0
    return 0.0
```

### 进阶形式：dense + 防 hack

```python
def code_rlvr_reward_v2(patch, task) -> float:
    # 防 hack
    if patch.modifies_test_files: return -1.0
    if "@pytest.mark.skip" in patch.added_lines: return -1.0
    if patch.has_excessive_branching(task.test_inputs): return -0.5

    test_result = pytest_run(apply_patch(task.repo, patch), task.target_tests)

    # Dense
    r = 0.0
    if test_result.compile_ok: r += 0.1
    if test_result.fail_to_pass_resolved > 0: r += 0.4
    if test_result.fail_to_pass_resolved == len(task.target_tests): r += 0.4
    if test_result.regression == 0: r += 0.1
    return r
```

→ 见 [4.4.3 三版 reward 示例](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md#-443-5-%E7%A7%8D-reward-%E7%B1%BB%E5%9E%8B%E6%9C%AC%E8%8A%82%E6%9C%80%E9%87%8D%E8%A6%81%E7%9A%84%E8%A1%A8)。

---

## ⚠️ 2.4.8 RLVR 的风险

| 风险 | 说明 | 防护 |
|---|---|---|
| **验证器有漏洞** | 测试覆盖不全，patch 能过测试但实际错 | hidden tests + regression tests |
| **Reward hacking** | agent 学会修测试、hard-code、skip | 静态检查 + 锁定测试文件 |
| **OOD 表现差** | 训练时 verifier 准，OOD 上 verifier 失效 | 多 verifier 交叉验证 |
| **Verifier shift** | 不同 verifier 给不同信号，模型学到投机 | 用 stable verifier 集合 |
| **任务难度不均** | 一组全 pass 或全 fail → 无信号 | 任务难度分层采样 |

→ 详见 [08-Reward-Hacking与Sandbox安全](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)

---

## 📌 2.4 节要点

| 问题 | 一句话答案 |
|---|---|
| RLVR 是什么？ | 用**客观验证器**（不是 RM）做 reward 的 RL |
| 为什么火？ | 便宜 + 客观 + 可扩展，不需要人标 |
| 适合什么任务？ | **数学 / 代码 / 工具 / 推理 / 形式系统** |
| 黄金组合？ | **RLVR + GRPO** |
| 代表作？ | DeepSeek-R1（reasoning）/ SWE-RL（code） |
| Code Agent 怎么用？ | pytest 通过率作 reward，配 GRPO 训练 |

---

## 🔗 延伸阅读

- 下一节：[05-PRM与ORM（过程vs结果奖励）](05-PRM%E4%B8%8EORM%EF%BC%88%E8%BF%87%E7%A8%8Bvs%E7%BB%93%E6%9E%9C%E5%A5%96%E5%8A%B1%EF%BC%89.md)
- 论文卡：[02-必学论文12篇精读卡](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md) #5 (DeepSeek-R1)
- 应用：[04-RL如何用在Code-Agent上](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md)
- 风险：[08-Reward-Hacking与Sandbox安全](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)
- 跨链：[Hello-Agents 11.2 奖励函数设计](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/02-%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8E%E5%A5%96%E5%8A%B1%E5%87%BD%E6%95%B0.md)
- 原始资料：`02-papers.md` §DeepSeek-R1 行、`05-glossary.md` §RLVR

---

⬅ [03-GRPO（组内相对优势）](03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md) | ➡ [05-PRM与ORM（过程vs结果奖励）](05-PRM%E4%B8%8EORM%EF%BC%88%E8%BF%87%E7%A8%8Bvs%E7%BB%93%E6%9E%9C%E5%A5%96%E5%8A%B1%EF%BC%89.md)
