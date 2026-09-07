---
tags: [agentic-rl, 第4章, code-agent, RLVR, execution-feedback, reward-design]
chapter: 4
section: 4.4
source: "agentic-rl-learning-map/code-agentic-rl/05-rl-for-code-agents.md (全文)"
---

# 4.4 RL 如何用在 Code Agent 上

⬅ [03-Observation-Action-Reward定义](03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md) | ➡ [05-必读论文10篇精读卡（SWE-bench线）](05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md)

---

## 🎬 故事比喻：代码任务为什么是 RLVR 的"主场"

```
传统 RLHF:
"哪个回答好？"  → 请人标 chosen/rejected → 训 RM → PPO
   贵 / 慢 / 主观 / 容易被骗

Code RLVR:
"这个 patch 对吗？" → pytest 跑一下 → 0/1 → 直接当 reward
   便宜 / 快 / 客观 / 不需要人
```

> 代码是少数几个**"答案有客观验证器"**的领域（数学、SQL、定理证明也是）。
> 这就是为什么 SWE-RL、CodeRL、SWE-Gym 全在搞 RLVR。

**但**——

> 「pytest 通过 ≠ 真正修好」。
> agent 可以**改测试**让 pytest 过；可以**hard-code** hidden pattern；可以**让测试 skip**。
> 这就是 [reward hacking](08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)，是本章的另一半内容。

---

## ✅ 4.4.1 用测试结果作为 reward 是否可行？

原始资料 §05-rl-for-code-agents.md 开篇明确：

> **可行，而且代码任务是最适合 RLVR 的领域之一。**

原因：

| 优势 | 说明 |
|---|---|
| 代码可执行 | 直接拿到 0/1 反馈 |
| 测试 / 编译 / type check / linter 都能给信号 | 反馈多样且自动化 |
| SWE-bench 有明确 issue / repo / patch / 测试 | 评测环境标准化 |
| CI 结果接近真实开发流程的成功标准 | 训练目标 ≈ 真实价值 |

**但 reward 不是真理**（资料原文）：

| 风险 | 说明 |
|---|---|
| 测试覆盖可能不足 | 通过测试 ≠ 没 bug |
| 隐藏测试可能不稳定 | flaky tests 干扰 reward |
| Agent 可能只修测试、不修需求 | reward hacking |
| Patch 可能破坏未覆盖功能 | hidden regression |
| 环境依赖 / 时间 / 网络 / 随机性 | reward 噪声 |

→ **设计 reward 时必须同时设计"防 hack 机制"**，详见 [08-Reward-Hacking与Sandbox安全](08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)。

---

## 🔄 4.4.2 Execution Feedback → 训练信号 的 8 种转换

原始资料 §"Execution feedback 如何转化为训练信号" 的核心表。

一次 coding agent episode 写成：

```
issue → observe repo → search → read file → edit patch → run tests
       → observe failure → edit again → final patch → reward
```

每个环节都能转成训练信号：

| 来源 | 可形成的信号 | 怎么用 |
|---|---|---|
| 最终测试全过 | **outcome reward** | PPO/GRPO 的 episode return |
| 编译 / 导入成功 | **dense reward** | 给中间步骤的 shaping reward |
| 部分测试 fail → pass | **shaped reward** | 阶段性奖励 |
| 新增 regression | **negative reward** | 惩罚 PASS_TO_PASS 失败 |
| traceback 被正确归因 | **process reward** | 给定位/分析步骤打分 |
| Patch 被 verifier 认为合理 | **verifier reward** | learned verifier 输出 |
| 多个候选 patch 测试结果不同 | **preference pair** | DPO 数据 |
| 成功轨迹 | **SFT / behavior cloning** | 复制成功 trajectory |

> 💡 **关键洞察**：同一次 rollout，可以**同时**抽出**多种**信号，喂给**多种**训练算法。
> 这就是 SWE-Gym、SWE-RL 数据 pipeline 的设计精髓。

---

## 🏆 4.4.3 5 种 reward 类型（本节最重要的表）

直接保留原始资料 §"Reward 类型" 表格：

| 类型 | 定义 | 优点 | 风险 | 适合阶段 |
|---|---|---|---|---|
| **Pass/fail reward** | 最终所有目标测试过给 1，否则 0 | 简单、客观、贴近 benchmark | **极稀疏，训练效率低** | 入门、评测 |
| **Dense reward** | 编译通过、部分测试通过、错误数减少等给中间分 | 学习信号更密 | reward shaping **可能偏离真实目标** | toy project、研究 |
| **Process reward** | 给定位、命令选择、错误归因、patch 步骤打分 | 缓解 credit assignment | 标注难，自动评分**可能不可靠** | 进阶研究 |
| **Verifier reward** | 训练 verifier 判断 patch / trajectory 是否正确 | 可用于训练、rerank、test-time scaling | verifier **会被利用**，可能学到 benchmark 偏差 | 研究 |
| **Rule reward** | 用规则或相似度，如 patch 与 gold diff 接近 | 便宜、稳定 | **会惩罚等价正确解** | 数据筛选 / 辅助 |

### 5 种 reward 的精确数学形式

记任务 $\tau = (x, \text{repo}, T_{\text{F2P}}, T_{\text{P2P}})$，trajectory $y = (a_0, \dots, a_T)$，最终 patch 应用后的 repo 状态为 $\text{repo}_T$。

#### V1. Pass/fail reward（outcome-only）

$$R_{\text{pass/fail}}(\tau, y) = \mathbb{1}\Big[ \big(\bigwedge_{t \in T_{\text{F2P}}} \text{pytest}(t, \text{repo}_T) = \text{pass}\big) \land \big(\bigwedge_{t \in T_{\text{P2P}}} \text{pytest}(t, \text{repo}_T) = \text{pass}\big) \Big]$$

**人话**：所有 FAIL_TO_PASS 测试通过 **且** 所有 PASS_TO_PASS 测试不破 → 1，否则 0。

#### V2. Dense reward（shaped）

$$R_{\text{dense}}(\tau, y) = \underbrace{R_{\text{pass/fail}}}_{\text{终局基础分}} + \sum_{k} w_k \cdot \phi_k(\tau, y)$$

每个 $\phi_k$ 是一个**子信号**（**安全的话应满足 [§1.5 potential-based shaping](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/05-Sparse-Reward%E4%B8%8E%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D.md)**）：

| $k$ | $\phi_k$ 含义 | 推荐 $w_k$ |
|---|---|---|
| 1 | $\mathbb{1}[\text{import OK}]$ | +0.2 |
| 2 | $\frac{|\text{F2P pass}|}{|T_{\text{F2P}}|}$（部分通过率） | +0.2 |
| 3 | $|\text{P2P fail}|$（regression 计数） | **-0.2** |
| 4 | $|\text{invalid actions}|$ | -0.1 |
| 5 | $\mathbb{1}[\text{edited test files}]$ | **-0.5**（防 hack） |

#### V3. Process reward (PRM-style)

$$R_{\text{process}}(\tau, y) = \sum_{t=0}^{T} \gamma^t \cdot \text{PRM}_\phi(\tau, a_{0:t})$$

每步给一个 PRM 分数（详见 [§2.5 PRM](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/05-PRM%E4%B8%8EORM%EF%BC%88%E8%BF%87%E7%A8%8Bvs%E7%BB%93%E6%9E%9C%E5%A5%96%E5%8A%B1%EF%BC%89.md)）：

| $t$ | $a_t$ | PRM 分 |
|---|---|---|
| 0 | `search("normalize_path")` | 0.8（定位对） |
| 1 | `read("src/paths.py")` | 0.7 |
| 2 | `edit(...)` | 0.3（修改方向错） |
| ... | ... | ... |

#### V4. Verifier reward (learned)

训练一个 verifier $V_\phi$（如 LEVER 论文）：

$$R_{\text{verifier}}(\tau, y) = V_\phi(\tau, y) \in [0, 1]$$

$V_\phi$ 的训练数据：$(\tau, y, \mathbb{1}[\text{真正修对}])$ 对，标准二分类 loss。

**Best-of-N 用法**：sample N 个 patch，选 $V_\phi$ 分最高的（同 [§2.5.5](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/05-PRM%E4%B8%8EORM%EF%BC%88%E8%BF%87%E7%A8%8Bvs%E7%BB%93%E6%9E%9C%E5%A5%96%E5%8A%B1%EF%BC%89.md) Best-of-N 公式）。

#### V5. Rule reward (similarity-based)

$$R_{\text{rule}}(\tau, y) = \text{sim}\big(\text{diff}(y), \text{gold\_diff}(\tau)\big) \in [0, 1]$$

其中 `sim` 可以是：

- **代码 BLEU**：n-gram 重叠
- **AST edit distance**：抽象语法树编辑距离
- **diff line overlap**：行级 Jaccard

**问题**：等价但写法不同的 patch 会被惩罚（如 `if x: return 1 else: return 2` vs `return 1 if x else 2`）。
**用途**：通常作**数据筛选辅助**而非主 reward。

### 三个版本 reward 示例（Toy Bug-Fix Gym 实战）

**V1 Binary**（最简单）：

```python
def reward_binary(test_result) -> float:
    return 1.0 if test_result.all_passed else 0.0
```

**V2 Dense**（最常用）：

```python
def reward_dense(test_before, test_after, action_log) -> float:
    r = 0.0
    if test_after.all_passed: r += 1.0
    if test_after.import_ok:  r += 0.2
    r += 0.2 * (test_after.passed - test_before.passed)
    r -= 0.2 * test_after.regression_count  # 破坏旧测试
    r -= 0.1 * action_log.invalid_action_count
    r -= 0.5 * action_log.edited_test_file  # 改测试文件！
    return r
```

**V3 Verifier-rerank**：

```python
def rerank_patches(candidates, verifier) -> Patch:
    # 采样多个 patch, 用 verifier 选最好
    scored = [(verifier.score(p), p) for p in candidates]
    return max(scored, key=lambda x: x[0])[1]
```

→ 每版的 **reward hacking 风险**详见 [08-Reward-Hacking与Sandbox安全](08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)。

---

## 🧪 4.4.4 RLVR 在代码任务中的具体用法

原始资料 §"RLVR 在代码任务中的作用"。

**RLVR 核心**：不用人类偏好，用**可验证信号**训练。

代码任务的 verifier 候选：

```
✓ pytest                ✓ 编译器        ✓ type checker
✓ linter (ruff/pylint)  ✓ 静态分析      ✓ fuzz tests
✓ hidden tests          ✓ benchmark harness    ✓ learned verifier
```

**典型 RLVR 用法（4 步）**：

```
1. 对同一 issue 采样多个 patch (例如 16 个)
2. 运行测试 / verifier
3. 将 pass/fail 转成 reward 或 preference pair
4. 用 PPO / GRPO / RLVR / DPO / SFT 训练
   - 训策略（agent 本身）
   - 训 reranker（选最好的 patch）
   - 训 verifier（判断 patch 对不对）
```

> 💡 这 4 步就是 **DeepSeek-R1**、**SWE-RL** 的工作流。
> 它和 [GRPO](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md) 配合极好——多个 patch 形成一个 group，组内相对奖励。

---

## 🌟 4.4.5 为什么 coding agent 特别适合 Agentic RL

资料原表：

| 特性 | 对 RL 的意义 |
|---|---|
| **可执行** | 能自动获得**客观**反馈 |
| **多步** | 适合 **trajectory-level learning** |
| **反馈丰富** | traceback / test logs / lint errors 都是 observation |
| **任务真实** | issue resolution 比单函数生成更接近**实际价值** |
| **可复现** | Docker + benchmark harness 让 evaluation 更稳定 |
| **数据来源多** | GitHub issue / PR / CI / tests / synthetic tasks |

> 这就是为什么 2024-2025 学术热点从"reasoning RL"快速扩散到"code RL"——SWE-RL、SWE-Gym、SWE-smith 几乎同时出来。

---

## ⚠️ 4.4.6 主要风险（4 类，必看）

来自资料 §"主要风险"。每一类都在 [08-Reward-Hacking与Sandbox安全](08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) 详展开，这里给速查：

### A. Reward Hacking

模型可能：
- 修改测试而不是修代码
- hard-code hidden pattern
- 删除失败路径
- 让测试 skip
- 输出符合 parser 的假 patch

**防护**：锁定测试文件、hidden tests、检查 regression、静态分析 patch、人工抽查。

### B. Benchmark Overfitting

风险：训练数据污染 SWE-bench；模型记住已知 issue；对 harness 投机。

**防护**：Verified / Live / 时间切分；报告 cost / pass@k；OOD repo 评测。

### C. 环境不稳定

风险：Docker build 失败；依赖版本漂移；测试 flaky；网络 / 时间依赖。

**防护**：固定镜像；记录环境 hash；重试 flaky；环境失败 vs 模型失败分开统计。

### D. Sandbox 安全

风险：agent 可执行任意命令；repo 可能有恶意脚本；rollout 并行执行大量未知代码。

**防护**：Docker / 容器隔离；禁网 / 白名单；文件权限；超时 / 配额 / 日志审计。

→ **完整版**：[08-Reward-Hacking与Sandbox安全](08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)

---

## 🆚 4.4.7 Inference-time loop vs Training-time RL

原始资料 §结尾的关键对照表（值得反复看）：

| 维度 | Inference-time loop | Training-time RL |
|---|---|---|
| 参数是否更新 | ❌ 否 | ✅ 是（或更新 verifier/reranker） |
| 数据来源 | 当前任务轨迹 | 大量任务轨迹 |
| 目标 | 单次任务成功 | 未来任务成功率提升 |
| 成本 | 推理成本 | 推理 + 训练 + 环境成本 |
| 代表 | **Aider / SWE-agent / OpenHands 常规用法** | **CodeRL / SWE-Gym / SWE-RL** |

> 💡 **入门顺序（资料原文，强烈推荐）**：
>
> 1. 先跑 inference-time loop，**理解失败模式**
> 2. 再保存 trajectory
> 3. 最后把 trajectory 转成 SFT / DPO / RLVR 数据

---

## 📌 4.4 节要点

| 问题 | 一句话答案 |
|---|---|
| Code Agent 为什么适合 RLVR？ | 测试 / 编译是天然 verifier，0/1 反馈客观、便宜、可批量 |
| Execution feedback 怎么用？ | 同一 rollout 可同时抽出 outcome / dense / process / preference 多种信号 |
| 5 种 reward 类型？ | pass/fail、dense、process、verifier、rule |
| Toy 项目用哪种？ | 入门用 V1 binary，加 V2 dense 看训练效果，研究做 V3 verifier rerank |
| 最大风险？ | reward hacking（agent 改测试不改代码） |
| 入门顺序？ | inference loop → 保存 trajectory → SFT/DPO → RLVR |

---

## 🔗 延伸阅读

- 下一节：[05-必读论文10篇精读卡（SWE-bench线）](05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md)
- 风险细节：[08-Reward-Hacking与Sandbox安全](08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)
- 算法前置：[04-RLVR（可验证奖励）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md)
- Toy 实操：[02-Toy-Bug-Fix-Agent-Gym（核心）](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)
- 训练落地：[04-RL](../../Code-Agent-Knowledge-Base/03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/04-RL.md)
- 长轨迹稳定性：[02-GRPO与长轨迹稳定性](../../Code-Agent-Knowledge-Base/05-Reward%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7/02-GRPO%E4%B8%8E%E9%95%BF%E8%BD%A8%E8%BF%B9%E7%A8%B3%E5%AE%9A%E6%80%A7.md)
- 跨链 GRPO 实战：[04-GRPO训练实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)
- 原始资料：`code-agentic-rl/05-rl-for-code-agents.md`

---

⬅ [03-Observation-Action-Reward定义](03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md) | ➡ [05-必读论文10篇精读卡（SWE-bench线）](05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md)
