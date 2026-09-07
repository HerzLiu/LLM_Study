---
tags: [agentic-rl, 第6章, 路径, 6周, code-agentic-rl, roadmap]
chapter: 6
section: 6.2
source: "agentic-rl-learning-map/code-agentic-rl/06-roadmap.md (全文)"
---

# 6.2 Code Agentic RL · 6 周学习路线

⬅ [01-Code-Agentic-RL-14天最短路径](01-Code-Agentic-RL-14%E5%A4%A9%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84.md) | ➡ [03-通用Agentic-RL最短路径](03-%E9%80%9A%E7%94%A8Agentic-RL%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84.md)

---

## 🎬 故事比喻：从读 SWE-bench 到设计 RL 实验

```
W1: Code Agent + SWE-bench 是啥
W2: 跑通一个 coding agent scaffold
W3: Repo localization + patch pipeline
W4: 执行反馈、测试反馈、self-repair
W5: RL / RLVR for code
W6: 走到研究入口（SWE-Gym / SWE-RL）
+W7-8: 扩展（跑 SWE-bench Lite + 训练）
```

> **这条路径来自原始资料 `code-agentic-rl/06-roadmap.md`**。
> 比 [6.1 14 天路径](01-Code-Agentic-RL-14%E5%A4%A9%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84.md) 更深，**每周有产出物**。
> 适合**真正系统进入** Code Agentic RL 方向、**4-8 周可投入**的人。

---

## 📅 6.2.0 路线总览

| 周 | 主题 | 必读论文 | 推荐 repo | 产出物 |
|---|---|---|---|---|
| W1 | Code Agent + SWE-bench 基础 | SWE-bench / SWE-bench Verified | SWE-bench | 3 个 SWE-bench task 手工拆解 |
| W2 | 跑通 Code Agent Scaffold | SWE-agent / OpenHands | Aider / SWE-agent / OpenHands | Aider 修一个 pytest bug + log |
| W3 | Repo Localization + Patch Pipeline | Agentless / AutoCodeRover | Agentless / AutoCodeRover | 3 个 bug 的 suspected_files.json |
| W4 | 执行反馈 + 测试反馈 + Self-Repair | Self-Debugging / CodeT / LEVER | CodeT / LEVER / SWE-ReX | 20 个 Python bug 的 trajectory |
| W5 | RL / RLVR for Code | CodeRL / SWE-Gym | CodeRL / SWE-Gym / TRL / verl | 3 版 reward 设计 + reward hacking 分析 |
| W6 | Code Agentic RL 研究入口 | SWE-RL / SWE-smith / R2E-Gym | SWE-RL / SWE-smith / SWE-Gym | 2 页实验设计 |
| (W7) | 跑 SWE-bench Lite | - | SWE-agent / OpenHands / Agentless | 比较 4 种方法的失败模式 |
| (W8) | SFT/DPO/verifier 训练 | - | TRL / verl / Agent Lightning | 在 Toy Gym trajectory 上跑训练 |

---

## 📖 6.2.1 各周详细安排

### 第 1 周：Code Agent + SWE-bench 基础

**学习目标**：

- 理解 Code Agent ≠ 单段代码生成 = **repo-level issue resolution**
- 理解 SWE-bench 的 task / patch / tests / resolved rate
- 掌握 FAIL_TO_PASS / PASS_TO_PASS / Verified

**必读**：

- SWE-bench
- SWE-bench Verified

**推荐 repo**：

- SWE-bench

**补的概念**：

- GitHub issue → patch → test validation
- Docker evaluation harness
- resolved rate
- hidden tests / regression tests

**小练习**：

```
手工阅读 3 个 SWE-bench task：
  - issue 描述
  - 相关文件
  - gold patch
  - 测试

写出每个 task 的 "reward 从哪里来"
```

→ 本库对应：[01-Code-Agent定义与边界](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/01-Code-Agent%E5%AE%9A%E4%B9%89%E4%B8%8E%E8%BE%B9%E7%95%8C.md) + [论文卡 1-2](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md)

---

### 第 2 周：跑通 Code Agent Scaffold

**学习目标**：

- 理解 Agent-Computer Interface (ACI)
- 跑通一个最小 coding agent loop
- 观察模型如何搜索 / 编辑 / 跑测试 / 失败恢复

**必读**：

- SWE-agent
- OpenHands

**推荐 repo**：

- Aider（最低成本）
- SWE-agent
- OpenHands

**补的概念**：

- Tool schema
- Shell command as action
- Observation from test logs
- Sandbox

**小练习**：

```
用 Aider 或 SWE-agent 在一个小 Python repo 上修复一个 failing pytest
保存完整过程：命令 / diff / 测试输出 / 最终结果
```

→ 本库对应：[03-Observation-Action-Reward定义](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md) + [07-Repo地图（SWE-agent-OpenHands等）](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/07-Repo%E5%9C%B0%E5%9B%BE%EF%BC%88SWE-agent-OpenHands%E7%AD%89%EF%BC%89.md)

---

### 第 3 周：Repo Localization + Patch Pipeline

**学习目标**：

- 理解为什么很多 SWE 任务的关键是**找对文件和函数**
- 掌握 localization → repair → rerank 的 pipeline
- 建立**强 baseline 观念**：复杂 agent loop 不是唯一解

**必读**：

- Agentless
- AutoCodeRover

**推荐 repo**：

- Agentless
- AutoCodeRover

**补的概念**：

- Fault localization
- Retrieval over code
- AST / symbol search
- Patch reranking

**小练习**：

```
对 3 个 bug 手工输出 suspected_files.json:
  - 文件
  - 理由
  - 置信度

固定 suspected files，让模型只生成 patch，对比成功率
```

→ 本库对应：[能力栈模块 1-3](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md)

---

### 第 4 周：执行反馈 + 测试反馈 + Self-Repair

**学习目标**：

- 学会把 traceback / failed assertion / lint error 转成下一步行动
- 理解 generated tests + verifier rerank
- 形成 "execution feedback → 训练信号" 的直觉

**必读**：

- Teaching LLMs to Self-Debug
- CodeT
- LEVER

**推荐 repo**：

- CodeT
- LEVER
- swe-rex（sandbox）

**补的概念**：

- Execution feedback
- Test-time scaling
- Verifier
- Preference from pass/fail candidates

**小练习**：

```
构造 20 个小 Python bug，每个 bug 有 failing test
对每个 bug 记录一条 trajectory:
  - initial fail
  - patch
  - rerun
  - final reward
```

→ 本库对应：[能力栈模块 4 + 7](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md) + [04-RL如何用在Code-Agent上](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md)

---

### 第 5 周：RL / RLVR for Code

**学习目标**：

- 理解 CodeRL 如何把 functional correctness 用作 RL 信号
- 理解 pass/fail / dense / process / verifier 4 类 reward
- 理解为什么 code agent 的 reward 既天然又危险

**必读**：

- CodeRL
- SWE-Gym

**推荐 repo**：

- CodeRL
- SWE-Gym
- TRL 或 verl

**补的概念**：

- Trajectory / Return
- Sparse reward
- Credit assignment
- RLVR

**小练习**：

```
为 toy bug-fix 环境设计三版 reward:
  - binary pass/fail
  - dense reward
  - verifier/rerank reward

写出每版可能的 reward hacking
```

→ 本库对应：[04-RL如何用在Code-Agent上](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md) + [08-Reward-Hacking与Sandbox安全](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)

---

### 第 6 周：Code Agentic RL 研究入口

**学习目标**：

- 理解 SWE-Gym / SWE-RL / SWE-smith 如何从 eval 走向训练
- 设计一个小型 Code Agentic RL 实验
- 明确数据 / 环境 / 奖励 / baseline / 评估指标

**必读**：

- SWE-RL
- SWE-smith
- R2E-Gym

**推荐 repo**：

- SWE-RL
- SWE-smith
- SWE-Gym

**补的概念**：

- Rollout collection
- Verifier training
- Synthetic SWE task generation
- Benchmark overfitting
- Sandbox safety

**小练习**：

```
写一份 2 页实验设计：
  - 任务集
  - agent
  - 工具
  - reward
  - baseline
  - 评估指标
  - 风险控制
```

→ 本库对应：[06-强推+可选论文索引（SWE-Gym-SWE-RL等）](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/06-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95%EF%BC%88SWE-Gym-SWE-RL%E7%AD%89%EF%BC%89.md) + [02-Toy-Bug-Fix-Agent-Gym（核心）](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)

---

## 🚀 6.2.2 可扩展到 8 周

### 第 7 周：跑 SWE-bench Lite

```
跑 SWE-bench Lite 或 Verified 的小批评估
比较 Aider / SWE-agent / OpenHands / Agentless 的失败模式
```

### 第 8 周：训练入门

```
用 toy bug-fix trajectories 做 SFT/DPO/verifier rerank
条件允许再接 TRL / verl / Agent Lightning 做小规模 RLVR 实验
```

→ 跨链训练实战：[Hello-Agents GRPO 实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)

---

## 🆚 6.2.3 6 周路线 vs 14 天最短路径

| 维度 | 14 天最短 | 6 周完整 |
|---|---|---|
| 论文数 | 10 | 13-15 + 通用论文 |
| Repo 跑通数 | 1-2 个 | 4-5 个 |
| Toy 项目 | 5 题 pilot | 30 题完整 + baseline 对比 |
| 训练接入 | ❌ 不接 | ✅ 可接 SFT/DPO（W8） |
| 输出物 | 论文笔记 + 简单 demo | 完整实验设计稿 + 训练 baseline |
| 适合 | 时间紧 | 系统学 |

---

## 📌 6.2 节要点

| 周 | 关键产出物 |
|---|---|
| W1 | 3 个 SWE-bench task 手工拆解 |
| W2 | Aider 修一个 bug 的完整 log |
| W3 | 3 个 bug 的 suspected_files.json |
| W4 | 20 个 Python bug 的 trajectory |
| W5 | 3 版 reward 设计 + hack 分析 |
| W6 | 2 页实验设计 |
| (W7) | 4 种方法在 SWE-bench Lite 上的对比 |
| (W8) | 在 Toy Gym 数据上跑通 SFT/DPO |

**核心原则**：每周必须有**可挂在 GitHub / 简历**的产出物。

---

## 🔗 延伸阅读

- 上一节：[01-Code-Agentic-RL-14天最短路径](01-Code-Agentic-RL-14%E5%A4%A9%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84.md)
- 下一节：[03-通用Agentic-RL最短路径](03-%E9%80%9A%E7%94%A8Agentic-RL%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84.md)
- Code 核心：[00-章节总览](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 训练落地路线：[Code-Agent落地路线](../../Code-Agent-Knowledge-Base/00-MOC/Code-Agent%E8%90%BD%E5%9C%B0%E8%B7%AF%E7%BA%BF.md)
- 通用 6 周路线：[05-6周通用路线](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/05-6%E5%91%A8%E9%80%9A%E7%94%A8%E8%B7%AF%E7%BA%BF.md)
- 自测题：[04-阅读检查清单（自测题）](04-%E9%98%85%E8%AF%BB%E6%A3%80%E6%9F%A5%E6%B8%85%E5%8D%95%EF%BC%88%E8%87%AA%E6%B5%8B%E9%A2%98%EF%BC%89.md)
- 原始资料：`code-agentic-rl/06-roadmap.md`

---

⬅ [01-Code-Agentic-RL-14天最短路径](01-Code-Agentic-RL-14%E5%A4%A9%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84.md) | ➡ [03-通用Agentic-RL最短路径](03-%E9%80%9A%E7%94%A8Agentic-RL%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84.md)
