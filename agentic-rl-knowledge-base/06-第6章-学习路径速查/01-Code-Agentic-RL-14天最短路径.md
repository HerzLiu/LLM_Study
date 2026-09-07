---
tags: [agentic-rl, 第6章, 路径, 14天, 最短路径, code-agentic-rl]
chapter: 6
section: 6.1
source: "agentic-rl-learning-map/code-agentic-rl/07-shortest-path.md"
---

# 6.1 Code Agentic RL · 14 天最短路径

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-Code-Agentic-RL-6周路线](02-Code-Agentic-RL-6%E5%91%A8%E8%B7%AF%E7%BA%BF.md)

---

## 🎬 故事比喻：两周从零进入 Code Agentic RL

```
Day 1-3: 建立任务概念（SWE-bench 是啥）
Day 4-5: 暖手（Aider 修一个 bug）
Day 6-7: 经典 scaffold（SWE-agent）
Day 8: 强 baseline（Agentless）
Day 9-10: 执行反馈 / 验证（Self-Debugging + CodeT）
Day 11-12: 训练入门（CodeRL + SWE-Gym）
Day 13-14: Toy Gym 5 题 pilot
```

> **这条路径来自原始资料 `code-agentic-rl/07-shortest-path.md`**。
> 它**故意精简**到 10 篇论文 + 5 个 repo + 3 个 RL 概念 + 1 个项目。
> 适合**目标明确 = Code Agentic RL** 的人，2 周内速成。

---

## 📚 6.1.1 10 篇以内论文（精选清单）

| 顺序 | 论文 | 为什么必读 |
|---|---|---|
| 1 | SWE-bench | **定义真实代码任务的评测范式** |
| 2 | SWE-bench Verified | 理解可靠测试 reward 和 benchmark 噪声 |
| 3 | SWE-agent | coding agent 基本 loop + ACI |
| 4 | OpenHands | 完整软件开发 agent 平台形态 |
| 5 | Agentless | 强 baseline，**不迷信复杂 agent** |
| 6 | AutoCodeRover | repo 结构检索 + fault localization |
| 7 | Teaching LLMs to Self-Debug | 执行反馈 + 错误恢复 |
| 8 | CodeT | generated tests + execution-based selection |
| 9 | CodeRL | RL for code generation 入门 |
| 10 | SWE-Gym | 从 benchmark 走向 training environment |

→ 完整精读卡见 [05-必读论文10篇精读卡（SWE-bench线）](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md)

### 如果只能读 5 篇（资料原文）

```
1. SWE-bench
2. SWE-agent
3. Agentless
4. Self-Debugging
5. SWE-Gym
```

---

## 🐙 6.1.2 5 个以内 GitHub repo

| 顺序 | Repo | 用法 |
|---|---|---|
| 1 | [Aider](https://github.com/Aider-AI/aider) | **最低成本**体验 coding agent loop |
| 2 | [SWE-bench](https://github.com/princeton-nlp/SWE-bench) | 学 benchmark harness 和测试 reward |
| 3 | [SWE-agent](https://github.com/SWE-agent/SWE-agent) | 跑真实 issue resolution agent |
| 4 | [OpenHands](https://github.com/All-Hands-AI/OpenHands) | 完整软件开发 agent 平台 |
| 5 | [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | 进入训练 / verifier / RL 研究 |

→ 详细 + 实操命令见 [07-Repo地图（SWE-agent-OpenHands等）](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/07-Repo%E5%9C%B0%E5%9B%BE%EF%BC%88SWE-agent-OpenHands%E7%AD%89%EF%BC%89.md)

---

## 🧠 6.1.3 必须补的 3 个 RL 概念（资料原文）

> 即使时间紧，**也必须**搞懂这 3 个：

1. **Trajectory / return**：一次 issue 修复过程如何变成 episode，最终测试结果如何变成 return
   → [01-MDP与轨迹（用Agent语言讲RL）](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/01-MDP%E4%B8%8E%E8%BD%A8%E8%BF%B9%EF%BC%88%E7%94%A8Agent%E8%AF%AD%E8%A8%80%E8%AE%B2RL%EF%BC%89.md)

2. **Sparse reward / credit assignment**：最终测试失败时，如何判断是哪一步导致失败
   → [05-Sparse-Reward与信用分配](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/05-Sparse-Reward%E4%B8%8E%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D.md)

3. **RLVR / verifier reward**：测试、执行、静态检查、learned verifier 如何转成训练信号
   → [04-RLVR（可验证奖励）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md)

> **3 个概念 = 第 1 章的 1.1 + 1.5 + 第 2 章的 2.4 共 3 节**。
> 加上必读 5 篇论文，2 天能搞定。

---

## 🎯 6.1.4 1 个建议复现项目

**Toy Bug-Fix Agent Gym**（资料原文）：

- 准备 30 个小 Python bug
- 每个 bug 有 failing pytest
- Agent 可以 read 文件 / search / edit / run pytest
- **Reward**：
  - `+1.0` 全测通过
  - `+0.2` 代码能 import/compile
  - `+0.2` 目标失败测试减少
  - `-0.1` 无效命令或无效 patch
  - `-0.2` 破坏已有 passing tests
- 先跑 prompt loop baseline
- 保存 `trajectories.jsonl`
- 再做 verifier rerank 或 DPO preference data

→ **完整规格**：[02-Toy-Bug-Fix-Agent-Gym（核心）](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)

---

## 📅 6.1.5 最短 14 天安排（资料原文）

| 天数 | 任务 |
|---|---|
| Day 1-2 | 读 **SWE-bench** + **SWE-bench Verified** |
| Day 3 | 跑/阅读 **SWE-bench** 一个 task 的 harness |
| Day 4-5 | 跑 **Aider** 修一个小 pytest bug |
| Day 6-7 | 读 **SWE-agent**，跑一个 SWE-bench Lite 单题或读官方 demo |
| Day 8 | 读 **Agentless**，手工做 localization |
| Day 9 | 读 **Self-Debugging**，整理执行反馈模板 |
| Day 10 | 读 **CodeT**，理解 generated tests/rerank |
| Day 11 | 读 **CodeRL**，补 trajectory / reward / RLVR |
| Day 12 | 读 **SWE-Gym**，理解 training environment |
| Day 13-14 | 写 **Toy Bug-Fix Agent Gym** 规格并跑 **3 个样例** |

---

## 🚀 6.1.6 14 天后你应该有

| 产出物 | 说明 |
|---|---|
| 5 篇论文笔记 | 每篇一句话定位 + 一个图 |
| 1 个 Aider bug-fix demo | 完整 trajectory + diff |
| 1 个 SWE-bench / SWE-agent 单题运行 log | 体验 evaluation harness |
| 1 份 Toy Bug-Fix Gym 设计稿 | 5 个 task 的 metadata + reward 函数 |
| 3 条 baseline trajectory | trajectories.jsonl |
| 失败模式初稿 | 至少 3 种 |

→ **你已经入门 Code Agentic RL**，可以正式 6 周深入。

---

## ⚠️ 6.1.7 14 天里**不要**做的事

| 不要 | 原因 |
|---|---|
| 试图跑 SWE-bench 全量 | Docker 慢，单题就够学 |
| 试图复现 SWE-RL 训练 | 算力要求高，**只读 reward / data pipeline** |
| 一上来读 PPO 推导 | 第 1 章故事比喻版 + 1.4 PPO 一节够用 |
| 想精通 GRPO 数学 | 看 [2.3](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md) 直觉即可 |
| 试图同时学 Web/APP/Code agent | Code 一个方向已经够大 |

---

## 📌 6.1 节要点

| 项 | 内容 |
|---|---|
| 论文 | 10 篇（紧缺时 5 篇） |
| Repo | 5 个 |
| RL 概念 | 3 个（trajectory / sparse / RLVR） |
| 项目 | 1 个（Toy Bug-Fix Gym 5 题 pilot） |
| 总时间 | 14 天 |
| 产出 | 论文笔记 + Aider demo + SWE-bench 单题 log + Gym 设计稿 + baseline trajectory |

---

## 🔗 延伸阅读

- 下一节：[02-Code-Agentic-RL-6周路线](02-Code-Agentic-RL-6%E5%91%A8%E8%B7%AF%E7%BA%BF.md)
- Code 核心：[00-章节总览](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 必读论文卡：[05-必读论文10篇精读卡（SWE-bench线）](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md)
- Toy 项目：[02-Toy-Bug-Fix-Agent-Gym（核心）](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)
- 自测题：[04-阅读检查清单（自测题）](04-%E9%98%85%E8%AF%BB%E6%A3%80%E6%9F%A5%E6%B8%85%E5%8D%95%EF%BC%88%E8%87%AA%E6%B5%8B%E9%A2%98%EF%BC%89.md)
- 原始资料：`code-agentic-rl/07-shortest-path.md`

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-Code-Agentic-RL-6周路线](02-Code-Agentic-RL-6%E5%91%A8%E8%B7%AF%E7%BA%BF.md)
