---
tags: [agentic-rl, 第6章, 路径, 通用, 最短路径]
chapter: 6
section: 6.3
source: "agentic-rl-learning-map/06-shortest-path.md (全文)"
---

# 6.3 通用 Agentic-RL 最短路径（不只 Code）

⬅ [02-Code-Agentic-RL-6周路线](02-Code-Agentic-RL-6%E5%91%A8%E8%B7%AF%E7%BA%BF.md) | ➡ [04-阅读检查清单（自测题）](04-%E9%98%85%E8%AF%BB%E6%A3%80%E6%9F%A5%E6%B8%85%E5%8D%95%EF%BC%88%E8%87%AA%E6%B5%8B%E9%A2%98%EF%BC%89.md)

---

## 🎬 故事比喻：广覆盖版的 14 天

如果你**不只**做 Code，还想兼顾 Web Agent / Tool Agent / Reasoning Agent，
**这条路径是通用版**：10 篇论文涵盖整个 Agentic RL 方向。

> 来自原始资料 `06-shortest-path.md`。
>
> ⚠️ **你目标明确 = Code Agentic RL**，更应该走 [6.1](01-Code-Agentic-RL-14%E5%A4%A9%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84.md)。
> 本节只在你**需要更宽视野**时回查。

---

## 📚 6.3.1 通用 10 篇论文（资料原文）

| 顺序 | 论文 | 读到什么程度 |
|---|---|---|
| 1 | ReAct | 读完整；重点看 thought/action/observation 格式 |
| 2 | PPO | 摘要 + 算法 + 目标函数直觉；推导可后补 |
| 3 | InstructGPT | 读完整；重点 SFT/RM/PPO 流水线 |
| 4 | DPO | 摘要 + 方法直觉 + 实验；先不深究数学 |
| 5 | Let's Verify Step by Step | 读完整；重点 PRM vs ORM |
| 6 | DeepSeekMath | 读 GRPO 和 RL 部分 |
| 7 | DeepSeek-R1 | 读训练流程、R1-Zero、RLVR、蒸馏 |
| 8 | Toolformer | 工具调用数据构造 + filtering |
| 9 | AgentBench | benchmark 任务设计 + 评价指标 |
| 10 | Agent Lightning | 系统架构 + 轨迹接口 + credit assignment |

→ 论文卡：[02-必学论文12篇精读卡](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md)

---

## 🐙 6.3.2 通用 5 个 repo

| 顺序 | Repo | 用法 |
|---|---|---|
| 1 | [CleanRL](https://github.com/vwxyzjn/cleanrl) | 单文件 PPO 补 RL loop |
| 2 | [TRL](https://github.com/huggingface/trl) | 跑小模型 PPO/DPO/GRPO 示例 |
| 3 | [AgentBench](https://github.com/THUDM/AgentBench) | 理解 agent benchmark + 环境 |
| 4 | [agent-lightning](https://github.com/microsoft/agent-lightning) | toy tool-use agent 接 RL |
| 5 | [verl](https://github.com/verl-project/verl) | 阅读生产级 RLVR/GRPO recipe |

→ 详细：[04-Repo地图与运行成本](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/04-Repo%E5%9C%B0%E5%9B%BE%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%88%90%E6%9C%AC.md)

---

## 🎯 6.3.3 极简版（只 5 篇 + 2 repo）

如果只能选 5 篇：

```
1. ReAct
2. InstructGPT
3. PPO
4. DeepSeek-R1
5. Agent Lightning
```

如果只能选 2 个 repo：

```
1. TRL
2. agent-lightning
```

---

## 📅 6.3.4 通用 14 天安排（资料原文）

| 天数 | 任务 |
|---|---|
| Day 1-2 | **ReAct** + **AgentBench**，建立 agent trajectory 和 benchmark 概念 |
| Day 3-4 | MDP/PPO 最小基础，读 PPO + CleanRL |
| Day 5-6 | **InstructGPT** + **TRL**，理解 RLHF |
| Day 7 | **DPO**，理解 preference optimization 轻量路线 |
| Day 8-9 | **DeepSeekMath** + **DeepSeek-R1**，理解 GRPO/RLVR |
| Day 10 | PRM/ORM，读 **Let's Verify Step by Step** |
| Day 11 | **Toolformer**，理解工具调用训练 |
| Day 12 | **Agent Lightning**，理解 agent 轨迹训练接口 |
| Day 13-14 | 做 toy project 设计 + baseline evaluation |

---

## 🆚 6.3.5 通用版 vs Code 版（怎么选）

| 维度 | 通用 14 天（6.3） | Code 14 天（6.1） |
|---|---|---|
| 覆盖任务 | Web / Tool / Math / 多领域 | **只 Code** |
| 论文重叠度 | InstructGPT / PPO / DPO / GRPO / RLVR / PRM | 几乎不重叠 |
| toy 项目 | Calculator / SQL / Code 任选 | **Bug-Fix Gym** |
| 适合 | 还在选方向 / 调研 | **目标明确 = Code** |

→ **你目标已定 = Code**，**强烈优先** [01-Code-Agentic-RL-14天最短路径](01-Code-Agentic-RL-14%E5%A4%A9%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84.md)。
→ 通用版只在你需要**写调研报告**时回看。

---

## 🎯 6.3.6 两条路径如何"互补"

如果你 4 周以上可投入，**可以混合两条**：

```
Week 1 (通用基础):
  - ReAct
  - PPO
  - InstructGPT
  - DPO
  - DeepSeek-R1

Week 2 (Code 入门):
  - SWE-bench
  - SWE-bench Verified
  - SWE-agent
  - Aider 跑一题

Week 3 (Code 深入):
  - Agentless
  - Self-Debugging
  - CodeT
  - CodeRL

Week 4 (动手):
  - Toy Bug-Fix Gym pilot
  - 3 版 reward 设计
```

→ 这就是混合 6.3 通用前置 + 6.1 Code 直击的版本。

---

## 📌 6.3 节要点

| 项 | 内容 |
|---|---|
| 论文 | 10 篇（极简 5 篇） |
| Repo | 5 个（极简 2 个） |
| 总时间 | 14 天 |
| 适合 | 还在选方向 / 通用调研 |
| 你的选择 | **直接走 6.1 Code 路径** |

---

## 🔗 延伸阅读

- 上一节：[02-Code-Agentic-RL-6周路线](02-Code-Agentic-RL-6%E5%91%A8%E8%B7%AF%E7%BA%BF.md)
- 下一节：[04-阅读检查清单（自测题）](04-%E9%98%85%E8%AF%BB%E6%A3%80%E6%9F%A5%E6%B8%85%E5%8D%95%EF%BC%88%E8%87%AA%E6%B5%8B%E9%A2%98%EF%BC%89.md)
- Code 直击路径：[01-Code-Agentic-RL-14天最短路径](01-Code-Agentic-RL-14%E5%A4%A9%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84.md)
- 通用论文卡：[02-必学论文12篇精读卡](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md)
- 原始资料：`06-shortest-path.md`

---

⬅ [02-Code-Agentic-RL-6周路线](02-Code-Agentic-RL-6%E5%91%A8%E8%B7%AF%E7%BA%BF.md) | ➡ [04-阅读检查清单（自测题）](04-%E9%98%85%E8%AF%BB%E6%A3%80%E6%9F%A5%E6%B8%85%E5%8D%95%EF%BC%88%E8%87%AA%E6%B5%8B%E9%A2%98%EF%BC%89.md)
