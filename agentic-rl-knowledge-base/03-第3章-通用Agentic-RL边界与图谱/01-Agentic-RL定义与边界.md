---
tags: [agentic-rl, 第3章, 定义, 边界, MARL, workflow, tool-use, LLM-Agent]
chapter: 3
section: 3.1
source: "agentic-rl-learning-map/01-boundaries.md (全文)"
---

# 3.1 Agentic RL 定义与边界

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-必学论文12篇精读卡](02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md)

---

## 🎬 故事比喻：什么是 Agentic RL，什么不是？

```
你做了下列哪些事，才算 Agentic RL？

[A] 用 ReAct + LangChain 拼一个 agent           → ❌ workflow agent
[B] 用 GPT-4 多步调工具修 issue                  → ❌ inference-time loop
[C] 收集 trajectory，再 SFT 一遍                  → ⚠️ 准 RL（无策略梯度）
[D] 用偏好对训 DPO                                → ⚠️ 偏好优化（非多轮）
[E] 用 GRPO + pytest reward 训 agent              → ✅ Agentic RL
[F] 多个 LLM agent 共享环境，所有 agent 都在学    → ✅ MARL（Agentic RL 子集）
```

> **关键判定**：
> 1. **是否有奖励优化训练**（不只是 prompting / SFT）？
> 2. **是否多轮环境交互**？
>
> 两个都 ✓ 才算 Agentic RL。

---

## 📐 3.1.1 紧凑定义（资料原文）

来自原始资料 §`01-boundaries.md`：

> **Agentic RL** 指用强化学习训练 LLM/AI Agent 在多轮环境中做决策。

更紧凑：

> **Agentic RL = LLM Agent + 多轮环境交互 + 强化学习式奖励优化 + 长程信用分配。**

### 动作类型不止生成 token

资料原文列出 Agentic RL 的"动作"：

- 自然语言回复
- 工具调用和参数选择
- 代码执行
- 网页、终端、数据库、APP 操作
- 检索、计划更新、记忆写入
- 子任务分配和多 agent 协作

### 奖励类型不止人类偏好

资料原文列出 Agentic RL 的"奖励来源"：

- 人类偏好或打分
- AI judge / RLAIF
- reward model
- process reward model
- outcome reward model
- 数学答案校验
- **单元测试和代码执行结果**  ← Code Agentic RL 主要来源
- 环境任务成功率
- 用户模拟器反馈
- self-play 或 verifier 产生的信号

---

## 🆚 3.1.2 与 11 个相邻概念的关系（完整对比）

直接保留原始资料 §"与相邻概念的关系"完整表格（**这是本节最重要的一张表**）：

| 概念 | 核心区别 | 与 Agentic RL 的关系 |
|---|---|---|
| **传统 RL** | 在 MDP/POMDP 中学策略，最大化长期回报；状态/动作通常更结构化 | 提供**数学语言和算法底座**（policy gradient、actor-critic、PPO） |
| **RLHF** | 人类偏好 → RM → PPO 优化 LLM；常用于单轮回答质量 | 是 LLM RL 的**入口**，但不一定有多轮环境交互 |
| **RLAIF** | 用 AI 反馈替代或减少人类反馈（Constitutional AI） | 可作为 Agentic RL 的**奖励来源或评价器** |
| **DPO** | 偏好对直接优化策略，不显式训 RM，不在线 rollout | 是 RLHF 的**轻量替代**；通常不是完整 Agentic RL，但常作为 post-training baseline |
| **PPO for LLM** | PPO 优化 LLM，加 KL 约束防偏离 | 很多 RLHF 和 agent training 系统的**标准起点** |
| **GRPO** | 同 prompt 多采样，组内相对奖励估 advantage，省 critic | 当前 reasoning/RLVR 系统的**高频算法** |
| **RLVR** | Reinforcement Learning with Verifiable Rewards | Agentic RL 的**重要支柱**，尤其工具/代码/数学/检索 |
| **LLM Agent** | 能规划、调用工具、用记忆、执行 workflow 的系统形态 | 不一定经过 RL；**Agentic RL 是训练这类系统的方法** |
| **Tool-use Agent** | 学会选工具、构造参数、读观察结果 | **Tool-use RL 是 Agentic RL 的核心子方向** |
| **Reasoning Agent** | 长链推理、搜索、反思、自检 | 可用 RLVR、PRM/ORM、verifier 或 self-improvement 训练 |
| **Multi-agent RL** | 多个 agent/策略同时学习，可能合作或竞争 | LLM multi-agent workflow **不自动等于 MARL**；只有多策略学习和交互回报时才接近 MARL |

---

## 🎯 3.1.3 Agentic RL 关心的 9 个问题

资料 §"这个方向解决什么问题" 原文：

| # | 问题 | 直觉 |
|---|---|---|
| 1 | **长期任务** | 最终成功可能要几十步到上百步行动 |
| 2 | **工具调用** | 何时调、选哪个、参数如何构造 |
| 3 | **环境交互** | action 改变 environment，env 返回 observation |
| 4 | **规划与重规划** | 失败后调整策略，不是一次性回答 |
| 5 | **反馈稀疏** | 只有最终 success/fail，缺中间监督 |
| 6 | **自动评估** | 用 verifier / unit test / execution / AI judge 扩大训练信号 |
| 7 | **多轮决策** | 每步影响后续 state 和可选 action |
| 8 | **Self-improvement** | 从自生成数据、反思、self-play 或 verifier 中改进 |
| 9 | **Agent benchmark** | Web / coding / database / app / OS / tool-use 环境的可复现评测 |

→ 这 9 个问题就是 Agentic RL 的"问题清单"。**所有论文都在解其中一两个**。

---

## 📚 3.1.4 必学 RL 基础（资料原文优先级）

资料原文按"必学"优先级：

| 优先级 | 概念 | 为什么必学 |
|---|---|---|
| 必学 | MDP/POMDP、state、action、reward、trajectory、return | 描述 agent 多轮任务 |
| 必学 | policy、value function、Q/V、advantage | PPO、GRPO、actor-critic 的共同语言 |
| 必学 | policy gradient | 理解为什么能从 reward 优化生成策略 |
| 必学 | actor-critic | 理解 PPO/RLHF 的 critic、value head、advantage 估计 |
| 必学 | **PPO** | RLHF 和多种 LLM RL 框架的核心算法 |
| 必学 | **KL regularization** | LLM post-training 防策略漂移的关键 |
| 必学 | **on-policy vs off-policy** | 理解 PPO/GRPO 为什么贵、DPO 为什么轻 |
| 必学 | sparse reward、credit assignment | Agentic RL 的主要困难 |
| 必学 | **reward hacking** | Agent 训练中**非常常见**的失败模式 |

→ 全部已在 [第 1 章](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) 详讲。

---

## 🚫 3.1.5 可以先跳过（资料原文）

资料明确建议**入门 Agentic RL 时跳过**：

- DQN 全家桶
- SAC、TD3、DDPG 等连续控制细节
- 复杂 Bellman 收敛证明
- 经典 MARL 博弈理论深水区
- model-based RL 深水区
- offline RL 理论
- control theory 背景

> 等你做**算法研究**或**大规模训练系统**时再补这些。

---

## 🎓 3.1.6 怎么"判断一个工作是不是 Agentic RL"

实用判定流程：

```
                 这个工作做了什么？
                       │
        ┌──────────────┴──────────────┐
        │                              │
   只是 prompting 或                有更新参数？
   workflow 编排？                       │
        │                       ┌───────┴────────┐
        ▼                       │                │
   ❌ 不是 Agentic RL          只 SFT?         有 policy gradient
   (是 LLM Agent 框架)          │             或 RL update?
                                ▼                │
                             ⚠️ 准 RL        多轮 trajectory?
                             (behavior         │
                              cloning)    ┌────┴────┐
                                          ▼          ▼
                                       否          是
                                       │           │
                                       ▼           ▼
                                  ⚠️ 单轮 RL    ✅ Agentic RL
                                  (RLHF 风格)
```

---

## 🌟 3.1.7 典型 Agentic RL 工作示例

| 工作 | 算法 | 多轮？ | 是 Agentic RL？ |
|---|---|---|---|
| InstructGPT | PPO + RM | ❌ 单轮 | ⚠️ 是 RLHF，**算 LLM RL 但不算 Agentic** |
| WebGPT | PPO + 浏览器 | ✅ | ✅ 早期 Agentic RL 雏形 |
| Toolformer | SFT | ❌ | ❌ 工具学习的 SFT，不是 RL |
| DeepSeek-R1 | GRPO + RLVR | ✅（reasoning trace） | ✅ 严格说是 reasoning RL，**算 Agentic** |
| SWE-RL | GRPO + RLVR | ✅ | ✅ Code Agentic RL 代表 |
| Agent Lightning | 任意 algo + agent trace | ✅ | ✅ **专门**为 Agentic RL 设计的训练框架 |
| ArCHer | 多轮 actor-critic | ✅ | ✅ |
| AutoGen / LangGraph | workflow | ✅ 但无 RL | ❌ Agent 框架，**不是 Agentic RL** |
| Reflexion | verbal memory（无参数更新） | ✅ | ⚠️ 资料叫 "verbal RL"，**非梯度 RL** |

---

## ⚠️ 3.1.8 4 个常见混淆

| 混淆 | 真相 |
|---|---|
| "用了 agent 框架就是 Agentic RL" | 错。必须有**奖励优化训练** |
| "RLHF 就是 Agentic RL" | 错。RLHF 通常单轮 |
| "多 agent workflow 就是 MARL" | 错。MARL 要求**多策略同时学习** |
| "Self-play 就是 self-improvement" | 错。self-play **不一定带来提升**，需要有效选择 + 验证信号 |

---

## 📌 3.1 节要点

| 问题 | 一句话 |
|---|---|
| Agentic RL 定义 | LLM Agent + 多轮环境 + RL 奖励优化 + 长程信用分配 |
| 与 RLHF 区别 | RLHF 通常单轮；Agentic RL 必多轮 |
| 与 LLM Agent 区别 | LLM Agent 是系统形态；Agentic RL 是**训练它的方法** |
| 与 MARL 区别 | MARL 是 Agentic RL 子集（多策略同时学习） |
| 必补 RL 基础 | MDP、policy、PPO、KL、sparse reward、reward hacking |
| 可跳 | DQN/SAC/TD3、Bellman 证明、control theory |

---

## 🔗 延伸阅读

- 下一节：[02-必学论文12篇精读卡](02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md)
- 算法谱系：[00-章节总览](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- RL 基础：[00-章节总览](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- Code 专题：[01-Code-Agent定义与边界](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/01-Code-Agent%E5%AE%9A%E4%B9%89%E4%B8%8E%E8%BE%B9%E7%95%8C.md)
- 原始资料：`01-boundaries.md`（全文）

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-必学论文12篇精读卡](02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md)
