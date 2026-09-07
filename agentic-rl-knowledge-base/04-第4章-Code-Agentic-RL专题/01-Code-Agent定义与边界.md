---
tags: [agentic-rl, 第4章, code-agent, 定义, 边界]
chapter: 4
section: 4.1
source: "agentic-rl-learning-map/code-agentic-rl/01-boundaries.md"
---

# 4.1 Code Agent 定义与边界

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-Code-Agent能力栈10模块](02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md)

---

## 🎬 故事比喻：三层抽象，三个角色

```
┌─────────────────────────────────────────────────────────┐
│  Level 1: Code LLM（"会写函数的实习生"）                  │
│  ─────────────────────────                              │
│  你说: "写一个 quicksort"                                │
│  它给: 一段 Python 代码                                   │
│  评估: HumanEval / MBPP 通过率                            │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  Level 2: Code Agent（"能上手干活的初级开发"）             │
│  ─────────────────────────                              │
│  你说: "修复这个 GitHub issue：…"                        │
│  它做:                                                   │
│    1. 读 issue → 2. 找文件 → 3. 改代码                   │
│    4. 跑测试 → 5. 看报错 → 6. 再改                       │
│    7. 提交 patch                                         │
│  评估: SWE-bench resolved rate                           │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  Level 3: Code Agentic RL（"会成长的开发者"）              │
│  ─────────────────────────                              │
│  Level 2 的所有动作 + 把每次成败的 trajectory             │
│  收集起来，用 pytest 通过率作 reward，反过来             │
│  更新模型参数 / verifier / reranker。                    │
│  评估: 训练后的 agent 在 hold-out task 上提升多少        │
└─────────────────────────────────────────────────────────┘
```

> 本章讲的就是 Level 3。Level 1-2 是它的前提。

---

## 📐 4.1.1 紧凑定义

来自原始资料的紧凑定义：

> **Code Agent = LLM + repo context + editing tools + execution environment + iterative debug loop。**

拆开看：

| 组成 | 含义 |
|---|---|
| LLM | 策略模型 π，通常是 Code LLM（CodeLlama / DeepSeek-Coder / Qwen-Coder 等） |
| repo context | 不只是单文件，能感知目录、依赖、调用链、风格 |
| editing tools | 文件读写、字符串替换、AST 修改、apply_patch |
| execution environment | shell、pytest、编译器、lint、type checker、git、有时还有浏览器 |
| iterative debug loop | 错了能看 traceback → 定位 → 再改，而不是一次性输出 |

**完整工作流**（来自资料 §01-boundaries.md 开头）：

```
1. 读取 issue / 需求 / failing test
2. 检索 repo，理解目录、依赖、调用链、已有风格
3. 制定修改计划
4. 编辑一个或多个文件
5. 运行测试、lint、type check 或示例命令
6. 根据报错定位原因并再次修改
7. 输出最终 patch / PR 描述 / 提交说明
```

> 这 7 步任何一步崩，整个任务就崩。
> 这也是为什么 SWE-bench resolved rate 长期在 20-50% 区间——**不是模型不会写代码，是这 7 步串起来太难稳定**。

---

## 🆚 4.1.2 与相邻概念对比

原始资料 §01-boundaries.md 表格的完整版：

| 概念 | 核心区别 | 和 Code Agent 的关系 |
|---|---|---|
| **Code LLM** | 主要做生成、补全、解释代码 | 是 Code Agent 的策略模型或子模块 |
| **LLM Agent** | 泛指会规划和调用工具的 LLM 系统 | Code Agent 是**面向软件工程任务**的 LLM Agent |
| **Tool-use Agent** | 重点是选择工具和参数 | Code Agent 的工具包括 shell / 编辑器 / 搜索 / 测试 / Git / 浏览器 |
| **Software Engineering Agent** | 更强调 issue resolution、repo 修改、CI、PR、代码质量 | 与 Code Agent **高度重叠**，语义更偏软件工程全流程 |
| **Autonomous Coding Agent** | 强调少人工干预完成开发任务 | Devin、OpenHands、SWE-agent、Aider 都属这条线 |
| **Code Agentic RL** | 用 RL / RLVR / verifier / 测试反馈**训练** Code Agent | 关注从**环境 trajectory** 中学习，而不是只靠 prompt loop |

→ 简单记忆：**Code LLM 是"会写代码"；Code Agent 是"会干活"；Code Agentic RL 是"会越干越好"。**

---

## 🎓 4.1.3 5 种训练方式对比（核心表）

来自原始资料 §01-boundaries.md "Code Agentic RL 和其他训练方式的区别"。**这张表是本章最重要的一张**：

| 方法 | 训练信号 | 多轮环境交互？ | 适合解决什么 |
|---|---|---|---|
| **Code SFT** | 代码 / 补全 / patch / instruction 数据 | 通常否 | 语法、风格、常见模式、单步 patch |
| **Code RLHF** | 人类偏好或质量评价 | 不一定 | 可读性、偏好、帮助性、安全性 |
| **Code DPO** | chosen / rejected 代码或回答 | 通常否 | 偏好优化、轻量 alignment |
| **Code RLVR** | 单测、编译、执行结果、静态检查 | 可以是单步或多步 | **可验证代码正确性** |
| **Code Agentic RL** ⭐ | **完整 repo 环境 trajectory + tests / CI / verifier reward** | **是** | **多步调试、repo navigation、patch 迭代、错误恢复** |

→ 你的目标方向 = **最后一行**。
→ 前 4 行是 Code Agentic RL 的"前置或子集"，不是替代。

---

## 🎯 4.1.4 哪些能力最适合用 RL 提升？

原始资料给的清单（按"为什么适合 RL"展开）：

| 能力 | 为什么适合 RL |
|---|---|
| **多步调试** | 最终成功依赖多个中间动作，天然有 trajectory 和 credit assignment |
| **测试反馈利用** | 测试结果可转成 objective reward 或 verifier signal |
| **Patch generation** | patch 有明确 pass/fail 反馈，能做采样、rerank 和训练 |
| **Repo navigation** | 搜索路径、打开文件、定位符号都是可学习动作 |
| **Tool calling** | shell、pytest、编辑器、rg、git 等工具使用有明确成败 |
| **Planning** | 复杂 issue 需要分解；计划质量影响后续成功率 |
| **错误恢复** | traceback、lint error、failed tests 提供 rich observation |
| **代码执行反馈** | 编译 / 运行 / 测试是天然 verifier |
| **Issue resolution** | SWE-bench 这类任务有**真实软件工程目标**和**可复现评测** |

→ 这 9 项就是 [4.2 能力栈](02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md) 的 RL 视角版。

---

## ⚠️ 4.1.5 不要混淆的两件事（重要！）

原始资料 §"不要混淆的两件事"原文：

### Inference-time agent loop
> 模型**不更新参数**，只是在推理时循环执行"读文件 → 改代码 → 跑测试 → 再改"。
> **代表**：SWE-agent、OpenHands、Aider 的多数用法。

### Training-time Code Agentic RL
> 把循环轨迹保存下来，**用 reward 更新模型、verifier 或策略**。
> **代表**：SWE-Gym、SWE-RL、CodeRL。

| 维度 | Inference-time loop | Training-time RL |
|---|---|---|
| 模型参数是否变 | ❌ 不变 | ✅ 变（或 verifier/reranker 变） |
| 数据来源 | 当前任务轨迹 | 大量任务轨迹 |
| 目标 | 单次任务成功 | 未来任务成功率提升 |
| 成本 | 推理成本 | 推理 + 训练 + 环境成本 |
| 代表 | Aider / SWE-agent / OpenHands 常规用法 | CodeRL / SWE-Gym / SWE-RL |

> **入门顺序**（资料 §05-rl-for-code-agents.md 结尾原文）：
> 先跑 inference-time loop，理解失败模式；
> 再保存 trajectory；
> 最后把 trajectory 转成 SFT / DPO / RLVR 数据。

**👉 这是本库强烈推荐的学习路径，第 5 章 Toy Bug-Fix Gym 也是这么设计的。**

---

## 🧭 4.1.6 在算法谱系里的定位

```
                  本章 Code Agentic RL
                          │
            ┌─────────────┼─────────────┐
            │             │             │
       【算法骨架】   【奖励来源】   【任务形态】
            │             │             │
       GRPO / PPO    pytest 通过    issue → patch
       (第2章已学)   编译成功       repo-level 多步
                    类型检查       trajectory 训练
                    linter
                    learned verifier
                    (= RLVR)
                    (第2章已学)
```

→ Code Agentic RL **不是新算法**，是**算法 × 奖励来源 × 任务形态**的特定组合。

---

## 📌 4.1 节要点

| 问题 | 一句话答案 |
|---|---|
| Code Agent 是什么？ | 在真实软件工程环境中完成 issue → patch → test 闭环的 LLM agent |
| 它和 Code LLM 区别？ | LLM 输出代码，Agent 输出**一整套动作序列**（含执行/调试/恢复） |
| 它和 SE Agent 区别？ | 几乎同义，SE Agent 更偏全流程（含 CI/PR/质量评估） |
| 它和 Code Agentic RL 区别？ | Code Agent = 系统形态；Code Agentic RL = **训练它的方法** |
| Inference loop vs Training RL？ | 前者不动模型参数，后者动；学习应先做前者再做后者 |
| 哪些能力最适合 RL 提升？ | 多步调试 / 测试反馈 / patch / 工具调用 / 错误恢复（共 9 项） |

---

## 🔗 延伸阅读

- 下一节：[02-Code-Agent能力栈10模块](02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md)
- 算法前置：[04-RLVR（可验证奖励）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md)
- 训练落地：[01-Code-Agent与Coder模型边界](../../Code-Agent-Knowledge-Base/01-Code-Agent%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/01-Code-Agent%E4%B8%8ECoder%E6%A8%A1%E5%9E%8B%E8%BE%B9%E7%95%8C.md)
- 原始资料：`agentic-rl-learning-map/code-agentic-rl/01-boundaries.md`
- 论文锚点：SWE-bench (4.5) / SWE-agent (4.5) / CodeRL (4.5)

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-Code-Agent能力栈10模块](02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md)
