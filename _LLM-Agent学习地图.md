---
tags: [学习地图, LLM, Agent, 入口]
created: 2026-06-03
---

# 🗺️ LLM-Agent 学习地图

> 本知识库关于 **LLM 与 Agent / Code Agent** 的学习路线 + 跨库联动。

---

## 📚 核心知识库一览

| 书 | 总览 | 主题 | 章数 | 状态 |
|---|---|---|---|---|
| **Happy-LLM** | [00-总览](Happy-LLM/00-%E6%80%BB%E8%A7%88.md) | LLM 是怎么造出来的（Transformer / 预训练 / RLHF） | 7 | ✅ |
| **Hello-Agents** | [00-总览](Hello-Agents/00-%E6%80%BB%E8%A7%88.md) | 怎么用 LLM 造 Agent（理论 + HelloAgents 框架） | 12 | ✅ |
| **Claude-Code-Source** | [00-总览](Claude-Code-Source/00-%E6%80%BB%E8%A7%88.md) | 工业级 Agent 长啥样（Claude Code v2.1.88 反编译） | 8 | ✅ |
| **Agentic-RL** | [00-总览](agentic-rl-knowledge-base/00-%E6%80%BB%E8%A7%88.md) | 从 RLHF 到 Code Agentic RL（SWE-bench / reward / sandbox） | 7+附录 | ✅ |
| **Code-Agent-Knowledge-Base** | [00-总览](Code-Agent-Knowledge-Base/00-%E6%80%BB%E8%A7%88.md) | 从通用 Base 到 Coder Model 的数据、训练、RL 全流程 | 5+附录 | ✅ |

---

## 🧭 核心知识库的关系

```
       Happy-LLM          Hello-Agents         Claude-Code-Source
       ──────             ────────             ─────────────────
       如何造 LLM    →    如何造 Agent    →    工业级 Agent 长啥样
       (神经网络 +        (理论 + 框架 +        (逆向 + 实战 +
        Transformer +     范式 + Memory +       工程化护栏 +
        预训练 + RLHF)    上下文工程)          安全 + SubAgent)
              │                   │                    │
              └──────────────┬────┴──────────────┬─────┘
                             ▼                   ▼
                    Agentic-RL            Code-Agent-Knowledge-Base
                    ──────────            ─────────────────────────
                    如何用 RL 训 Agent    如何训练 Coder Model
                    (RLVR / GRPO /        (数据工程 + Pre-train +
                     SWE-bench)            Mid-train + SFT + RL)
```

**类比**：
- Happy-LLM = **大学课**：你学物理学、电学、热力学
- Hello-Agents = **专业课 + 实验课**：你用这些原理造一个发动机原型
- Claude-Code-Source = **工厂参观**：看丰田怎么把发动机做成量产车

---

## 🎯 5 条学习路径

### 路径 A：完全新手（推荐 6-8 周）

```
Happy-LLM
   第 1-4 章（理解 LLM）
       ↓
Hello-Agents
   第 1 章（Agent 入门）
   第 4 章（手撕 ReAct / Plan / Reflection）
       ↓
Claude-Code-Source
   第 1-2 章（看工业版 Agent Loop）
   第 3 章（看工业版工具系统）
       ↓
Hello-Agents
   第 7-9 章（自建框架 + 记忆 + 上下文）
       ↓
Claude-Code-Source
   第 4-7 章（看工业版怎么处理同样问题）
       ↓
Claude-Code-Source
   第 8 章（跑通 MiniCC）
```

### 路径 B：有 LLM 基础，专攻 Agent（4 周）

```
Hello-Agents 1-4 章 → Claude-Code-Source 1-3 章
                    ↓
Hello-Agents 6-9 章 → Claude-Code-Source 4-7 章
                    ↓
Claude-Code-Source 第 8 章动手
```

### 路径 C：只想搞 LLM 训练，不做 Agent（3-4 周）

```
Happy-LLM 全书
   ↓
Hello-Agents 第 11 章（Agentic-RL）
```

### 路径 E：专攻 Code Agent / Coder Model 训练（4-6 周）

```
Happy-LLM 第 4-6 章（LLM 训练基础）
   ↓
Hello-Agents 第 4, 9, 11 章（Agent 范式 / 上下文 / Agentic-RL）
   ↓
Claude-Code-Source 第 2-4, 7 章（Agent Loop / Tool / Context / 安全）
   ↓
agentic-rl-knowledge-base 第 4 章（Code Agentic RL 专题）
   ↓
Code-Agent-Knowledge-Base（Coder Model 数据与训练全流程）
```

### 路径 D：想生产部署 Agent（重点 CC）

```
Hello-Agents 第 1, 4, 7 章打底
   ↓
Claude-Code-Source 全书
   ↓
最重要：CC 第 7 章（安全与权限）+ 第 8 章（MiniCC）
```

---

## 🔗 跨书联动

每个 Hello-Agents 章节末尾有"姊妹篇对照"块，指向对应 Claude-Code-Source 章节。
每个 Claude-Code-Source 章节总览也有"姊妹篇对照"块，指向 Hello-Agents 对应章节。

### 主要联动节点

| 主题 | Hello-Agents | Claude-Code-Source |
|---|---|---|
| Agent 定义 | 1.1 什么是智能体 | 1.1 Agent 是什么 |
| Agent Loop | 1.2 PEAS + T-A-O | 1.3 最简核心循环 |
| ReAct 范式 | 4.2 ReAct 完整实现 | 2.2 queryLoop 骨架 |
| 工具系统 | 7.5 BaseTool/ToolRegistry | 3.2 Tool 接口设计 |
| 多 Agent | 6.2 AutoGen / 6.4 CAMEL | 6.2 AgentTool |
| 上下文工程 | 9.1-9.3 ContextBuilder/GSSC | 4.1-4.5 上下文管理 |
| 记忆系统 | 8.2 四种记忆类型 | 4.3 TodoWrite / 4.4 CLAUDE.md |
| MCP 协议 | 10.2 MCP 详解 | 3.1 Tool Use 协议 |
| 自建框架综合实战 | 第 7 章 HelloAgents | 第 8 章 MiniCC |

### Code Agent 训练联动节点

| 主题 | 入口 |
|---|---|
| Coder Model 训练总览 | [00-总览](Code-Agent-Knowledge-Base/00-%E6%80%BB%E8%A7%88.md) |
| 四阶段数据形态 | [00-四阶段数据总览](Code-Agent-Knowledge-Base/02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/00-%E5%9B%9B%E9%98%B6%E6%AE%B5%E6%95%B0%E6%8D%AE%E6%80%BB%E8%A7%88.md) |
| Pre-train / Mid-train / SFT / RL | [Coder模型训练全流程地图](Code-Agent-Knowledge-Base/00-MOC/Coder%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E5%85%A8%E6%B5%81%E7%A8%8B%E5%9C%B0%E5%9B%BE.md) |
| Agent 工程闭环 | [01-Agent工程闭环总览](Code-Agent-Knowledge-Base/04-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF/01-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF%E6%80%BB%E8%A7%88.md) |
| Reward 与 Sandbox 安全 | [Reward-Hacking与Sandbox安全](Code-Agent-Knowledge-Base/05-Reward%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7/Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) |
| Code Agent 案例与论文卡 | [00-章节总览](Code-Agent-Knowledge-Base/06-%E6%A1%88%E4%BE%8B%E4%B8%8E%E8%AE%BA%E6%96%87%E5%8D%A1/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) |

---

