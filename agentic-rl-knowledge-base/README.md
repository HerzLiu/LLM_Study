---
tags: [agentic-rl, knowledge-base, README, code-agentic-rl]
created: 2026-06-05
source: "agentic-rl-learning-map/"
---

# Agentic RL / Code Agentic RL 知识库

> 这套知识库是在 `原始学习资料目录（未收录）/` 原始学习资料（15 篇 markdown）基础上消化、重组、配故事比喻、桥接已有 Obsidian 笔记后的精读版本。
> 不是简单的复制，也不是泛泛的总结，而是面向"懂 LLM/Agent、但 RL 基础薄弱、想进入 Code Agentic RL"的人重写的学习路径。

---

## 🎬 一句话定位

| 维度 | 内容 |
|---|---|
| **它是什么** | 把 Agentic RL（用 RL 训练能在多轮环境中行动的 LLM Agent）讲清楚的本地笔记书 |
| **它不是什么** | 不是 RL 教科书；不是论文复读机；不是开箱即用代码仓 |
| **谁适合读** | 已经会用 ReAct/工具/Memory，但 PPO/GRPO/RLVR 一脸懵的人 |
| **核心目的** | 让你在 4-6 周内能进入 Code Agentic RL（SWE-bench / SWE-agent / RLVR for code）方向 |

---

## 📑 知识库结构（33 篇）

```
agentic-rl-knowledge-base/
├── README.md (本文件)
├── 00-总览.md ⭐ 推荐第一篇读
│
├── 01-第1章-RL最小必要基础     (7 节：写给 LLM 背景的 RL 速成 + on/off-policy)
├── 02-第2章-从RLHF到Agentic-RL  (6 节：RLHF / DPO / GRPO / RLVR / PRM-ORM)
├── 03-第3章-通用Agentic-RL边界与图谱 (5 节：定义 + 论文 + Repo + 6周路线)
├── 04-第4章-Code-Agentic-RL专题 (8 节：本知识库核心重点) ⭐⭐⭐
├── 05-第5章-动手项目            (2 节：Calculator toy + Bug-Fix Agent Gym)
├── 06-第6章-学习路径速查        (4 节：14天/6周/检查清单)
│
└── 附录
    ├── 01-术语速查.md
    ├── 02-资料不一致与待确认清单.md ⚠️ 必读（已联网核查 13 篇论文状态）
    ├── 03-原始资料反向索引.md
    ├── 04-公式速查表.md ⭐ 写代码/论文时回查
    └── 05-训练诊断手册.md ⭐⭐ 训练崩了/曲线异常时回查
```

---

## 🚪 三条阅读入口

| 你的状态 | 入口 | 时间 |
|---|---|---|
| 完全没读过 → | [00-总览](00-%E6%80%BB%E8%A7%88.md) | 15 分钟看完路径选择 |
| 时间紧（1 天） → | [00-总览](00-%E6%80%BB%E8%A7%88.md) → [01-Code-Agentic-RL-14天最短路径](06-%E7%AC%AC6%E7%AB%A0-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E5%BE%84%E9%80%9F%E6%9F%A5/01-Code-Agentic-RL-14%E5%A4%A9%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84.md) → [00-章节总览](04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | 半天 |
| 系统学（6 周） → | 按 1→2→3→4→5→6 章顺序读 | 6 周 |

---

## 🚀 下一步：落地到 Coder Model 训练

本库讲清楚 **Agentic RL / Code Agentic RL 的理论底座**。如果你已经理解 GRPO、RLVR、O/A/R、reward hacking，下一步可以进入 [Code-Agent-Knowledge-Base](../Code-Agent-Knowledge-Base/00-%E6%80%BB%E8%A7%88.md)，看"从通用 Base 到 Coder Model"的数据与训练链路。

| 你在本库读到 | 下一步落地 |
|---|---|
| [01-Code-Agent定义与边界](04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/01-Code-Agent%E5%AE%9A%E4%B9%89%E4%B8%8E%E8%BE%B9%E7%95%8C.md) | [01-Code-Agent与Coder模型边界](../Code-Agent-Knowledge-Base/01-Code-Agent%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/01-Code-Agent%E4%B8%8ECoder%E6%A8%A1%E5%9E%8B%E8%BE%B9%E7%95%8C.md) |
| [02-Code-Agent能力栈10模块](04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md) | [02-Code-Agent能力栈与训练目标](../Code-Agent-Knowledge-Base/01-Code-Agent%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%88%E4%B8%8E%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87.md) |
| [03-Observation-Action-Reward定义](04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md) | [04-RL任务与验证器数据](../Code-Agent-Knowledge-Base/02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/04-RL%E4%BB%BB%E5%8A%A1%E4%B8%8E%E9%AA%8C%E8%AF%81%E5%99%A8%E6%95%B0%E6%8D%AE.md) |
| [04-RL如何用在Code-Agent上](04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md) | [04-RL](../Code-Agent-Knowledge-Base/03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/04-RL.md) |
| [08-Reward-Hacking与Sandbox安全](04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) | [Reward-Hacking与Sandbox安全](../Code-Agent-Knowledge-Base/05-Reward%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7/Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) |

完整双库映射：[双库桥接｜Agentic-RL到Code-Agent](../Code-Agent-Knowledge-Base/90-%E9%99%84%E5%BD%95/%E5%8F%8C%E5%BA%93%E6%A1%A5%E6%8E%A5%EF%BD%9CAgentic-RL%E5%88%B0Code-Agent.md)

---

## 🌉 与已有 Obsidian 知识的桥接

本库主动跨链到你 vault 里已有的笔记：

- **RL/RLHF 前置**：[Happy-LLM RLHF 章](../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07-RLHF%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.md)
- **SFT/GRPO 实战**：[Hello-Agents 第 11 章 Agentic-RL](../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- **MDP 形式化对比**：[01-从LLM训练到Agentic-RL](../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/01-%E4%BB%8ELLM%E8%AE%AD%E7%BB%83%E5%88%B0Agentic-RL.md)
- **GRPO 训练代码**：[04-GRPO训练实战](../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)

> 这些链接已经验证存在（2026-06-05），不会失效。

---

## ⚠️ 阅读前请知

1. **保留所有"需要进一步确认"标注**：原始资料里很多论文的会议状态、CCF-A 归属、机构归属都标了"需确认"。本库**原样保留**，没有自行编造。
2. **GitHub star 是 2026-06-05 的检索值**：会变。**这个日期看起来像未来时间**，是原始资料作者的检索时间戳，不是 typo。
3. **代码片段是教学用途**：标注「来自资料/可能需调整」的代码不保证开箱即用，跑实验请回原始 repo。
4. **资料中发现的不一致点**：已单独整理到 [02-资料不一致与待确认清单](%E9%99%84%E5%BD%95/02-%E8%B5%84%E6%96%99%E4%B8%8D%E4%B8%80%E8%87%B4%E4%B8%8E%E5%BE%85%E7%A1%AE%E8%AE%A4%E6%B8%85%E5%8D%95.md)，**强烈建议在精读前先看一眼**。

---

## 🔗 原始资料

- 总入口：`原始学习资料目录（未收录）/READING_GUIDE.md`
- 通用部分：`01-boundaries.md` ~ `07-toy-project-spec.md`（7 篇）
- Code 专题：`code-agentic-rl/01-boundaries.md` ~ `08-toy-bugfix-agent-gym.md`（8 篇）

→ 见 [03-原始资料反向索引](%E9%99%84%E5%BD%95/03-%E5%8E%9F%E5%A7%8B%E8%B5%84%E6%96%99%E5%8F%8D%E5%90%91%E7%B4%A2%E5%BC%95.md) 查每篇笔记对应原始文档的哪一段。

---

➡ 现在开始：[00-总览](00-%E6%80%BB%E8%A7%88.md)
