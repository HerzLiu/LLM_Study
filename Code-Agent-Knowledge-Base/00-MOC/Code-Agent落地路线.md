---
tags: [Code-Agent, MOC, 路线, 落地]
status: Active
source: "用户提供：训练Coder模型全流程.md; existing Obsidian vault"
stage: overview
---

# Code Agent 落地路线

⬅ [00-总览](../00-%E6%80%BB%E8%A7%88.md) | ➡ [Coder模型训练全流程地图](Coder%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E5%85%A8%E6%B5%81%E7%A8%8B%E5%9C%B0%E5%9B%BE.md)

## 路线 A：先建立心智模型

```text
Happy-LLM 训练基础
  -> Hello-Agents Agent 范式
  -> Claude-Code-Source 工业实现
  -> agentic-rl Code 专题
  -> 本库 Coder Model 训练链路
```

对应入口：

- [00-总览](../../Happy-LLM/00-%E6%80%BB%E8%A7%88.md)
- [00-总览](../../Hello-Agents/00-%E6%80%BB%E8%A7%88.md)
- [00-总览](../../Claude-Code-Source/00-%E6%80%BB%E8%A7%88.md)
- [00-章节总览](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- [00-总览](../00-%E6%80%BB%E8%A7%88.md)
- [双库桥接｜Agentic-RL到Code-Agent](../90-%E9%99%84%E5%BD%95/%E5%8F%8C%E5%BA%93%E6%A1%A5%E6%8E%A5%EF%BD%9CAgentic-RL%E5%88%B0Code-Agent.md)

## 路线 B：只想懂训练全流程

| Day | 内容 |
|---|---|
| Day 1 | [00-四阶段数据总览](../02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/00-%E5%9B%9B%E9%98%B6%E6%AE%B5%E6%95%B0%E6%8D%AE%E6%80%BB%E8%A7%88.md) |
| Day 2 | [01-Pre-train](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/01-Pre-train.md) |
| Day 3 | [02-Mid-train](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/02-Mid-train.md) |
| Day 4 | [03-SFT](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/03-SFT.md) |
| Day 5 | [04-RL](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/04-RL.md) |
| Day 6 | [Reward-Hacking与Sandbox安全](../05-Reward%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7/Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) |
| Day 7 | [00-章节总览](../06-%E6%A1%88%E4%BE%8B%E4%B8%8E%E8%AE%BA%E6%96%87%E5%8D%A1/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) |
| Day 8 | [自测清单](../90-%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E6%B8%85%E5%8D%95.md) |

## 路线 C：想做最小可运行实验

1. 先读 [02-Toy-Bug-Fix-Agent-Gym（核心）](../../agentic-rl-knowledge-base/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)。
2. 再读 [02-Scaffold-Sandbox-Proxy](../04-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF/02-Scaffold-Sandbox-Proxy.md)。
3. 对照 [01-SWE-bench与SWE-agent](../06-%E6%A1%88%E4%BE%8B%E4%B8%8E%E8%AE%BA%E6%96%87%E5%8D%A1/01-SWE-bench%E4%B8%8ESWE-agent.md) 理解评测和 ACI。
4. 用一个 Python 小仓库构造 `issue -> failing test -> patch -> pytest` 闭环。
5. 先做 SFT 轨迹采集，再考虑 RL rollout。

## 当前库的定位

本库更像一张工程蓝图：

- 它从训练数据说起，不只讲 agent 使用。
- 它把 Coder Model 当成一个被训练出来的 policy，而不只是 prompt + tool。
- 它默认最终目标是 repo-level bug fixing，而不是单函数代码生成。

## 和 Agentic-RL 知识库怎么配合

| 先读 Agentic-RL | 再读本库 |
|---|---|
| [03-GRPO（组内相对优势）](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md) | [02-GRPO与长轨迹稳定性](../05-Reward%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7/02-GRPO%E4%B8%8E%E9%95%BF%E8%BD%A8%E8%BF%B9%E7%A8%B3%E5%AE%9A%E6%80%A7.md) |
| [04-RLVR（可验证奖励）](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md) | [04-RL](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/04-RL.md) |
| [03-Observation-Action-Reward定义](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md) | [03-工具调用与轨迹Schema](../04-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF/03-%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E4%B8%8E%E8%BD%A8%E8%BF%B9Schema.md) |
| [02-Toy-Bug-Fix-Agent-Gym（核心）](../../agentic-rl-knowledge-base/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md) | 路线 C 的最小可运行实验 |

完整映射见 [双库桥接｜Agentic-RL到Code-Agent](../90-%E9%99%84%E5%BD%95/%E5%8F%8C%E5%BA%93%E6%A1%A5%E6%8E%A5%EF%BD%9CAgentic-RL%E5%88%B0Code-Agent.md)。
