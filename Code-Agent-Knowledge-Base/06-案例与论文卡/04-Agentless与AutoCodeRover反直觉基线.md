---
tags: [Code-Agent, Agentless, AutoCodeRover, 基线, Program-Repair]
status: Active
source: "https://arxiv.org/abs/2407.01489; https://arxiv.org/abs/2404.05427"
stage: cases
---

# 6.4 Agentless 与 AutoCodeRover 反直觉基线

⬅ [03-SWE-Gym与SWE-smith训练数据](03-SWE-Gym%E4%B8%8ESWE-smith%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE.md) | ➡ [术语表](../90-%E9%99%84%E5%BD%95/%E6%9C%AF%E8%AF%AD%E8%A1%A8.md)

## 🎬 故事比喻：不是每次都要派全自动机器人

如果水管漏了，你可以派一个全自动维修机器人，让它自己搜索、计划、找工具、拆墙。

但有时更稳的办法是：先让老师傅定位漏点，再让工人只修那一小段，最后用压力测试验收。

**Agentless** 和 **AutoCodeRover** 的价值就在这里：它们提醒我们，Code Agent 不一定越自主越好。结构化流程、检索、定位、验证，可能比“让模型自由行动”更便宜、更稳、更可解释。

## 🔧 技术拆解：Agentless 三阶段强基线

Agentless 把复杂 agent loop 拆成更固定的流程：

```text
localization -> repair -> patch validation
```

它不让 LLM 在每一步自由决定未来动作，而是把问题变成更可控的 pipeline。

这对本库的启发：

- SFT/RL 不是唯一方向，好的 pipeline 本身就是强 inductive bias。
- 训练 Coder Model 前，应该先有强检索、强定位、强验证 baseline。
- 如果复杂 agent 输给简单 pipeline，说明 agent loop 设计或工具接口可能有问题。

## 🔧 技术拆解：AutoCodeRover 的结构化搜索

AutoCodeRover 更强调程序结构：AST、类、方法、调用关系、测试反馈、fault localization。

它的核心不是让模型“随便翻仓库”，而是利用软件工程信息帮模型缩小上下文。

## 🧪 对比：自由 agent vs 结构化 pipeline

| 维度 | 自由 Code Agent | Agentless / AutoCodeRover |
|---|---|---|
| 动作选择 | 模型每轮决定 | 固定阶段或结构化搜索 |
| 可解释性 | 取决于轨迹质量 | 更容易定位失败阶段 |
| 成本 | 可能较高 | 通常更可控 |
| 上限 | 长程复杂任务潜力大 | 受 pipeline 设计约束 |
| 训练价值 | 产生丰富 trajectory | 产生高质量 localization / repair 数据 |

## ⚠️ 避坑

- 不要把“复杂 agent”当成天然先进。
- 如果定位阶段很弱，后续 RL 很可能只是在错误上下文里乱试。
- 结构化 pipeline 也可能过拟合 benchmark，需要真实 repo 验证。

## 🔗 跨库链接

- Code Agent 能力栈：[02-Code-Agent能力栈10模块](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md)
- Repo 理解与检索：[02-Code-Agent能力栈与训练目标](../01-Code-Agent%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%88%E4%B8%8E%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87.md)
- 工具与轨迹：[03-工具调用与轨迹Schema](../04-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF/03-%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E4%B8%8E%E8%BD%A8%E8%BF%B9Schema.md)
- SWE-bench 与 SWE-agent：[01-SWE-bench与SWE-agent](01-SWE-bench%E4%B8%8ESWE-agent.md)

## 资料锚点

- [Agentless paper](https://arxiv.org/abs/2407.01489)
- [AutoCodeRover paper](https://arxiv.org/abs/2404.05427)

## ❓ 自测题

- Agentless 为什么能成为 Code Agent 的强基线？
- AutoCodeRover 的结构化搜索给 Coder Model 数据工程什么启发？
- 什么时候该选择自由 agent loop，什么时候该选择固定 pipeline？
