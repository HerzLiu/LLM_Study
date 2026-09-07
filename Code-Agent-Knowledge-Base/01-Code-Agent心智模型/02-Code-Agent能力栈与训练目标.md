---
tags: [Code-Agent, 能力栈, 训练目标, SWE-bench]
status: Active
source: "用户提供：训练Coder模型全流程.md; agentic-rl-knowledge-base"
stage: concept
---

# Code Agent 能力栈与训练目标

⬅ [01-Code-Agent与Coder模型边界](01-Code-Agent%E4%B8%8ECoder%E6%A8%A1%E5%9E%8B%E8%BE%B9%E7%95%8C.md) | ➡ [00-四阶段数据总览](../02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/00-%E5%9B%9B%E9%98%B6%E6%AE%B5%E6%95%B0%E6%8D%AE%E6%80%BB%E8%A7%88.md)

## 能力栈

本库沿用 [Code Agent 能力栈 10 模块](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md)，但把它们映射到训练阶段：

| 能力 | 主要训练来源 |
|---|---|
| 代码语法、API、模式 | Pre-train |
| FIM 补全、跨文件引用 | Pre-train |
| 工具调用格式 | Mid-train + SFT |
| 长上下文 repo 理解 | Mid-train |
| 多轮行动协议 | SFT |
| 错误恢复、测试反馈利用 | SFT + RL |
| 真实修复成功率 | RL |
| reward hacking 抵抗 | RL + sandbox 规则 |

## 训练目标的层次

```text
next-token prediction
  -> agent trajectory imitation
  -> test-verified problem solving
```

## 为什么不能只靠 SFT

SFT 学的是教师已经走通的路径。推理时模型一旦走到教师轨迹之外，就会遇到 exposure bias：越走越偏、错误累积。

RL 的价值在于让模型在自己的轨迹里试错，用测试反馈更新策略。

## 和 Claude Code 的关系

Claude Code 这类产品体现的是 scaffold 和工程系统：

- Agent loop：[00-章节总览](../../Claude-Code-Source/02-%E7%AC%AC2%E7%AB%A0-Agent-Loop%E4%B8%BB%E5%BE%AA%E7%8E%AF/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 工具系统：[00-章节总览](../../Claude-Code-Source/03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 上下文管理：[00-章节总览](../../Claude-Code-Source/04-%E7%AC%AC4%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 权限与安全：[00-章节总览](../../Claude-Code-Source/07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

Coder Model 训练要做的是：让 policy 更适合这些 scaffold，而不是每次只靠 prompt 补救。

