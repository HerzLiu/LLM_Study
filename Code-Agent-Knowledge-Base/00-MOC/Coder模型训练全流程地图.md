---
tags: [Code-Agent, MOC, Coder-Model, 训练全流程, 地图]
status: Active
source: "用户提供：训练Coder模型全流程.md"
stage: overview
---

# Coder 模型训练全流程地图

⬅ [Code-Agent落地路线](Code-Agent%E8%90%BD%E5%9C%B0%E8%B7%AF%E7%BA%BF.md) | ➡ [00-四阶段数据总览](../02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/00-%E5%9B%9B%E9%98%B6%E6%AE%B5%E6%95%B0%E6%8D%AE%E6%80%BB%E8%A7%88.md)

## 阶段总表

| 阶段 | 输入数据 | 训练信号 | 产物 | 主笔记 |
|---|---|---|---|---|
| Pre-train | 海量代码、文档、repo-level 序列 | next-token CE | Code Base | [01-Pre-train](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/01-Pre-train.md) |
| Mid-train | 高质量代码、工具轨迹、长上下文样本 | next-token CE | 增强 Base | [02-Mid-train](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/02-Mid-train.md) |
| SFT | 教师模型成功轨迹 | assistant token CE | 会按 agent 协议行动的模型 | [03-SFT](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/03-SFT.md) |
| RL | 题面、repo、测试、docker 镜像 | sandbox reward | 能自我探索修 bug 的 Coder Model | [04-RL](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/04-RL.md) |

## Agentic-RL 前置概念

| 训练阶段 | 建议先回查的 Agentic-RL 概念 |
|---|---|
| Pre-train | [02-从交叉熵到next-token-loss](../../ml-dl-foundations/07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/02-%E4%BB%8E%E4%BA%A4%E5%8F%89%E7%86%B5%E5%88%B0next-token-loss.md) |
| Mid-train | [01-Agentic-RL定义与边界](../../agentic-rl-knowledge-base/03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/01-Agentic-RL%E5%AE%9A%E4%B9%89%E4%B8%8E%E8%BE%B9%E7%95%8C.md) |
| SFT | [01-RLHF三阶段（SFT-RM-PPO）](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/01-RLHF%E4%B8%89%E9%98%B6%E6%AE%B5%EF%BC%88SFT-RM-PPO%EF%BC%89.md) |
| RL | [03-GRPO（组内相对优势）](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md) + [04-RLVR（可验证奖励）](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md) |
| Reward / 安全 | [08-Reward-Hacking与Sandbox安全](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) |

## 数据形态演进

```text
纯文本 next-token
  -> 带工具格式的纯文本 next-token
  -> 带答案的多轮 messages 轨迹
  -> 没有答案、只有验证器的在线任务
```

## 关键转折点

- Pre-train 到 Mid-train：从“懂代码”变成“懂工具格式和长上下文”。
- Mid-train 到 SFT：从“会预测文本”变成“会按 agent 协议多轮行动”。
- SFT 到 RL：从“模仿教师轨迹”变成“自己探索并被测试反馈训练”。

## 最容易混淆的点

| 混淆 | 正确理解 |
|---|---|
| FIM 是新 loss | FIM 主要是序列重排，loss 仍是 next-token CE |
| 工具调用只要 SFT 学 | Mid-train 注入工具格式更能塑造 token 分布 |
| SFT 数据就是问答 | Coder SFT 的核心是多轮 tool trajectory |
| RL 数据也有答案 | RL 数据没有答案轨迹，只有任务和验证器 |
| pytest reward 足够 | pytest 是 anchor，但必须防 reward hacking |
