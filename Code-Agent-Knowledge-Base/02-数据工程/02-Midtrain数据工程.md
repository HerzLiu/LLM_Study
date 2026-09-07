---
tags: [Code-Agent, 数据工程, Midtrain, Tool-Use, Long-Context]
status: Active
source: "用户提供：训练Coder模型全流程.md §一.2"
stage: data
---

# Mid-train 数据工程

⬅ [01-Pretrain数据工程](01-Pretrain%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B.md) | ➡ [03-SFT轨迹数据](03-SFT%E8%BD%A8%E8%BF%B9%E6%95%B0%E6%8D%AE.md)

## 目标

在 code base 上继续训练，注入高质量代码、工具调用格式和长上下文能力，得到更适合 agent 训练的增强 base。

## 喂什么数据

| 数据类 | 作用 |
|---|---|
| 高质量代码 | 在退火阶段拉高能力下限 |
| 工具调用轨迹文本 | 让模型熟悉 tool_call JSON / XML / scaffold 格式 |
| 单格式 CoT | 固定 reasoning 格式，减少格式冲突 |
| 长上下文样本 | 训练 repo-level 长程依赖 |

## 为什么工具调用放在 Mid-train

Mid-train 学习率更高，模型塑性强，能真正学到工具调用 token 分布。

如果只放到 SFT，模型更容易学到表面格式，而不是稳定的 tool-use 先验。

## 长上下文课程

```text
4K -> 32K -> 128K -> 200K
RoPE base 同步放大
长样本配比从低到高退火
```

核心目的：避免窗口突然放大造成 loss spike 和短任务能力遗忘。

## 和已有库的回链

- 上下文工程理论：[00-章节总览](../../Hello-Agents/09-%E7%AC%AC9%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B7%A5%E7%A8%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- Claude Code 上下文管理：[00-章节总览](../../Claude-Code-Source/04-%E7%AC%AC4%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- RoPE 前置：[03-GQA与RoPE](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/03-GQA%E4%B8%8ERoPE.md)

## 自测问题

- 为什么 CoT 格式要尽量统一？
- 长上下文训练为什么要同时调窗口、RoPE base 和长样本配比？
- Mid-train 和 SFT 都能见工具调用，区别在哪里？

