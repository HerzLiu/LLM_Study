---
tags: [Code-Agent, 训练阶段, Midtrain, Tool-Use, Long-Context, Annealing]
status: Active
source: "用户提供：训练Coder模型全流程.md §三"
stage: midtrain
---

# Mid-train

⬅ [01-Pre-train](01-Pre-train.md) | ➡ [03-SFT](03-SFT.md)

## 🎬 故事比喻：从读手册到熟悉工具箱

实习生读完维修手册后，还不能马上独立修车。他需要熟悉车间工具：扳手放哪，升降台怎么用，检测仪输出怎么看，长工单怎么保持上下文。

Mid-train 就是给模型补“工具箱直觉”和“长任务耐力”。它仍然是在预测 token，但 token 里开始出现工具协议、长 repo、统一思考格式和更高质量的代码。

## 目标

把 Code Base 变成增强 Base：更高质量、更懂工具调用格式、更能处理长上下文。

## 喂什么数据

参考 [02-Midtrain数据工程](../02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/02-Midtrain%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B.md)：

- 高质量代码子集。
- 规范化工具调用轨迹文本。
- 单格式 reasoning 数据。
- 长文件、长 repo、长距离依赖样本。

## 训练信号

仍是 next-token cross entropy。

## 关键工程 trick

- WSD 调度：在 decay 阶段切换到高质量数据子集。
- 工具调用占比控制在小比例，如 5% 到 10%，用于塑造 token 分布。
- 长上下文采用长度课程：4K -> 32K -> 128K -> 200K。
- RoPE base 与窗口同步调整。
- 长样本配比渐增，避免 loss spike。

## 🔧 技术拆解：Mid-train 的三条线

| 线 | 做法 | 解决什么 |
|---|---|---|
| 高质量退火 | 后期提高精选代码比例 | 拉高下限，减少低质代码分布牵引 |
| 工具格式注入 | 混入 tool_call / tool_response 文本 | 让工具协议成为自然 token 分布 |
| 长上下文课程 | 逐步放大窗口和长样本占比 | 学 repo 级长程依赖，避免 loss spike |

## 🧪 例子：工具调用在 Mid-train 里的样子

```text
用户想确认失败测试。
<tool_call>
{"name": "bash", "arguments": {"cmd": "pytest tests/test_timeout.py -q"}}
</tool_call>
<tool_response>
1 failed, 23 passed
</tool_response>
失败集中在 timeout=0 的边界条件。
```

这里不是在训练 agent loop，只是在让模型把工具调用格式“看熟”。真正的多轮决策会在 [03-SFT](03-SFT.md) 和 [04-RL](04-RL.md) 里学。

## ⚠️ 工程坑

- 工具格式太多会互相干扰，最好规范为少数协议。
- reasoning / non-reasoning 混训容易让模型该想时不想、不该想时停不下来。
- 长上下文不能只拉 RoPE 参数，必须配合长样本和评测。

## 产物

增强 Base：

- 懂代码。
- 熟悉工具调用格式。
- 有更长上下文能力。
- 更适合接 SFT agent trajectory。

## 回链

- [00-章节总览](../../Claude-Code-Source/03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- [00-章节总览](../../Claude-Code-Source/04-%E7%AC%AC4%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- [00-章节总览](../../Hello-Agents/09-%E7%AC%AC9%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B7%A5%E7%A8%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- [03-GQA与RoPE](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/03-GQA%E4%B8%8ERoPE.md)

## ❓ 自测问题

- 为什么 Mid-train 不只是“继续 Pre-train”？
- 长上下文训练为什么需要课程学习？
- 工具调用注入和 SFT 工具轨迹有什么区别？
