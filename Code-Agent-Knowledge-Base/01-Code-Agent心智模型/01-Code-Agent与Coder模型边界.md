---
tags: [Code-Agent, Coder-Model, Code-LLM, 边界, 心智模型]
status: Active
source: "用户提供：训练Coder模型全流程.md"
stage: concept
---

# Code Agent 与 Coder 模型边界

⬅ [00-总览](../00-%E6%80%BB%E8%A7%88.md) | ➡ [02-Code-Agent能力栈与训练目标](02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%88%E4%B8%8E%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87.md)

## 三层能力

| 层级 | 一句话 | 典型任务 | 是否需要环境闭环 |
|---|---|---|---|
| Code LLM | 会写代码片段的模型 | HumanEval、MBPP、补全 | 不一定 |
| Code Agent | 会用工具在仓库里做事的系统 | 找文件、改 patch、跑测试 | 需要 |
| Coder Model | 被训练成适合 Code Agent 闭环的 policy | SWE-bench、真实 issue 修复 | 训练时也需要 |

## 🎬 故事比喻：三个角色

### Level 1：会写函数的实习生

你给他一张白纸：“写一个 `two_sum`。”他能写出来。

但你把一个 20 万行仓库扔给他，说“CI 里某个测试挂了，去修一下”，他会先愣住：文件在哪？测试怎么跑？改完怎么确认没破坏别的功能？这些都不是单函数生成题。

### Level 2：能干活的初级开发

他不只是写函数，还会：

1. 看 issue。
2. 用 `rg` 找相关文件。
3. 读实现和测试。
4. 改最小 patch。
5. 跑 pytest。
6. 看 traceback。
7. 再改。

这就是 Code Agent。它是一个系统，不只是一个模型。

### Level 3：会成长的开发者

他每次修 bug 都会被 CI 打分：这次 F2P 过了吗？P2P 破了吗？工具用得是否乱？有没有改测试骗分？

如果这些反馈能回到模型参数里，它就从 inference-time Code Agent 变成 training-time Coder Model / Code Agentic RL。

## 本库关注什么

本库关注的是第三层：**如何训练一个 Coder Model**。

它不是只研究 prompt，也不是只研究 agent scaffold，而是把模型训练、数据工程、工具环境、reward 和安全一起看。

## 和已有库的关系

- Agent 的基本公式 `LLM + Tools + Loop`：[00-章节总览](../../Claude-Code-Source/01-%E7%AC%AC1%E7%AB%A0-Agent%E5%85%A5%E9%97%A8%E4%B8%8ECC%E6%80%BB%E8%A7%88/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- ReAct、Plan-and-Solve、Reflection：[00-章节总览](../../Hello-Agents/04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- Code Agentic RL 的定义：[01-Code-Agent定义与边界](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/01-Code-Agent%E5%AE%9A%E4%B9%89%E4%B8%8E%E8%BE%B9%E7%95%8C.md)

## 关键判断标准

一个模型是否接近 Coder Model，不看它会不会写一个函数，而看它是否能稳定完成：

1. 读懂 issue。
2. 定位相关文件和符号。
3. 生成最小 patch。
4. 跑测试、读失败日志。
5. 根据反馈继续修。
6. 提交最终 patch，并尽量不引入 regression。

## 🔧 技术拆解：四个相邻概念

| 概念 | 关注点 | 典型代表 | 本库里对应位置 |
|---|---|---|---|
| Code LLM | 代码生成、补全、解释 | HumanEval / MBPP 模型 | [01-Pre-train](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/01-Pre-train.md) |
| Coding Assistant | 人和模型结对写代码 | Aider、IDE Copilot 类工具 | [02-OpenHands-Aider-ClaudeCode-Codex对比](../06-%E6%A1%88%E4%BE%8B%E4%B8%8E%E8%AE%BA%E6%96%87%E5%8D%A1/02-OpenHands-Aider-ClaudeCode-Codex%E5%AF%B9%E6%AF%94.md) |
| Code Agent | 自主执行多步软件工程任务 | SWE-agent、OpenHands、Claude Code、Codex | [01-Agent工程闭环总览](../04-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF/01-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF%E6%80%BB%E8%A7%88.md) |
| Coder Model 训练 | 用轨迹、测试、reward 更新策略 | SWE-Gym、SWE-smith、CodeRL | [04-RL](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/04-RL.md) |

## 🧪 例子：Code Agent 不是一次性 patch 生成器

```text
Issue: pandas 某个 groupby 边界条件结果错误。

Code LLM:
  读 prompt -> 生成一段可能的 patch。

Code Agent:
  读 issue
  -> 搜索 groupby 相关实现
  -> 读已有测试
  -> 复现失败
  -> 改实现
  -> 跑目标测试
  -> 跑回归测试
  -> 生成最终 diff

Coder Model 训练:
  记录上述 action / observation / patch
  -> 用 F2P/P2P 计算 reward
  -> 反向更新 policy 或训练 verifier
```

## ⚠️ 避坑：别把产品形态和训练方法混在一起

- Aider、Claude Code、OpenHands 常规使用时，多数是 **inference-time loop**：模型参数不变。
- SWE-Gym、SWE-smith、Code Agentic RL 讨论的是 **training-time learning**：轨迹会变成训练数据或 reward。
- 一个优秀的 scaffold 可以让普通模型变强，但如果模型没学过长程工具轨迹，它仍会在复杂 issue 上漂移。

## 自测问题

- 为什么 repo-level bug fixing 比单函数生成更像 agent 任务？
- Code Agent 的能力来自模型本身，还是 scaffold？两者怎么分工？
- Coder Model 训练为什么必须包含工具轨迹和执行反馈？

## 资料锚点

- [SWE-bench official](https://www.swebench.com/original.html)
- [SWE-agent paper](https://arxiv.org/abs/2405.15793)
- [OpenHands platform paper](https://openreview.net/forum?id=OJd3ayDDoF)
