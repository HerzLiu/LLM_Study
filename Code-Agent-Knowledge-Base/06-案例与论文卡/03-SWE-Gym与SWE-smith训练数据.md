---
tags: [Code-Agent, SWE-Gym, SWE-smith, 训练数据, RLVR]
status: Active
source: "https://arxiv.org/abs/2412.21139; https://swesmith.com/; https://arxiv.org/abs/2504.21798"
stage: cases
---

# 6.3 SWE-Gym 与 SWE-smith 训练数据

⬅ [02-OpenHands-Aider-ClaudeCode-Codex对比](02-OpenHands-Aider-ClaudeCode-Codex%E5%AF%B9%E6%AF%94.md) | ➡ [04-Agentless与AutoCodeRover反直觉基线](04-Agentless%E4%B8%8EAutoCodeRover%E5%8F%8D%E7%9B%B4%E8%A7%89%E5%9F%BA%E7%BA%BF.md)

## 🎬 故事比喻：考试场不够，还要训练基地

SWE-bench 像高考卷，能衡量能力，但不能天天拿来训练。

要培养会修 bug 的 Code Agent，需要训练基地：大量可执行任务、可复现环境、单元测试、失败轨迹、成功轨迹、verifier 数据。**SWE-Gym** 和 **SWE-smith** 就是在解决这个问题。

## 🔧 技术拆解：SWE-Gym 把真实 issue 变成训练环境

SWE-Gym 的核心价值是：不只是给题，还给可执行环境和轨迹，用于训练 agent 和 verifier。

| 要素 | 作用 |
|---|---|
| real-world Python task | 接近真实 repo issue |
| executable runtime | 能跑测试，能给 reward |
| unit tests | 形成可验证反馈 |
| agent trajectories | 可用于 SFT、RL、verifier |

这和本库 [04-RL](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/04-RL.md) 的训练环境是一一对应的。

## 🔧 技术拆解：SWE-smith 解决数据规模问题

真实 GitHub issue 很宝贵，但规模有限、环境重、构造成本高。

SWE-smith 的思路是：给定 Python codebase，自动构建执行环境，再合成会破坏现有测试的任务实例。这样可以把“训练题库”从几千扩到数万级。

## 🧪 数据生成链路

```text
真实/开源 Python repo
  -> 构建可执行环境
  -> 合成会 break tests 的任务
  -> agent 尝试修复
  -> 测试验证
  -> 生成轨迹 / reward / verifier 数据
```

## ⚠️ 避坑

- 合成任务要避免离真实 issue 太远，否则训练出来的 agent 会只适应“人造 bug”。
- execution environment 太重会限制并发和复现。
- verifier 如果只学训练分布，可能在真实 SWE-bench 上偏。
- 数据规模扩大后，去污染和任务去重更重要。

## 🔗 跨库链接

- RL 任务与验证器：[04-RL任务与验证器数据](../02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/04-RL%E4%BB%BB%E5%8A%A1%E4%B8%8E%E9%AA%8C%E8%AF%81%E5%99%A8%E6%95%B0%E6%8D%AE.md)
- RL 训练阶段：[04-RL](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/04-RL.md)
- Reward 稳定性：[Reward-Hacking与Sandbox安全](../05-Reward%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7/Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)
- Agentic RL 学习地图：[00-总览](../../agentic-rl-knowledge-base/00-%E6%80%BB%E8%A7%88.md)

## 资料锚点

- [SWE-Gym paper](https://arxiv.org/abs/2412.21139)
- [SWE-Gym GitHub](https://github.com/SWE-Gym/SWE-Gym)
- [SWE-Gym Apple research page](https://machinelearning.apple.com/research/training-software)
- [SWE-smith website](https://swesmith.com/)
- [SWE-smith paper](https://arxiv.org/abs/2504.21798)
- [SWE-smith GitHub](https://github.com/SWE-bench/SWE-smith)

## ❓ 自测题

- SWE-Gym 和 SWE-bench 的区别是什么？
- SWE-smith 为什么能缓解训练数据不足？
- 合成 bug 数据最大的泛化风险是什么？
