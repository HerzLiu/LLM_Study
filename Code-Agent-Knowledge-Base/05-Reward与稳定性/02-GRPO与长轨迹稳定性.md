---
tags: [Code-Agent, GRPO, KL, 长轨迹, 稳定性]
status: Active
source: "用户提供：训练Coder模型全流程.md §五.3-五.6"
stage: reward
---

# GRPO 与长轨迹稳定性

⬅ [Reward-Hacking与Sandbox安全](Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) | ➡ [术语表](../90-%E9%99%84%E5%BD%95/%E6%9C%AF%E8%AF%AD%E8%A1%A8.md)

## 为什么长轨迹难训

Code Agent 不是一次性输出答案，而是几十轮工具交互。

问题包括：

- reward 稀疏。
- credit assignment 难。
- token-level ratio 方差大。
- 一条慢轨迹拖住整个 batch。
- policy 漂移后工具行为会突然变坏。

## GRPO 的基本直觉

同一个 prompt 采样多条轨迹，在组内比较：

```text
比本组平均 reward 高 -> advantage 为正
比本组平均 reward 低 -> advantage 为负
```

这减少了额外 value model 的需求，也适合可验证 reward 场景。

## 稳定性三件套

| 方法 | 解决什么 |
|---|---|
| 序列级重要性比率 | 降低长序列 token ratio 方差 |
| KL-Cov | 只截断高偏移 token，防止熵坍塌 |
| KL 锚 | 与 reference 保持距离，防止 policy 漂移 |

## 工程调度

- rollout 与训练解耦。
- 轨迹完成即入队。
- 满 batch 即训练。
- horizon 从 5、15、30、64、128 逐步放开。

## 回链

- [03-GRPO（组内相对优势）](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md)
- [04b-On-Policy-vs-Off-Policy](../../agentic-rl-knowledge-base/01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/04b-On-Policy-vs-Off-Policy.md)
- [03-从SGD到PPO-GRPO](../../ml-dl-foundations/07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/03-%E4%BB%8ESGD%E5%88%B0PPO-GRPO.md)

