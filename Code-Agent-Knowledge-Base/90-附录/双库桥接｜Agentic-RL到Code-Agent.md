---
tags: [Code-Agent, agentic-rl, 双库桥接, 索引, 附录]
status: Active
source: "agentic-rl-knowledge-base 与 Code-Agent-Knowledge-Base 对照"
stage: appendix
---

# 双库桥接｜Agentic-RL 到 Code-Agent

⬅ [跨库索引](%E8%B7%A8%E5%BA%93%E7%B4%A2%E5%BC%95.md) | ➡ [自测清单](%E8%87%AA%E6%B5%8B%E6%B8%85%E5%8D%95.md)

## 🎬 一句话定位

这两个库不是重复关系，而是上下游关系：

```text
agentic-rl-knowledge-base
  = 理论底座：RL / GRPO / RLVR / OAR / reward / 风险

Code-Agent-Knowledge-Base
  = 训练落地：数据工程 / Pre-train / Mid-train / SFT / RL / scaffold / sandbox
```

## 🧭 什么时候读哪个库

| 你的问题 | 先读 |
|---|---|
| “GRPO / RLVR / PPO / KL 是什么？” | [00-章节总览](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) |
| “Code Agent 的 O/A/R 怎么形式化？” | [03-Observation-Action-Reward定义](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md) |
| “如何从 base 模型训成 Coder Model？” | [Coder模型训练全流程地图](../00-MOC/Coder%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E5%85%A8%E6%B5%81%E7%A8%8B%E5%9C%B0%E5%9B%BE.md) |
| “SFT 轨迹数据怎么长？” | [03-SFT轨迹数据](../02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/03-SFT%E8%BD%A8%E8%BF%B9%E6%95%B0%E6%8D%AE.md) |
| “RL 训练环境怎么接 scaffold / sandbox / proxy？” | [02-Scaffold-Sandbox-Proxy](../04-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF/02-Scaffold-Sandbox-Proxy.md) |
| “reward hacking 具体怎么防？” | 两边都读：[08-Reward-Hacking与Sandbox安全](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) + [Reward-Hacking与Sandbox安全](../05-Reward%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7/Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) |

## 🔁 章节映射

| Agentic-RL 理论底座 | Code-Agent 训练落地 |
|---|---|
| [01-MDP与轨迹（用Agent语言讲RL）](../../agentic-rl-knowledge-base/01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/01-MDP%E4%B8%8E%E8%BD%A8%E8%BF%B9%EF%BC%88%E7%94%A8Agent%E8%AF%AD%E8%A8%80%E8%AE%B2RL%EF%BC%89.md) | [03-工具调用与轨迹Schema](../04-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF/03-%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E4%B8%8E%E8%BD%A8%E8%BF%B9Schema.md) |
| [03-GRPO（组内相对优势）](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/03-GRPO%EF%BC%88%E7%BB%84%E5%86%85%E7%9B%B8%E5%AF%B9%E4%BC%98%E5%8A%BF%EF%BC%89.md) | [02-GRPO与长轨迹稳定性](../05-Reward%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7/02-GRPO%E4%B8%8E%E9%95%BF%E8%BD%A8%E8%BF%B9%E7%A8%B3%E5%AE%9A%E6%80%A7.md) |
| [04-RLVR（可验证奖励）](../../agentic-rl-knowledge-base/02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/04-RLVR%EF%BC%88%E5%8F%AF%E9%AA%8C%E8%AF%81%E5%A5%96%E5%8A%B1%EF%BC%89.md) | [04-RL](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/04-RL.md) |
| [01-Code-Agent定义与边界](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/01-Code-Agent%E5%AE%9A%E4%B9%89%E4%B8%8E%E8%BE%B9%E7%95%8C.md) | [01-Code-Agent与Coder模型边界](../01-Code-Agent%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/01-Code-Agent%E4%B8%8ECoder%E6%A8%A1%E5%9E%8B%E8%BE%B9%E7%95%8C.md) |
| [02-Code-Agent能力栈10模块](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md) | [02-Code-Agent能力栈与训练目标](../01-Code-Agent%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%88%E4%B8%8E%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87.md) |
| [03-Observation-Action-Reward定义](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md) | [04-RL任务与验证器数据](../02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/04-RL%E4%BB%BB%E5%8A%A1%E4%B8%8E%E9%AA%8C%E8%AF%81%E5%99%A8%E6%95%B0%E6%8D%AE.md) |
| [04-RL如何用在Code-Agent上](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md) | [04-RL](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/04-RL.md) |
| [05-必读论文10篇精读卡（SWE-bench线）](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md) | [00-章节总览](../06-%E6%A1%88%E4%BE%8B%E4%B8%8E%E8%AE%BA%E6%96%87%E5%8D%A1/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) |
| [08-Reward-Hacking与Sandbox安全](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) | [Reward-Hacking与Sandbox安全](../05-Reward%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7/Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) |
| [02-Toy-Bug-Fix-Agent-Gym（核心）](../../agentic-rl-knowledge-base/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md) | [Code-Agent落地路线](../00-MOC/Code-Agent%E8%90%BD%E5%9C%B0%E8%B7%AF%E7%BA%BF.md) |

## 🧩 概念映射

| 概念 | Agentic-RL 中怎么讲 | Code-Agent 中怎么落地 |
|---|---|---|
| Trajectory | episode 的 observation/action/reward 序列 | messages、tool_calls、observation、patch、reward 的轨迹 schema |
| Reward | pass/fail、dense、process、verifier、rule | F2P/P2P、工具合法性、过程分、GenRM 低权重 |
| RLVR | 可验证奖励范式 | pytest、编译、终态检查、浏览器渲染 |
| GRPO | 同 prompt 多采样组内相对优势 | 同 issue 采多条 patch trajectory |
| Credit assignment | 长程任务里判断哪步有效 | step-level progress、测试增量、定位命中 |
| Sandbox | 风险和防护原则 | Docker / OS 隔离 / proxy / scaffold 权限 |

## 🚦 推荐阅读路径

### 路径 A：从理论到训练

```text
Agentic-RL 第1章 MDP/轨迹
  -> 第2章 GRPO/RLVR
  -> 第4章 Code Agentic RL
  -> Code-Agent 四阶段数据总览
  -> Code-Agent SFT / RL / Reward
```

### 路径 B：从训练问题回查理论

```text
Code-Agent 训练全流程地图
  -> 卡在 reward：回 Agentic-RL 4.4 / 4.8
  -> 卡在 GRPO：回 Agentic-RL 2.3
  -> 卡在 OAR：回 Agentic-RL 4.3
  -> 卡在 toy：回 Agentic-RL 5.2
```

## ⚠️ 避免重复阅读

- 两库都讲 Code Agent 定义：Agentic-RL 负责边界和 O/A/R；Code-Agent 负责 Coder Model 训练视角。
- 两库都讲 reward hacking：Agentic-RL 负责风险分类；Code-Agent 负责训练 reward 和 sandbox 配置。
- 两库都讲 SWE-bench / SWE-agent：Agentic-RL 负责论文图谱；Code-Agent 负责案例如何服务数据与训练。

