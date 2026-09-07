---
tags: [Code-Agent, Tool-Use, Trajectory, Schema]
status: Active
source: "用户提供：训练Coder模型全流程.md §一.3, §五"
stage: engineering
---

# 工具调用与轨迹 Schema

⬅ [02-Scaffold-Sandbox-Proxy](02-Scaffold-Sandbox-Proxy.md) | ➡ [Reward-Hacking与Sandbox安全](../05-Reward%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7/Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)

## 最小轨迹结构

```json
{
  "task_id": "requests__requests-1142",
  "turns": [
    {"role": "assistant", "thought": "定位 timeout 处理", "action": {"tool": "bash", "args": "grep -rn timeout src/"}},
    {"role": "environment", "observation": "src/requests/adapters.py:45: timeout=None"},
    {"role": "assistant", "action": {"tool": "edit", "args": "修改 timeout 判断"}},
    {"role": "environment", "observation": "edit ok"},
    {"role": "assistant", "action": {"tool": "bash", "args": "pytest tests/test_timeout.py"}},
    {"role": "environment", "observation": "1 passed"}
  ],
  "final_patch": "...",
  "reward": 1.0
}
```

## SFT 与 RL 的差异

| 项 | SFT | RL |
|---|---|---|
| 轨迹来源 | 教师模型离线跑出的成功轨迹 | 当前 policy 在线 rollout |
| 环境反馈 | 写入数据，但 tool response masked | 实时返回 observation |
| 训练信号 | token CE | reward / advantage |
| 失败轨迹 | 通常丢弃 | 非常重要，用于负反馈 |

## 轨迹里要保留什么

- prompt 与 system 约束。
- 每轮 action。
- 每轮 observation。
- patch diff。
- 测试结果。
- 安全审计结果。
- reward 分解。

## 回链

- Tool use 协议：[01-Tool-Use协议基础](../../Claude-Code-Source/03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/01-Tool-Use%E5%8D%8F%E8%AE%AE%E5%9F%BA%E7%A1%80.md)
- O/A/R 定义：[03-Observation-Action-Reward定义](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md)

