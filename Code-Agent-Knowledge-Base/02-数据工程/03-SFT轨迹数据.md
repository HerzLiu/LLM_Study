---
tags: [Code-Agent, 数据工程, SFT, Trajectory, Tool-Masking]
status: Active
source: "用户提供：训练Coder模型全流程.md §一.3"
stage: data
---

# SFT 轨迹数据

⬅ [02-Midtrain数据工程](02-Midtrain%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B.md) | ➡ [04-RL任务与验证器数据](04-RL%E4%BB%BB%E5%8A%A1%E4%B8%8E%E9%AA%8C%E8%AF%81%E5%99%A8%E6%95%B0%E6%8D%AE.md)

## 目标

让模型模仿强教师模型的成功 agent 轨迹，学会按协议多轮行动。

## 数据形态

SFT 数据是完整 messages：

```json
{
  "messages": [
    {"role": "system", "content": "你是 coding agent，可用工具 bash / editor / submit"},
    {"role": "user", "content": "修复 timeout=0 永久阻塞的 bug"},
    {"role": "assistant", "content": "我先搜索 timeout 逻辑", "tool_calls": [{"name": "bash", "args": "grep -rn timeout src/"}]},
    {"role": "tool", "content": "src/requests/adapters.py:45: timeout=None", "content_mask": true},
    {"role": "assistant", "tool_calls": [{"name": "submit"}]}
  ],
  "channel": "swe_bench",
  "reasoning_content": "非 null"
}
```

## 数据从哪里来

1. 准备有验证标准的任务。
2. 强教师模型在 sandbox 里采样多条轨迹。
3. 只保留验证器通过的成功轨迹。
4. 整理成 messages 格式。

这相当于 rejection sampling / distillation。

## 四个训练细节

| 细节 | 作用 |
|---|---|
| 工具返回 masking | 不让模型背 shell 输出，只学何时调什么工具 |
| reasoning / standard 分流 | 避免是否思考的隐变量混乱 |
| document-level packing | 提升 token 利用率，同时隔离样本间 attention |
| channel-aware loss | 防止短 QA 淹没长 SWE 轨迹 |

## 和已有库的回链

- SFT 基础：[06-SFT有监督微调](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-SFT%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md)
- Chat Template 与 loss mask：[03-SFT实战要点](../../Happy-LLM/06-%E7%AC%AC6%E7%AB%A0-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%AE%9E%E8%B7%B5/03-SFT%E5%AE%9E%E6%88%98%E8%A6%81%E7%82%B9.md)
- Agent 工具协议：[01-Tool-Use协议基础](../../Claude-Code-Source/03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/01-Tool-Use%E5%8D%8F%E8%AE%AE%E5%9F%BA%E7%A1%80.md)

## 自测问题

- 为什么 tool message 要 mask？
- 为什么失败轨迹不直接作为 SFT 数据？
- SFT 为什么能教格式，但不能保证最终修复成功率？

