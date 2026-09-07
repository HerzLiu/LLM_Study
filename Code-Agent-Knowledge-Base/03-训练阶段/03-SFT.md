---
tags: [Code-Agent, 训练阶段, SFT, Trajectory, Behavior-Cloning]
status: Active
source: "用户提供：训练Coder模型全流程.md §四"
stage: sft
---

# SFT

⬅ [02-Mid-train](02-Mid-train.md) | ➡ [04-RL](04-RL.md)

## 🎬 故事比喻：跟着师傅看一遍标准维修录像

SFT 像让新人看师傅修车的录像。

录像里不只记录“最后换了哪个零件”，还记录师傅先看了什么、查了什么、怎么判断、用了哪个工具、测试失败后怎么回头定位。新人要模仿的是整条成功路线。

## 目标

让模型学会按 agent 协议多轮干活：思考、调工具、看观察、继续、提交。

## 喂什么数据

参考 [03-SFT轨迹数据](../02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/03-SFT%E8%BD%A8%E8%BF%B9%E6%95%B0%E6%8D%AE.md)：

- SWE 修 bug 成功轨迹。
- 代码理解轨迹。
- Terminal 任务轨迹。
- WebCoding 轨迹。
- 代码补全 / 生成任务。

## 训练信号

标准 causal LM 目标，但只对应该由 assistant 生成的 token 计算 loss。

工具返回、环境输出、测试日志要 mask。

## 🔧 技术拆解：SFT 学的是行为格式

```text
state = system + issue + previous tool observations
target = assistant thought / message / tool_call
masked = tool response / stdout / traceback 原文
```

模型学会的是“在这个状态下下一步该做什么”，不是背诵测试输出。

## 关键工程 trick

- 只保留真的被验证器判定成功的教师轨迹。
- 按 domain 分治，防止任务形态互相污染。
- reasoning 和 non-reasoning 分流。
- document-level packing 提升吞吐。
- channel-aware loss 保护长 SWE 任务。

## 🧪 例子：一条成功 SWE 轨迹的骨架

```text
Issue: timeout=0 应该立即超时。

assistant: 我先定位 timeout 处理。
tool: rg "timeout" src tests
assistant: 相关逻辑在 adapters.py，先读实现和测试。
tool: sed -n '1,140p' src/requests/adapters.py
assistant: 发现 timeout=0 被当成 None，修改判断。
tool: edit adapters.py
assistant: 跑目标测试。
tool: pytest tests/test_timeout.py -q
assistant: 目标测试通过，再跑相关回归。
tool: pytest tests/test_requests.py -q
assistant: submit
```

## ⚠️ 避坑

| 坑 | 现象 | 解法 |
|---|---|---|
| 把失败轨迹也当正样本 | 模型学会绕路和错误动作 | SFT 只保留成功或高质量修正轨迹 |
| tool response 不 mask | 模型试图预测环境输出 | loss mask |
| 轨迹太冗长 | 学到低效搜索习惯 | 同题多采后选短而干净的轨迹 |
| 只训 patch 不训过程 | 会改代码但不会恢复错误 | 保留 action-observation 过程 |

## 产物

会按 agent 格式多轮行动的模型。

它知道“该怎么像一个 coding agent 那样做事”，但仍主要是模仿教师，不保证能在没见过的 bug 上稳定试错成功。

## 回链

- [03-SFT实战要点](../../Happy-LLM/06-%E7%AC%AC6%E7%AB%A0-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%AE%9E%E8%B7%B5/03-SFT%E5%AE%9E%E6%88%98%E8%A6%81%E7%82%B9.md)
- [03-SFT训练实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/03-SFT%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)
- [00-章节总览](../../Claude-Code-Source/02-%E7%AC%AC2%E7%AB%A0-Agent-Loop%E4%B8%BB%E5%BE%AA%E7%8E%AF/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- [01-Tool-Use协议基础](../../Claude-Code-Source/03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/01-Tool-Use%E5%8D%8F%E8%AE%AE%E5%9F%BA%E7%A1%80.md)

## ❓ 自测问题

- SFT 的 exposure bias 在 Code Agent 里会怎么体现？
- 为什么同题多采样教师轨迹有价值？
- 为什么 channel-aware loss 能保护 SWE 能力？
