---
tags: [agentic-rl, 第5章, toy-project, calculator, tool-use-RL]
chapter: 5
section: 5.1
source: "agentic-rl-learning-map/07-toy-project-spec.md (全文)"
---

# 5.1 通用 Toy：Calculator Tool-use RL

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-Toy-Bug-Fix-Agent-Gym（核心）](02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)

---

## 🎬 故事比喻：用计算器学 Agentic RL

```
任务：自然语言数学题
agent 选择：
  A. 直接回答（可能算错）
  B. 调用 calculator 工具
  C. 中间结果再算一步
  D. 输出最终答案

reward：
  最终答案 exact match → +1.0
  否则 → 0.0
```

> **这是原始资料 §07-toy-project-spec.md 的通用 toy 项目**。
> 适合**还没确定方向**或**先用最简单环境理解 RL loop** 的人。
>
> ⚠️ **你目标是 Code Agentic RL**，可以**跳过本节**，直接去 [02-Toy-Bug-Fix-Agent-Gym（核心）](02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)。
> 这里只为完整性保留通用版的描述。

---

## 🎯 5.1.1 任务定义

输入：自然语言数学问题

```
"小明有 12 个苹果，吃掉 3 个，又买了 2 倍当前数量的苹果，现在有几个？"
"把 3.5 小时换算成分钟，再加上 45 分钟是多少？"
"一家店打 8 折后价格是 64 元，原价是多少？"
```

Agent 可以选择：

- 直接回答
- 调用 calculator 工具
- 对中间结果进行下一步计算
- 输出最终答案

---

## 🔌 5.1.2 环境接口

每个 episode：

```json
{
  "id": "episode-0001",
  "question": "小明有 12 个苹果...",
  "steps": [
    {
      "observation": "用户问题或工具返回结果",
      "action_type": "tool_call",
      "action": {"tool": "calculator", "expression": "12 - 3"},
      "reward": 0.0
    }
  ],
  "final_answer": "27",
  "gold_answer": "27",
  "success": true,
  "total_reward": 1.0
}
```

---

## 📐 5.1.3 OAR 形式化

| RL 概念 | 项目中的对应 |
|---|---|
| state / observation | 当前问题、历史步骤、工具返回结果 |
| action | 工具调用、参数、最终回答 |
| reward | 最终答案 exact match → 1，否则 0；可选加入工具格式错误惩罚 |
| trajectory | 一次完整解题过程 |
| policy | LLM agent |
| environment | calculator wrapper + answer checker |

---

## 🎯 5.1.4 Baseline 设计

资料原文 2 个 baseline：

1. **No-tool baseline**：只让 LLM 直接回答
2. **Prompted ReAct baseline**：允许 LLM 用 `Thought / Action / Observation / Final` 格式调用 calculator

**必须记录**：

```
- success rate
- average steps
- invalid tool-call rate
- common failure modes
```

---

## 🛤 5.1.5 训练路线

### 最小可行版本

```
1. 生成 200-1000 条可验证数学问题
2. 用 prompted agent rollout，保存 trajectories.jsonl
3. 用 exact match 计算 outcome reward
4. 对正确轨迹做 SFT，或对多个候选回答做 DPO 偏好对
5. 如有 GPU，再用 TRL/GRPO 或 Agent Lightning 接入在线训练
```

### 进阶版本

```
- 给每一步 tool call 加 process reward
- 对同一问题采样多个解法，用 group-relative reward 模拟 GRPO
- 加入错误工具返回、无关工具、单位换算等更复杂环境
```

---

## ✅ 5.1.6 验收标准

### 最低验收

- 能生成或加载一批问题
- 能运行 no-tool 和 ReAct 两个 baseline
- 能保存完整 trajectory
- 能计算 success rate 和 invalid tool-call rate
- 写出**至少 5 类失败模式**

### 理想验收

- 训练后相对 baseline 成功率提升
- 能展示 3 条从失败到成功的 trajectory 对比
- 能说明 reward hacking 风险（如答案解析漏洞、工具调用绕过、格式投机）

---

## 💻 5.1.7 最小代码骨架（概念示意，未验证）

```python
# calculator_toy.py
# 概念示意 - 不保证开箱即用

import json
from typing import Literal

# === Tool ===
def calculator(expression: str) -> str:
    """安全 eval（生产请用 ast.parse 校验）"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"ERROR: {e}"

# === Reward ===
def reward_fn(prediction: str, gold: str) -> float:
    # exact match
    return 1.0 if str(prediction).strip() == str(gold).strip() else 0.0

# === Episode runner ===
def run_episode(task: dict, policy, max_steps=8) -> dict:
    trajectory = {"id": task["id"], "steps": [], "gold_answer": task["gold"]}
    obs = task["question"]

    for t in range(max_steps):
        action = policy.act(obs, trajectory["steps"])  # 你的 LLM
        if action["type"] == "tool_call":
            result = calculator(action["expression"])
            trajectory["steps"].append({"t": t, "obs": obs,
                                        "action_type": "tool_call",
                                        "action": action, "result": result,
                                        "reward": 0.0})
            obs = result
        elif action["type"] == "final":
            r = reward_fn(action["answer"], task["gold"])
            trajectory["steps"].append({"t": t, "obs": obs,
                                        "action_type": "final",
                                        "action": action,
                                        "reward": r})
            trajectory["success"] = (r == 1.0)
            trajectory["total_reward"] = r
            return trajectory

    trajectory["success"] = False
    trajectory["total_reward"] = 0.0
    return trajectory

# === Baselines ===
def baseline_no_tool(task, llm):
    """直接问 LLM 算"""
    return llm.answer(task["question"])

def baseline_react(task, llm):
    """ReAct prompt + calculator"""
    return run_episode(task, react_policy(llm))
```

---

## ⚠️ 5.1.8 局限与提醒

| 局限 | 说明 |
|---|---|
| **不直击 Code 方向** | 学到的"工具调用"思路通用，但不涉及 repo navigation / patch / pytest |
| **任务太简单** | 数学题对现代 LLM 来说已基本不挑战 |
| **失败模式少** | 主要是 parser / 解析问题，没有 reward hacking 大舞台 |

> 💡 **你的方向是 Code Agentic RL**，强烈建议**直接做** [Toy Bug-Fix Agent Gym](02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)。
> Calculator toy 学到的东西 Bug-Fix Gym 全部覆盖，且更贴近你目标。

---

## 📌 5.1 节要点

| 项 | 内容 |
|---|---|
| 任务 | 多步算术 / 单位转换 |
| 工具 | calculator |
| Reward | exact match |
| Baseline | No-tool + ReAct |
| 进阶 | DPO（偏好对）/ GRPO（多采样组内） |
| 替代项目 | **Bug-Fix Gym（推荐你做这个）** |

---

## 🔗 延伸阅读

- 下一节（**推荐**）：[02-Toy-Bug-Fix-Agent-Gym（核心）](02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)
- OAR 基础：[03-Observation-Action-Reward定义](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md)
- 训练实战：[03-SFT训练实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/03-SFT%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)
- 原始资料：`07-toy-project-spec.md`

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-Toy-Bug-Fix-Agent-Gym（核心）](02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)
