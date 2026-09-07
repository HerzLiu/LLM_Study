---
tags: [Hello-Agents, 第4章, Plan-and-Solve, 范式, 实战]
chapter: 4
section: 4.3
---

# 4.3 Plan-and-Solve 范式(完整实现)

⬅ [02-ReAct范式](02-ReAct%E8%8C%83%E5%BC%8F.md)	|	➡ [04-Reflection范式](04-Reflection%E8%8C%83%E5%BC%8F.md)

> **"三思而后行"** 的范式 —— 先生成完整计划,再严格执行。
> Lei Wang 于 2023 年提出 [2]。

---

## 🎬 故事比喻:侦探 vs 建筑师

| 范式 | 比喻 | 工作方式 |
|---|---|---|
| **ReAct** | **侦探** | 边查线索边推理,随时调整 |
| **Plan-and-Solve** | **建筑师** | **先画完整蓝图**,再严格按图施工 |

→ 你日常用的"**Agent 模式**"(Claude / GPT-4 / Cursor 的高级模式)很多融入了这种思想。

---

## 🆚 4.3.1 ReAct vs Plan-and-Solve 本质差异

| 维度           | **ReAct**    | **Plan-and-Solve**  |
| ------------ | ------------ | ------------------- |
| **思考-行动关系**  | 紧密耦合(每步都想)   | **解耦**(先全想完再做)      |
| **规划粒度**     | 单步           | **全局**              |
| **适合任务**     | 需要探索的开放任务    | **结构清晰的多步任务**       |
| **抗"原地打转"**  | ❌ 容易陷入       | ✅ 有完整蓝图             |
| **抗中途变化**    | ✅ 能动态调整      | ⚠️ 计划僵化             |
| **LLM 调用次数** | T-A-O 每轮 1 次 | **规划 1 次 + 每步 1 次** |

---

## 🔧 4.3.2 工作原理(两阶段)

```
┌─────────────────────────────────────┐
│   阶段 1: 规划阶段 (Planner)        │
│   输入: 完整问题                      │
│   输出: 分步骤计划                    │
│   ["步骤 1", "步骤 2", ...]          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   阶段 2: 执行阶段 (Executor)        │
│   for 步骤 in 计划:                   │
│     基于历史结果执行当前步骤            │
│     记录结果到历史                     │
│   最终答案 = 最后一步的结果            │
└─────────────────────────────────────┘
```

### 形式化表达

**规划**:
```
Plan = [s_1, s_2, ..., s_n] = LLM_planner(question)
```

**执行**(第 i 步):
```
result_i = LLM_executor(question, Plan, results_{1...i-1}, s_i)
```

最终答案 = `result_n`(最后一步的结果)。

---

## 🎯 4.3.3 实战目标

我们这次**不用工具**,**纯靠提示工程**做一个逻辑推理任务:

> **"一个水果店周一卖出 15 个苹果。周二是周一的两倍。周三比周二少 5 个。三天总共卖了多少?"**

这个题:
- **结构清晰**(可分解为 4 个子步骤)
- **逻辑链条明确**
- **不需要外部信息**

→ 完美匹配 Plan-and-Solve 的优势场景。

---

## 🐍 4.3.4 Planner(规划器)

### Prompt 模板

```python
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的 AI 规划专家。你的任务是将用户提出的复杂问题分解成
一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务,并且严格按照逻辑顺序排列。

你的输出必须是一个 Python 列表,其中每个元素都是描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python 与 ``` 作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""
```

### 设计要点

| 设计点 | 解释 |
|---|---|
| **角色设定** | "顶级 AI 规划专家" → 激发专业能力 |
| **任务描述** | 明确"分解 + 独立 + 可执行 + 有序" |
| **格式约束** | **强制 Python 列表 + 代码块标记** → 方便代码解析 |

→ **格式约束是 Planner 的灵魂**。

### Planner 类实现

```python
import ast

class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """根据问题生成行动计划"""
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        # ⭐ 解析 LLM 输出的 Python 列表字符串
        try:
            # 提取 ```python 和 ``` 之间的内容
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # 用 ast.literal_eval 安全地解析为 Python 对象
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划失败: {e}")
            return []
```

### 📝 关键解读

#### 🔑 为什么用 `ast.literal_eval` 而不是 `eval`?

```python
plan = ast.literal_eval(plan_str)
```

**安全!**
- `eval()` 会执行**任意 Python 代码**(危险,LLM 可能输出恶意代码)
- `ast.literal_eval()` 只接受 **Python 字面值**(列表、字典、数字、字符串等)
- 遇到代码执行尝试会**报错**

→ **生产代码里永远别用 `eval()` 解析 LLM 输出**。

---

## 🐍 4.3.5 Executor(执行器)

### Prompt 模板

```python
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的 AI 执行专家。你的任务是严格按照给定的计划,一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤",并仅输出该步骤的最终答案,不要输出任何额外的解释。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""
```

### 设计要点

| 必含信息        | 作用             |
| ----------- | -------------- |
| **原始问题**    | 让 LLM 始终记得最终目标 |
| **完整计划**    | 让 LLM 知道当前步骤在哪 |
| **历史步骤与结果** | 提供上下文(关键!)     |
| **当前步骤**    | 明确这一步该做啥       |

### Executor 类实现

```python
class Executor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """按计划逐步执行"""
        history = ""

        print("\n--- 正在执行计划 ---")

        for i, step in enumerate(plan):
            print(f"\n-> 步骤 {i+1}/{len(plan)}: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages) or ""

            # ⭐ 关键:把当前步骤和结果加入历史
            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"

            print(f"✅ 步骤 {i+1} 完成,结果: {response_text}")

        # 最后一步的结果就是最终答案
        return response_text
```

### 📝 关键解读:**状态管理**

```python
history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
```

→ **状态管理是 Executor 的核心**。
- 每步**追加历史**
- 下一步 LLM 看到**完整历史**
- 保证**信息在子任务间流动**

否则 LLM 不知道前面算出了什么,无法做下一步。

---

## 🤖 4.3.6 整合:PlanAndSolveAgent

```python
class PlanAndSolveAgent:
    """协调 Planner 和 Executor 的主 Agent"""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)

    def run(self, question: str):
        """完整流程:先规划,后执行"""
        print(f"\n--- 开始处理 ---\n问题: {question}")

        # 阶段 1: 规划
        plan = self.planner.plan(question)
        if not plan:
            print("\n--- 任务终止 ---\n无法生成计划")
            return

        # 阶段 2: 执行
        final_answer = self.executor.execute(question, plan)

        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
```

### 📝 设计哲学:**组合优于继承**

```python
self.planner = Planner(llm_client)
self.executor = Executor(llm_client)
```

`PlanAndSolveAgent` 本身**没复杂逻辑**,只是**协调者(Orchestrator)**。

→ 这是好的工程设计:
- **职责单一**(Planner 只规划,Executor 只执行)
- **可独立测试**
- **可灵活替换**(比如换个更强的 Planner)

---

## 🎮 4.3.7 运行实例

### 真实运行结果

```
问题: 一个水果店周一卖出 15 个苹果。周二是周一的两倍。
      周三比周二少 5 个。三天总共卖了多少?

--- 正在生成计划 ---
✅ 计划已生成:
[
  "计算周一卖出的苹果数量: 15 个",
  "计算周二卖出的苹果数量: 周一数量 × 2 = 15 × 2 = 30 个",
  "计算周三卖出的苹果数量: 周二数量 - 5 = 30 - 5 = 25 个",
  "计算三天总销量: 周一 + 周二 + 周三 = 15 + 30 + 25 = 70 个"
]

--- 正在执行计划 ---
-> 步骤 1/4: 计算周一卖出...
✅ 结果: 15

-> 步骤 2/4: 计算周二卖出...
✅ 结果: 30

-> 步骤 3/4: 计算周三卖出...
✅ 结果: 25

-> 步骤 4/4: 计算总销量...
✅ 结果: 70

--- 任务完成 ---
最终答案: 70
```

→ **清晰、可追溯、答案正确**。

---

## ⭐ 4.3.8 Plan-and-Solve 的优势与局限

### ✅ 优势

| 优势 | 解释 |
|---|---|
| **全局目标一致性** | 不会跑偏 |
| **结构化推理** | 每步都明确,**易调试** |
| **避免原地打转** | ReAct 的常见问题 |
| **适合长任务** | 几十步的计划也能稳定执行 |

### ⚠️ 局限

| 局限 | 解释 |
|---|---|
| **计划僵化** | 中途遇变化,**难以调整** |
| **首次规划压力大** | Planner 要一次想到所有步骤,**对 LLM 能力要求高** |
| **计划错则全错** | Planner 出错,后续全跑偏 |
| **不适合开放探索** | 不知道接下来会发现什么的场景不适合 |

---

## 🔀 4.3.9 与 ReAct 的混合方案

实际生产中,**两者常常混用**:

| 混合模式                 | 思路                             |
| -------------------- | ------------------------------ |
| **Plan + ReAct 子任务** | 高层用 Plan,**每个子任务用 ReAct 动态执行** |
| **Plan 后允许 Replan**  | 执行中发现计划错了,**触发 Re-planning**   |
| **多 Plan 投票**        | 生成 N 个不同 Plan,**让 LLM 投票选最优**  |

→ 这就是为什么 **LangGraph** 这种支持复杂控制流的框架越来越火(第 6 章会讲)。

---

## ⚠️ 小白避坑

1. **Planner 输出格式必须严格**
   - 不严格 → `ast.literal_eval` 失败
   - 加 few-shot 示例提升稳定性
2. **Executor 别让步骤太长**
   - 一步太复杂 → LLM 处理不好
   - 应该让 Planner 把任务**切细**
3. **历史太长会爆 token**
   - 长任务后期 prompt 越来越长
   - 解法:**摘要历史** + **保留关键结果**
4. **`ast.literal_eval` 仅接受字面值**
   - 不能解析含函数调用的字符串
   - LLM 输出要纯净

---

## 📌 4.3 节要点

| 知识点 | 一句话 |
|---|---|
| **两阶段** | 规划(Planner)+ 执行(Executor) |
| **格式约束** | Planner 输出 **Python 列表 + 代码块** |
| **`ast.literal_eval`** | **安全**解析 LLM 输出 |
| **状态管理** | Executor 累积历史,**信息在步骤间流动** |
| **组合优于继承** | Agent = 协调者,Planner/Executor 各司其职 |
| **适用** | 结构化多步任务,**逻辑链清晰** |
| **不适用** | 开放探索 / 中途多变 |

---

## 🔗 延伸阅读

- 上一节:[02-ReAct范式](02-ReAct%E8%8C%83%E5%BC%8F.md)
- 下一节:[04-Reflection范式](04-Reflection%E8%8C%83%E5%BC%8F.md) —— "做完再想"
- 进阶混合:[第 6 章 LangGraph](../06-%E7%AC%AC6%E7%AB%A0-%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E5%AE%9E%E8%B7%B5/05-LangGraph-%E5%9B%BE%E9%A9%B1%E5%8A%A8.md)

---

⬅ [02-ReAct范式](02-ReAct%E8%8C%83%E5%BC%8F.md)	|	➡ [04-Reflection范式](04-Reflection%E8%8C%83%E5%BC%8F.md)
