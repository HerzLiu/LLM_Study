---
tags: [Hello-Agents, 第7章, SimpleAgent, ReActAgent, Reflection, Plan-and-Solve]
chapter: 7
section: 7.4
---

# 7.4 Agent 范式的框架化实现

⬅ [03-框架核心接口](03-%E6%A1%86%E6%9E%B6%E6%A0%B8%E5%BF%83%E6%8E%A5%E5%8F%A3.md)	|	➡ [05-工具系统](05-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F.md)

> 把 [第 4 章](../04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) 手撕的 3 大范式,**用 HelloAgents 框架重构** + **新增 2 个范式**。

---

## 🎬 故事比喻:同样的菜,不同的厨房

| | 第 4 章手撕 | 第 7 章框架化 |
|---|---|---|
| 比喻 | 露天烧烤(每次都要从零搭) | **专业厨房**(灶台/锅碗/调料都齐) |
| 工作流 | 每个 Agent 都要写完整循环 | **继承基类 + 实现 run** |
| 工具调用 | 自己实现 | **统一工具系统** |
| 历史管理 | 每个都写一遍 | **基类自动管** |

→ **同样的菜,框架化做 = 更快 + 更稳 + 更易维护**。

---

## 🎯 7.4 节 5 大重构目标

| 目标 | 解释 |
|---|---|
| **提示词系统优化** | 从特定任务 → **通用化设计** |
| **接口标准化** | 所有 Agent 同样的初始化 + `run` 方法 |
| **可配置** | 支持自定义 prompt / config |
| **新增 2 个范式** | `SimpleAgent`(基础)+ `FunctionCallAgent`(OpenAI 标准) |
| **集成工具系统** | 统一 `tool_registry` 接入 |

---

## 🤖 7.4.1 SimpleAgent —— 基础对话

> 最基础的 Agent,**展示如何在框架基础上构建**。
> **支持可选工具调用** —— 新手友好。

### 关键代码骨架

```python
from typing import Optional, Iterator
from hello_agents import SimpleAgent, HelloAgentsLLM, Config, Message

class MySimpleAgent(SimpleAgent):
    """重写的简单对话 Agent"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        enable_tool_calling: bool = True,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """对话主流程"""
        # 1. 构建消息列表(系统提示 + 历史 + 当前)
        messages = [
            {"role": "system", "content": self._get_enhanced_system_prompt()},
            *[{"role": m.role, "content": m.content} for m in self._history],
            {"role": "user", "content": input_text},
        ]

        # 2. 无工具:简单对话
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            return response

        # 3. 有工具:进入工具调用循环
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)
```

### ⭐ 关键设计 1:**工具调用是可选的**

```python
enable_tool_calling=False → 退化为简单对话 Agent
enable_tool_calling=True  → 支持工具调用
```

→ **一个类两种模式**,**新手不用一上来就理解工具调用**。

### ⭐ 关键设计 2:**自定义工具调用格式**

```
[TOOL_CALL:tool_name:parameters]

例:
  [TOOL_CALL:search:Python 编程]
  [TOOL_CALL:memory:recall=用户信息]
  [TOOL_CALL:calculator:2+3*4]
```

→ HelloAgents 用**自定义标记格式**,**比 OpenAI Function Calling 更简单直观**(虽然不如它工业级)。

### ⭐ 关键设计 3:**便利方法 `add_tool`**

```python
agent = MySimpleAgent(name="助手", llm=llm)
agent.add_tool(CalculatorTool())   # 一行加工具
```

→ **极致简化**用户操作。

---

## 🔄 7.4.2 ReActAgent —— 框架化版本

> 把 [第 4 章 ReAct](../04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/02-ReAct%E8%8C%83%E5%BC%8F.md) 重构为框架版。

### 关键改进:**通用化 Prompt 模板**

```python
MY_REACT_PROMPT = """你是一个具备推理和行动能力的 AI 助手。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应,每次只能执行一个步骤:

Thought: 分析当前问题,思考需要什么信息或采取什么行动。
Action: 选择一个行动,格式必须是以下之一:
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含 Thought 和 Action 两部分
2. 工具调用必须严格遵循: 工具名[参数]
3. 只有当你确信有足够信息回答时,才使用 Finish
4. 如果工具返回的信息不够,继续使用其他工具

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""
```

→ 对比 [第 4 章版本](../04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/02-ReAct%E8%8C%83%E5%BC%8F.md#step-1%E7%B2%BE%E5%BF%83%E8%AE%BE%E8%AE%A1%E7%9A%84-prompt-%E6%A8%A1%E6%9D%BF),**改进点**:
- 角色定义更通用("具备推理和行动能力的 AI 助手")
- **多重提醒**避免 LLM 跑偏
- 模板**支持任何任务**,不绑死"旅行助手"

### 框架化结构

```python
import re
from typing import Optional, List
from hello_agents import ReActAgent, HelloAgentsLLM, Config, Message, ToolRegistry

class MyReActAgent(ReActAgent):
    """重写的 ReAct Agent"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None,    # ⭐ 支持自定义 prompt
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt or MY_REACT_PROMPT

    def run(self, input_text: str, **kwargs) -> str:
        """ReAct 主循环"""
        self.current_history = []

        for step in range(1, self.max_steps + 1):
            # 1. 构建 prompt
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str,
            )

            # 2. 调用 LLM
            response = self.llm.invoke([{"role": "user", "content": prompt}], **kwargs)

            # 3. 解析 + 执行
            thought, action = self._parse_output(response)

            if action.startswith("Finish"):
                return self._parse_action_input(action)

            tool_name, tool_input = self._parse_action(action)
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        return "已达到最大步数"
```

### ⭐ 改进 1:**`tool_registry` 替代 `tool_executor`**

```
第 4 章: self.tool_executor.getTool(tool_name)
第 7 章: self.tool_registry.execute_tool(tool_name, tool_input)
```

→ 框架统一的工具系统(详见 [7.5 节](05-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F.md))。

### ⭐ 改进 2:**`custom_prompt` 参数**

```python
my_prompt = "你是金融领域的 ReAct Agent..."
agent = MyReActAgent(name="金融助手", llm=llm, tool_registry=tools, custom_prompt=my_prompt)
```

→ **不同场景用不同 prompt**,**通用框架的灵魂**。

---

## 🔁 7.4.3 ReflectionAgent —— 框架化版本

把 [第 4 章 Reflection](../04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/04-Reflection%E8%8C%83%E5%BC%8F.md) 框架化:

```python
class MyReflectionAgent(ReflectionAgent):
    """框架化的 Reflection Agent"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        max_iterations: int = 3,
        **kwargs,
    ):
        super().__init__(name, llm, **kwargs)
        self.max_iterations = max_iterations

    def run(self, task: str, **kwargs) -> str:
        # 1. 执行
        draft = self._execute(task)

        # 2. 反思 + 修订循环
        for i in range(1, self.max_iterations + 1):
            feedback = self._reflect(task, draft)
            if "无需修改" in feedback:
                break
            draft = self._refine(task, draft, feedback)

        return draft
```

→ **核心逻辑没变**,但**继承自基类 + 用统一 LLM 接口**。

---

## 📋 7.4.4 PlanAndSolveAgent —— 框架化版本

```python
class MyPlanAndSolveAgent(PlanAndSolveAgent):
    """框架化的 Plan-and-Solve Agent"""

    def __init__(self, name, llm, **kwargs):
        super().__init__(name, llm, **kwargs)
        self.planner = Planner(llm)       # ⭐ 内部 Planner
        self.executor = Executor(llm)     # ⭐ 内部 Executor

    def run(self, question: str, **kwargs) -> str:
        # 1. 规划
        plan = self.planner.plan(question)

        # 2. 执行
        return self.executor.execute(question, plan)
```

→ **"协调者模式"** —— Agent 本身只协调,真活儿给 Planner / Executor。
→ 来自[第 4 章设计](../04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/03-Plan-and-Solve%E8%8C%83%E5%BC%8F.md#-436-%E6%95%B4%E5%90%88planandsolveagent)。

---

## ⚡ 7.4.5 FunctionCallAgent —— OpenAI 标准方式

> 新增的范式,**用 OpenAI Function Calling**(更标准 + 更稳定)。

### 与自定义工具调用格式对比

| | 自定义格式(SimpleAgent) | **Function Calling** |
|---|---|---|
| 格式 | `[TOOL_CALL:name:params]` | OpenAI 原生 JSON schema |
| 解析 | 正则匹配 | **OpenAI SDK 自动** |
| 稳定性 | 中(依赖 prompt) | **高**(模型原生支持) |
| 兼容性 | 仅自家框架 | **行业标准** |

### 简化实现

```python
class MyFunctionCallAgent(FunctionCallAgent):
    """基于 OpenAI Function Calling 的 Agent"""

    def run(self, input_text: str, **kwargs) -> str:
        # 1. 工具转 OpenAI 格式
        tools_schema = self.tool_registry.get_openai_tools_schema()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text},
        ]

        # 2. 多轮工具调用循环
        while True:
            response = self.llm.invoke_with_tools(
                messages=messages,
                tools=tools_schema,
            )

            # 3. 如果有工具调用
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    result = self.tool_registry.execute_tool(
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    messages.append({"role": "tool", "content": result})
                continue   # 把工具结果给 LLM,继续循环

            # 4. 没有工具调用,这是最终回答
            return response.content
```

→ **生产环境推荐用这种**,**比自定义格式稳定 10 倍**。

---

## 🎯 7.4.6 5 大范式对比

| 范式 | 适用场景 | 工具调用 | 复杂度 |
|---|---|---|---|
| **SimpleAgent** | 基础对话 / 入门 | 可选 | ⭐ |
| **ReActAgent** | 推理 + 工具调用 | 必须 | ⭐⭐ |
| **ReflectionAgent** | 高质量创作 / 代码 | 否 | ⭐⭐ |
| **PlanAndSolveAgent** | 结构化多步任务 | 否 | ⭐⭐⭐ |
| **FunctionCallAgent** | **生产级工具调用** | 必须 | ⭐⭐⭐ |

→ HelloAgents 框架**提供 5 种范式开箱即用**。

---

## ⚠️ 小白避坑

1. **不要把所有范式都用一遍**
   - 根据任务选 1-2 种
   - 例:简单查询用 `SimpleAgent`,复杂规划用 `PlanAndSolveAgent`
2. **自定义工具调用格式 vs OpenAI 标准**
   - **学习用**自定义(理解原理)
   - **生产用** Function Calling
3. **`custom_prompt` 是强大的扩展点**
   - 写好领域专属 prompt → 同一个 ReActAgent 适配任何领域
4. **继承基类时记得 `super().__init__()`**
   - 漏了 → 父类初始化没跑 → 各种诡异 bug

---

## 📌 7.4 节要点

| 知识点 | 一句话 |
|---|---|
| **5 种范式** | Simple/ReAct/Reflection/Plan-Solve/FunctionCall |
| **统一接口** | 同样的 `__init__` 和 `run` |
| **`custom_prompt`** | 不同领域用不同提示词 |
| **工具系统集成** | 通过 `tool_registry` 统一接入 |
| **协调者模式** | Plan-Solve = 协调 Planner + Executor |
| **FunctionCall** | OpenAI 原生,**生产首选** |

---

## 🔗 延伸阅读

- 上一节:[03-框架核心接口](03-%E6%A1%86%E6%9E%B6%E6%A0%B8%E5%BF%83%E6%8E%A5%E5%8F%A3.md)
- 下一节:[05-工具系统](05-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F.md) —— 工具基类 + 注册机制
- 第 4 章对应:[手撕原版](../04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

⬅ [03-框架核心接口](03-%E6%A1%86%E6%9E%B6%E6%A0%B8%E5%BF%83%E6%8E%A5%E5%8F%A3.md)	|	➡ [05-工具系统](05-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F.md)
