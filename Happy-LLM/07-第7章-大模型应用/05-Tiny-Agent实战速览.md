---
tags: [Happy-LLM, 第7章, Agent, 实战速览, tool_calls]
chapter: 7
section: 7.3.3
---

# 7.3.3 Tiny-Agent 实战速览

⬅ [04-LLM-Agent原理](04-LLM-Agent%E5%8E%9F%E7%90%86.md)	|	➡ [06-章节小结](06-%E7%AB%A0%E8%8A%82%E5%B0%8F%E7%BB%93.md)

> ⚡ **本节速览**：用 OpenAI `tool_calls` 写一个 Agent，**核心逻辑 ~60 行**。

---

## 🎯 我们要做什么

实现一个**任务导向型 Agent**，能：
- 知道当前时间
- 做数学运算
- 比较数字大小
- 数字符串中字母出现次数

→ 用户问"strawberry 里有几个 r？"，Agent 自动调用 `count_letter_in_string` 工具回答。

---

## 🏗 四步走

```
Step 1: 初始化 OpenAI 客户端 + 模型
Step 2: 定义工具函数（带 docstring）
Step 3: 构造 Agent 类（核心）
Step 4: 运行交互循环
```

---

## 🐍 Step 1：初始化客户端

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",  # 替换为你的 API Key
    base_url="https://api.siliconflow.cn/v1",  # 使用硅基流动
)

model_name = "Qwen/Qwen2.5-32B-Instruct"
```

> 💡 用国内服务（如硅基流动）避免网络问题。

---

## 🐍 Step 2：定义工具函数（带文档）

```python
# src/tools.py
from datetime import datetime
import wikipedia

def get_current_datetime() -> str:
    """
    获取当前日期和时间。
    :return: 当前日期和时间的字符串表示。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def count_letter_in_string(a: str, b: str) -> str:
    """
    统计字符串中某个字母的出现次数。
    :param a: 要搜索的字符串。
    :param b: 要统计的字母。
    :return: 字母在字符串中出现的次数。
    """
    return str(a.count(b))


def add(a: float, b: float) -> float:
    """
    两数相加。
    """
    return a + b


def compare(a: float, b: float) -> str:
    """
    比较两数大小，返回较大者。
    """
    return f"{max(a, b)} 比 {min(a, b)} 更大"


def search_wikipedia(query: str) -> str:
    """
    在维基百科中搜索指定查询的前三个页面摘要。
    """
    page_titles = wikipedia.search(query)
    summaries = []
    for title in page_titles[:3]:
        try:
            page = wikipedia.page(title=title, auto_suggest=False)
            summaries.append(f"页面: {title}\n摘要: {page.summary}")
        except Exception:
            pass
    return "\n\n".join(summaries) if summaries else "维基百科没有搜索到合适的结果"
```

### 🔑 关键设计

- **每个函数必须有 docstring**：会被自动解析成 OpenAI tool schema
- **参数加类型注解**：`def add(a: float, b: float)` → JSON schema 自动推断
- **return 一律是字符串**：方便给 LLM 看

---

## 🐍 自动转 OpenAI Tool Schema

```python
# src/utils.py
import inspect

def function_to_json(func) -> dict:
    """
    把 Python 函数自动转成 OpenAI tool schema。
    解析签名 + docstring → JSON。
    """
    # 解析参数和类型
    sig = inspect.signature(func)
    parameters = {}
    required = []
    for name, param in sig.parameters.items():
        # 类型映射...
        parameters[name] = {"type": "string"}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

→ 这样**不用手写 schema**，新增工具只需写函数 + docstring。

---

## 🐍 Step 3：Agent 类（核心）

```python
# src/core.py
from openai import OpenAI
from utils import function_to_json
from tools import get_current_datetime, add, compare, count_letter_in_string

SYSTEM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""


class Agent:
    def __init__(self, client: OpenAI, model: str, tools: list, verbose: bool = True):
        self.client = client
        self.model = model
        self.tools = tools
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.verbose = verbose

    def get_tool_schema(self) -> list:
        """所有工具转成 OpenAI 格式"""
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        """⭐ 真正执行工具函数"""
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        # ⚠️ 安全提示：实际生产不要用 eval，应该用查表
        function_result = eval(f"{function_name}(**{function_args})")

        return {
            "role": "tool",
            "content": function_result,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt: str) -> str:
        """⭐ 核心方法：处理一次用户输入"""
        self.messages.append({"role": "user", "content": prompt})

        # 第 1 次 LLM 调用：让模型决定要不要调工具
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )

        # ⭐ 检查模型是否要调工具
        if response.choices[0].message.tool_calls:
            # 把 LLM 的"想调工具"消息记录到历史
            self.messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content,
            })

            # 执行每个工具调用
            for tool_call in response.choices[0].message.tool_calls:
                tool_result = self.handle_tool_call(tool_call)
                self.messages.append(tool_result)

            if self.verbose:
                tool_names = [tc.function.name for tc in
                             response.choices[0].message.tool_calls]
                print("调用工具：", tool_names)

            # ⭐ 第 2 次 LLM 调用：把工具结果给模型，生成最终回复
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # 记录最终回复
        self.messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content,
        })

        return response.choices[0].message.content
```

### 🔑 核心理解：两次 LLM 调用

```
第 1 次 LLM 调用:
   输入: 用户问题 + 工具列表
   输出:
      - 如果不需要工具 → 直接回答
      - 如果需要工具 → 返回 tool_calls（"我要调这个工具，参数是...")

第 2 次 LLM 调用（仅当调了工具）:
   输入: 用户问题 + 工具列表 + 工具调用结果
   输出: 基于工具结果的最终回答
```

→ 这就是 **Function Calling 的标准两步走**。

---

## 🐍 Step 4：运行 Agent

```python
# demo.py
if __name__ == "__main__":
    client = OpenAI(
        api_key="YOUR_API_KEY",
        base_url="https://api.siliconflow.cn/v1",
    )

    agent = Agent(
        client=client,
        model="Qwen/Qwen2.5-32B-Instruct",
        tools=[get_current_datetime, add, compare, count_letter_in_string],
        verbose=True,
    )

    while True:
        prompt = input("\033[94mUser: \033[0m")
        if prompt.lower() == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)
```

---

## 🎮 真实交互示例

```
User: 你好
Assistant: 你好！有什么可以帮助你的吗？

User: 9.12 和 9.2 哪个更大？
调用工具： ['compare']
Assistant: 9.2 比 9.12 更大。

User: strawberry 里有几个 r？
调用工具： ['count_letter_in_string']
Assistant: 单词 "strawberry" 中有 3 个字母 'r'。

User: 现在几点了？
调用工具： ['get_current_datetime']
Assistant: 当前的时间是 2025 年 4 月 26 日 17:01:33。

User: exit
```

→ Agent **自动识别**何时调用工具，**自动选**调用哪个工具，**自动整合**结果回答。

---

## 💡 这个 Agent 体现了什么 Agent 能力？

| 能力 | 体现 |
|---|---|
| **目标理解** | 理解用户问题（"几点了"= 查时间） |
| **工具使用** | 自动调对应工具 |
| **记忆** | `self.messages` 保存对话历史 |
| **规划** | 简单：单步工具调用 |
| **反思** | 无（这是任务导向型，不需要） |

→ 这是**最简单的任务导向型 Agent**。
进阶版可以加 ReAct 循环、Plan-and-Execute、多 Agent 协作等。

---

## 🚀 升级到更复杂 Agent

| 进阶方向 | 工具 |
|---|---|
| **多步推理** | LangChain `AgentExecutor` |
| **多 Agent 协作** | AutoGen / CrewAI |
| **可视化** | 改成 Streamlit / Gradio Web UI |
| **加 RAG** | 工具里加一个 `search_knowledge_base` |
| **加记忆** | 用向量库存历史对话 |

---

## ⚠️ 小白避坑

1. **`eval` 危险，生产用查表**
   ```python
   tool_map = {"add": add, "compare": compare, ...}
   tool_map[function_name](**function_args)
   ```
2. **docstring 一定要写**
   - LLM 通过 docstring 知道工具能干啥
3. **return 值要是字符串**
   - 不然 JSON 序列化会出错
4. **工具调用失败要处理**
   - try/except 包起来，把错误返回给 LLM
5. **`temperature` 设小（如 0.1）**
   - Agent 要确定性，不要随机

---

## 📌 7.3.3 节要点

- Tiny-Agent = **OpenAI client + 工具函数 + Agent 类（两次 LLM 调用）**
- 核心代码 **~60 行**
- 关键技术：**Function Calling 两步走**
- 工业级用 LangChain / AutoGen / Dify

---

## 🔗 延伸阅读

- 上一节：[04-LLM-Agent原理](04-LLM-Agent%E5%8E%9F%E7%90%86.md)
- 下一节：[06-章节小结](06-%E7%AB%A0%E8%8A%82%E5%B0%8F%E7%BB%93.md)
- 完整代码：[Happy-LLM Tiny-Agent](https://github.com/datawhalechina/happy-llm)

---

⬅ [04-LLM-Agent原理](04-LLM-Agent%E5%8E%9F%E7%90%86.md)	|	➡ [06-章节小结](06-%E7%AB%A0%E8%8A%82%E5%B0%8F%E7%BB%93.md)
