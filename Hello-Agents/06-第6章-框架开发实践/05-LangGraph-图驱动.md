---
tags: [Hello-Agents, 第6章, LangGraph, 状态图, 图驱动]
chapter: 6
section: 6.5
---

# 6.5 LangGraph —— 图驱动 + 灵活控制流

⬅ [04-CAMEL-角色扮演](04-CAMEL-%E8%A7%92%E8%89%B2%E6%89%AE%E6%BC%94.md)	|	➡ [06-四大框架对比](06-%E5%9B%9B%E5%A4%A7%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94.md)

> **LangChain 生态扩展**,把 Agent 流程建模为**状态图**(State Graph)。
> **天然支持循环、分支、回溯** —— **最适合 Reflection / Plan 类复杂工作流**。

---

## 🎬 故事比喻:用流程图编辑器画 Agent

| 框架 | 比喻 |
|---|---|
| **AutoGen** | Slack 工作群(对话驱动) |
| **AgentScope** | 企业级消息总线(事件驱动) |
| **CAMEL** | 两个专家结对(角色驱动) |
| **LangGraph** | **流程图编辑器**(图驱动) |

→ LangGraph **像 Visio 画流程图**,但**每个节点是 AI**。

---

## 🎯 6.5.1 LangGraph 的核心理念

### 革命性洞察:**Agent 流程 = 图**

```
传统链式结构(LangChain Chain):
    输入 → A → B → C → 输出
    
    特点:单向流动,信息只能往前走
    问题:无法处理 循环 / Reflection / 复杂分支

LangGraph 图结构:
    ┌───┐     ┌───┐
    │ A │ ──▶ │ B │ ──┐
    └───┘     └───┘   │
       ▲              ▼
       │            ┌───┐
       └────────────│ C │  ← 可以循环回 A
                    └───┘
                       │
                       ▼
                     输出
```

→ **天然支持循环 / 分支 / 回溯** —— **就像真实软件流程图**。

---

## 🧱 6.5.2 三大核心概念

### 概念 1:**节点(Node)**

> 每一步操作 = 一个节点

```python
def call_llm(state):
    """LLM 调用节点"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state):
    """工具调用节点"""
    tool_result = tool.invoke(state["last_action"])
    return {"tool_output": tool_result}
```

→ 节点可以是:**LLM 调用 / 工具执行 / 数据处理 / 任何 Python 函数**。

### 概念 2:**边(Edge)**

> 定义节点之间的**跳转逻辑**

```python
# 简单边:从 A 到 B
graph.add_edge("A", "B")

# 条件边:根据状态决定走哪
graph.add_conditional_edges(
    "decide_node",
    lambda state: "use_tool" if state["need_tool"] else "respond",
    {
        "use_tool": "tool_node",
        "respond": "respond_node"
    }
)
```

### 概念 3:**状态(State)**

> 在节点间流动的"数据载体"

```python
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[str]
    last_action: str
    tool_output: str
```

→ **每个节点修改 state 的一部分,下游节点接收完整 state**。

---

## 🆚 6.5.3 LangGraph vs 其他框架

### 与 LangChain 的关系

```
LangChain (第 1 代)
   ↓ 单向链式 → 处理简单流程
   ↓ 不支持循环 → Reflection 难实现
   ↓
LangGraph (扩展)
   ↓ 状态图 → 任意控制流
   ↓ 支持循环 → Reflection 轻松实现
```

→ **LangGraph 不是替代 LangChain**,而是**补足复杂控制流的能力**。

### 与其他多 Agent 框架的差异

| 框架 | 控制流方式 |
|---|---|
| **AutoGen** | 顺序 RoundRobin(线性) |
| **CAMEL** | 双 Agent 对话(双向) |
| **AgentScope** | Pipeline(多种,但需手写代码) |
| **LangGraph** | **任意图**(可视化 + 灵活) |

→ **LangGraph 是控制流最灵活的框架**,适合复杂场景。

---

## 🛠 6.5.4 实战:三步问答助手

### 任务设定

构建一个**三步问答 Agent**:
1. **检索**(Retrieve):用搜索引擎查相关信息
2. **判断**(Decide):看检索结果是否足够,不够再检索
3. **回答**(Answer):基于检索结果给最终答案

```
            ┌──────────────────────────┐
            │   用户问题                  │
            └──────────────┬──────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │   Retrieve 节点  │  ← 检索
                  └────────┬────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │   Decide 节点    │  ← 是否够?
                  └────────┬────────┘
                            │
            ┌───────────────┼───────────────┐
            │ 不够                            │ 足够
            ▼                                ▼
   (回到 Retrieve,循环检索)           ┌──────────┐
                                       │ Answer 节点│
                                       └──────────┘
                                           │
                                           ▼
                                       最终答案
```

→ **典型的"图驱动 Agent"**:决策节点 + 循环 + 分支。

### 关键代码

#### Step 1:**定义 State**

```python
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages

class QAState(TypedDict):
    question: str                        # 用户问题
    retrieved_docs: List[str]            # 检索到的文档
    retrieval_count: int                 # 检索次数
    is_enough: bool                      # 是否足够
    final_answer: str                    # 最终答案
```

→ **State 是图的"血液"**,数据通过它在节点间流动。

#### Step 2:**定义节点**

```python
def retrieve(state: QAState) -> QAState:
    """检索节点:调用搜索引擎"""
    print(f"🔍 第 {state['retrieval_count']+1} 次检索: {state['question']}")
    docs = search_engine.query(state['question'])
    return {
        "retrieved_docs": state['retrieved_docs'] + docs,
        "retrieval_count": state['retrieval_count'] + 1
    }

def decide(state: QAState) -> QAState:
    """决策节点:让 LLM 判断信息是否足够"""
    prompt = f"问题:{state['question']}\n已有信息:{state['retrieved_docs']}\n这些信息足够回答吗?(yes/no)"
    response = llm.invoke(prompt)
    is_enough = "yes" in response.lower()
    return {"is_enough": is_enough}

def answer(state: QAState) -> QAState:
    """回答节点:生成最终答案"""
    prompt = f"问题:{state['question']}\n依据:{state['retrieved_docs']}\n请回答:"
    response = llm.invoke(prompt)
    return {"final_answer": response}
```

#### Step 3:**构建图**

```python
from langgraph.graph import StateGraph, END

# 创建图
graph = StateGraph(QAState)

# 添加节点
graph.add_node("retrieve", retrieve)
graph.add_node("decide", decide)
graph.add_node("answer", answer)

# 添加边
graph.set_entry_point("retrieve")           # 起点:检索
graph.add_edge("retrieve", "decide")        # 检索后判断

# ⭐ 条件边:核心控制流
def route_after_decide(state):
    if state["retrieval_count"] >= 3:
        return "answer"   # 检索 3 次了,强制回答
    return "answer" if state["is_enough"] else "retrieve"

graph.add_conditional_edges(
    "decide",
    route_after_decide,
    {
        "retrieve": "retrieve",   # 不够 → 回去再检索(循环!)
        "answer": "answer"        # 够了 → 去回答
    }
)

graph.add_edge("answer", END)

# 编译图
app = graph.compile()
```

#### Step 4:**运行**

```python
# 调用图
result = app.invoke({
    "question": "2024 年中国 GDP 是多少?",
    "retrieved_docs": [],
    "retrieval_count": 0,
    "is_enough": False,
    "final_answer": ""
})

print(result["final_answer"])
```

→ **图自动按定义的边跳转**,你不用写主循环。

---

## ⭐ 6.5.5 LangGraph 的核心优势:控制流灵活

### 优势 1:**天然支持循环**

```python
# 经典场景:Reflection 自我修订循环
graph.add_conditional_edges(
    "critic",
    lambda s: "execute" if s["needs_revision"] else "end",
    {"execute": "execute_node", "end": END}
)
```

→ 一个条件边搞定 Reflection。
→ 在 [第 4 章 Reflection](../04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/04-Reflection%E8%8C%83%E5%BC%8F.md) 我们用 for 循环模拟,LangGraph **直接用图天然表达**。

### 优势 2:**天然支持分支**

```python
# 根据用户问题类型路由到不同处理节点
graph.add_conditional_edges(
    "classify",
    lambda s: s["question_type"],
    {
        "math": "math_solver",
        "code": "code_writer",
        "chat": "casual_chat"
    }
)
```

### 优势 3:**Human-in-the-loop**

```python
from langgraph.checkpoint.memory import MemorySaver

# 加入检查点,可暂停 + 人工干预
graph_with_human = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["answer"]   # 在生成答案前暂停,等人工确认
)
```

→ **真实生产中"人机协作"的优雅实现**。

### 优势 4:**可视化**

```python
# 自动画图
graph.get_graph().draw_mermaid_png(output_file_path="agent.png")
```

→ **得到 mermaid 流程图**,**老板汇报神器**。

---

## 🎯 6.5.6 LangGraph 的优势与局限

### ✅ 优势

| 优势 | 解释 |
|---|---|
| **控制流最灵活** | 循环 / 分支 / 回溯 / 并行 任意组合 |
| **Reflection 完美适配** | 第 4 章手写循环 → 一个条件边 |
| **状态管理清晰** | TypedDict + 节点自动合并 |
| **可视化** | 自动生成流程图 |
| **Human-in-the-loop** | 内置检查点 |
| **生态强大** | LangChain 工具/RAG/Memory **直接用** |
| **持久化** | Checkpointer 支持状态保存 |

### ⚠️ 局限

| 局限 | 解释 |
|---|---|
| **学习曲线陡** | 需要"图"思维 |
| **简单任务过度** | 单步 Agent 用图太复杂 |
| **State 设计有讲究** | 字段定义不好 → 节点合并出错 |
| **依赖 LangChain 生态** | 想用纯 LangGraph 不太可能 |
| **调试稍复杂** | 多分支 + 循环 → 容易迷失 |

---

## 🌟 LangGraph 的"杀手锏"应用

### 应用 1:**Reflection Agent**(完美匹配)

```
   生成 ── 反思 ──┐
     ▲             │
     │             ▼
     └── 是否够好? (条件边)
```

### 应用 2:**Plan-and-Execute**(完美匹配)

```
   规划 → 执行步骤 1 → 执行步骤 2 → ...
              │              │
              ▼              ▼
          失败重试       失败重试
```

### 应用 3:**Multi-Agent + 决策路由**

```
   接收问题 → 分类 ─┬─→ 专家 A
                    ├─→ 专家 B
                    └─→ 专家 C
                         │
                         ▼
                    汇总回答
```

### 应用 4:**长时运行 Agent**(配 checkpointer)

```
   工作 → 检查点保存 → 工作 → 检查点保存 → ...
                ↑                            
                └── 中断恢复
```

---

## ⚠️ 小白避坑

1. **State 字段命名要小心**
   - 不同节点修改同一字段 → 后写覆盖前面
   - 用 `Annotated[List, add]` 实现累积
2. **条件边逻辑要清晰**
   - 复杂的 `if-else` → 分多个条件边
3. **循环必须有终止条件**
   - 否则无限循环烧 LLM 调用
4. **不要把所有逻辑塞进一个节点**
   - 一个节点干一件事,符合"图"思维
5. **善用可视化**
   - 设计完图先画出来,**人脑过一遍**

---

## 🎯 6.5.7 何时选 LangGraph?

```
✅ 适合:
- Reflection / 自我修订循环
- Plan-and-Execute 多步规划
- 需要分支路由(分类后处理)
- Human-in-the-loop 场景
- 长时运行任务(检查点)
- 已经在用 LangChain 生态

❌ 不适合:
- 简单单步 Agent(过度)
- 纯多 Agent 协作(不如 AutoGen 直观)
- 不愿学"图"思维
```

---

## 📌 6.5 节要点

| 知识点 | 一句话 |
|---|---|
| **核心哲学** | 图驱动 + 灵活控制流 |
| **3 大概念** | 节点(Node)+ 边(Edge)+ 状态(State) |
| **杀手锏** | **天然支持循环和分支**(Reflection 完美) |
| **State 管理** | TypedDict 定义,节点局部修改 |
| **Human-in-the-loop** | 内置 checkpointer |
| **可视化** | 自动生成 mermaid 图 |
| **典型应用** | Reflection / Plan / 多专家路由 |
| **底层** | LangChain 生态扩展 |

---

## 🔗 延伸阅读

- 上一节:[04-CAMEL-角色扮演](04-CAMEL-%E8%A7%92%E8%89%B2%E6%89%AE%E6%BC%94.md)
- 下一节:[06-四大框架对比](06-%E5%9B%9B%E5%A4%A7%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94.md) —— 总结选型
- 第 4 章对应:[Reflection 范式](../04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/04-Reflection%E8%8C%83%E5%BC%8F.md)
- 官方:https://langchain-ai.github.io/langgraph/

---

⬅ [04-CAMEL-角色扮演](04-CAMEL-%E8%A7%92%E8%89%B2%E6%89%AE%E6%BC%94.md)	|	➡ [06-四大框架对比](06-%E5%9B%9B%E5%A4%A7%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94.md)
