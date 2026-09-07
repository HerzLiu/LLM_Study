---
tags: [Hello-Agents, 第9章, ContextBuilder, GSSC, 流水线]
chapter: 9
section: 9.3
---

# 9.3 ContextBuilder + GSSC 流水线

⬅ [01-什么是上下文工程](01-%E4%BB%80%E4%B9%88%E6%98%AF%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B7%A5%E7%A8%8B.md)	|	➡ [03-NoteTool结构化笔记](03-NoteTool%E7%BB%93%E6%9E%84%E5%8C%96%E7%AC%94%E8%AE%B0.md)

> HelloAgents 的**上下文工程核心组件** —— 把上下文构建过程**工程化**。

---

## 🎬 故事比喻:菜单 vs 大杂烩

| | 大杂烩 | **精致菜单(GSSC)** |
|---|---|---|
| 做法 | 把所有食材一起煮 | 食材分类 → 选好的 → 摆盘 → 装盘控制 |
| 比喻 | 第 8 章的 Memory+RAG 直接拼 | **ContextBuilder 的 GSSC 流水线** |

→ **ContextBuilder = 上下文的"米其林大厨"**。

---

## 🎯 9.3.1 设计动机与目标

### 4 大目标

| 目标 | 含义 |
|---|---|
| **统一入口** | GSSC 抽象为可复用流水线,**减少重复模板代码** |
| **稳定形态** | 输出**固定骨架**的上下文模板,便于调试 + A/B 测试 |
| **预算守护** | 在 token 预算内保留高价值信息,**超限有兜底压缩** |
| **最小规则** | 不引入"来源/优先级"等分类,**避免复杂度爆炸** |

---

## 🏗 9.3.2 GSSC 流水线总览

> **GSSC = Gather → Select → Structure → Compress**(4 阶段)

```
┌─────────────────────────────────────┐
│  1. Gather (汇集)                    │
│     - 系统指令(最高优先)               │
│     - Memory 检索结果                 │
│     - RAG 检索结果                    │
│     - 对话历史(最近 N 条)              │
│     - 自定义信息包                     │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  2. Select (选择)                    │
│     - 评分: 相关性 × 0.7 + 新近性 × 0.3 │
│     - 过滤: min_relevance 阈值         │
│     - 贪心: token 预算内填充            │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  3. Structure (结构化)                │
│     [Role & Policies]                 │
│     [Task]                            │
│     [Evidence]                        │
│     [Context]                         │
│     [Output]                          │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  4. Compress (兜底压缩)               │
│     - 超限时分区压缩                   │
│     - 保留结构 + 截断 + 标记            │
└─────────────────────────────────────┘
```

---

## 📦 9.3.3 核心数据结构

### `ContextPacket`:候选信息包(统一单元)

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class ContextPacket:
    """候选信息包 —— 系统中信息的基本单元"""
    content: str                                     # 内容
    timestamp: datetime                              # 时间戳
    token_count: int                                 # token 数
    relevance_score: float = 0.5                     # 相关性 0.0-1.0
    metadata: Optional[Dict[str, Any]] = None        # 元数据

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # 确保相关性在合法范围
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))
```

→ **统一封装**:Memory / RAG / 历史 / 自定义 都变成 `ContextPacket`,**简化后续处理**。

### `ContextConfig`:配置管理

```python
@dataclass
class ContextConfig:
    """上下文构建配置"""
    max_tokens: int = 3000                # 最大 token 数
    reserve_ratio: float = 0.2             # 为系统指令预留比例
    min_relevance: float = 0.1             # 最低相关性阈值
    enable_compression: bool = True        # 是否启用压缩

    # ⭐ 核心:综合分数权重
    recency_weight: float = 0.3            # 新近性权重
    relevance_weight: float = 0.7          # 相关性权重

    def __post_init__(self):
        # 权重和必须等于 1.0
        assert abs(self.recency_weight + self.relevance_weight - 1.0) < 1e-6
```

#### 关键参数解读

| 参数 | 作用 |
|---|---|
| `max_tokens` | LLM 上下文窗口预算 |
| `reserve_ratio` | **预留给系统指令**(避免被挤占) |
| `min_relevance` | **过滤低质量信息** |
| `recency_weight + relevance_weight = 1` | 综合评分 |

---

## 🔧 9.3.4 阶段 1:Gather(多源汇集)

> 从多个来源汇集候选信息,**核心要求:容错性**。

```python
def _gather(
    self,
    user_query: str,
    conversation_history=None,
    system_instructions=None,
    custom_packets=None,
) -> List[ContextPacket]:
    """汇集所有候选信息"""
    packets = []

    # 1. 系统指令 (最高优先级,relevance=1.0 永远保留)
    if system_instructions:
        packets.append(ContextPacket(
            content=system_instructions,
            timestamp=datetime.now(),
            token_count=self._count_tokens(system_instructions),
            relevance_score=1.0,
            metadata={"type": "system_instruction", "priority": "high"}
        ))

    # 2. Memory 检索(try/except 容错)
    if self.memory_tool:
        try:
            memory_results = self.memory_tool.run({
                "action": "search",
                "query": user_query,
                "limit": 10,
                "min_importance": 0.3,
            })
            packets.extend(self._parse_memory_results(memory_results, user_query))
        except Exception as e:
            print(f"[WARNING] 记忆检索失败: {e}")  # ⭐ 不中断流程

    # 3. RAG 检索(同样容错)
    if self.rag_tool:
        try:
            rag_results = self.rag_tool.run({"action": "search", "query": user_query, "limit": 5})
            packets.extend(self._parse_rag_results(rag_results, user_query))
        except Exception as e:
            print(f"[WARNING] RAG 检索失败: {e}")

    # 4. 对话历史 (只取最近 5 条,避免占满)
    if conversation_history:
        for msg in conversation_history[-5:]:
            packets.append(ContextPacket(
                content=f"{msg.role}: {msg.content}",
                timestamp=msg.timestamp,
                token_count=self._count_tokens(msg.content),
                relevance_score=0.6,
                metadata={"type": "conversation_history", "role": msg.role}
            ))

    # 5. 自定义信息包
    if custom_packets:
        packets.extend(custom_packets)

    return packets
```

### 关键设计

| 设计 | 解释 |
|---|---|
| **容错** | 每个外部源 try/except,单源失败不影响整体 |
| **优先级** | 系统指令标记为高优先 + relevance=1.0 |
| **历史限制** | 只取最近 5 条(防止历史占满) |
| **元数据 type** | 标记类型,Structure 阶段会用 |

---

## 🎯 9.3.5 阶段 2:Select(智能选择 ⭐ 核心)

> 根据相关性 + 新近性评分,**贪心填充** token 预算。

```python
def _select(
    self,
    packets: List[ContextPacket],
    user_query: str,
    available_tokens: int,
) -> List[ContextPacket]:
    """选择最相关的信息包"""

    # 1. 分离系统指令(永远保留)和其他
    system_packets = [p for p in packets if p.metadata.get("type") == "system_instruction"]
    other_packets = [p for p in packets if p.metadata.get("type") != "system_instruction"]

    # 2. 算系统指令占用的 tokens
    system_tokens = sum(p.token_count for p in system_packets)
    remaining = available_tokens - system_tokens

    # 3. 为其他信息计算综合分数
    scored = []
    for packet in other_packets:
        # 算相关性(如果还没算过)
        if packet.relevance_score == 0.5:
            packet.relevance_score = self._calculate_relevance(packet.content, user_query)

        # 算新近性
        recency = self._calculate_recency(packet.timestamp)

        # ⭐ 综合分数公式
        combined = (
            self.config.relevance_weight * packet.relevance_score
            + self.config.recency_weight * recency
        )

        # 过滤低于阈值
        if packet.relevance_score >= self.config.min_relevance:
            scored.append((combined, packet))

    # 4. 按分数降序排序
    scored.sort(key=lambda x: x[0], reverse=True)

    # 5. ⭐ 贪心填充
    selected = system_packets.copy()
    current_tokens = system_tokens
    for score, packet in scored:
        if current_tokens + packet.token_count <= available_tokens:
            selected.append(packet)
            current_tokens += packet.token_count
        else:
            break   # 预算满了

    return selected
```

### 相关性计算(简化版)

```python
def _calculate_relevance(self, content: str, query: str) -> float:
    """Jaccard 相似度(生产环境可换向量相似度)"""
    content_words = set(content.lower().split())
    query_words = set(query.lower().split())
    intersection = content_words & query_words
    union = content_words | query_words
    return len(intersection) / len(union) if union else 0.0
```

### 新近性计算(指数衰减)

```python
def _calculate_recency(self, timestamp: datetime) -> float:
    """24 小时内保持高分,之后衰减"""
    import math
    age_hours = (datetime.now() - timestamp).total_seconds() / 3600
    decay = 0.1
    return max(0.1, min(1.0, math.exp(-decay * age_hours / 24)))
```

---

## 🎨 9.3.6 阶段 3:Structure(结构化输出)

> 把选中的信息组织成**固定骨架**的上下文模板。

```python
def _structure(self, selected_packets, user_query: str) -> str:
    """组织成结构化模板"""

    # 按类型分组
    system_instructions = []
    evidence = []
    context = []

    for packet in selected_packets:
        ptype = packet.metadata.get("type", "general")
        if ptype == "system_instruction":
            system_instructions.append(packet.content)
        elif ptype in ["rag_result", "knowledge"]:
            evidence.append(packet.content)
        else:
            context.append(packet.content)

    # ⭐ 构建固定骨架
    sections = []
    if system_instructions:
        sections.append("[Role & Policies]\n" + "\n".join(system_instructions))

    sections.append(f"[Task]\n{user_query}")

    if evidence:
        sections.append("[Evidence]\n" + "\n---\n".join(evidence))

    if context:
        sections.append("[Context]\n" + "\n".join(context))

    sections.append("[Output]\n请基于以上信息,提供准确、有据的回答。")

    return "\n\n".join(sections)
```

### 固定 5 分区的优势

| 分区 | 作用 |
|---|---|
| `[Role & Policies]` | 角色 + 行为准则 |
| `[Task]` | 当前任务 |
| `[Evidence]` | RAG 检索的证据 |
| `[Context]` | 历史对话 + 记忆 |
| `[Output]` | 输出要求 |

→ **可读性 + 可调试 + 可扩展 + 可 A/B 测试**。

---

## 🗜 9.3.7 阶段 4:Compress(兜底压缩)

> 即使前面控制好了,**意外超限时**有兜底。

```python
def _compress(self, context: str, max_tokens: int) -> str:
    """超限的上下文压缩"""
    current = self._count_tokens(context)
    if current <= max_tokens:
        return context   # 不超 = 不压

    print(f"[ContextBuilder] 超限({current} > {max_tokens}),压缩中")

    # ⭐ 分区压缩:保持结构完整性
    sections = context.split("\n\n")
    compressed = []
    used = 0

    for section in sections:
        sec_tokens = self._count_tokens(section)
        if used + sec_tokens <= max_tokens:
            compressed.append(section)
            used += sec_tokens
        else:
            # 部分保留
            remaining = max_tokens - used
            if remaining > 50:
                truncated = self._truncate_text(section, remaining)
                compressed.append(truncated + "\n[... 内容已压缩 ...]")
            break

    return "\n\n".join(compressed)
```

→ **保持结构** + **截断标记** —— LLM 知道"这里有省略"。

---

## 🚀 9.3.8 完整使用示例

```python
from hello_agents.context import ContextBuilder, ContextConfig
from hello_agents.tools import MemoryTool, RAGTool
from hello_agents.core.message import Message

# 1. 创建工具
memory_tool = MemoryTool(user_id="user123")
rag_tool = RAGTool(knowledge_base_path="./kb")

# 2. 创建 ContextBuilder
config = ContextConfig(
    max_tokens=3000,
    reserve_ratio=0.2,
    relevance_weight=0.7,
    recency_weight=0.3,
)
builder = ContextBuilder(
    memory_tool=memory_tool,
    rag_tool=rag_tool,
    config=config,
)

# 3. 构建上下文
context = builder.build(
    user_query="如何优化 Pandas 内存占用?",
    conversation_history=[
        Message("我在开发数据分析工具", "user"),
        Message("用什么技术栈?", "assistant"),
        Message("Python + Pandas", "user"),
    ],
    system_instructions="你是 Python 数据工程顾问,要求: 1) 具体可行 2) 解释原理 3) 给代码",
)

print(context)
```

### 输出示例

```
[Role & Policies]
你是 Python 数据工程顾问,要求: 1) 具体可行 2) 解释原理 3) 给代码

[Task]
如何优化 Pandas 内存占用?

[Evidence]
Pandas 内存优化的核心策略包括:
1. 使用合适的数据类型(category 代替 object)
2. 分块读取大文件
3. 使用 chunksize 参数
---
数据类型优化可显著减少内存占用,如 int64 降级 int32 可节省 50%。

[Context]
user: 我在开发数据分析工具
assistant: 用什么技术栈?
user: Python + Pandas
记忆: 用户正在开发数据分析工具

[Output]
请基于以上信息,提供准确、有据的回答。
```

→ **干净、结构化、可追溯**。

---

## 🤖 9.3.9 与 Agent 集成

```python
class ContextAwareAgent(SimpleAgent):
    """具有上下文感知能力的 Agent"""

    def __init__(self, name, llm, **kwargs):
        super().__init__(name=name, llm=llm, system_prompt=kwargs.get("system_prompt", ""))

        # 上下文构建器
        self.memory_tool = MemoryTool(user_id=kwargs.get("user_id", "default"))
        self.rag_tool = RAGTool(knowledge_base_path=kwargs.get("kb", "./kb"))
        self.context_builder = ContextBuilder(
            memory_tool=self.memory_tool,
            rag_tool=self.rag_tool,
            config=ContextConfig(max_tokens=4000),
        )
        self.conversation_history = []

    def run(self, user_input: str) -> str:
        # 1. 构建优化上下文
        optimized_context = self.context_builder.build(
            user_query=user_input,
            conversation_history=self.conversation_history,
            system_instructions=self.system_prompt,
        )

        # 2. 调 LLM
        messages = [
            {"role": "system", "content": optimized_context},
            {"role": "user", "content": user_input},
        ]
        response = self.llm.invoke(messages)

        # 3. 更新历史
        self.conversation_history.extend([
            Message(user_input, "user"),
            Message(response, "assistant"),
        ])

        # 4. 重要交互入记忆
        self.memory_tool.run({
            "action": "add",
            "content": f"Q: {user_input}\nA: {response[:200]}...",
            "memory_type": "episodic",
            "importance": 0.6,
        })

        return response
```

→ **Agent 自动拥有"上下文管理大脑"**。

---

## 💡 9.3.10 最佳实践

| 建议 | 解释 |
|---|---|
| **动态调整预算** | 简单任务 1500 tokens,复杂 5000+ |
| **生产用向量相似度** | 替换 Jaccard,**质量大幅提升** |
| **缓存系统指令** | 不变内容缓存,**省 tokens 重算** |
| **监控 + 日志** | 记录"选中数/token 使用率",便于优化 |
| **A/B 测试参数** | `relevance_weight` / `min_relevance` 等关键参数 |

---

## ⚠️ 小白避坑

1. **`reserve_ratio` 别设 0**
   - 系统指令会被挤占
2. **`min_relevance` 别设太高**
   - 太高 → 没东西能通过
   - 一般 0.1~0.3
3. **Jaccard 是简化版**
   - 中文分词不友好
   - 生产用 jieba + 向量
4. **`token_count` 是估算**
   - 生产用 tiktoken 精确计算

---

## 📌 9.3 节要点

| 知识点 | 一句话 |
|---|---|
| **ContextBuilder** | 上下文工程的工程化实现 |
| **GSSC 4 阶段** | Gather → Select → Structure → Compress |
| **ContextPacket** | 统一信息包,**简化处理** |
| **综合评分** | `相关性 × 0.7 + 新近性 × 0.3` |
| **贪心填充** | 按分数高低,**token 预算内填满** |
| **5 分区模板** | Role/Task/Evidence/Context/Output |
| **兜底压缩** | 超限时**保持结构 + 截断标记** |

---

## 🔗 延伸阅读

- 上一节:[01-什么是上下文工程](01-%E4%BB%80%E4%B9%88%E6%98%AF%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B7%A5%E7%A8%8B.md)
- 下一节:[03-NoteTool结构化笔记](03-NoteTool%E7%BB%93%E6%9E%84%E5%8C%96%E7%AC%94%E8%AE%B0.md) —— 长程任务的"外脑"
- Memory:[02-记忆系统四种类型](../08-%E7%AC%AC8%E7%AB%A0-%E8%AE%B0%E5%BF%86%E4%B8%8E%E6%A3%80%E7%B4%A2/02-%E8%AE%B0%E5%BF%86%E7%B3%BB%E7%BB%9F%E5%9B%9B%E7%A7%8D%E7%B1%BB%E5%9E%8B.md)

---

⬅ [01-什么是上下文工程](01-%E4%BB%80%E4%B9%88%E6%98%AF%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B7%A5%E7%A8%8B.md)	|	➡ [03-NoteTool结构化笔记](03-NoteTool%E7%BB%93%E6%9E%84%E5%8C%96%E7%AC%94%E8%AE%B0.md)
