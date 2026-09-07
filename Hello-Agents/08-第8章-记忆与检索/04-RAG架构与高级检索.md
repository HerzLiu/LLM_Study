---
tags: [Hello-Agents, 第8章, RAG, RAGTool, MQE, HyDE]
chapter: 8
section: 8.4
---

# 8.4 RAG 系统:HelloAgents 架构 + 高级检索

⬅ [03-RAG基础与原理](03-RAG%E5%9F%BA%E7%A1%80%E4%B8%8E%E5%8E%9F%E7%90%86.md)	|	➡ [05-综合实战与小结](05-%E7%BB%BC%E5%90%88%E5%AE%9E%E6%88%98%E4%B8%8E%E5%B0%8F%E7%BB%93.md)

---

## 🎬 故事比喻:朴素图书管理员 vs 智能图书管理员

| | 朴素 RAG | HelloAgents RAG |
|---|---|---|
| 检索方式 | 单一向量相似度 | **向量 + MQE + HyDE 三路检索** |
| 文档处理 | 简单按字数切 | **多格式智能解析** |
| 结果整合 | 直接拼接 | **智能片段合并 + 截断** |
| 比喻 | **新手图书管理员**(只会按目录找) | **资深图书管理员**(会多角度推荐) |

---

## 🏗 8.4.1 HelloAgents RAG 整体架构

```
┌─────────────────────────────────────────┐
│  RAGTool (Agent 的检索能力)              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   智能问答层 (Q&A)                       │
│   ├── 多策略检索: 向量 + MQE + HyDE        │
│   ├── 上下文构建: 智能合并 + 截断           │
│   └── LLM 增强生成: 基于上下文准确问答      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   向量存储层 (Vector Storage)            │
│   └── QdrantVectorStore (命名空间隔离)   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   嵌入表示层 (Embedding)                 │
│   └── 复用记忆系统的嵌入服务              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   文档处理层 (Document Processing)       │
│   ├── DocumentProcessor (多格式解析)     │
│   ├── Document (元数据管理)              │
│   └── Pipeline (端到端处理)              │
└─────────────────────────────────────────┘
```

→ **4 层架构,职责清晰**。

---

## 🛠 8.4.2 RAGTool 使用接口

### 基本使用(3 步)

```python
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import RAGTool

# 1. 创建 RAGTool(指定知识库路径)
rag_tool = RAGTool(
    knowledge_base_path="./knowledge_base",   # 文档目录
    namespace="default",                       # 命名空间隔离
    collection_name="rag_knowledge_base",     # Qdrant 集合名
)

# 2. 注册到 Agent
agent = SimpleAgent(name="智能助手", llm=HelloAgentsLLM())
tool_registry = ToolRegistry()
tool_registry.register_tool(rag_tool)
agent.tool_registry = tool_registry

# 3. 问答(自动 RAG)
response = agent.run("HelloAgents 框架是怎么设计的?")
print(response)
```

### 文档加载

```python
# 添加单个文档
rag_tool.execute("add_document", file_path="./docs/intro.md")

# 批量加载目录
rag_tool.execute("load_directory", directory="./docs/")
# 自动识别 PDF / MD / TXT / DOCX
```

---

## 📄 8.4.3 多格式文档处理

### `DocumentProcessor` 支持的格式

| 格式 | 库 | 用途 |
|---|---|---|
| **PDF** | `pypdf` / `pdfplumber` | 论文 / 报告 |
| **Markdown** | 内置 | 技术文档 |
| **TXT** | 内置 | 纯文本 |
| **DOCX** | `python-docx` | Word 文档 |
| **HTML** | `beautifulsoup4` | 网页 |
| **CSV / JSON** | 内置 | 结构化数据 |

### 文档分块策略

```python
class DocumentProcessor:
    def chunk(self, text: str, max_tokens: int = 500, overlap: int = 100):
        """
        智能分块:
        - 优先按 段落 → 句子 → 字符 切分(递归)
        - 保持语义完整
        - 块间重叠 100 token,防止关键信息被切断
        """
        ...
```

→ **关键参数**:
- `max_tokens=500` → 块大小适中
- `overlap=100` → 5/1 重叠,保证连贯

---

## 🎯 8.4.4 高级检索三大策略(核心)

### 策略 1:**向量检索**(基础)

```python
results = vector_store.search(
    query_vector=embedding.encode(user_query),
    top_k=5,
)
```

**问题**:用户提问可能**和文档措辞差异大**,直接向量检索可能漏召回。

---

### 策略 2:**MQE**(Multi-Query Expansion,多查询扩展)

> 让 LLM **生成多个等价问法**,**多角度检索**。

```python
def mqe_retrieval(user_query, llm, vector_store, top_k=5):
    """多查询扩展检索"""

    # 1. 让 LLM 生成 3-5 个等价问法
    prompt = f"""请把下面的用户问题改写成 3 个不同表达方式的等价问题。
    用户问题: {user_query}

    输出格式:
    1. ...
    2. ...
    3. ..."""

    expansions = llm.invoke([{"role": "user", "content": prompt}])
    questions = [user_query] + parse_questions(expansions)

    # 2. 并行检索所有问法
    all_results = []
    for q in questions:
        results = vector_store.search(embedding.encode(q), top_k=top_k)
        all_results.extend(results)

    # 3. 去重 + 重排
    return rerank_and_dedup(all_results, top_k=top_k)
```

#### 例子

```
用户问: "Python 怎么学最快?"
   ↓ MQE 扩展
1. Python 怎么学最快?           (原问题)
2. Python 最高效的学习方法是什么?
3. 如何快速入门 Python?
4. 推荐 Python 学习路径
   ↓
4 个问法都检索 → 召回率显著提升
```

→ **解决"提问表达"和"文档表达"不一致** 的问题。

---

### 策略 3:**HyDE**(Hypothetical Document Embedding,假设性文档嵌入)

> **让 LLM 先生成"假设答案",再用"假设答案"去检索**。

```python
def hyde_retrieval(user_query, llm, vector_store, top_k=5):
    """HyDE 检索"""

    # 1. 让 LLM 生成"假设性答案"(可能错,但风格接近真文档)
    prompt = f"""请为下面的问题生成一个详细的、看起来真实的答案。
    问题: {user_query}

    要求: 答案应该像专业文档中的段落,即使你不确定。"""

    hypothetical_answer = llm.invoke([{"role": "user", "content": prompt}])

    # 2. ⭐ 用"假设答案"的向量去检索(不是用问题向量)
    answer_vector = embedding.encode(hypothetical_answer)
    results = vector_store.search(answer_vector, top_k=top_k)

    return results
```

#### 为什么有效?

```
问题: "什么是 ReAct?"  → 很短,信息量小
   ↓ 直接检索 → 匹配度不高

LLM 生成假答: "ReAct 是 2022 年提出的智能体范式,
              结合推理(Reasoning)和行动(Acting),
              通过 Thought-Action-Observation 循环..."
   ↓ 用假答检索 → 文档库里真有讲 ReAct 的文档,**完美匹配**!
```

→ **以毒攻毒** —— 用 LLM 的"想象"匹配文档的"风格"。

---

## 🤝 8.4.5 三大策略融合

> HelloAgents 把三种策略**并行使用 + 结果融合**。

```python
def multi_strategy_retrieval(user_query, top_k=5):
    """多策略融合检索"""

    # 三路并行
    vector_results = vector_retrieval(user_query)
    mqe_results    = mqe_retrieval(user_query)
    hyde_results   = hyde_retrieval(user_query)

    # 融合 + 去重 + 重排
    all_results = vector_results + mqe_results + hyde_results
    deduped = dedup_by_content(all_results)
    reranked = rerank_by_relevance(deduped, user_query)

    return reranked[:top_k]
```

**效果**:**召回率显著高于单一策略**,**生产级 RAG 必备**。

---

## 🔧 8.4.6 上下文构建优化

### 问题:**简单拼接太傻**

```python
# ❌ 朴素拼接
context = "\n".join([doc.content for doc in results])
# 问题:
# - 重复内容
# - 总长度可能超 LLM 上下文窗口
# - 不同来源的文档没区分
```

### 解法:**智能片段合并 + 截断**

```python
def build_context(documents, max_tokens=3000):
    """智能构建上下文"""

    # 1. 去除重复内容
    documents = dedup_documents(documents)

    # 2. 按相关性排序
    documents.sort(key=lambda x: x.score, reverse=True)

    # 3. 按 token 预算累加(超限就截断)
    context_parts = []
    used_tokens = 0
    for doc in documents:
        doc_tokens = count_tokens(doc.content)
        if used_tokens + doc_tokens > max_tokens:
            # 截断:取部分内容
            remaining = max_tokens - used_tokens
            context_parts.append(doc.content[:remaining * 2])
            break

        # 加入,标注来源(可追溯)
        context_parts.append(f"【来源: {doc.source}】\n{doc.content}")
        used_tokens += doc_tokens

    return "\n\n---\n\n".join(context_parts)
```

→ **保证不超 token 限制,保留最重要的内容,可追溯来源**。

---

## 💡 8.4.7 Memory + RAG 协同

> HelloAgents 的设计精妙:**RAG 结果可自动存入记忆**。

```python
# 用户问问题
agent.run("HelloAgents 怎么设计 Memory?")

# 内部流程:
# 1. RAG 检索文档
# 2. LLM 基于文档回答
# 3. ⭐ 把高质量回答自动存入 SemanticMemory
# 4. 下次类似问题 → 优先从 Memory 答,**不需要再检索**(快)
```

→ **检索一次,反复利用** —— **类似人类"学过的知识"**。

---

## ⭐ 8.4.8 与 Coze/Dify 的 RAG 对比

| 维度 | **HelloAgents RAG** | **Coze/Dify RAG** |
|---|---|---|
| 部署 | 自部署 | 平台代管 |
| 检索策略 | 向量 + MQE + HyDE | 通常单向量 |
| 灵活性 | 完全控制 | 受限 |
| 学习价值 | **高(源码可见)** | 低(黑盒) |
| 适合 | 学习 + 深度定制 | 快速搭建 |

→ HelloAgents RAG 是**学习级 + 中等生产级**;真正企业级看 LangChain RAG / LlamaIndex。

---

## ⚠️ 小白避坑

1. **MQE 和 HyDE 都靠 LLM,有成本**
   - 每次检索多调 1-2 次 LLM
   - 高频场景考虑缓存
2. **Vector DB 必须建好**
   - 没数据就检索 → **空结果**
   - 先 `load_directory` 喂数据
3. **`namespace` 防止数据串台**
   - 多用户 → 必用 namespace
   - 默认 namespace="default" 是大家共用
4. **Embedding 切换要清库**
   - 换 embedding 模型 → 旧向量失效
   - 必须**清空 + 重新嵌入**所有文档

---

## 📌 8.4 节要点

| 知识点 | 一句话 |
|---|---|
| **RAGTool** | Agent 的 RAG 能力工具 |
| **4 层架构** | 文档处理 + 嵌入 + 向量存储 + 智能问答 |
| **多格式文档** | PDF/MD/TXT/DOCX/HTML/CSV/JSON |
| **MQE** | 多查询扩展,**解决提问表达差异** |
| **HyDE** | 假设答案检索,**以毒攻毒** |
| **上下文构建** | 智能合并 + token 预算 + 来源追溯 |
| **Memory + RAG 协同** | RAG 结果可存入 Memory |

---

## 🔗 延伸阅读

- 上一节:[03-RAG基础与原理](03-RAG%E5%9F%BA%E7%A1%80%E4%B8%8E%E5%8E%9F%E7%90%86.md)
- 下一节:[05-综合实战与小结](05-%E7%BB%BC%E5%90%88%E5%AE%9E%E6%88%98%E4%B8%8E%E5%B0%8F%E7%BB%93.md)
- Happy-LLM:[Tiny-RAG 实战](../../Happy-LLM/07-%E7%AC%AC7%E7%AB%A0-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8/03-Tiny-RAG%E5%AE%9E%E6%88%98%E9%80%9F%E8%A7%88.md)

---

⬅ [03-RAG基础与原理](03-RAG%E5%9F%BA%E7%A1%80%E4%B8%8E%E5%8E%9F%E7%90%86.md)	|	➡ [05-综合实战与小结](05-%E7%BB%BC%E5%90%88%E5%AE%9E%E6%88%98%E4%B8%8E%E5%B0%8F%E7%BB%93.md)
