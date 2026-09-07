---
tags: [Happy-LLM, 第7章, RAG, 实战速览]
chapter: 7
section: 7.2.2
---

# 7.2.2 Tiny-RAG 实战速览

⬅ [02-RAG原理深入](02-RAG%E5%8E%9F%E7%90%86%E6%B7%B1%E5%85%A5.md)	|	➡ [04-LLM-Agent原理](04-LLM-Agent%E5%8E%9F%E7%90%86.md)

> ⚡ **本节速览**：用 ~200 行代码搭一个完整 RAG。完整源码见 Happy-LLM `code/RAG/`。

---

## 🎯 五大模块

```
Tiny-RAG/
├── 1. ReadFiles      ─ 文档加载 + 切分
├── 2. Embedding      ─ 向量化模型
├── 3. VectorStore    ─ 向量数据库（带相似度检索）
├── 4. LLM            ─ 大模型生成
└── 5. demo.py        ─ 串起来运行
```

---

## 🐍 Step 1：文档加载和切分

```python
class ReadFiles:
    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据扩展名选择读取方式
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        """
        按 token 长度切分，带重叠
        - max_token_len: 每块最大 token 数
        - cover_content: 重叠区大小（保证关键信息不被切断）
        """
        # ... 按行遍历，累积到 max_token_len 切一块
        # 相邻块之间保留 cover_content 字符重叠
```

### 🔑 关键设计

- **按行切分**：保持句子完整性
- **重叠区**：防止关键信息被切断（如人名跨段）
- **长行单独切**：超长行按 token 强制切分

→ 经验值：**chunk=500~1000 token，overlap=100~200 token**

---

## 🐍 Step 2：Embedding 基类 + OpenAI 实现

```python
import numpy as np
from typing import List

class BaseEmbeddings:
    """所有 Embedding 模型的基类"""

    def __init__(self, path: str, is_api: bool):
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError   # 子类必须实现

    @classmethod
    def cosine_similarity(cls, v1: List[float], v2: List[float]) -> float:
        """余弦相似度（继承下来给所有子类用）"""
        v1, v2 = np.array(v1), np.array(v2)
        dot = np.dot(v1, v2)
        return dot / (np.linalg.norm(v1) * np.linalg.norm(v2))


class OpenAIEmbedding(BaseEmbeddings):
    """OpenAI / 硅基流动 API 调用"""

    def __init__(self, path='', is_api=True):
        super().__init__(path, is_api)
        if is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:
        text = text.replace("\n", " ")
        return self.client.embeddings.create(
            input=[text], model=model
        ).data[0].embedding
```

### 🔑 设计要点

- **基类抽象**：换 Embedding 模型只需新建子类
- **`cosine_similarity` 写在基类**：所有子类共享
- **默认用 `bge-m3`**：中文友好、开源免费

---

## 🐍 Step 3：向量数据库（核心）

```python
class VectorStore:
    def __init__(self, document: List[str] = ['']):
        self.document = document   # 文档片段列表
        self.vectors = []           # 对应的向量列表

    def get_vector(self, EmbeddingModel: BaseEmbeddings):
        """把所有文档向量化"""
        self.vectors = [
            EmbeddingModel.get_embedding(doc) for doc in self.document
        ]
        return self.vectors

    def persist(self, path: str = 'storage'):
        """保存到本地（避免每次重算）"""
        # 保存 self.document 和 self.vectors 到 JSON / Pickle

    def load_vector(self, path: str = 'storage'):
        """从本地加载"""
        ...

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        """⭐ 核心：根据问题检索 top-k 文档"""
        # 1. 把 query 向量化
        query_vector = EmbeddingModel.get_embedding(query)

        # 2. 算 query 和每个文档的相似度
        scores = np.array([
            EmbeddingModel.cosine_similarity(query_vector, v)
            for v in self.vectors
        ])

        # 3. 取 top-k 相似度最高的文档
        return np.array(self.document)[scores.argsort()[-k:][::-1]].tolist()
```

### 🔑 检索逻辑（核心一行）

```python
np.array(self.document)[scores.argsort()[-k:][::-1]]
```

拆解：
- `scores.argsort()` → 按相似度从小到大的索引
- `[-k:]` → 取最大的 k 个
- `[::-1]` → 倒序（最大的在前）
- `np.array(self.document)[...]` → 取对应文档

→ **简单的 numpy 实现**，生产可换 FAISS / Milvus。

---

## 🐍 Step 4：LLM 模块

```python
RAG_PROMPT_TEMPLATE = """
使用以上下文回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。

问题: {question}
可参考的上下文：
···
{context}
···
有用的回答:
"""


class OpenAIChat:
    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct"):
        self.model = model

    def chat(self, prompt: str, history: list, content: str) -> str:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        client.base_url = os.getenv("OPENAI_BASE_URL")

        # ⭐ 把检索到的 content 填到 prompt 模板里
        history.append({
            "role": "user",
            "content": RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)
        })

        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=2048,
            temperature=0.1,    # 低温度，更确定的回答
        )
        return response.choices[0].message.content
```

### 🔑 关键设计

- **`temperature=0.1`**：低温度，避免随机性
- **RAG_PROMPT 模板**：明示"基于上下文回答"
- **历史对话**：支持多轮（虽然简化版很简单）

---

## 🐍 Step 5：把一切串起来

### 第一次运行（建库）

```python
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# 1. 加载文档 + 切分
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)

# 2. 向量化
vector = VectorStore(docs)
embedding = OpenAIEmbedding()
vector.get_vector(EmbeddingModel=embedding)

# 3. 持久化（下次不用重算）
vector.persist(path='storage')

# 4. 问问题
question = 'RAG 的原理是什么？'
content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))
```

### 下次运行（直接加载）

```python
vector = VectorStore()
vector.load_vector('./storage')   # 不用重新向量化

# ... 同上
```

---

## 🎯 完整效果

```
User: RAG 的原理是什么？
Assistant: RAG（Retrieval-Augmented Generation）是检索增强生成技术。
其工作流程分为三个核心步骤：① 索引（将文档库分割成片段并构建向量索引）；
② 检索（根据问题相似度找出相关文档片段）；③ 生成（LLM 基于检索到的上下文
生成回答）。RAG 有效缓解了大语言模型的幻觉问题，因为生成的内容基于真实
文档，使答案具有可追溯性和可信度。
```

→ **回答完全基于你的知识库**，不会瞎编。

---

## 🚀 进阶建议

### 升级路径

```
本章 Tiny-RAG（学习版）
    ↓
LangChain / LlamaIndex（工程化）
    ↓
+ Rerank 模型（提升检索质量）
    ↓
+ Multi-Query Rewrite（查询优化）
    ↓
+ Agentic RAG（让 Agent 决定怎么检索）
```

### 推荐工业级框架

| 框架 | 特点 |
|---|---|
| **LangChain** | 最流行，生态完善 |
| **LlamaIndex** | 专注 RAG，文档处理强 |
| **Haystack** | 老牌，企业级 |
| **Dify / FastGPT** | 国产 RAG 平台，开箱即用 |

---

## ⚠️ 小白避坑

1. **API Key 别提交到 GitHub**
   - 用环境变量 `os.getenv("OPENAI_API_KEY")`
2. **第一次跑很慢**
   - 大量文档的 Embedding 调用慢
   - 必须 `persist()` 保存，避免下次重算
3. **k 值别取太大**
   - k=1~3 通常够用
   - k=10+ 会引入大量噪声
4. **不同 Embedding 模型不能混用**
   - 建库时用啥模型，查询时也得用啥
5. **本地小 LLM 效果可能差**
   - RAG 对 LLM 的"指令遵循能力"要求高
   - 推荐至少 7B+ 的 Chat 模型

---

## 📌 7.2.2 节要点

- Tiny-RAG = **ReadFiles + Embedding + VectorStore + LLM** 五大模块
- 核心代码 **~200 行**就能跑通
- 工业级用 LangChain / LlamaIndex / Dify
- 进阶方向：**Rerank、Query Rewrite、Agentic RAG**

---

## 🔗 延伸阅读

- 上一节：[02-RAG原理深入](02-RAG%E5%8E%9F%E7%90%86%E6%B7%B1%E5%85%A5.md)
- 下一节：[04-LLM-Agent原理](04-LLM-Agent%E5%8E%9F%E7%90%86.md)
- 完整代码：[Happy-LLM Chapter7 RAG](https://github.com/datawhalechina/happy-llm)

---

⬅ [02-RAG原理深入](02-RAG%E5%8E%9F%E7%90%86%E6%B7%B1%E5%85%A5.md)	|	➡ [04-LLM-Agent原理](04-LLM-Agent%E5%8E%9F%E7%90%86.md)
