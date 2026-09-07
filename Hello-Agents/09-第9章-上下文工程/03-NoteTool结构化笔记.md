---
tags: [Hello-Agents, 第9章, NoteTool, 结构化笔记, 长程任务]
chapter: 9
section: 9.4
---

# 9.4 NoteTool:结构化笔记

⬅ [02-ContextBuilder与GSSC](02-ContextBuilder%E4%B8%8EGSSC.md)	|	➡ [04-TerminalTool文件访问](04-TerminalTool%E6%96%87%E4%BB%B6%E8%AE%BF%E9%97%AE.md)

> **长时程任务的"外脑"** —— Markdown + YAML 格式的结构化外部记忆。

---

## 🎬 故事比喻:工程师的工作笔记本

| | 没有 NoteTool | **有 NoteTool** |
|---|---|---|
| 工程师状态 | 全靠脑子记 | **有笔记本**(可翻可改可搜) |
| Agent 状态 | 全靠上下文窗口 | **有持久外部记忆** |
| 长程任务 | 几小时就忘 | **几天后还能接着干** |

→ NoteTool = Agent 的"**笔记本**"。

---

## 🎯 9.4.1 为什么需要 NoteTool?

### 与 MemoryTool 的差异

| | **MemoryTool**(第 8 章) | **NoteTool**(本章) |
|---|---|---|
| **关注点** | 对话式记忆(短期/情景/语义) | **项目式记录**(长期追踪) |
| **存储** | Qdrant + SQLite + Neo4j | **Markdown + YAML 文件** |
| **可读性** | 机器友好 | **人机双友好** |
| **版本控制** | 难 | ✅ Git 友好 |
| **典型用途** | 对话历史 / 用户偏好 | TODO / 项目状态 / 阻塞 |

→ **互补不替代**:**MemoryTool 是大脑,NoteTool 是笔记本**。

### 4 大特性

| 特性 | 解释 |
|---|---|
| **结构化记录** | Markdown + YAML,机器 + 人类都看得懂 |
| **版本友好** | 纯文本,**Git 跟踪修改** |
| **低开销** | 无数据库,**轻量级**状态追踪 |
| **灵活分类** | `type` + `tags` **多维组织** |

---

## 💼 9.4.2 典型应用场景

### 场景 1:**长期项目追踪**

```
重构一个大型代码库(可能几周)
   ↓ NoteTool 记录
- task_state: 当前阶段进度
- conclusion: 阶段性结论
- blocker:   遇到的问题
- action:    下一步行动
```

### 场景 2:**研究任务管理**

```
做文献综述
   ↓ NoteTool 记录
- 每篇论文的核心观点 (conclusion)
- 待深入调研的主题 (action)
- 重要参考文献 (reference)
```

### 场景 3:**与 ContextBuilder 配合**

```python
# Agent 的 run() 方法里
def run(self, user_input):
    # 1. 检索相关笔记
    relevant_notes = self.note_tool.run({
        "action": "search",
        "query": user_input,
        "limit": 3,
    })

    # 2. 转成 ContextPacket
    note_packets = [
        ContextPacket(
            content=note['content'],
            timestamp=note['updated_at'],
            token_count=count(note['content']),
            relevance_score=0.7,
            metadata={"type": "note"},
        )
        for note in relevant_notes
    ]

    # 3. 构建上下文时传入
    context = self.context_builder.build(
        user_query=user_input,
        custom_packets=note_packets,
    )
```

→ **NoteTool + ContextBuilder = 长程记忆 + 智能注入**。

---

## 📄 9.4.3 存储格式

### 笔记文件(`.md`)

每个笔记是一个独立的 Markdown 文件:

```markdown
---
id: note_20250119_153000_0
title: 项目进展 - 第一阶段
type: task_state
tags: [refactoring, phase1, backend]
created_at: 2025-01-19T15:30:00
updated_at: 2025-01-19T15:30:00
---

# 项目进展 - 第一阶段

## 完成情况
已完成数据模型层的重构,主要改动包括:
1. 统一了实体类的命名规范
2. 引入类型提示
3. 优化了数据库查询性能

## 测试覆盖
- 单元测试覆盖率: 85%
- 集成测试覆盖率: 70%

## 下一步计划
1. 重构业务逻辑层
2. 解决依赖冲突问题
3. 提升集成测试覆盖率至 85%
```

### 格式优势

| 部分 | 优势 |
|---|---|
| **YAML 元数据** | 机器精确解析 (id/type/tags/timestamps) |
| **Markdown 正文** | 人类可读 + 富文本(标题/列表/代码块) |
| **文件名即 ID** | 简化管理,**一文件一笔记** |

### 索引文件(`notes_index.json`)

```json
{
  "note_20250119_153000_0": {
    "id": "note_20250119_153000_0",
    "title": "项目进展 - 第一阶段",
    "type": "task_state",
    "tags": ["refactoring", "phase1", "backend"],
    "created_at": "2025-01-19T15:30:00",
    "updated_at": "2025-01-19T15:30:00",
    "file_path": "./notes/note_20250119_153000_0.md"
  }
}
```

**作用**:
- 快速检索(无需打开每个文件)
- 元数据集中管理
- 完整性校验

---

## 🛠 9.4.4 七大核心操作

| 操作 | 用途 |
|---|---|
| **`create`** | 创建笔记 |
| **`read`** | 读取笔记 |
| **`update`** | 更新笔记 |
| **`delete`** | 删除笔记 |
| **`search`** | 搜索笔记 |
| **`list`** | 列出笔记 |
| **`stats`** | 统计信息 |

### 操作 1:**`create`(创建)**

```python
from hello_agents.tools import NoteTool

notes = NoteTool(workspace="./project_notes")

note_id = notes.run({
    "action": "create",
    "title": "重构项目 - 第一阶段",
    "content": """## 完成情况
已完成数据模型层重构,测试覆盖 85%。

## 下一步
重构业务逻辑层""",
    "note_type": "task_state",
    "tags": ["refactoring", "phase1"],
})
print(f"✅ 笔记创建成功, ID: {note_id}")
```

**支持的 7 种笔记类型**:
- `task_state` —— 任务状态
- `conclusion` —— 结论
- `blocker` —— 阻塞点
- `action` —— 行动项
- `reference` —— 参考资料
- `general` —— 通用
- `note` —— 笔记(自定义)

### 操作 2:**`read`(读取)**

```python
note = notes.run({"action": "read", "note_id": note_id})
print(note["metadata"])    # YAML 元数据
print(note["content"])     # Markdown 正文
```

### 操作 3:**`update`(更新)**

```python
notes.run({
    "action": "update",
    "note_id": note_id,
    "content": "已完成业务逻辑层重构,覆盖率 90%",  # 新内容
    "tags": ["refactoring", "phase1", "completed"],  # 更新 tags
})
# updated_at 自动更新
```

### 操作 4:**`search`(搜索)**

```python
# 关键词搜索 + 多维过滤
results = notes.run({
    "action": "search",
    "query": "重构",
    "note_type": "task_state",          # 按类型
    "tags": ["phase1"],                  # 按标签(交集)
    "limit": 10,
})
```

### 操作 5:**`list`(列出)**

```python
# 列出所有(按更新时间倒序)
results = notes.run({"action": "list", "limit": 20})

# 按类型筛选
results = notes.run({"action": "list", "note_type": "blocker"})
```

---

## ⭐ 9.4.5 与 ContextBuilder 深度集成

### 完整流程

```
┌────────────────────────────────────┐
│  Agent.run(user_input)              │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  1. NoteTool.search(user_input)    │
│     → 找到相关笔记 (top 3-5)        │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  2. 转成 ContextPacket             │
│     标记 metadata.type="note"      │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  3. ContextBuilder.build()          │
│     - GSSC 流水线                    │
│     - 笔记作为 custom_packets        │
│     - 自动评分 + 选择 + 结构化         │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  4. LLM 调用                        │
│     基于优化上下文回答               │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  5. (可选) NoteTool.create(...)     │
│     把重要的新发现记下来              │
└────────────────────────────────────┘
```

→ **闭环**:**读笔记 → 上下文 → LLM → 写笔记**。

---

## 🎮 9.4.6 实战示例:长程项目助手

```python
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import NoteTool, MemoryTool

llm = HelloAgentsLLM()
notes = NoteTool(workspace="./project_notes")
memory = MemoryTool(user_id="dev_team")

agent = SimpleAgent(
    name="项目助手",
    llm=llm,
    system_prompt="""你是项目助手,可以使用 NoteTool 记录项目进展:
- task_state: 当前进度
- conclusion: 阶段结论
- blocker: 阻塞问题
- action: 下一步行动
请主动维护这些笔记""",
)

registry = ToolRegistry()
registry.register_tool(notes)
registry.register_tool(memory)
agent.tool_registry = registry

# 第一天
agent.run("今天完成了用户认证模块,但 OAuth 集成有问题。下一步需要解决 OAuth")
# → Agent 自动创建多条笔记:
#    - task_state: "完成用户认证模块"
#    - blocker:    "OAuth 集成问题"
#    - action:     "解决 OAuth 集成"

# 三天后(中间无对话)
agent.run("OAuth 进展如何?")
# → Agent 搜索笔记 → 找到 blocker → 回答:
#   "您之前提到 OAuth 集成有问题。这是阻塞项。让我帮您梳理..."
```

→ **3 天前的事情还记得**,**真正长程**。

---

## ⚠️ 小白避坑

1. **`workspace` 路径要稳定**
   - 换路径 → 笔记丢失(其实只是找不到)
   - 用绝对路径或固定相对路径
2. **YAML 元数据不能乱写**
   - 字段顺序可调,**字段名固定**
   - 手动改 `.md` 文件容易破坏 YAML
3. **`tags` 要有体系**
   - 别每次随机加新 tag
   - 维护一个**标签词表**
4. **不要把所有事都记成 `general`**
   - 用对 `note_type` → 后续筛选才方便
5. **Git 友好不等于自动 commit**
   - 重要笔记记得手动 `git commit`

---

## 📌 9.4 节要点

| 知识点 | 一句话 |
|---|---|
| **NoteTool 定位** | 长时程任务的**结构化外部记忆** |
| **存储格式** | Markdown + YAML(人机双友好) |
| **7 种笔记类型** | task_state / conclusion / blocker / action / ... |
| **核心操作** | create / read / update / search / list / ... |
| **与 ContextBuilder 集成** | 笔记 → ContextPacket → 智能注入 |
| **vs MemoryTool** | 项目式记录 vs 对话式记忆 |

---

## 🔗 延伸阅读

- 上一节:[02-ContextBuilder与GSSC](02-ContextBuilder%E4%B8%8EGSSC.md)
- 下一节:[04-TerminalTool文件访问](04-TerminalTool%E6%96%87%E4%BB%B6%E8%AE%BF%E9%97%AE.md)
- 对比:[MemoryTool](../08-%E7%AC%AC8%E7%AB%A0-%E8%AE%B0%E5%BF%86%E4%B8%8E%E6%A3%80%E7%B4%A2/02-%E8%AE%B0%E5%BF%86%E7%B3%BB%E7%BB%9F%E5%9B%9B%E7%A7%8D%E7%B1%BB%E5%9E%8B.md)

---

⬅ [02-ContextBuilder与GSSC](02-ContextBuilder%E4%B8%8EGSSC.md)	|	➡ [04-TerminalTool文件访问](04-TerminalTool%E6%96%87%E4%BB%B6%E8%AE%BF%E9%97%AE.md)
