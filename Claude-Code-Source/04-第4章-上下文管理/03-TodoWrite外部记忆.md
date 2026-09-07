---
tags: [Claude-Code, 第4章, TodoWrite, 外部记忆, Prompt工程]
chapter: 4
section: 4.3
---

# 4.3 解药二：TodoWriteTool 外部任务记忆

⬅ [02-AutoCompact压缩](02-AutoCompact%E5%8E%8B%E7%BC%A9.md)　|	➡ [04-ClaudeMD长期记忆](04-ClaudeMD%E9%95%BF%E6%9C%9F%E8%AE%B0%E5%BF%86.md)

> 📂 源码：`src/tools/TodoWriteTool/prompt.ts`（184 行 prompt！）

---

## 🎬 故事比喻：实习生的小本本

实习生的工作记忆有限，但他有个"小本本"专门记任务清单：

```
□ 看代码
☑ 改 utils.foo()  ← 已完成
■ 改 utils.bar()  ← 正在做
□ 改 utils.baz()
□ 跑测试
```

只要小本本在手，**就算他忘了所有细节，也不会丢任务**。
Auto-compact 会"丢细节"，但 todo list 保住"框架"。

---

## 🔧 TodoWrite 的设计

工具签名：

```python
TodoWrite(todos=[
    {"content": "重构 utils.foo()", "status": "completed", "priority": "high"},
    {"content": "重构 utils.bar()", "status": "in_progress", "priority": "high"},
    {"content": "重构 utils.baz()", "status": "pending", "priority": "high"},
    {"content": "跑测试", "status": "pending", "priority": "medium"},
])
```

**关键设计**：

| 设计 | 为什么 |
|---|---|
| **每次全量替换**（不是增量） | 简化模型心智，永远操作完整列表 |
| **状态：pending / in_progress / completed / cancelled** | 4 种状态足以表达 |
| **同时只能有 1 个 in_progress** | 模型一次只干一件事 |
| **CLI 实时渲染** | 用户能看进度 |
| **priority 字段** | 帮模型做轻重缓急 |

---

## 🔧 Prompt 设计（最值得抄）

`src/tools/TodoWriteTool/prompt.ts`（节选）：

```
## When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list
4. User provides multiple tasks - When users provide a list of things to be done
5. After receiving new instructions - Immediately capture user requirements as todos
6. When you start working on a task - Mark it as in_progress BEFORE beginning work.
   Ideally you should only have one todo as in_progress at a time
7. After completing a task - Mark it as completed and add any new follow-up tasks
```

```
## When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do.
```

---

## ⭐ 加深理解：这套 prompt 教训

### 教训 1：明确"何时用 / 何时不用"

不告诉模型"何时不用"，它会**过度使用**（连"打印 hello world"也建 todo）或**完全不用**（任务太长它忘了拆）。

> 你写自己工具的 prompt 也要写"反例"。

### 教训 2：用大量 `<example>` 演示

```
<example>
User: I want to add a dark mode toggle to the application settings.
Make sure you run the tests and build when you're done!
Assistant: *Creates todo list with the following items:*
1. Creating dark mode toggle component in Settings page
2. Adding dark mode state management (context/store)
3. Implementing CSS-in-JS styles for dark theme
4. Updating existing components to support theme switching
5. Running tests and build process
*Begins working on the first task*
</example>
```

**示例比规则更能教会模型**。一个好示例顶 10 条 bullet。

### 教训 3：强调"实时更新"

```
Mark it as in_progress BEFORE beginning work.
Mark it as completed as soon as you finish it.
Do not batch up multiple tasks before marking them as completed.
```

否则模型会"集中标完"或者忘了更新，用户看不到进度。

---

## 🔧 TodoWrite 为什么是"破解健忘症"的关键

LLM 常见问题：做着做着忘了"用户最初让我干啥"。

有了 todo list：
1. **模型不停"低头看清单"** → 不会跑偏
2. **即使触发 compact，summary 也会保留 todo 状态**
3. **用户也能看到进度** → 体验更好

**你今天看到我在写 8 篇文档时一直在用 TodoWrite** —— 就是为了 8 篇写下来不跑题。

---

## 🐍 Python 简化实现

```python
# Todo 全局状态（实际应该按 session 存）
TODOS: list[dict] = []

class TodoWriteTool(Tool):
    name = "TodoWrite"
    description = "Manage task todo list"
    prompt = """Use this tool to track tasks for the current coding session.

## When to use:
- Tasks with 3+ steps
- User provides multiple things to do
- After starting work (mark in_progress)
- After finishing (mark completed)

## When NOT to use:
- Single trivial tasks
- Conversational/informational queries

## Status values:
- pending / in_progress / completed / cancelled

## Rules:
- Only ONE todo can be in_progress at a time
- Each call REPLACES the full list (not append)
"""
    input_schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content":  {"type": "string"},
                        "status":   {"type": "string",
                                     "enum": ["pending", "in_progress",
                                              "completed", "cancelled"]},
                        "priority": {"type": "string",
                                     "enum": ["high", "medium", "low"]},
                    },
                    "required": ["content", "status", "priority"],
                }
            }
        },
        "required": ["todos"],
    }

    async def call(self, todos: list) -> str:
        global TODOS
        TODOS = todos
        # 渲染当前 todos
        return f"OK: {len(TODOS)} todos updated"
```

---

## ⭐ 加深理解：TodoWrite vs auto-compact 协同

| 机制 | 治什么 |
|---|---|
| **auto-compact** | 丢历史细节，保整体连贯 |
| **TodoWrite** | 保任务框架，**哪怕 compact 也丢不了** |

具体怎么协同：
1. 跑长任务，模型边干边维护 todo
2. token 涨到 167K → 触发 compact
3. compact prompt 里强调"Pay attention to: ...Any pending todos"
4. 压缩后 messages 短了，但 todo state 通过 compact summary 被保留
5. 模型继续工作，看 todo 知道下一步该干啥

**两个机制必须配合使用**，单用任何一个都会出问题。

---

## 🔗 延伸阅读

- 上一节：[02-AutoCompact压缩](02-AutoCompact%E5%8E%8B%E7%BC%A9.md)
- 下一节：[04-ClaudeMD长期记忆](04-ClaudeMD%E9%95%BF%E6%9C%9F%E8%AE%B0%E5%BF%86.md)
- 真源码：`_source/.../src/tools/TodoWriteTool/prompt.ts:1-184`
- 工具设计 5 条：[06-设计哲学5条](../03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/06-%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A65%E6%9D%A1.md)

---

## ❓ 小测验

> 1. TodoWrite 为什么要"每次全量替换"而不是增量？
> 2. 为什么 prompt 里必须写"When NOT to use"？
> 3. TodoWrite 和 auto-compact 怎么协同？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#43-todowrite-%E5%A4%96%E9%83%A8%E8%AE%B0%E5%BF%86)

---

⬅ [02-AutoCompact压缩](02-AutoCompact%E5%8E%8B%E7%BC%A9.md)　|	➡ [04-ClaudeMD长期记忆](04-ClaudeMD%E9%95%BF%E6%9C%9F%E8%AE%B0%E5%BF%86.md)
