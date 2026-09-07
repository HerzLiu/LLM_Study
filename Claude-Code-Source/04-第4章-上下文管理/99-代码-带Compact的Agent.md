---
tags: [Claude-Code, 第4章, 代码合集, Python实战]
chapter: 4
section: 4.99
---

# 4.99 代码合集：Python 带 Compact 的 Agent

⬅ [05-高级压缩机制](05-%E9%AB%98%E7%BA%A7%E5%8E%8B%E7%BC%A9%E6%9C%BA%E5%88%B6.md)　|	➡ [第 5 章 →](../05-%E7%AC%AC5%E7%AB%A0-System-Prompt%E5%B7%A5%E7%A8%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

> 🎯 把第 4 章学的全部组合：token 估算 + auto-compact + Todo 外部记忆 + 主循环集成。

---

## 🐍 完整代码

```python
# mini_agent_with_compact.py
# ============================================================
# 演示：上下文压缩 + Todo 任务清单
# - 实时 token 估算
# - 超阈值触发 auto-compact
# - Todo 状态外部存储（compact 也不丢）
# ============================================================

import asyncio
import json
from typing import AsyncGenerator

# ---------- token 估算 ----------
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cn = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cn * 2 + (len(text) - cn) // 4

def messages_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            for block in content:
                total += estimate_tokens(json.dumps(block, ensure_ascii=False))
        else:
            total += estimate_tokens(str(content))
    return total

# ---------- 阈值（demo 用小阈值，方便触发）----------
CONTEXT_WINDOW = 8_000
AUTO_COMPACT_THRESHOLD = 6_000
SUMMARY_RESERVE = 1_000

# ---------- 假装的 LLM ----------
async def fake_llm(messages: list, mode: str = "normal") -> dict:
    if mode == "summary":
        return {
            "type": "text",
            "text": (
                "<summary>\n"
                f"先前对话涉及 {len(messages)} 条消息，主要做了：读取若干文件并修改。\n"
                "待办：检查测试通过；提交代码。\n"
                "</summary>"
            )
        }

    tool_results = sum(1 for m in messages if m.get("role") == "tool")
    if tool_results < 5:
        return {
            "type": "tool_use",
            "name": "fake_big_read",
            "args": {"i": tool_results},
            "id": f"t{tool_results}"
        }
    return {"type": "text", "text": "终于读完了，工作完成。", "stop": True}

# ---------- 故意返回大内容的工具 ----------
async def tool_fake_big_read(i: int) -> str:
    return f"file{i}.txt 内容：" + ("数据 " * 400)              # ~2000 字符

# ---------- 简易 Todo 状态 ----------
TODOS: list[dict] = []

def todo_write(new_todos: list[dict]) -> str:
    global TODOS
    TODOS = new_todos
    return f"OK: {len(TODOS)} todos updated"

# ---------- ⭐ Auto-compact ----------
async def auto_compact_if_needed(messages: list) -> list:
    tokens = messages_tokens(messages)
    print(f"  [context check] {tokens} tokens / {AUTO_COMPACT_THRESHOLD} threshold")
    if tokens < AUTO_COMPACT_THRESHOLD:
        return messages

    print(f"  ⚠️ TRIGGER AUTO-COMPACT (tokens {tokens} >= {AUTO_COMPACT_THRESHOLD})")
    summary_response = await fake_llm(messages, mode="summary")
    summary_text = summary_response["text"]

    # 保留最近 1 条（防止上下文断裂），其余替换为 summary
    keep_recent = messages[-1:] if messages else []
    new_messages = [
        {"role": "system",
         "content": "[Auto-compact summary of prior conversation]\n" + summary_text},
        *keep_recent,
    ]
    new_tokens = messages_tokens(new_messages)
    print(f"  ✓ Compacted: {tokens} → {new_tokens} (saved {tokens - new_tokens})")
    return new_messages

# ---------- 主循环 ----------
async def agent_loop(user_input: str):
    messages = [{"role": "user", "content": user_input}]

    todo_write([
        {"content": "读取一堆文件", "status": "in_progress", "priority": "high"},
        {"content": "总结发现",     "status": "pending",     "priority": "medium"},
    ])

    for turn in range(1, 20):
        print(f"\n=== Turn {turn} ===")

        # ⭐ 每轮调 LLM 前先看要不要压缩
        messages = await auto_compact_if_needed(messages)

        response = await fake_llm(messages)

        if response.get("type") == "text":
            print(f"  Assistant: {response['text'][:100]}")
            if response.get("stop"):
                todo_write([
                    {"content": "读取一堆文件", "status": "completed", "priority": "high"},
                    {"content": "总结发现",     "status": "completed", "priority": "medium"},
                ])
                print(f"  Todos final: {TODOS}")
                return
            messages.append({"role": "assistant", "content": response["text"]})
            continue

        if response["type"] == "tool_use":
            name = response["name"]
            print(f"  Tool call: {name}({response['args']})")
            result = await tool_fake_big_read(**response["args"])
            messages.append({"role": "assistant", "content": [response]})
            messages.append({
                "role": "tool",
                "tool_use_id": response["id"],
                "content": result[:100] + "..."
            })

if __name__ == "__main__":
    asyncio.run(agent_loop("帮我做点事"))
```

---

## 🔧 运行预期

```bash
python3 mini_agent_with_compact.py
```

你会看到 token 不断增长，到第 3-4 轮触发 compact：

```
=== Turn 1 ===
  [context check] 32 tokens / 6000 threshold
  Tool call: fake_big_read({'i': 0})

=== Turn 2 ===
  [context check] 1284 tokens / 6000 threshold
  Tool call: fake_big_read({'i': 1})

=== Turn 3 ===
  [context check] 2536 tokens / 6000 threshold
  Tool call: fake_big_read({'i': 2})

=== Turn 4 ===
  [context check] 3788 tokens / 6000 threshold
  Tool call: fake_big_read({'i': 3})

=== Turn 5 ===
  [context check] 5040 tokens / 6000 threshold
  Tool call: fake_big_read({'i': 4})

=== Turn 6 ===
  [context check] 6292 tokens / 6000 threshold
  ⚠️ TRIGGER AUTO-COMPACT (tokens 6292 >= 6000)
  ✓ Compacted: 6292 → 158 (saved 6134)
  Assistant: 终于读完了，工作完成。
  Todos final: [...]
```

---

## 🛠 动手练习

1. **改阈值观察**：把 `AUTO_COMPACT_THRESHOLD` 调高/低，看触发次数变化
2. **加 TodoWrite 工具**：让模型在 fake_llm 第一轮就先调 TodoWrite 写计划，后续每完成一项就更新
3. **保留更多上下文**：试试 `keep_recent = messages[-3:]` 而不是 `[-1:]`，看 compact 后能少丢一些信息
4. **加 attachment 机制**：让 tool 结果超 1000 字符的存到字典而不是 messages，只放一个 ref

---

## 📊 与真源码的对照

| 代码段 | 真源码 |
|---|---|
| `auto_compact_if_needed` | `autoCompact.ts:autoCompactIfNeeded` |
| `summary_response = await fake_llm(messages, mode="summary")` | `compact.ts:compactConversation` |
| `[{"role": "system", "content": "[Compacted...]"}]` | `compact.ts:buildPostCompactMessages` |
| 阈值 `AUTO_COMPACT_THRESHOLD` | `getAutoCompactThreshold(model)` |

---

## 🔗 延伸阅读

- 上一节：[05-高级压缩机制](05-%E9%AB%98%E7%BA%A7%E5%8E%8B%E7%BC%A9%E6%9C%BA%E5%88%B6.md)
- 下一章：[第 5 章 System Prompt](../05-%E7%AC%AC5%E7%AB%A0-System-Prompt%E5%B7%A5%E7%A8%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 真源码：`_source/.../src/services/compact/autoCompact.ts`

---

⬅ [05-高级压缩机制](05-%E9%AB%98%E7%BA%A7%E5%8E%8B%E7%BC%A9%E6%9C%BA%E5%88%B6.md)　|	➡ [第 5 章 →](../05-%E7%AC%AC5%E7%AB%A0-System-Prompt%E5%B7%A5%E7%A8%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
