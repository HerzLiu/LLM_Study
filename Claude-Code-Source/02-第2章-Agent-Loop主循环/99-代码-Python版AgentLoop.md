---
tags: [Claude-Code, 第2章, 代码合集, Python实战]
chapter: 2
section: 2.99
---

# 2.99 代码合集：Python 版 AgentLoop

⬅ [05-工业级护栏](05-%E5%B7%A5%E4%B8%9A%E7%BA%A7%E6%8A%A4%E6%A0%8F.md)　|	➡ [第 3 章 →](../03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

> 🎯 把第 2 章学的全部组合：async generator + 5 段循环 + 并发工具 + 轮次护栏。

---

## 🐍 完整代码

```python
# mini_agent_loop.py
# ============================================================
# Claude Code 主循环的 Python 简化版
# - async generator 流式
# - 区分"模型回复"和"工具结果"
# - 轮次上限护栏
# - 并发工具执行
# ============================================================

import asyncio
import json
import os
from typing import AsyncGenerator

# ---------- 1. 假工具集 ----------
async def tool_read_file(path: str) -> str:
    """异步读文件。"""
    await asyncio.sleep(0.1)                            # 模拟 IO
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"ERROR: {e}"

async def tool_list_dir(path: str) -> str:
    """列目录。"""
    await asyncio.sleep(0.1)
    try:
        return "\n".join(os.listdir(path))
    except Exception as e:
        return f"ERROR: {e}"

TOOLS = {
    "read_file": tool_read_file,
    "list_dir":  tool_list_dir,
}

# ---------- 2. 假装的流式 LLM ----------
async def fake_llm_stream(messages: list) -> AsyncGenerator[dict, None]:
    """模拟流式 LLM 输出，按当前 messages 状态决策。"""
    tool_rounds = sum(1 for m in messages if m.get("role") == "tool")

    if tool_rounds == 0:
        yield {"type": "text", "text": "我先看看目录里有什么文件..."}
        await asyncio.sleep(0.2)
        yield {"type": "tool_use", "name": "list_dir",
               "args": {"path": "."}, "id": "t1"}
        yield {"type": "stop"}
    elif tool_rounds == 1:
        yield {"type": "text", "text": "好的，我读一下 README.md"}
        await asyncio.sleep(0.2)
        yield {"type": "tool_use", "name": "read_file",
               "args": {"path": "README.md"}, "id": "t2"}
        yield {"type": "stop"}
    else:
        yield {"type": "text", "text": "已经读完了，这看起来是一个学习项目。"}
        yield {"type": "stop"}

# ---------- 3. 工具执行器（并发）----------
async def run_tools(tool_uses: list) -> list:
    """并发执行多个工具调用。模仿 toolOrchestration.runTools()"""
    async def run_one(tu):
        func = TOOLS[tu["name"]]
        result = await func(**tu["args"])
        return {
            "role": "tool",
            "tool_use_id": tu["id"],
            "name": tu["name"],
            "content": result[:300]
        }
    # ⭐ asyncio.gather 并发等所有任务
    return await asyncio.gather(*[run_one(tu) for tu in tool_uses])

# ---------- 4. 主循环（async generator）----------
async def agent_loop(
    user_input: str,
    max_turns: int = 10
) -> AsyncGenerator[dict, None]:
    """模仿 src/query.ts 的 queryLoop()。"""
    messages = [{"role": "user", "content": user_input}]
    turn = 1

    while True:                                          # ⭐ 主循环
        yield {"event": "turn_start", "turn": turn}

        # === A. 流式收 LLM 响应 ===
        tool_uses = []
        async for chunk in fake_llm_stream(messages):
            if chunk["type"] == "text":
                yield {"event": "assistant_text", "text": chunk["text"]}
            elif chunk["type"] == "tool_use":
                tool_uses.append(chunk)
                yield {"event": "tool_call",
                       "tool": chunk["name"], "args": chunk["args"]}

        messages.append({"role": "assistant", "tool_uses": tool_uses})

        # === B. 没工具 = 退出 ===
        if not tool_uses:
            yield {"event": "end", "reason": "end_turn", "turn": turn}
            return

        # === C. 执行工具 ===
        results = await run_tools(tool_uses)
        for r in results:
            yield {"event": "tool_result", "name": r["name"],
                   "content": r["content"][:80]}
            messages.append(r)

        # === D. 轮次上限 ===
        if turn + 1 > max_turns:
            yield {"event": "end", "reason": "max_turns", "turn": turn}
            return

        # === E. 进入下一轮 ===
        turn += 1

# ---------- 5. 消费方 ----------
async def main():
    async for event in agent_loop("帮我看看这个项目"):
        print(json.dumps(event, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🔧 运行

```bash
cd ~/llm-study
python3 mini_agent_loop.py
```

预期输出（节选）：
```json
{"event": "turn_start", "turn": 1}
{"event": "assistant_text", "text": "我先看看目录里有什么文件..."}
{"event": "tool_call", "tool": "list_dir", "args": {"path": "."}}
{"event": "tool_result", "name": "list_dir", "content": "..."}
{"event": "turn_start", "turn": 2}
{"event": "assistant_text", "text": "好的，我读一下 README.md"}
{"event": "tool_call", "tool": "read_file", "args": {"path": "README.md"}}
{"event": "tool_result", "name": "read_file", "content": "..."}
{"event": "turn_start", "turn": 3}
{"event": "assistant_text", "text": "已经读完了..."}
{"event": "end", "reason": "end_turn", "turn": 3}
```

---

## 📊 与 Claude Code 真源码对照

| 代码段 | 对应真源码 |
|---|---|
| `while True:` | `query.ts:307 while (true) {` |
| `async for chunk in fake_llm_stream` | `query.ts` 中 `for await (chunk of ...)` |
| `if not tool_uses: return` | `query.ts:557 toolUseBlocks.length === 0` 判断 |
| `await run_tools(tool_uses)` | `query.ts:1382 runTools(...)` |
| `if turn > max_turns:` | `query.ts:1705 if (maxTurns && nextTurnCount > maxTurns)` |
| `state` 字典 | `query.ts:268 let state: State` |

**你的 100 行 Python 把 1729 行 TS 的骨架完整还原了**。后续章节的进阶机制（权限、压缩、SubAgent）都是在这个骨架上加料。

---

## 🛠 动手练习

### ★ 基础
1. **修改 `fake_llm_stream`**：让它在某一轮一次性返回 3 个 tool_use，观察并发执行
2. **加 tool_write_file 工具**：让模型在第 3 轮调它写文件

### ★★ 进阶
3. **加 token 估算 + auto-compact**：模仿真源码，token 超阈值时调一次"假压缩"
4. **加 max_turns=1 测试**：触发轮次上限，看会发生什么
5. **加 try/except**：让某个工具偶尔抛异常，看主循环是否还能跑

### ★★★ 高级
6. **换成真 Claude API**：把 `fake_llm_stream` 换成 `anthropic.AsyncAnthropic().messages.stream(...)`
7. **加权限 callback**：在 `run_tools` 里加 `can_use_tool(tool, args)` 钩子

> 💡 第 8 章 [00-章节总览](../08-%E7%AC%AC8%E7%AB%A0-%E5%8A%A8%E6%89%8B%E5%AE%9E%E7%8E%B0MiniCC/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) 会综合做所有这些。

---

## 🔗 延伸阅读

- 上一节：[05-工业级护栏](05-%E5%B7%A5%E4%B8%9A%E7%BA%A7%E6%8A%A4%E6%A0%8F.md)
- 下一章：[第 3 章 工具系统](../03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 真源码导航：[源码地图](../%E9%99%84%E5%BD%95/%E6%BA%90%E7%A0%81%E5%9C%B0%E5%9B%BE.md)

---

⬅ [05-工业级护栏](05-%E5%B7%A5%E4%B8%9A%E7%BA%A7%E6%8A%A4%E6%A0%8F.md)　|	➡ [第 3 章 →](../03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
