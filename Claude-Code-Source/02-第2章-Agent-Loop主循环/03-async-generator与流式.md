---
tags: [Claude-Code, 第2章, Python语法, 异步编程, 流式]
chapter: 2
section: 2.3
---

# 2.3 async generator + yield：流式响应的秘密

⬅ [02-queryLoop骨架](02-queryLoop%E9%AA%A8%E6%9E%B6.md)　|	➡ [04-工具执行的两种模式](04-%E5%B7%A5%E5%85%B7%E6%89%A7%E8%A1%8C%E7%9A%84%E4%B8%A4%E7%A7%8D%E6%A8%A1%E5%BC%8F.md)

> ⭐ 没有它，你的 Agent 会"卡 30 秒不动"。

---

## 🎬 故事比喻：厨师 vs 服务员的两种端菜方式

| 方式 | 类比 | 用户体验 |
|---|---|---|
| **整盘端** | 厨师炒完整盘菜才端 | 你坐着等 30 分钟，啥都看不到 |
| **小碟试** | 每炒好一勺就用小碟端给你"先尝尝" | 你一直在吃，全程不无聊 |

普通函数 = 整盘端。
**Generator（生成器）= 小碟试**。

LLM 生成 30 秒文本时，如果用"整盘端"，用户看到空白终端 30 秒；如果用"小碟试"，用户看到字一个个蹦出来，体验天差地别。

---

## 🔧 核心概念三件套

### 1. `async def`：声明协程

```python
async def foo():           # "协程函数"。调用 foo() 不会真跑，只返回协程对象
    await asyncio.sleep(1) # await：等另一个协程，期间让出 CPU
    return 42

# 怎么调用？
result = foo()              # ❌ 没真跑，只拿到协程对象
result = await foo()        # ✅ 真跑，但只能在 async def 里写
asyncio.run(foo())          # ✅ 从同步代码进入异步世界的标准入口
```

### 2. `yield`：让函数变成生成器

```python
def gen():                 # 普通函数 + yield = 生成器
    yield 1
    yield 2
    yield 3

for x in gen():
    print(x)               # 输出 1 2 3
```

`yield` 的魔法：函数执行到 `yield` 就**暂停**，返回值给调用方；下次循环再从暂停处继续。

### 3. `async def` + `yield` = **async generator**

```python
async def my_gen():        # 既是协程，又是生成器
    yield "hello"
    await asyncio.sleep(1)
    yield "world"

# 消费时必须用 async for
async def consume():
    async for x in my_gen():
        print(x)
```

**Claude Code 主循环就是这种**：`async function* queryLoop(...)`（TypeScript 语法，等价于 Python 的 `async def` + `yield`）。

---

## ⭐ 加深理解：为什么用 generator 而不是普通函数？

### 问题：普通函数必须"全部跑完才返回"

```python
async def run_agent_bad(user_input):
    messages = [...]
    while True:
        response = await call_llm(messages)
        # ... 干一堆活 ...
        if done:
            return final_text             # 30 秒后才 return
```

外层调用者：
```python
result = await run_agent_bad("...")        # 阻塞 30 秒
print(result)                               # 30 秒后才打印
```

用户感受：**程序假死 30 秒**。

### 用 generator 改造

```python
async def run_agent_good(user_input):
    messages = [...]
    while True:
        async for chunk in stream_llm(messages):
            yield {"type": "text", "data": chunk}    # 每收到一个 chunk 立刻吐出去
        # 工具调用进度也 yield
        yield {"type": "tool_call", "tool": "Read"}
        yield {"type": "tool_result", "ok": True}
        if done:
            yield {"type": "end"}
            return
```

外层：
```python
async for event in run_agent_good("..."):
    render(event)                           # 实时渲染每个事件
```

用户感受：**字一个个蹦出来，工具执行进度也能看到**。

---

## 🐍 完整可运行 demo

```python
# async_gen_demo.py
import asyncio

# ---------- 模拟一个"流式" LLM ----------
async def stream_llm():
    """每 0.3 秒吐一个字。"""
    for char in "Hello! 我是 Claude。":
        await asyncio.sleep(0.3)            # 模拟网络延迟
        yield char

# ---------- Agent loop（async generator）----------
async def agent():
    yield {"event": "start"}
    text = ""
    async for ch in stream_llm():
        text += ch
        yield {"event": "delta", "char": ch}   # 每个字单独发事件
    yield {"event": "end", "final": text}

# ---------- 消费方（外层 UI）----------
async def main():
    async for ev in agent():
        if ev["event"] == "start":
            print("[开始]", end="", flush=True)
        elif ev["event"] == "delta":
            print(ev["char"], end="", flush=True)   # 实时打印
        elif ev["event"] == "end":
            print(f"\n[结束] 完整文本: {ev['final']}")

asyncio.run(main())
```

跑一下：
```bash
python3 async_gen_demo.py
```

你会看到字**一个个**蹦出来，全程 ~5 秒。这就是流式体验。

---

## 🔧 Claude Code 主循环里的 yield

回看上一节的骨架：

```typescript
yield { type: 'stream_request_start' };       // 告诉 UI "我开始请求 API 了"

for await (const chunk of streamFromClaude(...)) {
  yield chunk;                                // 每个 chunk 立刻吐
}

for await (const update of runTools(...)) {
  yield update.message;                       // 工具进度也吐
}

return { reason: 'end_turn' };                // generator 也能 return（标志结束）
```

**外层（CLI / SDK / Web UI）就根据这些事件类型实时渲染**。比如 CLI 看到 `stream_request_start` 显示菊花图标，看到 `chunk` 就打印文字，看到 `tool_call` 就显示工具图标。

---

## ⚠️ 小白避坑

1. **协程对象 ≠ 已运行**
   ```python
   foo()                                # 只拿到协程对象
   asyncio.create_task(foo())           # 才真正调度它跑
   await foo()                          # 跑并等结果（只能在 async def 里）
   ```

2. **`for` vs `async for`**
   - 普通 generator 用 `for`
   - async generator 用 `async for`
   - 用错会报 `TypeError: object is not iterable` 或类似

3. **`yield` 在 `async def` 里就变成 async generator**
   - 不需要特殊导入
   - 但**不能再用 `return value`**（只能 `return` 不带值，或抛 `StopAsyncIteration`）

4. **不要在同步代码里 await**
   - 所有 await 必须在 `async def` 里
   - 入口处 `asyncio.run(main())` 桥接

---

## 🔗 延伸阅读

- 上一节：[02-queryLoop骨架](02-queryLoop%E9%AA%A8%E6%9E%B6.md) —— `yield` 在主循环里的位置
- 下一节：[04-工具执行的两种模式](04-%E5%B7%A5%E5%85%B7%E6%89%A7%E8%A1%8C%E7%9A%84%E4%B8%A4%E7%A7%8D%E6%A8%A1%E5%BC%8F.md) —— 流式工具执行
- 完整代码：[99-代码-Python版AgentLoop](99-%E4%BB%A3%E7%A0%81-Python%E7%89%88AgentLoop.md)
- 真源码：`_source/.../src/query.ts:219` 的 `async function*`

---

## ❓ 小测验

> 1. `async def` + `yield` 组合叫什么？
> 2. 为什么 Claude Code 主循环必须用 async generator 而不是普通 async 函数？
> 3. 消费 async generator 用什么关键字？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#23-async-generator)

---

⬅ [02-queryLoop骨架](02-queryLoop%E9%AA%A8%E6%9E%B6.md)　|	➡ [04-工具执行的两种模式](04-%E5%B7%A5%E5%85%B7%E6%89%A7%E8%A1%8C%E7%9A%84%E4%B8%A4%E7%A7%8D%E6%A8%A1%E5%BC%8F.md)
