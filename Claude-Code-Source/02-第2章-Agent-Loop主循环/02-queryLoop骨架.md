---
tags: [Claude-Code, 第2章, 源码拆解, 核心架构]
chapter: 2
section: 2.2
---

# 2.2 queryLoop 骨架拆解

⬅ [01-从30行到工业级](01-%E4%BB%8E30%E8%A1%8C%E5%88%B0%E5%B7%A5%E4%B8%9A%E7%BA%A7.md)　|	➡ [03-async-generator与流式](03-async-generator%E4%B8%8E%E6%B5%81%E5%BC%8F.md)

> ⭐ 把 1729 行的 `queryLoop` 剥成 30 行精华，逐行讲。

---

## 📂 源码定位

- 文件：`_source/.../src/query.ts`
- 入口：第 219 行 `export async function* query(...)`
- 主体：第 241 行 `async function* queryLoop(...)`
- 主循环：第 307 行 `while (true) {`
- 循环结束：第 1728 行 `} // while (true)`

---

## 🎬 故事比喻：外科手术现场

主刀医生（你）站在解剖台前，面对一个 1729 行的庞然大物。
你要做两件事：

1. **找到"心脏"**：哪段是真正驱动循环的核心？
2. **切除"装饰"**：日志、监控、断路器、A/B 标记…全是辅助器官

切完之后剩下的 30 行就是真正的 Agent 引擎。

---

## 🔧 极简骨架（30 行精华）

```typescript
// src/query.ts 简化版
export async function* queryLoop(params, ...) {
  let state = {
    messages: params.messages,        // 短期记忆
    toolUseContext: params.toolUseContext,
    turnCount: 1,                     // 轮次计数（防死循环）
    // ... 其他字段
  };

  while (true) {                      // ⭐ 核心：无限循环
    const { messages, turnCount } = state;

    // === A. 调 Claude API，流式收响应 ===
    yield { type: 'stream_request_start' };

    const toolUseBlocks: ToolUseBlock[] = [];   // 本轮模型想用的工具
    const assistantMessages = [];                // 本轮模型说的话

    for await (const chunk of streamFromClaude(messagesForQuery)) {
      yield chunk;                              // 边收边吐给外层（流式）
      if (chunk.type === 'tool_use') {
        toolUseBlocks.push(chunk);
      }
    }

    // === B. 没有工具调用 = 模型说完了，退出循环 ===
    if (toolUseBlocks.length === 0) {
      return { reason: 'end_turn', turnCount };
    }

    // === C. 有工具调用 = 执行工具 ===
    const toolResults = [];
    for await (const update of runTools(toolUseBlocks, ..., canUseTool, ...)) {
      yield update.message;                     // 工具进度也流式吐出去
      toolResults.push(update.message);
    }

    // === D. 检查轮次上限（防止 Agent 跑飞）===
    if (maxTurns && turnCount + 1 > maxTurns) {
      yield { type: 'max_turns_reached' };
      return { reason: 'max_turns', turnCount };
    }

    // === E. 把模型回复 + 工具结果都追加到 messages，进入下一轮 ===
    state = {
      ...state,
      messages: [...messagesForQuery, ...assistantMessages, ...toolResults],
      turnCount: turnCount + 1,
    };
    // 回到 while(true) 顶部
  }
}
```

**5 个段落 A-E**，每个段落对应 [1.3 节](../01-%E7%AC%AC1%E7%AB%A0-Agent%E5%85%A5%E9%97%A8%E4%B8%8ECC%E6%80%BB%E8%A7%88/03-%E6%9C%80%E7%AE%80%E6%A0%B8%E5%BF%83%E5%BE%AA%E7%8E%AF.md) 的 5 步：

| 段 | 干啥 | 对应 1.3 节哪一步 |
|---|---|---|
| A | 调 LLM + 收流式响应 | 第 2 步 |
| B | 看有没有 tool_use | 第 3 步 |
| C | 执行工具 | 第 4 步 |
| D | 轮次上限检查 | （护栏，1.3 没讲） |
| E | 状态更新 → 下一轮 | 第 5 步 |

---

## ⭐ 加深理解：3 个关键设计

### 设计 1：`State` 对象集中放循环变量

```typescript
let state: State = {
  messages,
  toolUseContext,
  maxOutputTokensOverride,
  autoCompactTracking,
  stopHookActive,
  maxOutputTokensRecoveryCount,
  hasAttemptedReactiveCompact,
  turnCount,
  pendingToolUseSummary,
  transition,
};
```

源码注释解释：
> "Mutable cross-iteration state. The loop body destructures this at the top of each iteration so reads stay bare-name (`messages`, `toolUseContext`). Continue sites write `state = { ... }` instead of 9 separate assignments."

**翻译**：跨轮次的可变状态打包成一个对象。每轮开头解构出来当本地变量用，要更新时整个对象重新赋值——避免 9 个变量分别赋值时漏一个。

**学习点**：当一个循环有很多状态变量时，**集中管理比散落更安全**。

### 设计 2：用 `toolUseBlocks.length === 0` 而不是 `stop_reason` 判断退出

源码第 554 行的神注释：
```typescript
// @see https://docs.claude.com/en/docs/build-with-claude/tool-use
// Note: stop_reason === 'tool_use' is unreliable -- it's not always set correctly.
```

> 翻译：API 有时候不会正确设置 `stop_reason`，所以用更稳的判断。

**这是工程师踩坑后写的注释**——记住，**生产代码里凡是"以防万一"的逻辑，背后都有故事**。

### 设计 3：流式 + 累积同时进行

注意 A 段的关键：

```typescript
for await (const chunk of streamFromClaude(messagesForQuery)) {
  yield chunk;                              // 立刻吐给外层（流式 UI）
  if (chunk.type === 'tool_use') {
    toolUseBlocks.push(chunk);              // 同时累积，等会儿执行
  }
}
```

**一边流出去（用户能看到打字效果），一边累积（等流完后处理工具）**。这是流式 agent loop 的核心技巧。

---

## 🐍 Python 版极简骨架

把上面 TypeScript 翻译成 Python（带详细注释）：

```python
async def query_loop(messages: list, max_turns: int = 30):
    """主循环骨架。yield 各种事件给外层消费。"""
    state = {"messages": messages, "turn_count": 1}

    while True:                                          # ⭐ 主循环
        turn = state["turn_count"]
        msgs = state["messages"]

        # === A. 调 LLM 流式收响应 ===
        yield {"type": "stream_start"}
        tool_uses, assistant_text = [], []
        async for chunk in stream_from_llm(msgs):
            yield chunk
            if chunk["type"] == "tool_use":
                tool_uses.append(chunk)
            elif chunk["type"] == "text":
                assistant_text.append(chunk["text"])

        # === B. 没有工具 = 退出 ===
        if not tool_uses:
            yield {"type": "end", "reason": "end_turn"}
            return

        # === C. 执行工具 ===
        tool_results = []
        async for update in run_tools(tool_uses):
            yield update
            tool_results.append(update)

        # === D. 轮次上限 ===
        if turn + 1 > max_turns:
            yield {"type": "end", "reason": "max_turns"}
            return

        # === E. 状态更新 → 下一轮 ===
        state = {
            "messages": msgs + [{"role": "assistant", "content": assistant_text}] + tool_results,
            "turn_count": turn + 1,
        }
```

完整可运行版：[99-代码-Python版AgentLoop](99-%E4%BB%A3%E7%A0%81-Python%E7%89%88AgentLoop.md)

---

## 📝 一句话小结

> **5 段动作 + State 集中管理 + tool_use 长度判退出 + 流式累积同时进行 = Claude Code 主循环精华。**

---

## 🔗 延伸阅读

- 上一节：[01-从30行到工业级](01-%E4%BB%8E30%E8%A1%8C%E5%88%B0%E5%B7%A5%E4%B8%9A%E7%BA%A7.md) —— 为什么需要工业级
- 下一节：[03-async-generator与流式](03-async-generator%E4%B8%8E%E6%B5%81%E5%BC%8F.md) —— `yield` 是怎么工作的
- 完整代码：[99-代码-Python版AgentLoop](99-%E4%BB%A3%E7%A0%81-Python%E7%89%88AgentLoop.md)
- 真源码：`_source/.../src/query.ts:241-1728`

---

## ❓ 小测验

> 1. 主循环 5 段（A-E）分别做什么？
> 2. 为什么把状态打包成 `State` 对象而不是用一堆散变量？
> 3. `toolUseBlocks.length === 0` 在循环里扮演什么角色？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#22-queryloop-%E9%AA%A8%E6%9E%B6)

---

⬅ [01-从30行到工业级](01-%E4%BB%8E30%E8%A1%8C%E5%88%B0%E5%B7%A5%E4%B8%9A%E7%BA%A7.md)　|	➡ [03-async-generator与流式](03-async-generator%E4%B8%8E%E6%B5%81%E5%BC%8F.md)
