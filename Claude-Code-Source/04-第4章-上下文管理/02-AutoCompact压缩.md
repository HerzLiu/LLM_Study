---
tags: [Claude-Code, 第4章, AutoCompact, Prompt工程]
chapter: 4
section: 4.2
---

# 4.2 解药一：Auto-Compact 历史压缩

⬅ [01-Token与ContextWindow](01-Token%E4%B8%8EContextWindow.md)　|	➡ [03-TodoWrite外部记忆](03-TodoWrite%E5%A4%96%E9%83%A8%E8%AE%B0%E5%BF%86.md)

> 📂 源码：`src/services/compact/autoCompact.ts` + `compact.ts` + `prompt.ts`

---

## 🎬 故事比喻：会议纪要替换原始录音

开了一上午会，60 分钟原始录音 → 转写后几万字。
你下午要继续开会，但不能再带这几万字进会议室。
**怎么办？让秘书写一份"上午会议纪要"（500 字），下午开会时给在场的人看纪要而不是听录音。**

Auto-compact = 让另一个 LLM 给你的对话历史"做会议纪要"。

---

## 🔧 完整流程（5 步）

```
当前 messages（150K tokens）
        ↓
1. 检测：tokens >= 167K？是 → 触发
2. 派一个"压缩小代理"（同模型或便宜模型）
3. 给它特殊 prompt："总结上面对话，保留关键信息"
4. 小代理输出 <summary>...</summary>
5. 用 summary 替换原 messages
        ↓
新 messages（5K-15K tokens）
        ↓
主循环继续
```

视觉化：

```
压缩前：
[user: 帮我重构 utils]
[assistant: 我先看一下]  [tool_use Read utils.ts]
[tool_result: <500 行代码>]
[assistant: 嗯...]  [tool_use Edit ...]
[tool_result: edited]
[assistant: 跑测试]  [tool_use Bash npm test]
[tool_result: <5000 行>]
... 又 30 条

           ↓ 压缩 ↓

压缩后：
[system: 这是之前对话的总结]
[assistant: <压缩摘要：用户要重构 utils；
            我已经修改了 foo() 和 bar()；
            测试通过；待办：处理 baz() 边界>]
[user: 继续]
```

通常 **150K → 5-15K，省 90%**。

---

## 🔧 关键 prompt（源码摘抄）

`src/services/compact/prompt.ts` 的核心：

```
Before providing your final summary, wrap your analysis in <analysis> tags
to organize your thoughts and ensure you've covered all necessary points.
In your analysis process:

1. Chronologically analyze each message and section of the conversation.
   For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like: file names, full code snippets, function signatures, file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback...
2. Double-check for technical accuracy and completeness...
```

**学习点**：
- 用 `<analysis>` 块让模型"思考"，后面剥离只保留 `<summary>`
- **明确告诉模型保留什么**：文件名、代码片段、函数签名、错误。否则它会泛泛而谈
- 强调"按时间顺序" → 输出连贯不跳跃

---

## ⭐ 加深理解：阻止"二次工具调用"

源码里有这段神奇 preamble：

```typescript
const NO_TOOLS_PREAMBLE = `CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.
- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text...
```

**注释亮点**：
> "Sonnet 4.6 sometimes attempts a tool call despite the weaker trailer instruction. With maxTurns: 1, a denied tool call means no text output → falls through to the streaming fallback (2.79% on 4.6 vs 0.01% on 4.5)."

**翻译**：
- Claude 4.6 比 4.5 更喜欢调工具
- 压缩任务里它会乱调 → 浪费唯一机会
- 工程师把"绝对不要调工具"放最前 + 解释拒绝后果

**这是 prompt 工程的真实工业级细节**：
- 不能客气地说"请不要调工具"
- 要直接喊 `CRITICAL`
- 要解释**后果**："Tool calls will be REJECTED and will waste your only turn"

---

## 🔧 失败的断路器

```typescript
const MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3
// BQ 2026-03-10: 1,279 sessions had 50+ consecutive failures (up to 3,272)
// in a single session, wasting ~250K API calls/day globally.
```

**真实数据驱动决策**：
- 工程师查 BigQuery 发现：1279 个会话连续压缩失败 50+ 次
- 最严重的一个会话失败 3272 次
- 每天浪费 25 万 API 调用

**对策**：连续失败 3 次就停手，避免无限烧钱。

**学习点**：
- 阈值不是拍脑袋，是查数据定的
- 注释里写日期"BQ 2026-03-10" → 可追溯
- 任何重试逻辑都要有"断路器"

---

## 🐍 Python 简化实现

```python
async def auto_compact(client, messages: list) -> list:
    """触发 auto-compact。"""
    print("⚠️ Triggering auto-compact...")

    # 把对话拼成纯文本
    convo_text = "\n".join(
        f"[{m['role']}] " + str(m.get('content', ''))[:500]
        for m in messages
    )

    # ⭐ 关键 prompt：禁止工具调用
    summary_prompt = (
        "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n"
        "Summarize the conversation above. Preserve:\n"
        "- User's original intent\n"
        "- Key file paths and code snippets\n"
        "- Important errors and how they were resolved\n"
        "- Any pending todos\n\n"
        "Wrap your summary in <summary></summary> tags."
    )

    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        # ⚠️ 不传 tools 参数 → 物理上不让它调
        messages=[{
            "role": "user",
            "content": f"Conversation so far:\n{convo_text}\n\n{summary_prompt}"
        }],
    )
    summary = resp.content[0].text
    print(f"✓ Compacted to {len(summary)} chars")

    # 用 summary 替换全部历史
    return [{
        "role": "user",
        "content": f"[Compacted summary of prior conversation]\n{summary}"
    }]

# 主循环里集成
while True:
    if estimate_tokens(messages) > AUTO_COMPACT_THRESHOLD:
        messages = await auto_compact(client, messages)
    response = call_llm(messages, tools=TOOLS)
    # ... 正常处理 ...
```

完整版见 [99-代码-带Compact的Agent](99-%E4%BB%A3%E7%A0%81-%E5%B8%A6Compact%E7%9A%84Agent.md)。

---

## ⭐ 加深理解：什么信息保留 / 什么可以丢

| 必须保留 | 可以丢 |
|---|---|
| 用户原始意图 | 中间思考过程 |
| 文件名和路径 | 完整代码片段（可压成摘要）|
| 关键技术决策 | 重复确认的对话 |
| 错误信息和修复方法 | 工具结果的细节（保留结论）|
| 用户反馈和偏好 | 临时变量、调试输出 |
| 待办事项 | 失败的尝试细节（保留教训）|

**指导思想**：保留"下次决策需要的信息"，丢掉"过程性细节"。

---

## 🔗 延伸阅读

- 上一节：[01-Token与ContextWindow](01-Token%E4%B8%8EContextWindow.md)
- 下一节：[03-TodoWrite外部记忆](03-TodoWrite%E5%A4%96%E9%83%A8%E8%AE%B0%E5%BF%86.md)
- 完整 Python 实现：[99-代码-带Compact的Agent](99-%E4%BB%A3%E7%A0%81-%E5%B8%A6Compact%E7%9A%84Agent.md)
- 真源码：`_source/.../src/services/compact/prompt.ts:30-100`

---

## ❓ 小测验

> 1. Auto-compact 5 步流程
> 2. 为什么 NO_TOOLS_PREAMBLE 要解释"后果"？
> 3. `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3` 的依据？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#42-auto-compact)

---

⬅ [01-Token与ContextWindow](01-Token%E4%B8%8EContextWindow.md)　|	➡ [03-TodoWrite外部记忆](03-TodoWrite%E5%A4%96%E9%83%A8%E8%AE%B0%E5%BF%86.md)
