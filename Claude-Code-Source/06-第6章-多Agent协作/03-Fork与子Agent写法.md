---
tags: [Claude-Code, 第6章, Fork, SubAgent, Prompt写法]
chapter: 6
section: 6.3
---

# 6.3 Fork vs Subagent + 写好 prompt

⬅ [02-AgentTool与内置Agent](02-AgentTool%E4%B8%8E%E5%86%85%E7%BD%AEAgent.md)　|	➡ [04-并发与反模式](04-%E5%B9%B6%E5%8F%91%E4%B8%8E%E5%8F%8D%E6%A8%A1%E5%BC%8F.md)

> 📂 源码：`src/tools/AgentTool/prompt.ts:100-220` + `forkSubagent.ts`

---

## 🎬 故事比喻：派"陌生人"还是派"克隆"

主管要派人去做事，两种选择：

| 派谁 | 类比 |
|---|---|
| **陌生新员工**（Subagent） | 完全不知道公司情况，必须把背景全告诉他 |
| **你的克隆**（Fork） | 知道你知道的一切，只用告诉他"做什么" |

Claude Code 的 Task 工具支持这两种：
- 指定 `subagent_type` → 派陌生新员工（专业工种）
- 不指定 → Fork（克隆自己）

---

## 📊 Fork vs Subagent 对比

| 维度 | Fork | Subagent |
|---|---|---|
| **上下文** | 继承主 Agent 全部历史 | 空白重新开始 |
| **工具集** | 同主 Agent | 子类型限定 |
| **Prompt cache** | ✅ 共享主 Agent 缓存（省钱）| ❌ 独立缓存 |
| **prompt 风格** | "做什么"（背景不解释） | "做什么 + 全部背景" |
| **典型用途** | 任务延展、研究 | 专门化（搜索/验证）|
| **是否能改主 context** | 否 | 否 |

---

## 🔧 Fork 的源码描述

`src/tools/AgentTool/prompt.ts` 中（当 `isForkSubagentEnabled()` 为 true）：

```
## When to fork

Fork yourself (omit `subagent_type`) when the intermediate tool output
isn't worth keeping in your context.
The criterion is qualitative — "will I need this output again" — not task size.

- Research: fork open-ended questions
- Implementation: prefer to fork implementation work that requires more than a couple of edits

Forks are cheap because they share your prompt cache.
Don't set `model` on a fork — a different model can't reuse the parent's cache.

**Don't peek.** The tool result includes an output_file path — do NOT Read
or tail it unless the user explicitly asks for a progress check. You get
a completion notification; trust it.

**Don't race.** After launching, you know nothing about what the fork found.
Never fabricate or predict fork results.
```

**3 条铁律**：
1. **Don't peek**：派完别去看进度文件，会破坏隔离
2. **Don't race**：别假设 fork 的结果
3. **Don't change model**：换模型会破坏 cache 共享

---

## 🔧 怎么写 SubAgent 的 prompt（金句）

`prompt.ts:103` 中：

```
Brief the agent like a smart colleague who just walked into the room —
it hasn't seen this conversation, doesn't know what you've tried,
doesn't understand why this task matters.

- Explain what you're trying to accomplish and why.
- Describe what you've already learned or ruled out.
- Give enough context about the surrounding problem that the agent can
  make judgment calls rather than just following a narrow instruction.
- If you need a short response, say so ("report in under 200 words").
- Lookups: hand over the exact command.
  Investigations: hand over the question.

Terse command-style prompts produce shallow, generic work.

**Never delegate understanding.** Don't write "based on your findings, fix the bug"
or "based on the research, implement it." Those phrases push synthesis onto
the agent instead of doing it yourself. Write prompts that prove you understood:
include file paths, line numbers, what specifically to change.
```

---

## ⭐ 加深理解：3 条 prompt 写法精华

### 精华 1："新员工不知道历史"

SubAgent 没有主 Agent 的对话历史。
**所有相关背景必须塞进 prompt**。

❌ 坏 prompt（fork 可以这么写，subagent 不行）：
```
"Continue from where I left off"
"Fix the bug we discussed"
```

✅ 好 prompt：
```
"Fix the off-by-one bug at src/utils.py:42.
The function `compute_offset` returns N+1 instead of N.
Context: we're refactoring pagination logic; the bug appeared after 
PR #123 changed the index base from 0 to 1."
```

### 精华 2："请求具体"

模糊请求 → 模糊答案。

❌ "找一下相关代码"
✅ "找 src/ 下所有定义 HTTP 路由的文件。报告格式: file:line + 一句话用途"

### 精华 3："不要委托理解"

最致命的反模式：把综合判断扔给 SubAgent。

❌ "看完了帮我修 bug"
- SubAgent 不知道你的判断
- SubAgent 可能修错地方
- 你失去了对 bug 根因的把控

✅ "在 src/auth.py:42 把 `==` 改成 `===`，原因是字符串比较应该用严格相等"
- 你已经做了诊断
- SubAgent 只负责机械执行

---

## 📋 写 SubAgent prompt 的 5 句模板

```
1. [What] 我希望你 [具体动作]
2. [Why] 因为 [上下文/原因]
3. [Constraints] 边界：[做什么 / 不做什么]
4. [Format] 报告格式：[字数 / 结构]
5. [Examples] （可选）类似任务的例子
```

例子：
```python
Task(
  subagent_type="Explore",
  prompt="""
[What] 找 src/ 下所有定义 HTTP 路由的文件。
[Why] 我们要审查认证模块，需要先盘清入口。
[Constraints] 只看 Python 文件；忽略 tests/ 和 vendor/。
[Format] 列表，每行：`file:line - 一句话用途`，总共不超过 200 字。
"""
)
```

---

## 🐍 Python Fork 实现

```python
async def fork_self(prompt: str) -> str:
    """Fork: 继承主 Agent 的 messages，但跑独立循环。"""
    # ⭐ 复制主 Agent 当前 messages（不是引用）
    sub_messages = self.messages.copy()
    # 加一条 user prompt 作为新任务
    sub_messages.append({"role": "user", "content": prompt})

    sub_agent = Agent(
        config=self.config,        # 同主 Agent 配置
        messages=sub_messages      # ⭐ 继承上下文
    )
    result = await sub_agent.run_one_task()
    return result                  # 只返回字符串

async def spawn_subagent(self, type: str, prompt: str) -> str:
    """Subagent: 全新 context，专门 prompt。"""
    cfg = SUBAGENTS[type]
    sub_messages = [
        # ⭐ 不继承主 messages，从专门 system prompt 开始
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": prompt}
    ]
    sub_agent = Agent(config=cfg, messages=sub_messages)
    return await sub_agent.run_one_task()
```

完整版见 [99-代码-递归Agent框架](99-%E4%BB%A3%E7%A0%81-%E9%80%92%E5%BD%92Agent%E6%A1%86%E6%9E%B6.md)。

---

## ⚠️ 关键陷阱

### 陷阱 1：Fork 改模型 → cache miss

```python
# ❌ 错误
fork_self(prompt="...", model="haiku")     # 父用 sonnet 子用 haiku → 完全 cache miss

# ✅ 正确
fork_self(prompt="...")                     # 继承父模型
```

### 陷阱 2：偷看 SubAgent 进度

```python
# ❌ 错误
sub_result = await spawn_subagent(...)
mid_progress = await read("/tmp/sub_progress.log")  # 把所有原始输出拉回主 context
```

### 陷阱 3：让 SubAgent 自己派 SubAgent

容易嵌套爆炸。Explore Agent 通过 `disallowedTools: [AGENT_TOOL_NAME]` 物理禁掉。

---

## 🔗 延伸阅读

- 上一节：[02-AgentTool与内置Agent](02-AgentTool%E4%B8%8E%E5%86%85%E7%BD%AEAgent.md)
- 下一节：[04-并发与反模式](04-%E5%B9%B6%E5%8F%91%E4%B8%8E%E5%8F%8D%E6%A8%A1%E5%BC%8F.md)
- 真源码：`_source/.../src/tools/AgentTool/prompt.ts:100-220`

---

## ❓ 小测验

> 1. Fork vs Subagent 最大区别？
> 2. "Don't peek" 是什么意思？为什么？
> 3. "Never delegate understanding" 怎么理解？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#63-fork-vs-subagent)

---

⬅ [02-AgentTool与内置Agent](02-AgentTool%E4%B8%8E%E5%86%85%E7%BD%AEAgent.md)　|	➡ [04-并发与反模式](04-%E5%B9%B6%E5%8F%91%E4%B8%8E%E5%8F%8D%E6%A8%A1%E5%BC%8F.md)
