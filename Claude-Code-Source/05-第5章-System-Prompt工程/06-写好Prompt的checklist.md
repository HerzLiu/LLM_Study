---
tags: [Claude-Code, 第5章, 实战, checklist]
chapter: 5
section: 5.6
---

# 5.6 写好自己 Agent prompt 的 checklist

⬅ [05-PromptCache技巧](05-PromptCache%E6%8A%80%E5%B7%A7.md)　|	➡ [99-代码-Prompt拼装器](99-%E4%BB%A3%E7%A0%81-Prompt%E6%8B%BC%E8%A3%85%E5%99%A8.md)

> 🎯 把全章学的浓缩成"上线 checklist"。直接抄到自己项目。

---

## ✅ 必备 7 块

写一个新 Agent system prompt，**至少**要包含这 7 块。缺一块都可能出问题。

### ☐ 1. 身份（1 句话）
```
You are <NAME>, <ROLE>.
```
例：`You are MyBot, a database management assistant.`

**作用**：定调整段对话。**模型据此选语气、能力边界**。

---

### ☐ 2. 能力边界（能做什么/不能做什么）
```
You can: query schemas, design indexes, optimize SQL.
You cannot: execute DDL on production without explicit confirmation.
```

**作用**：减少"超纲发挥"和"过度推卸"。

---

### ☐ 3. 沟通风格（emoji / 长度 / 格式）
```
- Be concise; responses under 200 words.
- Don't use emojis unless asked.
- Use file_path:line_number for code references.
```

**作用**：让输出符合产品 UX 预期。

---

### ☐ 4. 工具使用规则（什么时候用哪个）
```
- Prefer dedicated tools over shell commands.
- Make parallel tool calls when independent.
- Mark TodoWrite items as completed immediately.
```

**作用**：避免模型 "无脑 Bash 一切"。

---

### ☐ 5. 任务流程（先 X 再 Y）
```
- Read schema before writing migration.
- Run tests before claiming task complete.
- If approach fails, diagnose before retry.
```

**作用**：教模型"工作流"。

---

### ☐ 6. 红线 & 安全（NEVER / ALWAYS）
```
NEVER:
- Modify production data without explicit confirmation
- Bypass safety checks (--no-verify, --force)
- Commit secrets to repo

ALWAYS:
- Verify changes work before claiming success
- Investigate root cause, not symptoms
```

**作用**：硬约束。

---

### ☐ 7. 环境信息（动态）
```
# Environment
- CWD: /Users/me/projects
- Date: 2026-06-03
- DB engine: PostgreSQL 16
- ...
```

**作用**：让模型知道"今天"的上下文。

---

## ⭐ 加分项（可选但显著提升质量）

### ☐ 8. Few-shot 示例

用 `<example>` 块演示理想行为：

```xml
<example>
User: "找一下所有 N+1 query"
Assistant: I'll search for ORM patterns that may cause N+1.
<tool_call: Grep(pattern="\.all\(\)\.|filter\(.*\).first\(\)")>
...
</example>
```

**1 个好 example 顶 10 条 bullet**。

---

### ☐ 9. 错误恢复指南
```
When a tool fails:
1. Read the error message carefully
2. Check input format
3. Try a more targeted fix
4. If still failing, ask user for guidance
```

---

### ☐ 10. 版本标记
```typescript
// @[MODEL LAUNCH]: Update for Sonnet 4.7
const FRONTIER_MODEL = "claude-sonnet-4-7"
```

模型升级时全局搜这个 tag，找到所有要同步更新的位置。

---

## ❌ 反模式（不要做）

| ❌ 反模式 | 后果 |
|---|---|
| 几千字散文 | 模型读不进，效果差 |
| "你要努力" 空话 | 完全没指导 |
| "尽量"、"也许" 软语 | 模型自由发挥 |
| 不写反例 | 模型不知道边界 |
| 不留版本号 | 模型升级时找不到要改哪 |
| 假设模型懂业务 | 当面对生面孔实习生 |

---

## 🔧 写 prompt 的"5 个动作"

1. **bullet 不是段落**
   - 坏：`"Always be careful with destructive operations and always confirm with the user before executing them..."`
   - 好：`" - Confirm before destructive ops\n - Confirm before pushing\n - Confirm before deleting"`

2. **NEVER/ALWAYS 不是"请"**
   - 坏：`"请不要泄露密钥"`
   - 好：`"NEVER reveal API keys in output"`

3. **解释"为什么"**
   - 坏：`"不要 force push"`
   - 好：`"NEVER force push — it can overwrite teammates' work and is hard to recover"`

4. **给具体反例**
   - 坏：`"不要懒"`
   - 好：`"Don't write 'based on your findings, fix the bug' — that delegates synthesis instead of doing it"`

5. **结构化分节**
   - 用 `# 标题` + `- bullet`
   - 模型更容易索引、用户更容易 review

---

## 📋 上线 checklist 完整版

写完 prompt 后逐条勾：

```
基础 7 块:
- [ ] 身份 1 句话
- [ ] 能力边界
- [ ] 沟通风格
- [ ] 工具使用规则
- [ ] 任务流程
- [ ] 红线（NEVER/ALWAYS）
- [ ] 环境信息

质量:
- [ ] 用 bullet 不是段落
- [ ] 每条 NEVER/ALWAYS 都有"为什么"
- [ ] 至少 1 个 few-shot example
- [ ] 版本/模型 tag
- [ ] 缓存切分（静态/动态分开）

测试:
- [ ] 跑过 5 个典型用例
- [ ] 跑过 2 个对抗用例（用户试图让 Agent 做坏事）
- [ ] 长会话不丢规则
- [ ] 内外用户差异（如果有）
```

---

## 🎯 模板：你的第一个 Agent prompt

直接抄这个开始：

```python
SYSTEM_PROMPT = """You are <NAME>, an interactive CLI <ROLE> agent.

# Tone and style
- Be concise. Reference code with file_path:line_number.
- Don't use emojis unless asked.
- Don't include colons before tool calls.

# Doing tasks
- Read files before modifying them.
- Don't expand scope beyond user request.
- Before claiming complete, verify it works.
- Report failures faithfully — don't hide errors.

# Using your tools
- Prefer dedicated tools (Read/Edit/Grep) over Bash.
- Call multiple tools in parallel when independent.
- Mark TodoWrite items as completed immediately.

# Safety
- NEVER bypass safety checks (--no-verify, --force).
- ALWAYS confirm before destructive ops affecting shared state.
- Investigate root cause, not symptoms.

# Environment
- CWD: {cwd}
- Date: {date}
"""
```

后续每个项目按需 fork 这个模板。

---

## 🔗 延伸阅读

- 上一节：[05-PromptCache技巧](05-PromptCache%E6%8A%80%E5%B7%A7.md)
- 下一节：[99-代码-Prompt拼装器](99-%E4%BB%A3%E7%A0%81-Prompt%E6%8B%BC%E8%A3%85%E5%99%A8.md)
- 完整 MiniCC system prompt：`~/llm-study/minicc/minicc.py:build_system_prompt`

---

## ❓ 小测验

> 1. 默写必备 7 块
> 2. 反模式中最致命的一条是什么？为什么？
> 3. "NEVER" 比 "请不要" 强在哪？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#56-checklist)

---

⬅ [05-PromptCache技巧](05-PromptCache%E6%8A%80%E5%B7%A7.md)　|	➡ [99-代码-Prompt拼装器](99-%E4%BB%A3%E7%A0%81-Prompt%E6%8B%BC%E8%A3%85%E5%99%A8.md)
