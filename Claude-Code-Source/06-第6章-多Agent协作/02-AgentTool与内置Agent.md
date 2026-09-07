---
tags: [Claude-Code, 第6章, AgentTool, 内置Agent]
chapter: 6
section: 6.2
---

# 6.2 AgentTool 设计 + 内置 SubAgent

⬅ [01-为什么需要SubAgent](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81SubAgent.md)　|	➡ [03-Fork与子Agent写法](03-Fork%E4%B8%8E%E5%AD%90Agent%E5%86%99%E6%B3%95.md)

> 📂 源码：`src/tools/AgentTool/`（15 个文件）+ `src/tools/AgentTool/built-in/`（6 个内置）

---

## 🔧 AgentTool 工具签名

模型这样调用：

```json
{
  "name": "Task",
  "input": {
    "description": "Find auth endpoints",
    "subagent_type": "Explore",
    "prompt": "Find all files in src/ that define HTTP routes for authentication. List them with file:line and a one-sentence purpose."
  }
}
```

| 参数 | 含义 |
|---|---|
| `description` | 短描述（UI 显示） |
| `subagent_type` | 选哪种 SubAgent（Explore/Verify/...） |
| `prompt` | ⭐ 给 SubAgent 的指令 |

主 Agent 调完会**等 SubAgent 完成**，拿到 SubAgent 的最终字符串作为 tool_result。

---

## 🗂️ 6 个内置 SubAgent

`src/tools/AgentTool/built-in/`：

| 名字 | 用途 |
|---|---|
| `exploreAgent.ts` | 搜索专家（READ-ONLY） |
| `generalPurposeAgent.ts` | 通用助手 |
| `planAgent.ts` | 规划专家 |
| `verificationAgent.ts` | 验证专家（跑测试） |
| `claudeCodeGuideAgent.ts` | Claude Code 使用问答 |
| `statuslineSetup.ts` | 状态栏配置助手 |

---

## ⭐ Explore Agent 详解（最有教学价值）

`exploreAgent.ts` 是最值得抄的样板。

### System Prompt（节选）

```
You are a file search specialist for Claude Code.
You excel at thoroughly navigating and exploring codebases.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to search and analyze existing code.
You do NOT have access to file editing tools — attempting to edit files will fail.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents
```

### 配置（双保险）

```typescript
export const EXPLORE_AGENT = {
  agentType: 'Explore',
  disallowedTools: [               // ⭐ 物理上禁这些工具
    AGENT_TOOL_NAME,                // 不能再嵌套派 agent
    EXIT_PLAN_MODE_TOOL_NAME,
    FILE_EDIT_TOOL_NAME,
    FILE_WRITE_TOOL_NAME,
    NOTEBOOK_EDIT_TOOL_NAME,
  ],
  // Ant 用 inherit（主 Agent 模型），外部用 haiku（最便宜）
  model: process.env.USER_TYPE === 'ant' ? 'inherit' : 'haiku',
  omitClaudeMd: true,              // 不加载项目 CLAUDE.md
  getSystemPrompt: () => getExploreSystemPrompt(),
}
```

---

## 💡 学习点：双保险设计

```
Prompt 上禁止 + 工具集禁止 = 双保险
```

**为啥要双保险**：
- 单靠 prompt：模型可能"忘了规则"或被 prompt injection 绕过
- 单靠工具集：模型不知道为啥不能用 → 死循环找替代品
- **双保险**：模型既知道边界，也物理上做不到

**抄作业**：你设计任何"受限 Agent"时都这么做。

---

## 💰 模型选择：用便宜的够用

```typescript
model: process.env.USER_TYPE === 'ant' ? 'inherit' : 'haiku'
```

外部 Explore Agent 用 **Haiku**（最便宜的 Claude）。
**理由**：搜索任务对模型能力要求低，不需要 Sonnet/Opus。

| 模型 | 大致定价（input） | 适合 |
|---|---|---|
| Opus | $15 / M | 复杂推理、规划 |
| Sonnet | $3 / M | 通用 coding |
| Haiku | $0.25-1 / M | 简单搜索、分类 |

**Haiku 比 Sonnet 便宜 12 倍**。
对"派 100 个 Explore SubAgent" 的大任务，**省钱效果指数级**。

---

## 🗑 `omitClaudeMd: true` 的含义

```typescript
omitClaudeMd: true,    // 不加载项目 CLAUDE.md（专注搜索）
```

为啥？
- CLAUDE.md 通常含项目业务规则、commit 风格、PR 流程
- **搜索任务不需要这些** → 加载等于浪费 token
- 主 Agent 已经有完整 CLAUDE.md，会综合处理结果

---

## 🐍 Python 简化版

```python
from dataclasses import dataclass, field

@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    model: str = "claude-sonnet-4-5"
    when_to_use: str = ""

EXPLORE_AGENT = AgentConfig(
    name="Explore",
    system_prompt=(
        "You are a READ-ONLY search specialist.\n"
        "STRICTLY PROHIBITED from: creating, modifying, or deleting files.\n"
        "Use only `search` and `read` to find information."
    ),
    allowed_tools=["search", "read", "glob"],      # 白名单
    disallowed_tools=["write", "edit", "bash"],    # 黑名单（冗余但更保险）
    model="claude-haiku-4-5",                       # ⭐ 便宜模型
    when_to_use="Search codebase for files/symbols. Returns summary.",
)

VERIFY_AGENT = AgentConfig(
    name="Verify",
    system_prompt="You verify changes by running tests.",
    allowed_tools=["read", "bash"],
    when_to_use="Verify changes pass tests. Returns pass/fail report.",
)
```

完整版见 [99-代码-递归Agent框架](99-%E4%BB%A3%E7%A0%81-%E9%80%92%E5%BD%92Agent%E6%A1%86%E6%9E%B6.md)。

---

## 🔧 AgentTool prompt：给主 Agent 看的"如何用"

`src/tools/AgentTool/prompt.ts:202` 节选：

```
Launch a new agent to handle complex, multi-step tasks autonomously.

When using the Task tool, specify a subagent_type to use a specialized agent,
or omit it to fork yourself.

Available agent types:
- Explore: Fast read-only search. Specify thoroughness: quick/medium/very thorough.
- Verify: Run tests and check outputs.
- Plan: Generate a plan but don't execute.
- ...
```

主 Agent 看完这段就知道:
- 何时派 vs 自己干
- 派哪个工种
- 怎么写 prompt

详细 prompt 写法见 [03-Fork与子Agent写法](03-Fork%E4%B8%8E%E5%AD%90Agent%E5%86%99%E6%B3%95.md)。

---

## 📂 用户自定义 SubAgent

`.claude/agents/code-reviewer.md`：

```markdown
---
name: code-reviewer
when_to_use: Review code for quality and security issues
allowed_tools: [Read, Grep, Glob]
---

You are a senior code reviewer. Focus on:
- Security vulnerabilities (OWASP top 10)
- Performance issues
- Maintainability

Always include file:line references in your feedback.
```

放进项目后，主 Agent 可以 `Task(subagent_type="code-reviewer", prompt="...")`。
**自定义 Agent = SubAgent 系统的扩展点**。

---

## 🔗 延伸阅读

- 上一节：[01-为什么需要SubAgent](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81SubAgent.md)
- 下一节：[03-Fork与子Agent写法](03-Fork%E4%B8%8E%E5%AD%90Agent%E5%86%99%E6%B3%95.md)
- 真源码：`_source/.../src/tools/AgentTool/built-in/exploreAgent.ts`

---

## ❓ 小测验

> 1. Explore Agent 用 Haiku 而不是 Sonnet 的原因？
> 2. "双保险设计"是哪两层？为什么需要两层？
> 3. `omitClaudeMd: true` 在 Explore 上为什么合理？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#62-agenttool-%E4%B8%8E%E5%86%85%E7%BD%AE)

---

⬅ [01-为什么需要SubAgent](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81SubAgent.md)　|	➡ [03-Fork与子Agent写法](03-Fork%E4%B8%8E%E5%AD%90Agent%E5%86%99%E6%B3%95.md)
