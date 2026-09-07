---
tags: [Claude-Code, 第5章, PromptCache, 成本优化]
chapter: 5
section: 5.5
---

# 5.5 Prompt Cache：90% token 省钱秘诀

⬅ [04-工具使用与红线](04-%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E4%B8%8E%E7%BA%A2%E7%BA%BF.md)　|	➡ [06-写好Prompt的checklist](06-%E5%86%99%E5%A5%BDPrompt%E7%9A%84checklist.md)

> 📂 源码：`src/constants/prompts.ts:104` 边界标记 + `src/utils/api.ts:splitSysPromptPrefix`

---

## 🎬 故事比喻：餐厅"今日菜单"的两部分

餐厅菜单分两部分：
- **固定菜单**（一年不变）：印刷精美的硬卡片，永远在桌上
- **今日特餐**（每天变）：A4 纸塞在硬卡片里

如果客人每次进店都重新印整本菜单 → 浪费纸。
聪明做法：**固定部分印一次反复用，每天只换那张 A4**。

Prompt Cache = 同样的逻辑。

---

## 🔧 Anthropic Prompt Cache 是什么

**机制**：发请求时标记某段 prompt 为"可缓存"，下次请求如果**前缀完全一样**，缓存命中 → **只算 10% 的费用**。

```python
client.messages.create(
    model="claude-sonnet-4-5",
    system=[
        {
            "type": "text",
            "text": "You are Claude Code...\n\n# Tone and style\n - Be concise...",
            "cache_control": {"type": "ephemeral"}    # ⭐ 标记为可缓存
        }
    ],
    messages=[...]
)
```

**省钱效果**：
- 普通 token: $3 / 百万 input tokens
- 缓存命中: $0.30 / 百万 (10% 价格)
- 缓存写入（首次）: $3.75 / 百万

**对长会话尤其明显**：每轮发 10K 固定 system prompt，1000 轮就省下数百美刀。

---

## 🔧 Claude Code 怎么用 Prompt Cache

`prompts.ts:104` 定义了一个**边界标记**：

```typescript
export const SYSTEM_PROMPT_DYNAMIC_BOUNDARY =
  '__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__'

// WARNING: Do not remove or reorder this marker without updating cache logic in:
// - src/utils/api.ts (splitSysPromptPrefix)
// - src/services/api/claude.ts (buildSystemPromptBlocks)
```

把 system prompt 切成两段：

```
┌──────────────────────────────┐
│ 静态部分（可缓存）           │
│ - 身份                       │
│ - 行为规则                   │
│ - 工具说明                   │
├──────────────────────────────┤
│  SYSTEM_PROMPT_DYNAMIC_BOUNDARY  ← 分界
├──────────────────────────────┤
│ 动态部分（每次变）           │
│ - CWD / 日期                 │
│ - CLAUDE.md（项目相关）       │
│ - MCP 连接（可能变）         │
│ - Skills（可能变）           │
└──────────────────────────────┘
```

**好处**：静态部分能稳定命中缓存。

---

## ⭐ 加深理解：什么内容能缓存？

```
能缓存的特征：
✅ 内容稳定（几小时不变）
✅ 长度够大（≥1024 tokens 才值得缓存）
✅ 出现在 prompt 前面
✅ 多次会话复用

不能/不该缓存：
❌ CWD、日期（每次会话变）
❌ MCP 连接（中途连断会变）
❌ 用户特定动态内容
```

---

## 🔧 systemPromptSection 包装器

`src/constants/systemPromptSections.ts` 提供两个工厂：

```typescript
// 普通：可缓存
systemPromptSection('memory', () => loadMemoryPrompt())

// 危险标记：不可缓存（必须显式写理由）
DANGEROUS_uncachedSystemPromptSection(
  'mcp_instructions',
  () => getMcpInstructionsSection(mcpClients),
  'MCP servers connect/disconnect between turns'   // ← 必须解释为啥不缓存
)
```

**强制工程师在"不能缓存时写明原因"** —— 这是非常细节的设计。后来者读代码时立刻能懂。

---

## 🐍 Python 实战：用 Anthropic SDK 的缓存

```python
import anthropic

client = anthropic.Anthropic()

# 静态部分（可缓存，约 5000 tokens）
STATIC_SYSTEM = """You are MyAgent, ...

# Tone and style
 - Be concise
 - ...

# Doing tasks
 - Read before modifying
 - ...

# Using your tools
 - Prefer dedicated tools over Bash
 - ...
"""

# 动态部分（每次重新算）
def get_dynamic_section():
    return f"""
# Environment
 - CWD: {os.getcwd()}
 - Date: {datetime.now().strftime('%Y-%m-%d')}
"""

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": STATIC_SYSTEM,
            "cache_control": {"type": "ephemeral"}    # ⭐ 标记缓存
        },
        {
            "type": "text",
            "text": get_dynamic_section()
            # 不加 cache_control → 不缓存
        }
    ],
    messages=[...]
)

# 看缓存命中
print("cache_read_tokens:", response.usage.cache_read_input_tokens)
print("cache_create_tokens:", response.usage.cache_creation_input_tokens)
```

第二次调用同一 prompt 时 `cache_read_input_tokens` 会很大，**实际花费打 10% 折**。

---

## ⭐ 加深理解：缓存的隐藏陷阱

### 陷阱 1：前缀必须**完全一致**

哪怕多一个空格，缓存就 miss。所以静态部分要**真的不变**。

### 陷阱 2：缓存有 TTL

Anthropic ephemeral cache 默认 **5 分钟** 不用就过期。
长间隔的会话可能命不中。

### 陷阱 3：太短不缓存

少于 1024 tokens 的 prompt 不缓存（API 限制）。

### 陷阱 4：动态内容塞到静态部分会破坏缓存

```python
# ❌ 错误：CWD 塞到静态前面 → 每次 CWD 变就 miss
STATIC = f"You are X. CWD: {os.getcwd()} ..."

# ✅ 正确：CWD 单独动态 section
STATIC = "You are X..."
DYNAMIC = f"CWD: {os.getcwd()}"
```

Claude Code 的 `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` 就是为了**强制分离**。

---

## 📊 真实成本对比

假设场景：
- 静态 system prompt: 8000 tokens
- 动态后缀: 500 tokens
- messages 历史: 平均 5000 tokens
- 一次会话 50 轮

|  | 不缓存 | 缓存 |
|---|---|---|
| 每轮 input | 13500 tokens | 8000 cached + 5500 fresh |
| 每轮费用 (Sonnet $3/M) | $0.0405 | $0.024 + cache write 0.03 (首轮)<br>之后 $0.024/轮 |
| 50 轮总费 | ~$2.03 | ~$1.20 (省 41%) |

实际有的项目省到 70%+，看 system prompt 长度和会话长度。

---

## 🔗 延伸阅读

- 上一节：[04-工具使用与红线](04-%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E4%B8%8E%E7%BA%A2%E7%BA%BF.md)
- 下一节：[06-写好Prompt的checklist](06-%E5%86%99%E5%A5%BDPrompt%E7%9A%84checklist.md)
- Anthropic Cache 文档：https://docs.claude.com/en/docs/build-with-claude/prompt-caching
- 真源码：`_source/.../src/constants/prompts.ts:104`

---

## ❓ 小测验

> 1. Prompt Cache 命中可以省多少费用？
> 2. 为什么 CWD 不能塞到静态 prompt 里？
> 3. `DANGEROUS_uncached...` 为什么要求传"原因"参数？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#55-prompt-cache)

---

⬅ [04-工具使用与红线](04-%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E4%B8%8E%E7%BA%A2%E7%BA%BF.md)　|	➡ [06-写好Prompt的checklist](06-%E5%86%99%E5%A5%BDPrompt%E7%9A%84checklist.md)
