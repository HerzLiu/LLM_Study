---
tags: [Claude-Code, 第4章, Token, ContextWindow]
chapter: 4
section: 4.1
---

# 4.1 Token 与 Context Window 基础

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-AutoCompact压缩](02-AutoCompact%E5%8E%8B%E7%BC%A9.md)

---

## 🎬 故事比喻：A4 纸大小的工作记忆

研究表明：人类工作记忆约能同时记 7±2 个项目。
LLM 比这强很多，但也有上限——这个上限叫 **context window**。

类比：你的助理工作记忆 = **一张 A4 纸**。
- 每说一句话，都要写到 A4 上
- 做事时要"看着 A4 做"
- 写满了，就忘掉最早的

Claude Sonnet 4.5 的"A4 纸"大约能装 **200K tokens**，但用得快爆。

---

## 🔧 Token 是什么？

**Token ≠ 字符 ≠ 单词**。Token 是 LLM 处理文本时的最小单位。

| 类型 | 估算 |
|---|---|
| 英文 | 1 token ≈ 0.75 个单词 |
| 中文 | 1 个汉字 ≈ 1.5-2 tokens |
| 代码 | 1 行约 8-15 tokens |

**例子**：
- "Hello, world!" ≈ 4 tokens
- "你好世界" ≈ 6-8 tokens
- 一个 100 行 Python 文件 ≈ 800-1500 tokens

> 💡 真实计数用 Anthropic 的 `count_tokens` API，但日常估算 `字符数 / 3.5` 就够了。

---

## 📊 Claude Code 的阈值结构

`src/services/compact/autoCompact.ts` 里的常量：

```typescript
// 输出预留（最长 compact 摘要的 p99.99）
const MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000

// auto-compact 触发阈值的 buffer
export const AUTOCOMPACT_BUFFER_TOKENS = 13_000
export const WARNING_THRESHOLD_BUFFER_TOKENS = 20_000
export const ERROR_THRESHOLD_BUFFER_TOKENS = 20_000
```

**阈值计算**（以 200K context window 为例）：

```
0 ──────────────────────────────────── 200K
                                              │
                                有效窗口 ≈ 180K  │  预留 20K 给输出
                                              │
                          auto-compact 阈值 ≈ 167K  │  - 13K buffer
                                              │
                       警告/错误阈值 ≈ 147K  │  - 20K buffer
                                              │
   <─── 正常工作 ──>│<─ 警告区 ─>│<─ 触发压缩 ─>│
```

| 阶段 | token | 行为 |
|---|---|---|
| 正常 | < 147K | 啥也不做 |
| 警告 | 147K-167K | CLI 黄字警告 |
| **触发压缩** | **≥ 167K** | **自动 auto-compact** |
| 硬上限 | 180K | API 报 prompt_too_long |
| 物理 | 200K | 完全爆 |

**为啥这么多 buffer？**
- 模型生成回复要空间（≤ 20K）
- compact 自己也要发 prompt，要算开销
- 防 API 报错

---

## 🐍 Python token 估算

```python
def estimate_tokens(text: str) -> int:
    """粗略估算 token。生产用 anthropic.count_tokens()"""
    if not text:
        return 0
    cn = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')   # 中文字符
    return cn * 2 + (len(text) - cn) // 4                    # 英文按 1/4

def messages_tokens(messages: list) -> int:
    """整个 messages 的估算。"""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):                        # 结构化（工具调用块）
            for block in content:
                total += estimate_tokens(json.dumps(block, ensure_ascii=False))
        else:
            total += estimate_tokens(str(content))
    return total
```

---

## ⭐ 加深理解：为什么 token 涨这么快？

设想一个普通任务流程：

| 行为 | 增加 tokens |
|---|---|
| 用户问 | 50 |
| Read 一个 500 行文件 | ~5,000 |
| Grep 100 个匹配 | ~3,000 |
| 模型分析回复 | ~2,000 |
| Edit 文件（含原文+新文） | ~2,000 |
| Bash npm test（5000 行输出） | ~30,000 |
| 又 Read 5 个相关文件 | ~10,000 |
| 模型综合回复 | ~3,000 |
| **一轮下来** | **~55K** |

跑 3-4 轮就接近阈值。**所以 auto-compact 不是可选项，是必需品**。

---

## 🔧 真计数 vs 估算

源码 `src/utils/tokens.ts:tokenCountWithEstimation`：
- **真计数**：调 `client.messages.count_tokens(...)`（需要 API 调用，慢）
- **估算**：本地按字符数算（快但粗略）

**策略**：
- 估算用于"是否快爆了"的实时判断（每轮都算）
- 真计数用于关键决策（要不要 compact）

---

## 🔗 延伸阅读

- 上一节：[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一节：[02-AutoCompact压缩](02-AutoCompact%E5%8E%8B%E7%BC%A9.md) —— 真正怎么压
- 真源码：`_source/.../src/services/compact/autoCompact.ts:30-90`

---

## ❓ 小测验

> 1. Claude Sonnet 4.5 的 context window 大约多大？
> 2. Claude Code 为什么留这么多 buffer？
> 3. 一轮普通编程任务大约消耗多少 token？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#41-token-%E4%B8%8E-contextwindow)

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-AutoCompact压缩](02-AutoCompact%E5%8E%8B%E7%BC%A9.md)
