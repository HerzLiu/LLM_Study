---
tags: [Claude-Code, 第5章, 代码合集, Python实战]
chapter: 5
section: 5.99
---

# 5.99 代码合集：分节式 Prompt 拼装器

⬅ [06-写好Prompt的checklist](06-%E5%86%99%E5%A5%BDPrompt%E7%9A%84checklist.md)　|	➡ [第 6 章 →](../06-%E7%AC%AC6%E7%AB%A0-%E5%A4%9AAgent%E5%8D%8F%E4%BD%9C/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

> 🎯 把第 5 章学的全部组合：分节 + 边界 + 标记 + 渲染。

---

## 🐍 完整代码

```python
# system_prompt_builder.py
# ============================================================
# 模仿 Claude Code 的 system prompt 分节式拼装器
# - 每节是 PromptSection
# - 区分 cacheable 和 dangerous_uncached
# - 渲染时输出 prompt + debug 信息
# ============================================================

import os
import datetime
from dataclasses import dataclass
from typing import Callable

# ============================================================
# 1. Section 数据类
# ============================================================
@dataclass
class PromptSection:
    name: str                                   # 唯一名字（调试用）
    content_fn: Callable[[], str | None]        # 返回内容的函数（懒求值）
    cacheable: bool = True
    reason_uncacheable: str = ""                # 不可缓存时必须填原因

def section(name: str, fn: Callable[[], str | None]) -> PromptSection:
    """可缓存 section。"""
    return PromptSection(name=name, content_fn=fn, cacheable=True)

def dangerous_uncached_section(name: str, fn, reason: str) -> PromptSection:
    """不可缓存 section，强制要求理由。"""
    if not reason:
        raise ValueError(f"Must provide reason for uncached section '{name}'")
    return PromptSection(name=name, content_fn=fn,
                         cacheable=False, reason_uncacheable=reason)

# ============================================================
# 2. 各 section 内容
# ============================================================
DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"

def get_identity():
    return "You are MiniCC, an interactive CLI coding agent."

def get_tone_and_style():
    rules = [
        "Only use emojis if the user explicitly requests it.",
        "Your responses should be short and concise.",
        "When referencing code, use file_path:line_number format.",
        "Do not use a colon before tool calls.",
    ]
    return "# Tone and style\n" + "\n".join(f" - {r}" for r in rules)

def get_doing_tasks():
    rules = [
        "Don't add features beyond what was asked.",
        "Read files before modifying them.",
        "Don't create files unless absolutely necessary.",
        "If an approach fails, diagnose before switching tactics.",
        "Before reporting task complete, verify it actually works.",
        "Report outcomes faithfully — never claim success when output shows failure.",
    ]
    return "# Doing tasks\n" + "\n".join(f" - {r}" for r in rules)

def get_using_tools():
    rules = [
        "Prefer dedicated tools over Bash (Read > cat, Edit > sed, Grep > grep).",
        "Call multiple tools in parallel when independent.",
        "Mark each todo as completed as soon as you finish it.",
    ]
    return "# Using your tools\n" + "\n".join(f" - {r}" for r in rules)

def get_safety():
    rules = [
        "Dangerous commands (rm -rf /, mkfs, fork bombs) are auto-blocked.",
        "NEVER bypass safety checks (--no-verify, --force).",
        "ALWAYS confirm before destructive ops affecting shared state.",
    ]
    return "# Safety\n" + "\n".join(f" - {r}" for r in rules)

# ---- 动态 sections ----

def get_env_info():
    return (
        "# Environment\n"
        f" - CWD: {os.getcwd()}\n"
        f" - Date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n"
        f" - Platform: {os.uname().sysname}"
    )

def get_claude_md():
    """从当前目录或 ~/CLAUDE.md 加载。"""
    for path in ["CLAUDE.md", os.path.expanduser("~/CLAUDE.md")]:
        if os.path.isfile(path):
            with open(path) as f:
                return f"# Project Instructions (from {path})\n{f.read()}"
    return None

# ============================================================
# 3. 拼装器
# ============================================================
def build_system_prompt() -> tuple[str, dict]:
    """返回 (拼好的字符串, debug 元信息)"""
    sections = [
        # ===== 静态部分（可缓存）=====
        section("identity",     get_identity),
        section("tone_style",   get_tone_and_style),
        section("doing_tasks",  get_doing_tasks),
        section("using_tools",  get_using_tools),
        section("safety",       get_safety),

        # ===== 边界标记 =====
        section("dynamic_boundary", lambda: DYNAMIC_BOUNDARY),

        # ===== 动态部分（不缓存）=====
        dangerous_uncached_section("env_info", get_env_info,
                                    "CWD/date changes every session"),
        dangerous_uncached_section("claude_md", get_claude_md,
                                    "Per-project file may be edited"),
    ]

    parts = []
    debug = {"sections": [], "total_chars": 0, "static_chars": 0, "dynamic_chars": 0}
    crossed_boundary = False

    for s in sections:
        content = s.content_fn()
        if content is None:
            continue
        parts.append(content)
        info = {
            "name": s.name,
            "chars": len(content),
            "cacheable": s.cacheable,
            "uncacheable_reason": s.reason_uncacheable or None,
        }
        debug["sections"].append(info)
        debug["total_chars"] += len(content)
        if s.name == "dynamic_boundary":
            crossed_boundary = True
        elif crossed_boundary:
            debug["dynamic_chars"] += len(content)
        else:
            debug["static_chars"] += len(content)

    return "\n\n".join(parts), debug

# ============================================================
# 4. Demo
# ============================================================
if __name__ == "__main__":
    import json
    prompt, debug = build_system_prompt()
    print("==== SYSTEM PROMPT ====")
    print(prompt)
    print("\n==== DEBUG ====")
    print(json.dumps(debug, indent=2, ensure_ascii=False))
```

---

## 🔧 运行

```bash
cd ~/llm-study
python3 system_prompt_builder.py
```

输出节选：

```
==== SYSTEM PROMPT ====
You are MiniCC, an interactive CLI coding agent.

# Tone and style
 - Only use emojis if the user explicitly requests it.
 - Your responses should be short and concise.
 - When referencing code, use file_path:line_number format.
 - Do not use a colon before tool calls.

# Doing tasks
 - ...

__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__

# Environment
 - CWD: /Users/reader
 - Date: 2026-06-03
 - Platform: Darwin

==== DEBUG ====
{
  "sections": [
    {"name": "identity",   "chars": 49,  "cacheable": true},
    {"name": "tone_style", "chars": 280, "cacheable": true},
    ...
    {"name": "env_info",   "chars": 95,  "cacheable": false,
     "uncacheable_reason": "CWD/date changes every session"},
    ...
  ],
  "total_chars": 1247,
  "static_chars": 800,
  "dynamic_chars": 400
}
```

---

## 🐍 Python 新语法

| 写法 | 含义 |
|---|---|
| `@dataclass` | 自动生成 `__init__`、`__repr__` 等 |
| `Callable[[], str \| None]` | 类型注解：无参函数返回 str 或 None |
| `lambda: DYNAMIC_BOUNDARY` | 匿名函数，等价 `def _(): return DYNAMIC_BOUNDARY` |
| `tuple[str, dict]` | Python 3.9+ 内置泛型 |
| `if (x := compute()):` | 海象运算符（Python 3.8+），在条件里赋值 |

---

## 🛠 动手练习

1. **加一个 section**：写 `get_task_management()` 包含 TodoWrite 使用规则，注册到 sections 列表
2. **真接入 Anthropic SDK**：
   - 把 static 部分放 system 字段并加 `cache_control: {"type": "ephemeral"}`
   - 跑两次同一请求，观察 `cache_read_input_tokens`
3. **测试 dangerous_uncached_section 检查**：不传 reason 看会不会报错

---

## 🔗 延伸阅读

- 上一节：[06-写好Prompt的checklist](06-%E5%86%99%E5%A5%BDPrompt%E7%9A%84checklist.md)
- 下一章：[第 6 章](../06-%E7%AC%AC6%E7%AB%A0-%E5%A4%9AAgent%E5%8D%8F%E4%BD%9C/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 真源码：`_source/.../src/constants/systemPromptSections.ts`

---

⬅ [06-写好Prompt的checklist](06-%E5%86%99%E5%A5%BDPrompt%E7%9A%84checklist.md)　|	➡ [第 6 章 →](../06-%E7%AC%AC6%E7%AB%A0-%E5%A4%9AAgent%E5%8D%8F%E4%BD%9C/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
