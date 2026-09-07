---
tags: [Claude-Code, 第6章, 代码合集, Python实战]
chapter: 6
section: 6.99
---

# 6.99 代码合集：Python 递归 Agent 框架

⬅ [04-并发与反模式](04-%E5%B9%B6%E5%8F%91%E4%B8%8E%E5%8F%8D%E6%A8%A1%E5%BC%8F.md)　|	➡ [第 7 章 →](../07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

> 🎯 把第 6 章学的全部组合：主 Agent + 内置 SubAgent + 并发派发 + 深度限制 + 上下文隔离。

---

## 🐍 完整代码

```python
# mini_subagent.py
# ============================================================
# 演示 SubAgent 机制：主 Agent 派生子 Agent，上下文隔离
# - AgentConfig 定义"工种"
# - Agent 类支持递归派生
# - 并发派多个 SubAgent
# - 深度限制防爆
# ============================================================

import asyncio
from dataclasses import dataclass, field

# ============================================================
# 1. Agent 配置
# ============================================================
@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)   # 空 = 所有
    when_to_use: str = ""

# ============================================================
# 2. 假装的工具
# ============================================================
async def tool_read(path: str) -> str:
    return f"<contents of {path}>"

async def tool_write(path: str, content: str) -> str:
    return f"wrote {len(content)} chars to {path}"

async def tool_search(query: str) -> str:
    await asyncio.sleep(0.3)
    return f"found 5 matches for '{query}'"

ALL_TOOLS = {
    "read":   tool_read,
    "write":  tool_write,
    "search": tool_search,
}

# ============================================================
# 3. 内置 SubAgent 定义
# ============================================================
EXPLORE_AGENT = AgentConfig(
    name="Explore",
    system_prompt=(
        "You are a READ-ONLY search specialist.\n"
        "STRICTLY PROHIBITED from: creating, modifying, or deleting files.\n"
        "Use only `search` and `read` to find information."
    ),
    allowed_tools=["search", "read"],          # ⭐ 物理禁 write
    when_to_use="Search the codebase. Returns a summary."
)

VERIFY_AGENT = AgentConfig(
    name="Verify",
    system_prompt="You verify whether a change works by running tests.",
    allowed_tools=["read"],
    when_to_use="Run verification on a change. Returns pass/fail."
)

SUBAGENTS = {
    "Explore": EXPLORE_AGENT,
    "Verify":  VERIFY_AGENT,
}

# ============================================================
# 4. Agent 类（支持递归）
# ============================================================
class Agent:
    def __init__(self, config: AgentConfig, depth: int = 0):
        self.config = config
        self.depth = depth                       # ⭐ 深度（防爆）
        self.messages: list[dict] = []           # ⭐ 独立 context
        self.indent = "  " * depth

    def _log(self, msg: str):
        print(f"{self.indent}[{self.config.name}@d{self.depth}] {msg}")

    async def call_tool(self, name: str, args: dict) -> str:
        # 检查工具白名单
        if self.config.allowed_tools and name not in self.config.allowed_tools:
            if name != "Task":      # Task 是特殊的"派生"工具
                return f"ERROR: tool '{name}' not allowed for {self.config.name}"

        if name == "Task":
            return await self._spawn_subagent(**args)

        func = ALL_TOOLS.get(name)
        if not func:
            return f"ERROR: unknown tool {name}"
        return await func(**args)

    async def _spawn_subagent(self, subagent_type: str, prompt: str, **_) -> str:
        """派生 SubAgent 跑独立循环。"""
        if self.depth >= 3:                      # ⭐ 深度限制
            return "ERROR: max subagent depth reached"
        cfg = SUBAGENTS.get(subagent_type)
        if not cfg:
            return f"ERROR: unknown subagent type {subagent_type}"
        self._log(f"→ spawning {subagent_type}: {prompt[:50]}")

        # ⭐ SubAgent 有独立的 messages
        sub = Agent(config=cfg, depth=self.depth + 1)
        result = await sub.run(prompt)
        self._log(f"← {subagent_type} returned: {result[:60]}")
        return result

    async def run(self, user_input: str) -> str:
        """跑 agent loop，返回最终字符串。"""
        self.messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user",   "content": user_input},
        ]
        return await self._fake_llm_loop()

    async def _fake_llm_loop(self) -> str:
        """模拟 agent loop。"""
        if self.config.name == "Explore":
            # SubAgent: 搜索 + read，返回摘要
            r1 = await self.call_tool("search", {"query": "auth"})
            r2 = await self.call_tool("read", {"path": "src/auth.py"})
            return f"Summary: searched & read. {r1}; {r2[:30]}"

        elif self.config.name == "Verify":
            r = await self.call_tool("read", {"path": "tests/test.py"})
            return f"Summary: verified — tests look OK"

        else:
            # 主 Agent: 并发派两个 SubAgent
            self._log("Decision: spawn 2 parallel subagents")
            results = await asyncio.gather(
                self.call_tool("Task", {
                    "subagent_type": "Explore",
                    "prompt": "Find auth-related files. Report in 100 words.",
                }),
                self.call_tool("Task", {
                    "subagent_type": "Verify",
                    "prompt": "Verify the test suite passes.",
                }),
            )
            return f"Final answer:\n - {results[0]}\n - {results[1]}"

# ============================================================
# 5. Demo
# ============================================================
MAIN_AGENT_CONFIG = AgentConfig(
    name="Main",
    system_prompt="You are the main coordinator. Use Task to delegate.",
)

async def main():
    main_agent = Agent(MAIN_AGENT_CONFIG, depth=0)
    final = await main_agent.run("分析这个项目的认证模块并验证")
    print("\n========== FINAL OUTPUT ==========")
    print(final)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🔧 运行预期

```bash
python3 mini_subagent.py
```

输出：
```
[Main@d0] Decision: spawn 2 parallel subagents
  [Explore@d1] → spawning Explore (递归调用 .run)
  [Verify@d1] → ...
[Main@d0] ← Explore returned: Summary: searched & read...
[Main@d0] ← Verify returned: Summary: verified — tests look OK

========== FINAL OUTPUT ==========
Final answer:
 - Summary: searched & read. found 5 matches for 'auth'; <contents of src/auth...
 - Summary: verified — tests look OK
```

注意：主 Agent 的 `self.messages` 里**完全没有**子 Agent 的搜索结果、read 内容、verify 细节——只有"我派出去 X 个 SubAgent，他们说 Y" 这种摘要。

**上下文隔离实现了**！

---

## ⭐ 关键观察

### 观察 1：递归用的是同一个类

`Agent` 类既是主 Agent 也是 SubAgent。**只有 `config` 和 `depth` 不同**。
真实 Claude Code 也是这样——`query()` 函数递归调用。

### 观察 2：独立 messages 就是隔离

```python
sub = Agent(config=cfg, depth=self.depth + 1)
# sub.messages 是空 list，从 system prompt 开始
```

不传 messages → 全新 context。这就是隔离。

### 观察 3：返回字符串是关键

SubAgent 的 `run()` 只返回**最终字符串**，所有中间过程留在 `sub.messages` 里随对象销毁而消失。

---

## 🛠 动手练习

1. **加 Plan SubAgent**：
   ```python
   PLAN_AGENT = AgentConfig(
       name="Plan",
       system_prompt="You generate a plan. DO NOT execute.",
       allowed_tools=["read"],
   )
   ```
   主 Agent 任务前先派 Plan。

2. **触发深度限制**：让 Explore agent 尝试派 Task，看 `depth >= 3` 怎么挡。

3. **观察上下文隔离**：在 `_fake_llm_loop` 末尾打印 `len(self.messages)`，比较主 Agent 和 SubAgent 的 context 大小。

4. **加 Fork 模式**：
   ```python
   async def _fork(self, prompt: str) -> str:
       # Fork: 继承父 messages
       sub = Agent(self.config, depth=self.depth + 1)
       sub.messages = self.messages.copy()      # ⭐ 继承
       sub.messages.append({"role": "user", "content": prompt})
       return await sub._fake_llm_loop()
   ```

---

## 📊 与真源码对照

| 代码段 | 真源码 |
|---|---|
| `Agent` 类递归 | `src/tools/AgentTool/runAgent.ts` |
| `EXPLORE_AGENT` 定义 | `src/tools/AgentTool/built-in/exploreAgent.ts` |
| `depth >= 3` 防爆 | `disallowedTools: [AGENT_TOOL_NAME]` 物理禁 |
| `asyncio.gather` 并发 | 主 Agent 一次发多个 tool_use |
| `sub.messages = []` 隔离 | `forkSubagent.ts` 中 forkedAgent 处理 |

---

## 🔗 延伸阅读

- 上一节：[04-并发与反模式](04-%E5%B9%B6%E5%8F%91%E4%B8%8E%E5%8F%8D%E6%A8%A1%E5%BC%8F.md)
- 下一章：[第 7 章](../07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 真源码：`_source/.../src/tools/AgentTool/`

---

⬅ [04-并发与反模式](04-%E5%B9%B6%E5%8F%91%E4%B8%8E%E5%8F%8D%E6%A8%A1%E5%BC%8F.md)　|	➡ [第 7 章 →](../07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
