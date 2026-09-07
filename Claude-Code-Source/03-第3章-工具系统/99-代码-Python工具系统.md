---
tags: [Claude-Code, 第3章, 代码合集, Python实战]
chapter: 3
section: 3.99
---

# 3.99 代码合集：Python 工具系统

⬅ [06-设计哲学5条](06-%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A65%E6%9D%A1.md)　|	➡ [第 4 章 →](../04-%E7%AC%AC4%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

> 🎯 把第 3 章学的全部组合：Tool 基类 + 3 个具体工具 + 注册表 + 执行器（带权限和并发）。

---

## 🐍 完整代码

```python
# mini_tool_system.py
# ============================================================
# 模仿 Claude Code 工具系统的 Python 简化版
# - Tool 抽象基类
# - 3 个具体工具（Read / Write / Bash）
# - 注册表 + Plan Mode 过滤
# - 执行器：权限检查 + 并发
# ============================================================

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Callable

# ============================================================
# 1. Tool 抽象基类
# ============================================================
class Tool(ABC):
    """所有工具的基类。模仿 src/Tool.ts 的 Tool 接口。"""

    # ---- 基础元信息 ----
    name: str = ""
    description: str = ""               # 简短，给 UI
    prompt: str = ""                    # 详细，给 LLM
    input_schema: dict = {}             # JSON Schema

    # ---- 行为标志 ----
    is_read_only: bool = False
    is_concurrency_safe: bool = True
    is_destructive: bool = False

    @abstractmethod
    async def call(self, **kwargs) -> str:
        """子类必须实现。"""
        ...

    def to_api_spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.prompt or self.description,
            "input_schema": self.input_schema,
        }

# ============================================================
# 2. 具体工具实现
# ============================================================
class ReadTool(Tool):
    name = "read_file"
    description = "Read a file."
    prompt = """Read a file from the local filesystem.
Usage:
- The path parameter must be an absolute path.
- Results are returned with line numbers prefixed in `<num>: <content>` format.
- Maximum 2000 lines per read."""
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute file path"},
        },
        "required": ["path"],
    }
    is_read_only = True

    async def call(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()[:2000]
            return "".join(f"{i+1}: {line}" for i, line in enumerate(lines))
        except Exception as e:
            return f"ERROR: {e}"


class WriteTool(Tool):
    name = "write_file"
    description = "Write content to a file (overwrite)."
    prompt = """Write text to a file. Overwrites if exists. Path must be absolute."""
    input_schema = {
        "type": "object",
        "properties": {
            "path":    {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    is_read_only = False
    is_destructive = True               # 触发权限

    async def call(self, path: str, content: str) -> str:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"OK: wrote {len(content)} chars to {path}"
        except Exception as e:
            return f"ERROR: {e}"


class BashTool(Tool):
    name = "bash"
    description = "Run a shell command."
    prompt = """Execute a bash command and return its stdout/stderr.
Usage:
- Avoid destructive commands without explicit confirmation.
- Timeout: 30 seconds default."""
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command"},
        },
        "required": ["command"],
    }
    is_read_only = False
    is_concurrency_safe = False         # ⚠️ Bash 不能乱并发
    is_destructive = True

    async def call(self, command: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            return "ERROR: Command timed out after 30s"
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        return f"[exit={proc.returncode}]\nSTDOUT:\n{out}\nSTDERR:\n{err}"

# ============================================================
# 3. 工具注册表
# ============================================================
class ToolRegistry:
    """模仿 src/tools.ts 的注册和过滤逻辑。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_for_api(self, plan_mode: bool = False) -> list[dict]:
        """Plan Mode 下只返回 read-only 工具。"""
        tools = list(self._tools.values())
        if plan_mode:
            tools = [t for t in tools if t.is_read_only]
        return [t.to_api_spec() for t in tools]

# ============================================================
# 4. 工具执行器
# ============================================================
class ToolExecutor:
    """模仿 services/tools/StreamingToolExecutor.ts 的简化版。"""

    def __init__(self, registry: ToolRegistry,
                 can_use_tool: Callable = None):
        self.registry = registry
        self.can_use_tool = can_use_tool or (lambda t, i: True)

    async def execute_one(self, tool_use: dict) -> dict:
        tool = self.registry.get(tool_use["name"])
        if tool is None:
            return self._error(tool_use["id"], f"Unknown tool: {tool_use['name']}")

        # ⭐ 权限检查
        if not self.can_use_tool(tool, tool_use["input"]):
            return self._error(tool_use["id"], "Permission denied")

        try:
            result = await tool.call(**tool_use["input"])
        except Exception as e:
            result = f"ERROR: {e}"

        return {
            "type": "tool_result",
            "tool_use_id": tool_use["id"],
            "content": str(result)[:5000],          # 截断防爆
        }

    async def execute_batch(self, tool_uses: list[dict]) -> list[dict]:
        """能并发的并发，不能并发的串行。"""
        parallel, serial = [], []
        for tu in tool_uses:
            tool = self.registry.get(tu["name"])
            if tool and tool.is_concurrency_safe:
                parallel.append(tu)
            else:
                serial.append(tu)

        parallel_results = await asyncio.gather(*[
            self.execute_one(tu) for tu in parallel
        ])
        serial_results = []
        for tu in serial:
            serial_results.append(await self.execute_one(tu))
        return parallel_results + serial_results

    def _error(self, tu_id: str, msg: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tu_id,
                "content": msg, "is_error": True}

# ============================================================
# 5. Demo
# ============================================================
async def demo():
    reg = ToolRegistry()
    reg.register(ReadTool())
    reg.register(WriteTool())
    reg.register(BashTool())

    def permission_check(tool: Tool, input: dict) -> bool:
        if tool.is_destructive:
            print(f"[PERMISSION] {tool.name} is destructive. Input={input}. Auto-allow in demo.")
        return True

    executor = ToolExecutor(reg, permission_check)

    tool_uses = [
        {"id": "t1", "name": "read_file", "input": {"path": "/etc/hosts"}},
        {"id": "t2", "name": "bash",      "input": {"command": "echo hello"}},
    ]
    results = await executor.execute_batch(tool_uses)
    for r in results:
        print(json.dumps(r, ensure_ascii=False, indent=2))

    print("\nAvailable in Plan Mode:")
    for spec in reg.list_for_api(plan_mode=True):
        print(f"  - {spec['name']}")

if __name__ == "__main__":
    asyncio.run(demo())
```

---

## 🐍 Python 语法新增点

| 写法 | 解释 |
|---|---|
| `from abc import ABC, abstractmethod` | 抽象基类。`@abstractmethod` 的方法子类必须实现 |
| `dict[str, Tool]` | Python 3.9+ 内置泛型 |
| `Tool \| None` | Python 3.10+ union 类型 |
| `asyncio.create_subprocess_shell` | 异步启子进程 |
| `asyncio.wait_for(..., timeout=30)` | 给协程加超时 |
| `[t for t in tools if cond]` | 列表推导式 |
| `**tu["input"]` | 字典展开为关键字参数 |

---

## 🛠 动手练习

1. **加 GlobTool**：用 `glob.glob` 实现文件名 pattern 匹配，注册到 registry
2. **改 permission_check**：destructive 时让用户输入 y/n 确认
3. **观察并发**：让 Read 工具加 `print(f"{path} start"); ...; print(f"{path} end")`，跑并发批量看顺序
4. **加 Edit 工具**：参考 [03-FileReadTool解剖](03-FileReadTool%E8%A7%A3%E5%89%96.md) 思路，实现"必须唯一匹配的字符串替换"

---

## 🔗 延伸阅读

- 上一节：[06-设计哲学5条](06-%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A65%E6%9D%A1.md)
- 下一章：[第 4 章 上下文管理](../04-%E7%AC%AC4%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 完整 MiniCC：`~/llm-study/minicc/minicc.py`

---

⬅ [06-设计哲学5条](06-%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A65%E6%9D%A1.md)　|	➡ [第 4 章 →](../04-%E7%AC%AC4%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
