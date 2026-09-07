---
tags: [Claude-Code, 第3章, 接口设计, 源码拆解]
chapter: 3
section: 3.2
---

# 3.2 Claude Code 的 Tool 接口设计

⬅ [01-Tool-Use协议基础](01-Tool-Use%E5%8D%8F%E8%AE%AE%E5%9F%BA%E7%A1%80.md)　|	➡ [03-FileReadTool解剖](03-FileReadTool%E8%A7%A3%E5%89%96.md)

> 📂 源码：`src/Tool.ts:362`（Tool 接口定义）

---

## 🎬 故事比喻：从"路边摊菜单"到"五星酒店菜单"

**路边摊菜单**（裸 API 协议）：
- 菜名
- 价格
- 描述

**五星酒店菜单**（Claude Code 的 Tool 接口）：
- 菜名 + 别名
- 详细描述（给客人）
- **何时推荐 / 何时不推荐**
- **是否素食 / 是否含麸质 / 是否含坚果**
- **能否快速上 / 是否限量**
- **是否破坏性烹饪**（用了就回不去）
- ...

Claude Code 的 Tool 多了一堆"标签"，**每一个标签都对应一种生产场景的需要**。

---

## 🔧 Tool 接口的 18 个字段

`src/Tool.ts:362` 的 Tool 类型（精简版）：

```typescript
export type Tool<Input, Output> = {
  // ===== 基础元信息 =====
  readonly name: string                                 // 工具名（API 用）
  description(input, options): Promise<string>          // UI 显示用的描述
  prompt(): Promise<string>                             // ⭐ 给 LLM 看的详细说明
  readonly inputSchema: ZodSchema                       // 输入参数 schema
  outputSchema?: ZodSchema                              // 输出 schema（可选）

  // ===== 行为标志 =====
  isEnabled(): boolean                                  // 当前会话是否启用
  isReadOnly(input): boolean                            // 是否只读？
  isConcurrencySafe(input): boolean                     // 能否并发？
  isDestructive?(input): boolean                        // 是否不可逆？

  // ===== 权限相关 =====
  preparePermissionMatcher(input): Promise<...>         // 权限规则匹配器

  // ===== 核心执行函数 =====
  call(args, context, canUseTool, parentMessage, onProgress?)
    : Promise<ToolResult<Output>>

  // ===== UX 元数据 =====
  searchHint?: string                                   // 工具搜索关键词
  aliases?: string[]                                    // 别名（重命名时兼容）
  userFacingName?(input): string                        // UI 友好名
  getActivityDescription?(input): string                // "Reading file.py" 这种实时描述

  // ===== 高级 =====
  maxResultSizeChars: number                            // 结果过大要持久化
  shouldDefer?: boolean                                 // 是否延迟加载（ToolSearch）
  alwaysLoad?: boolean                                  // 永远在第一轮 prompt 里
  inputsEquivalent?(a, b): boolean                      // 两次调用是否等价（缓存用）
  isSearchOrReadCommand?(input): {...}                  // UI 折叠分类
  interruptBehavior?(): 'cancel' | 'block'              // 中断时怎么办
}
```

---

## ⭐ 加深理解：为什么裸 API 不够？

裸 Anthropic API 只关心 3 个字段：name / description / input_schema。
Claude Code 多出来的 15 个字段都是为了解决**工程问题**：

| 字段 | 解决什么生产问题 |
|---|---|
| `isReadOnly` | Plan Mode 下只允许只读工具运行 |
| `isConcurrencySafe` | 多工具并发时，决定是否能放进并发池 |
| `isDestructive` | 触发权限弹窗确认 |
| `preparePermissionMatcher` | `--allowedTools Bash(npm test:*)` 这种规则匹配 |
| `maxResultSizeChars` | grep 出几千行时，存到磁盘+给模型一个路径 |
| `searchHint` | 工具被 defer 时，模型搜索关键词找它 |
| `onProgress` | npm install 等长任务实时进度推到 UI |
| `aliases` | 工具重命名时，旧名字还能用 |
| `interruptBehavior` | 用户按 Ctrl+C 时这个工具该 cancel 还是 block |

**每一条都对应"一种用户场景"或"一类 bug 修复"**。

---

## 📊 description vs prompt 的差别（最容易混）

| 函数 | 给谁看 | 长度 | 例子 |
|---|---|---|---|
| `description(input, options)` | **UI** 显示 | 1 行 | `"Reading /etc/hosts"` |
| `prompt()` | **LLM** 决策 | 几百字 | `"Reads a file from the local filesystem. The file_path must be absolute..."` |

**为什么分开？**
- UI 要简短：用户在终端看一眼就明白
- LLM 要详细：模型需要全部边界、规则、示例才能用对

如果合并：UI 太啰嗦或 LLM 信息不够，二选一都不好。

---

## 🐍 Python 简化版 Tool 基类

```python
from abc import ABC, abstractmethod
from typing import Any

class Tool(ABC):
    """Tool 抽象基类，模仿 src/Tool.ts 的 Tool 接口。"""

    # 基础元信息
    name: str = ""
    description: str = ""               # 给 UI（简短）
    prompt: str = ""                    # 给 LLM（详细）
    input_schema: dict = {}             # JSON Schema

    # 行为标志
    is_read_only: bool = False
    is_concurrency_safe: bool = True
    is_destructive: bool = False

    @abstractmethod
    async def call(self, **kwargs) -> str:
        """子类必须实现：真正干活的函数。"""
        ...

    def to_api_spec(self) -> dict:
        """转成 Anthropic API 期望的 tool 定义格式。"""
        return {
            "name": self.name,
            "description": self.prompt or self.description,
            "input_schema": self.input_schema,
        }
```

完整实现见 [99-代码-Python工具系统](99-%E4%BB%A3%E7%A0%81-Python%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F.md)。

---

## 🔧 真实工具的"目录式"组织

Claude Code 不把工具写成一个文件，而是**每个工具一个目录**。看 `src/tools/FileReadTool/`：

```
FileReadTool/
├── FileReadTool.ts      # 主实现（1183 行）
├── prompt.ts            # 给 LLM 的说明书模板
├── limits.ts            # 文件大小/行数限制
├── imageProcessor.ts    # 图像文件处理
└── UI.tsx               # CLI 终端渲染
```

**为什么这样组织？**
一个像样的工具至少需要：实现 + prompt + UI + 边界处理。分文件比塞一个大文件可维护。

下一节 [03-FileReadTool解剖](03-FileReadTool%E8%A7%A3%E5%89%96.md) 会逐文件讲。

---

## 🔗 延伸阅读

- 上一节：[01-Tool-Use协议基础](01-Tool-Use%E5%8D%8F%E8%AE%AE%E5%9F%BA%E7%A1%80.md)
- 下一节：[03-FileReadTool解剖](03-FileReadTool%E8%A7%A3%E5%89%96.md) —— case study
- 完整 Python 基类：[99-代码-Python工具系统](99-%E4%BB%A3%E7%A0%81-Python%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F.md)
- 真源码：`_source/.../src/Tool.ts:362-792`

---

## ❓ 小测验

> 1. Claude Code 的 Tool 接口比裸 API 多了哪些"行为标志"？至少说 3 个
> 2. `description` 和 `prompt` 给谁看？为什么分开？
> 3. 为什么每个工具用一个目录而不是一个文件？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#32-tool-%E6%8E%A5%E5%8F%A3%E8%AE%BE%E8%AE%A1)

---

⬅ [01-Tool-Use协议基础](01-Tool-Use%E5%8D%8F%E8%AE%AE%E5%9F%BA%E7%A1%80.md)　|	➡ [03-FileReadTool解剖](03-FileReadTool%E8%A7%A3%E5%89%96.md)
