---
tags: [Claude-Code, 第3章, 案例分析, FileReadTool]
chapter: 3
section: 3.3
---

# 3.3 FileReadTool 完整解剖

⬅ [02-Tool接口设计](02-Tool%E6%8E%A5%E5%8F%A3%E8%AE%BE%E8%AE%A1.md)　|	➡ [04-工具清单总览](04-%E5%B7%A5%E5%85%B7%E6%B8%85%E5%8D%95%E6%80%BB%E8%A7%88.md)

> 🎯 一个真实工具的完整 case study。学完它，你能照着写自己的工具。

---

## 📂 源码定位

```
_source/.../src/tools/FileReadTool/
├── FileReadTool.ts      # 主实现（1183 行）
├── prompt.ts            # LLM 说明书（49 行）
├── limits.ts            # 大小限制（92 行）
├── imageProcessor.ts    # 图像处理（94 行）
└── UI.tsx               # 终端渲染（184 行）
```

---

## 🎬 故事比喻：为啥读个文件要 1183 行？

你以为 "读文件" 就是 `cat file.txt`？
真实生产场景里它要处理：
- 路径是绝对还是相对？非绝对要警告
- 文件不存在？错误提示要有用
- 文件是目录？要提示用 ls
- 文件是图片？要返回 base64
- 文件是 PDF？要分页处理
- 文件是 Jupyter notebook？要解析单元格
- 文件超大？要支持 offset/limit
- 文件刚读过？要提示"file unchanged"避免重复读
- 行号怎么加？要让 Edit 工具能精确匹配
- 编码问题？非 UTF-8 要回退
- ……

**每条都对应一个真实用户场景**。1183 行不是冗余，是覆盖。

---

## 🔧 5 个文件逐一看

### 文件 1：`prompt.ts` — 给 LLM 的说明书

```typescript
export function renderPromptTemplate(...) {
  return `Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 2000 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files)
- Results are returned using cat -n format, with line numbers starting at 1
- This tool allows Claude Code to read images (eg PNG, JPG, etc).
- This tool can read PDF files. For large PDFs (more than 10 pages), you MUST provide the pages parameter
- This tool can read Jupyter notebooks (.ipynb files)
- This tool can only read files, not directories. To read a directory, use an ls command via the Bash tool.
- ...`
}
```

**写好 tool prompt 的真理**（从这段抄）：
1. **用 bullet 列表**比段落好读
2. **明确边界**："can only read files, not directories"
3. **强制规则用大写**："MUST provide the pages parameter"
4. **指向其他工具**："use an ls command via the Bash tool"——选错时引导
5. **解释返回格式**："cat -n format, with line numbers starting at 1"——后面 Edit 才能精确匹配

### 文件 2：`FileReadTool.ts:227` — inputSchema

```typescript
const inputSchema = z.strictObject({
  file_path: z.string().describe('The absolute path to the file to read'),
  offset: z.number().int().nonnegative().optional().describe(
    'The line number to start reading from. Only provide if the file is too large to read at once'
  ),
  limit: z.number().int().positive().optional().describe(
    'The number of lines to read. Only provide if the file is too large to read at once.'
  ),
  pages: z.string().optional().describe(
    `Page range for PDF files (e.g., "1-5", "3", "10-20")...`
  ),
})
```

**关键点**：每个字段都带 `.describe(...)`，**这些描述会自动变成 JSON Schema 的 description 发给 LLM**。

> 给 LLM 写 schema 的核心原则：**每个参数都要有清晰的 description，告诉它什么时候用、用什么值**。

### 文件 3：`FileReadTool.ts:337` — 行为标志

```typescript
export const FileReadTool = buildTool({
  name: FILE_READ_TOOL_NAME,                  // "Read"
  searchHint: 'read files, images, PDFs, notebooks',
  maxResultSizeChars: Infinity,               // 结果不持久化（自己已截断）
  strict: true,                                // 严格 schema 校验
  isReadOnly() { return true },               // 只读 → Plan Mode 也能用
  isConcurrencySafe() { return true },        // 多个 Read 可并发
  isSearchOrReadCommand() {
    return { isSearch: false, isRead: true }  // UI 折叠分类
  },
  // ...
})
```

### 文件 4：`call()` — 真正干活

精简后的逻辑：

```typescript
async call(input, context, canUseTool, parentMessage, onProgress) {
  // 1. 验证路径（绝对路径、不越权）
  if (!isAbsolutePath(input.file_path)) {
    return errorResult("path must be absolute")
  }

  // 2. 询问权限（canUseTool 回调，可能弹窗）
  const perm = await canUseTool(this, input, context)
  if (!perm.allowed) return errorResult(perm.reason)

  // 3. 真正读文件
  const content = await readFileWithLimits(input.file_path, input.offset, input.limit)

  // 4. 加行号前缀（cat -n 格式）
  const numbered = addLineNumbers(content, input.offset ?? 1)

  // 5. 检查是否要触发"file unchanged"提示
  if (sameAsLastRead(...)) return { content: FILE_UNCHANGED_STUB }

  // 6. 返回结构化结果
  return { type: 'text', content: numbered, ... }
}
```

注意第 2 步：**`canUseTool` 是权限关卡**，执行前必须问"我能跑吗"。详见 [第 7 章](../07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)。

### 文件 5：`UI.tsx` — 终端渲染

用 [Ink](https://github.com/vadimdemedes/ink)（React for CLI）写的组件，负责把"读取中…读了 N 行"显示成漂亮的进度条。
**和模型决策无关，纯 UX 优化**。

---

## ⭐ 加深理解：行号是为 Edit 服务的

注意 prompt 里这条：
> "Results are returned using cat -n format, with line numbers starting at 1"

Read 工具**强制返回**带行号的内容：
```
1: import os
2: import sys
3:
4: def main():
5:     pass
```

为什么？**因为 Edit 工具需要基于精确字符串匹配做编辑**。模型必须先看到行号、看到原文，才能写出**唯一匹配**的 oldString。

```
模型决策流程：
  1. Read("main.py") → 看到第 4 行 `def main():`
  2. 决定改成 `def main() -> None:`
  3. Edit(old="def main():", new="def main() -> None:")
  4. Edit 工具检查 → "def main():" 在文件中只出现 1 次 → 替换
```

如果没有行号，模型可能：
- 不确定要改哪一行
- 写出能匹配多处的 oldString → Edit 报错

**两个工具协同设计** —— 这是细节，但极其重要。

---

## 🐍 Python 版 ReadTool（带详细注释）

```python
from pathlib import Path

class ReadTool(Tool):
    name = "Read"
    description = "Read a file."
    prompt = """Read a file from the local filesystem.
Usage:
- The path parameter must be an absolute path.
- Results are returned with line numbers in `<num>: <content>` format.
- Maximum 2000 lines per read."""

    input_schema = {
        "type": "object",
        "properties": {
            "path":   {"type": "string", "description": "Absolute file path"},
            "offset": {"type": "integer", "description": "Start line (1-indexed)"},
            "limit":  {"type": "integer", "description": "Max lines"},
        },
        "required": ["path"],
    }
    is_read_only = True

    async def call(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        p = Path(path).expanduser()
        # 路径验证
        if not p.is_absolute():
            return f"ERROR: path must be absolute, got: {path}"
        if not p.exists():
            return f"ERROR: file not found: {p}"
        if p.is_dir():
            return f"ERROR: is a directory, use Bash `ls` instead"
        # 读 + 截断
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            return f"ERROR: {e}"
        end = min(offset - 1 + limit, len(lines))
        selected = lines[offset-1:end]
        # ⭐ 关键：加行号
        return "\n".join(f"{offset+i}: {line}" for i, line in enumerate(selected))
```

完整实现见 [99-代码-Python工具系统](99-%E4%BB%A3%E7%A0%81-Python%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F.md)。

---

## 📝 一句话小结

> **一个真实工具 = prompt（教 LLM）+ schema（约束输入）+ 行为标志（生产标签）+ call（实现）+ UI（渲染）。每一层都为"少 bug、好维护、给 LLM 用对"服务。**

---

## 🔗 延伸阅读

- 上一节：[02-Tool接口设计](02-Tool%E6%8E%A5%E5%8F%A3%E8%AE%BE%E8%AE%A1.md)
- 下一节：[04-工具清单总览](04-%E5%B7%A5%E5%85%B7%E6%B8%85%E5%8D%95%E6%80%BB%E8%A7%88.md) —— 40 个工具速览
- Edit 工具的协同：[06-设计哲学5条](06-%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A65%E6%9D%A1.md#%E5%93%B2%E5%AD%A6-2%E8%A1%8C%E5%8F%B7%E6%98%AF%E4%B8%BA-edit-%E6%9C%8D%E5%8A%A1%E7%9A%84)
- 真源码：`_source/.../src/tools/FileReadTool/FileReadTool.ts:337`

---

## ❓ 小测验

> 1. FileReadTool 为什么要 1183 行？至少举 3 个边界场景
> 2. Read 强制返回行号是为了配合哪个工具？
> 3. `description` 和 `prompt` 在这个工具里分别是什么？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#33-filereadtool-%E8%A7%A3%E5%89%96)

---

⬅ [02-Tool接口设计](02-Tool%E6%8E%A5%E5%8F%A3%E8%AE%BE%E8%AE%A1.md)　|	➡ [04-工具清单总览](04-%E5%B7%A5%E5%85%B7%E6%B8%85%E5%8D%95%E6%80%BB%E8%A7%88.md)
