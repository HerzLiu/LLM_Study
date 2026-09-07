---
tags: [Claude-Code, 第7章, Bash安全, 危险模式, 黑名单]
chapter: 7
section: 7.3
---

# 7.3 Bash 危险模式检测（26+ 种）

⬅ [02-canUseTool过滤器](02-canUseTool%E8%BF%87%E6%BB%A4%E5%99%A8.md)　|	➡ [04-物理沙箱与Hooks](04-%E7%89%A9%E7%90%86%E6%B2%99%E7%AE%B1%E4%B8%8EHooks.md)

> 📂 源码：`src/tools/BashTool/bashSecurity.ts` —— 整个工程**最硬核**的安全核心

---

## 🎬 故事比喻：海关查毒品的"经验清单"

资深海关有一份"内部清单"，记录历年来走私贩用过的所有花招：
- 普通包装 → 看一眼
- 双层底夹层 → 检查
- 真空塑封 → 警觉
- 藏在罐头里 → 经验
- 藏在玩具里 → 经验
- 藏在轮胎里 → 经验
- ……

每一条都对应一次真实截获。
**Claude Code 的危险模式清单也是 26+ 种**，每条对应一种攻击或绕过。

---

## 🔧 危险模式的 4 大类

### 类 1：命令替换（最常见）

`bashSecurity.ts` 中：

```typescript
const COMMAND_SUBSTITUTION_PATTERNS = [
  { pattern: /<\(/, message: 'process substitution <()' },
  { pattern: />\(/, message: 'process substitution >()' },
  { pattern: /=\(/, message: 'Zsh process substitution =()' },
  { pattern: /(?:^|[\s;&|])=[a-zA-Z_]/, message: 'Zsh equals expansion (=cmd)' },
  { pattern: /\$\(/, message: '$() command substitution' },
  { pattern: /\$\{/, message: '${} parameter substitution' },
  { pattern: /\$\[/, message: '$[] legacy arithmetic expansion' },
  { pattern: /~\[/, message: 'Zsh-style parameter expansion' },
  { pattern: /\(e:/, message: 'Zsh-style glob qualifiers' },
  { pattern: /\(\+/, message: 'Zsh glob qualifier with command execution' },
  { pattern: /\}\s*always\s*\{/, message: 'Zsh always block' },
  { pattern: /<#/, message: 'PowerShell comment syntax' },
]
```

**为啥连 PowerShell 语法都堵**？
源码注释说：**"防御性编程，万一以后改了执行 shell"**。

---

### 类 2：Zsh 危险模块

```typescript
const ZSH_DANGEROUS_COMMANDS = new Set([
  'zmodload',     // 加载 zsh 模块的入口
  'emulate',      // 仿真模式（eval 等价）
  'sysopen', 'sysread', 'syswrite', 'sysseek',   // zsh/system（隐形 IO）
  'zpty',         // zsh/zpty（伪终端执行命令）
  'ztcp', 'zsocket',   // zsh/net（网络外传）
  'mapfile',      // 关联数组隐形读文件
  'zf_rm', 'zf_mv', 'zf_ln', 'zf_chmod',   // zsh/files（绕过二进制检查）
  'zf_chown', 'zf_mkdir', 'zf_rmdir', 'zf_chgrp',
])
```

**翻译**：
- Zsh 有一堆模块化能力，可以**绕过对二进制命令的检查**
- 你以为禁了 `rm`，但 `zf_rm` 是 zsh 内建的 rm，绕过！
- Claude Code 把所有逃逸口都列出来禁

---

### 类 3：跨平台代码执行入口

`src/utils/permissions/dangerousPatterns.ts`：

```typescript
export const CROSS_PLATFORM_CODE_EXEC = [
  // 解释器
  'python', 'python3', 'python2',
  'node', 'deno', 'tsx',
  'ruby', 'perl', 'php', 'lua',
  // 包运行器
  'npx', 'bunx', 'npm run', 'yarn run', 'pnpm run', 'bun run',
  // Shell
  'bash', 'sh',
]
```

**为啥这些要特殊处理**？

源码注释：
```
// An allow rule like `Bash(python:*)` or `PowerShell(node:*)` lets the model
// run arbitrary code via that interpreter, bypassing the auto-mode classifier.
```

**翻译**：用户写 `Bash(python:*)` 允许所有 python 命令 → 模型 `python -c "import os; os.system('rm -rf /')"` 任意代码全过。
所以这种"代码解释器"前缀的 allow 规则需要**特殊警告**。

---

### 类 4：其他细节检查

源码 `BASH_SECURITY_CHECK_IDS` 把所有检查编号：

```typescript
const BASH_SECURITY_CHECK_IDS = {
  INCOMPLETE_COMMANDS: 1,
  JQ_SYSTEM_FUNCTION: 2,            // jq 的 system() 函数能执行命令
  JQ_FILE_ARGUMENTS: 3,
  OBFUSCATED_FLAGS: 4,              // 混淆 flag
  SHELL_METACHARACTERS: 5,
  DANGEROUS_VARIABLES: 6,
  NEWLINES: 7,                       // 多行命令注入
  DANGEROUS_PATTERNS_COMMAND_SUBSTITUTION: 8,
  IFS_INJECTION: 11,                 // IFS 注入
  GIT_COMMIT_SUBSTITUTION: 12,
  PROC_ENVIRON_ACCESS: 13,           // /proc/PID/environ 读敏感信息
  MALFORMED_TOKEN_INJECTION: 14,
  BACKSLASH_ESCAPED_WHITESPACE: 15,  // 反斜杠转义空格绕过
  BRACE_EXPANSION: 16,               // {a,b,c} 展开
  CONTROL_CHARACTERS: 17,
  UNICODE_WHITESPACE: 18,            // unicode 空格字符绕过
  MID_WORD_HASH: 19,
  ZSH_DANGEROUS_COMMANDS: 20,
  BACKSLASH_ESCAPED_OPERATORS: 21,
  COMMENT_QUOTE_DESYNC: 22,
  QUOTED_NEWLINE: 23,
  // ... 还有更多
}
```

**26+ 种检查**，每一种都对应一种攻击手法。

---

## ⭐ 加深理解：3 个绝妙的攻击案例

### 案例 1：jq 的 system() 函数

```bash
$ jq -r '.x | @sh "rm -rf /tmp/test"' < data.json
```

你以为 jq 只是 JSON 处理工具？错，它内建 `system()` 函数能执行命令。
**Claude Code 专门检测 `jq` 调用里有没有 `system()`**。

### 案例 2：IFS 注入

```bash
$ IFS=$'\n'; cmd="ls\nrm -rf /"; $cmd
```

通过修改 IFS（字段分隔符），让 shell 把字符串拆成多个命令。
**Claude Code 检测 IFS 设置**。

### 案例 3：Unicode 空格绕过

```bash
$ rm　-rf　/         # 这里用的是全角空格 U+3000，不是 ASCII 空格
```

如果检测脚本只匹配 ASCII 空格，全角空格就绕过了。
**Claude Code 检测 unicode 空格**。

---

## 🐍 Python 简化版（10 条最常见）

```python
import re

DANGEROUS_PATTERNS = [
    (r"\brm\s+-rf\s+/(?:\s|$)",       "rm -rf root"),
    (r"\bmkfs\b",                      "format filesystem"),
    (r":\(\)\s*\{\s*:\|:&\s*\};\s*:",  "fork bomb"),
    (r"\bdd\s+.*of=/dev/[sh]d",        "raw disk write"),
    (r">\s*/dev/sd",                   "raw disk redirect"),
    (r"\bchmod\s+-R\s+777\s+/",        "chmod 777 root"),
    (r"\$\(",                          "command substitution"),
    (r"`",                              "backtick substitution"),
    (r"\bcurl\s+.*\|\s*(sh|bash)",     "curl pipe to shell"),
    (r"\bwget\s+.*\|\s*(sh|bash)",     "wget pipe to shell"),
]

DANGEROUS_INTERPRETERS = {
    "python", "python3", "node", "ruby", "perl", "bash", "sh",
    "deno", "tsx", "lua", "php"
}

def check_dangerous(cmd: str) -> str | None:
    """返回危险名字，安全返回 None。"""
    for pattern, name in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            return name
    return None

def is_dangerous_interpreter_rule(rule_pattern: str) -> bool:
    """检测 'Bash(python:*)' 这种危险 allow 规则。"""
    base = rule_pattern.split(":")[0].split()[0]
    return base in DANGEROUS_INTERPRETERS
```

---

## ⚠️ 重要提醒

危险模式**永远是黑名单**，意味着：
- 新型攻击需要持续更新
- 不可能 100% 覆盖
- **必须配合其他防线**（沙箱、用户确认）

**不要**以为做了危险检测就高枕无忧。下一节 [04-物理沙箱与Hooks](04-%E7%89%A9%E7%90%86%E6%B2%99%E7%AE%B1%E4%B8%8EHooks.md) 讲终极防线。

---

## 🔗 延伸阅读

- 上一节：[02-canUseTool过滤器](02-canUseTool%E8%BF%87%E6%BB%A4%E5%99%A8.md)
- 下一节：[04-物理沙箱与Hooks](04-%E7%89%A9%E7%90%86%E6%B2%99%E7%AE%B1%E4%B8%8EHooks.md)
- 真源码：`_source/.../src/tools/BashTool/bashSecurity.ts`（神级注释，强烈推荐）

---

## ❓ 小测验

> 1. 危险模式 4 大类是什么？
> 2. 为什么 `Bash(python:*)` 这种 allow 规则危险？
> 3. 全角空格能绕过什么检查？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#73-bash-%E5%8D%B1%E9%99%A9%E6%A8%A1%E5%BC%8F)

---

⬅ [02-canUseTool过滤器](02-canUseTool%E8%BF%87%E6%BB%A4%E5%99%A8.md)　|	➡ [04-物理沙箱与Hooks](04-%E7%89%A9%E7%90%86%E6%B2%99%E7%AE%B1%E4%B8%8EHooks.md)
