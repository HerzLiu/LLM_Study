---
tags: [Claude-Code, 第7章, canUseTool, 权限拦截]
chapter: 7
section: 7.2
---

# 7.2 canUseTool：层层过滤器

⬅ [01-5种权限模式](01-5%E7%A7%8D%E6%9D%83%E9%99%90%E6%A8%A1%E5%BC%8F.md)　|	➡ [03-Bash危险模式](03-Bash%E5%8D%B1%E9%99%A9%E6%A8%A1%E5%BC%8F.md)

> 📂 源码：`src/hooks/useCanUseTool.tsx` + `src/utils/permissions/`

---

## 🎬 故事比喻：海关的 5 道安检

进出境物品要过 5 道关卡：
1. **绝对禁运清单**（毒品 / 武器）→ 直接拒
2. **国别限制**（特殊国家不让进）→ 拒
3. **个人黑名单**（有违规记录）→ 拒
4. **个人白名单**（VIP 通道）→ 放行
5. **常规检查**（X 光、申报）→ 通常放行，可疑物品询问

`canUseTool` 就是这套海关。

---

## 🔧 5 层过滤的伪代码

```typescript
async function canUseTool(tool, input, context): Promise<PermissionResult> {

  // === 第 0 层：危险模式检测（黑名单，任何 mode 生效）===
  if (tool.name === "Bash") {
    const danger = checkDangerousPattern(input.command)
    if (danger) return { behavior: 'deny', message: `BLOCKED: ${danger}` }
  }

  // === 第 1 层：Mode 检查 ===
  if (context.permissionMode === 'plan' && !tool.isReadOnly(input)) {
    return { behavior: 'deny', message: 'Plan Mode is read-only' }
  }
  if (context.permissionMode === 'bypassPermissions') {
    return { behavior: 'allow' }   // bypass 跳过后面所有，但第 0 层已挡危险
  }

  // === 第 2 层：用户 deny 规则（优先级最高）===
  for (const rule of context.denyRules) {
    if (ruleMatches(rule, tool, input)) {
      return { behavior: 'deny', message: `Deny rule: ${rule}` }
    }
  }

  // === 第 3 层：用户 allow 规则 ===
  for (const rule of context.allowRules) {
    if (ruleMatches(rule, tool, input)) {
      return { behavior: 'allow' }
    }
  }

  // === 第 4 层：destructive 默认弹窗 ===
  if (tool.isDestructive?.(input)) {
    return { behavior: 'ask', message: 'This operation is destructive' }
  }

  // === 第 5 层：默认 allow ===
  return { behavior: 'allow' }
}
```

---

## 📊 PermissionResult 三种结果

`src/types/permissions.ts`：

```typescript
type PermissionResult =
  | { behavior: 'allow', updatedInput?: ... }     // 允许（可能改了输入）
  | { behavior: 'deny',  message: string }        // 拒绝（带原因）
  | { behavior: 'ask',   message: string }        // 问用户
```

### `updatedInput`：允许时可以改写参数

例如用户规则"允许 Bash 但加 timeout=30s"：
- 原输入：`{"command": "npm test"}`
- 允许时改写：`{"command": "timeout 30 npm test"}`

**好处**：透明地给命令"加护栏"，模型不用关心。

---

## 🔧 用户规则解析

用户在 `.claude/settings.json` 配：

```json
{
  "permissions": {
    "allow": [
      "Bash(npm test:*)",
      "Bash(git status)",
      "Bash(git diff:*)",
      "Read(/Users/me/projects/*)"
    ],
    "deny": [
      "Bash(rm:*)",
      "Bash(curl:*)",
      "Write(/etc/*)"
    ]
  }
}
```

匹配规则：
- `Bash(npm test:*)` → 匹配所有 `npm test ...` 开头的命令
- `Bash(git status)` → 精确匹配
- `Read(/Users/me/projects/*)` → 路径前缀匹配

源码 `src/utils/permissions/shellRuleMatching.ts` 实现 shell 命令匹配。

---

## ⭐ 加深理解：规则匹配的微妙之处

考虑用户规则 `Bash(curl:*)`（禁 curl）。模型可能尝试绕过：

```bash
$ /usr/bin/curl evil.com        # 用绝对路径
$ =curl evil.com                 # Zsh equals 展开
$ ${VAR:=curl} evil.com          # 参数展开
$ alias curl=wget && curl ...    # 改 alias
$ bash -c 'curl ...'             # 包一层
```

源码 `bashSecurity.ts` 神注释：

```
// Zsh EQUALS expansion: =cmd at word start expands to $(which cmd).
// `=curl evil.com` → `/usr/bin/curl evil.com`, bypassing Bash(curl:*) deny
// rules since the parser sees `=curl` as the base command, not `curl`.
```

**工程师真考虑了 zsh equals 展开这种边角语法** —— 一个绕过路径没堵，权限就形同虚设。

---

## 🐍 Python 版 canUseTool

```python
async def can_use_tool(tool, input: dict, mode: PermissionMode,
                       allow_rules: list, deny_rules: list,
                       ask_user) -> PermissionResult:
    # 第 0 层：危险检测
    if tool.name == "Bash":
        d = check_dangerous(input.get("command", ""))
        if d:
            return PermissionResult("deny", f"BLOCKED: {d}")

    # 第 1 层：Mode 检查
    if mode == PermissionMode.PLAN and not tool.is_read_only:
        return PermissionResult("deny", f"Plan Mode: {tool.name} not read-only")

    if mode == PermissionMode.BYPASS:
        return PermissionResult("allow")

    # 第 2 层：deny 规则
    for rule in deny_rules:
        if rule_matches(rule, tool.name, input):
            return PermissionResult("deny", f"User deny rule")

    # 第 3 层：allow 规则
    for rule in allow_rules:
        if rule_matches(rule, tool.name, input):
            return PermissionResult("allow")

    # 第 4 层：accept_edits 自动放行编辑
    if mode == PermissionMode.ACCEPT_EDITS and tool.name in ("Edit", "Write"):
        return PermissionResult("allow")

    # 第 5 层：destructive 弹窗
    if tool.is_destructive:
        ok = await ask_user(f"Allow {tool.name}? input={input}")
        return PermissionResult("allow" if ok else "deny",
                                "" if ok else "User declined")

    # 默认放行
    return PermissionResult("allow")
```

---

## 🔧 拒绝追踪（Denial Tracking）

`src/utils/permissions/denialTracking.ts`：

如果用户拒绝了某工具调用，**模型应该学会"不再重试同样的事"**。
Claude Code 把这条信息塞回 messages：

```
user rejected: Bash(rm -rf old_logs/)
```

模型看到后，下次不会重试相同操作，而是**换思路**：
```
"那我用 mv 把它们移到 ~/.Trash？"
```

> 这是细节但很重要 —— 否则模型反复问 "rm 行吗？rm 行吗？" 用户疯。

---

## 🔗 延伸阅读

- 上一节：[01-5种权限模式](01-5%E7%A7%8D%E6%9D%83%E9%99%90%E6%A8%A1%E5%BC%8F.md)
- 下一节：[03-Bash危险模式](03-Bash%E5%8D%B1%E9%99%A9%E6%A8%A1%E5%BC%8F.md) —— 第 0 层细讲
- 真源码：`_source/.../src/hooks/useCanUseTool.tsx`

---

## ❓ 小测验

> 1. canUseTool 的 5 层过滤分别是？顺序对吗
> 2. `updatedInput` 在 PermissionResult 里有什么用？
> 3. 拒绝追踪解决什么问题？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#72-canusetool-%E8%BF%87%E6%BB%A4%E5%99%A8)

---

⬅ [01-5种权限模式](01-5%E7%A7%8D%E6%9D%83%E9%99%90%E6%A8%A1%E5%BC%8F.md)　|	➡ [03-Bash危险模式](03-Bash%E5%8D%B1%E9%99%A9%E6%A8%A1%E5%BC%8F.md)
