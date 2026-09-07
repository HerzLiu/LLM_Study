---
tags: [Claude-Code, 第7章, 沙箱, Hooks, 物理隔离]
chapter: 7
section: 7.4
---

# 7.4 物理沙箱 + Hooks

⬅ [03-Bash危险模式](03-Bash%E5%8D%B1%E9%99%A9%E6%A8%A1%E5%BC%8F.md)　|	➡ [05-安全设计哲学](05-%E5%AE%89%E5%85%A8%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A6.md)

> 📂 源码：`vendor/sandbox/` + `node_modules/@anthropic-ai/sandbox-runtime/`

---

## 🎬 故事比喻：进展厅前的"金属探测门"

之前的检查都是"看清单 + 问问题"（软件层）。
物理沙箱 = **金属探测门 + 防爆玻璃 + 押送警卫**（硬件层）。

哪怕坏人混过了清单检查、骗过了海关问询，**金属门一响 + 玻璃挡着 + 警卫看着**——根本搞不出大事。

---

## 🛡️ 沙箱：终极防线

prompt 约束、工具过滤、危险模式检测都是**软件层**。
万一某个攻击绕过了上层，**沙箱在最底层兜底**。

### Claude Code 的沙箱实现

`vendor/sandbox/` + `@anthropic-ai/sandbox-runtime` 包：

| 平台 | 沙箱机制 |
|---|---|
| **macOS** | `sandbox-exec` + sandbox profile（系统级） |
| **Linux** | `seccomp` filter + namespace 隔离（系统调用过滤） |
| **网络** | HTTP proxy 强制走代理（可审计 + 白名单） |

### 沙箱限制什么

子进程跑在沙箱里：
- ✅ 只能写**白名单目录**（如 `/tmp`、当前项目）
- ❌ 不能联网（除非允许的域名）
- ❌ 不能 fork 太多子进程
- ❌ 不能访问其他用户数据
- ❌ 不能读敏感系统文件（`/etc/shadow` 等）
- 限 CPU / 内存

### 视觉化

```
┌────────────────────────────────────────────┐
│              Claude Code 主进程            │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │     Bash 子进程（在沙箱里）          │  │
│  │                                      │  │
│  │  ┌────────────────────────────────┐  │  │
│  │  │  实际命令（如 npm install）     │  │  │
│  │  │  - 文件 IO 限制                 │  │  │
│  │  │  - 网络白名单                   │  │  │
│  │  │  - 资源上限                     │  │  │
│  │  └────────────────────────────────┘  │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
            ↑ 沙箱边界 ↑
   即使命令"坏"，也跑不出这个边界
```

---

## 🔧 Hooks：用户自定义拦截

除了系统级权限，Claude Code 还提供**用户级 hook 机制**。

### 配置例子

`.claude/settings.json`：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {"type": "command", "command": "/path/to/audit.sh"}
        ]
      }
    ]
  }
}
```

**含义**：每次 Bash 工具要执行前，先跑 `audit.sh` 脚本。

### Hook 脚本协议

```bash
#!/bin/bash
# audit.sh

# stdin 收到工具输入（JSON 格式）
INPUT=$(cat)

# 通过 jq 提取命令
CMD=$(echo "$INPUT" | jq -r '.command')

# 你的审计逻辑
if [[ "$CMD" =~ "production" ]]; then
    # 写日志
    echo "$(date) BLOCKED: $CMD" >> /var/log/claude_audit.log
    # 返回非 0 拒绝
    exit 1
fi

# 0 = 放行
exit 0
```

### Hook 类型

| Hook 名 | 何时触发 | 用途 |
|---|---|---|
| `PreToolUse` | 工具执行前 | 审计 / 拦截 |
| `PostToolUse` | 工具执行后 | 日志 / 通知 |
| `UserPromptSubmit` | 用户输入时 | 输入审计 |
| `Stop` | 主循环每轮 | 中断条件 |
| `PostSampling` | LLM 输出后 | 内容过滤 |

### Hooks 的用途

- **企业审计**：所有 Bash 命令记录到 SIEM
- **合规检查**：禁止访问生产数据
- **自动化集成**：触发 CI / 通知 Slack
- **个人定制**：本地规则（如禁用某些路径）

---

## ⭐ 加深理解：纵深防御的完整图景

把第 7 章所有机制串起来：

```
┌────────────────────────────────────────────┐
│ 1. Prompt 教育（第 5 章）                  │
│    "不要做危险操作"                        │
├────────────────────────────────────────────┤
│ 2. 工具过滤（第 3 章）                     │
│    Plan Mode 删掉所有写工具                │
├────────────────────────────────────────────┤
│ 3. 用户规则（本章 7.2）                    │
│    allow / deny 列表                       │
├────────────────────────────────────────────┤
│ 4. 危险模式检测（本章 7.3）                │
│    26+ 种正则黑名单                        │
├────────────────────────────────────────────┤
│ 5. canUseTool 弹窗（本章 7.2）             │
│    destructive 默认问                      │
├────────────────────────────────────────────┤
│ 6. Hooks（本章 7.4）                       │
│    用户自定义审计                          │
├────────────────────────────────────────────┤
│ 7. 物理沙箱（本章 7.4）                    │
│    seccomp / sandbox-exec                  │
├────────────────────────────────────────────┤
│ 8. Remote killswitch                       │
│    Anthropic 紧急远程禁用                  │
└────────────────────────────────────────────┘
        ↑
   任何一层失败，下一层还能挡
```

**8 层防御** —— 这就是"纵深防御"的真实样貌。

---

## ⚠️ Hooks 的注意事项

1. **Hook 脚本本身的安全**：放可写目录会被改 → 放只读位置或加签名
2. **Hook 超时**：脚本卡死会卡住主循环 → 加超时
3. **Hook 失败处理**：脚本崩了默认应该 **fail-safe**（拒绝）还是 **fail-open**（放行）？看场景

---

## 🐍 Python Hook 简化示意

```python
import asyncio
import subprocess
import json

async def run_pre_tool_hook(tool_name: str, input: dict,
                              hook_script: str) -> tuple[bool, str]:
    """跑用户配置的 PreToolUse hook。"""
    payload = json.dumps({"tool": tool_name, "input": input})
    try:
        result = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                hook_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=5    # ⭐ 防 hook 卡死
        )
        stdout, stderr = await result.communicate(input=payload.encode())
        if result.returncode == 0:
            return True, ""           # 放行
        else:
            return False, stderr.decode()[:200]   # 拒绝 + 原因
    except asyncio.TimeoutError:
        return False, "Hook timed out"
```

---

## 🔗 延伸阅读

- 上一节：[03-Bash危险模式](03-Bash%E5%8D%B1%E9%99%A9%E6%A8%A1%E5%BC%8F.md)
- 下一节：[05-安全设计哲学](05-%E5%AE%89%E5%85%A8%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A6.md) —— 全章总结
- 真源码：`_source/.../node_modules/@anthropic-ai/sandbox-runtime/`
- Hooks 文档：Anthropic Claude Code docs

---

## ❓ 小测验

> 1. 物理沙箱比软件层安全好在哪？
> 2. PreToolUse hook 收到什么 / 返回什么决定放行？
> 3. Hook 超时该怎么处理？fail-safe 还是 fail-open？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#74-%E7%89%A9%E7%90%86%E6%B2%99%E7%AE%B1%E4%B8%8E-hooks)

---

⬅ [03-Bash危险模式](03-Bash%E5%8D%B1%E9%99%A9%E6%A8%A1%E5%BC%8F.md)　|	➡ [05-安全设计哲学](05-%E5%AE%89%E5%85%A8%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A6.md)
