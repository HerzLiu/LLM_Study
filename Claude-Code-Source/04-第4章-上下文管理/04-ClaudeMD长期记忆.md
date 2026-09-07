---
tags: [Claude-Code, 第4章, CLAUDE.md, 长期记忆]
chapter: 4
section: 4.4
---

# 4.4 解药三：CLAUDE.md 项目长期记忆

⬅ [03-TodoWrite外部记忆](03-TodoWrite%E5%A4%96%E9%83%A8%E8%AE%B0%E5%BF%86.md)　|	➡ [05-高级压缩机制](05-%E9%AB%98%E7%BA%A7%E5%8E%8B%E7%BC%A9%E6%9C%BA%E5%88%B6.md)

---

## 🎬 故事比喻：办公桌墙上的便利贴

实习生工位上贴着几张大字便利贴：
```
【咖啡机】二楼东侧
【打印机】密码 1234
【上班打卡】10:00 前
【提交代码】先 pre-commit hook
```

这些规则**每天都要查**，不能每次都问主管。贴墙上 → 顺手看 → 不占脑子。

**CLAUDE.md 就是这种便利贴**：项目级规则、命令、约束写一次，**每次启动 Claude Code 自动加载**到 system prompt。

---

## 🔧 怎么用 CLAUDE.md

### 创建位置

Claude Code 启动时**按顺序**查找：

| 位置 | 作用范围 |
|---|---|
| `当前目录/CLAUDE.md` | 项目级 |
| `父目录/.../CLAUDE.md` | 子项目继承父项目规则 |
| `~/.claude/CLAUDE.md` | 全局（所有项目共享） |

**所有命中的文件都被加载**，按从近到远顺序拼到 system prompt 里。

### 典型内容

```markdown
# 我的 SaaS 项目

## 技术栈
- Python 3.11 + FastAPI
- 测试用 pytest
- 部署：Docker → AWS ECS

## 规范
- 所有函数都要类型注解
- 异常用 logger.exception 而不是 logger.error
- 不要直接修改 main 分支

## 常用命令
- 运行测试：`pytest -xvs`
- 启动开发：`uvicorn app.main:app --reload`
- 构建：`make build`
- 部署 staging：`./deploy.sh staging`

## 注意事项
- `models/legacy/` 目录是历史代码，不要重构
- 数据库迁移要先在 staging 跑通
```

效果：以后再让 Claude 跑测试，它直接 `pytest -xvs`，不会问"用什么测试框架"。

---

## ⭐ 加深理解：CLAUDE.md vs messages

```
最终发给 API 的 system prompt =
    Claude Code 默认 system prompt
  + CLAUDE.md 内容（所有命中文件）
  + 用户的 --system-prompt 参数（如果有）
  + 工具说明
```

**为什么不放进 messages？**

| 维度 | system prompt | messages |
|---|---|---|
| 何时变化 | 项目级，几天一变 | 每轮都变 |
| 能否缓存 | ✅ Prompt Cache 命中省 90% 费 | ❌ 动态内容不缓存 |
| 模型注意力 | 高（明示身份和规则） | 一般（混在历史里） |

**Prompt Cache 命中**是巨大的成本优势。固定内容堆 system prompt，动态内容放 messages，能省一大笔。

---

## 📊 三种"记忆"对照

| 名称 | 位置 | 作用范围 | 谁写 |
|---|---|---|---|
| `CLAUDE.md` | 项目根 / 父目录 | 项目级 | 你 |
| `~/.claude/CLAUDE.md` | 用户主目录 | 全局 | 你 |
| `.claude/settings.json` | 项目根 | 项目配置（非文本记忆）| 你 |
| MemDir (`src/memdir/`) | 内部存储 | 跨会话结构化记忆 | Claude Code 自动 |
| Session 持久化 | `~/.claude/projects/...` | 会话恢复 | Claude Code 自动 |

---

## 🔧 写好 CLAUDE.md 的 5 条原则

### 1️⃣ 简洁但具体

❌ 不好：
```
请遵循最佳实践
```
✅ 好：
```
- 异常用 logger.exception 不是 logger.error
- 数据库 ORM 用 SQLAlchemy 2.0 风格（select() 不是 query()）
```

### 2️⃣ 写"为什么"

```
## 不要重构 models/legacy/
原因：这部分代码是 2019 年的，业务方还在用旧 API，未经协调重构会导致线上故障。
```

模型知道**原因**才能在边界情况下做合理判断。

### 3️⃣ 列出命令

```
## 常用命令
- 跑测试：`pytest -xvs`
- 跑 lint：`ruff check .`
- 构建：`make build`
```

否则模型会 `pytest -v` 或 `python -m pytest`，不一定符合你项目的惯例。

### 4️⃣ 标注"禁区"

```
## NEVER
- 不要修改 .env.production
- 不要 git push --force
- 不要删除任何 migrations/* 文件
```

强约束用 NEVER，模型会更服从。

### 5️⃣ 不超过 200 行

> 太长 → token 涨 → 缓存命中前的开销变大。
> 把"经常用的"放 CLAUDE.md，"偶尔用的"留给用户当场说。

---

## 🐍 Python Mini 版

```python
def load_claude_md(cwd: str = ".") -> str:
    """从当前目录和家目录加载 CLAUDE.md。"""
    parts = []

    # 项目级
    project_md = Path(cwd) / "CLAUDE.md"
    if project_md.exists():
        parts.append(f"# Project Instructions (from {project_md})\n{project_md.read_text()}")

    # 全局
    global_md = Path.home() / ".claude" / "CLAUDE.md"
    if global_md.exists():
        parts.append(f"# Global Instructions (from {global_md})\n{global_md.read_text()}")

    return "\n\n".join(parts)

def build_system_prompt(...):
    sections = [
        "You are MiniCC, ...",
        # ... 默认 sections ...
    ]
    # 把 CLAUDE.md 加在最后（高优先级位置）
    claude_md = load_claude_md()
    if claude_md:
        sections.append(claude_md)
    return "\n\n".join(sections)
```

---

## ⚠️ 小白避坑

1. **路径敏感**：CLAUDE.md 在哪个目录决定它的作用范围
2. **会被缓存**：改了 CLAUDE.md 后下次启动才生效（或 `/reload`）
3. **太长会爆 token**：建议 < 200 行
4. **不要写敏感信息**：CLAUDE.md 会进 prompt，等于"告诉模型"

---

## 🔗 延伸阅读

- 上一节：[03-TodoWrite外部记忆](03-TodoWrite%E5%A4%96%E9%83%A8%E8%AE%B0%E5%BF%86.md)
- 下一节：[05-高级压缩机制](05-%E9%AB%98%E7%BA%A7%E5%8E%8B%E7%BC%A9%E6%9C%BA%E5%88%B6.md)
- System Prompt 全貌：[00-章节总览](../05-%E7%AC%AC5%E7%AB%A0-System-Prompt%E5%B7%A5%E7%A8%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

## ❓ 小测验

> 1. CLAUDE.md 和 messages 的本质差别？
> 2. CLAUDE.md 为什么不会爆 token？（提示：Prompt Cache）
> 3. 写好 CLAUDE.md 的 5 条原则

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#44-claudemd)

---

⬅ [03-TodoWrite外部记忆](03-TodoWrite%E5%A4%96%E9%83%A8%E8%AE%B0%E5%BF%86.md)　|	➡ [05-高级压缩机制](05-%E9%AB%98%E7%BA%A7%E5%8E%8B%E7%BC%A9%E6%9C%BA%E5%88%B6.md)
