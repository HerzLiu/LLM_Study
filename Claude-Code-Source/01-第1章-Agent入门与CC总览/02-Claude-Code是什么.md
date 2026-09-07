---
tags: [Claude-Code, 第1章, 产品介绍]
chapter: 1
section: 1.2
---

# 1.2 Claude Code 是什么 + 与同类产品对比

⬅ [01-Agent是什么](01-Agent%E6%98%AF%E4%BB%80%E4%B9%88.md)　|	➡ [03-最简核心循环](03-%E6%9C%80%E7%AE%80%E6%A0%B8%E5%BF%83%E5%BE%AA%E7%8E%AF.md)

---

## 🎬 一句话定义

> **Claude Code = Anthropic 官方出的"住在终端里的 Coding Agent"。**

你在终端里输："帮我把这个 bug 修了"——它自己读代码、跑测试、改文件、提 commit。

---

## 🔧 它和同类产品的差别

| 工具 | 形态 | 主要交互 | 能"动手"吗 |
|---|---|---|---|
| ChatGPT 网页 | Web | 复制粘贴问问题 | ❌ |
| GitHub Copilot | IDE 插件 | 行内自动补全 | 🟡 写代码不执行 |
| Cursor / Windsurf | 改版 VSCode | Chat 面板 + 文件编辑 | ✅ 在 IDE 内 |
| **Claude Code** | **终端 CLI** | **自然语言对话** | ✅ 可跑任意命令 |
| OpenHands / Aider | 终端/Web | 类似 | ✅ |

### Claude Code 的"出圈点"

1. **终端原生** —— 不依赖 IDE，SSH 进服务器也能用
2. **长任务能力强** —— 能持续工作几十分钟，自己规划自己执行
3. **真·能干活** —— 工具集里有 `Bash`，几乎能做命令行能做的任何事（在你授权前提下）

---

## 📂 源码统计（来自反编译仓库）

| 项 | 数值 |
|---|---|
| 源文件数 | ~1,884 个 .ts/.tsx |
| 代码行数 | ~512,664 行 |
| 最大单文件 | `main.tsx` (4683 行, 785KB) |
| 内置工具 | 40+ 个 |
| 斜杠命令 | 80+ 个 |
| 依赖包 | 192 个 |
| 运行时 | Bun（编译为 Node.js bundle） |

> ⚠️ Claude Code **本体是 TypeScript**，不是 Python。
> 我们用 Python 重写它的核心思想，是为了让 Python 学习者能跟着动手。

---

## 🗂️ 源码目录速览

```
src/
├── main.tsx                # CLI 入口（4683 行，React/Ink 终端 UI）
├── QueryEngine.ts          # SDK/无头模式入口
├── query.ts                # ⭐ 主 Agent 循环（1729 行，最核心）
├── Tool.ts                 # 工具接口定义
├── Task.ts                 # 任务/SubAgent 基类
├── tools.ts                # 工具注册与过滤
├── tools/                  # 40+ 工具实现
│   ├── BashTool/           #   执行 shell 命令
│   ├── FileReadTool/       #   读文件
│   ├── FileEditTool/       #   编辑文件
│   ├── GrepTool/           #   代码搜索
│   ├── GlobTool/           #   文件名匹配
│   ├── AgentTool/          #   派生子 Agent（Task 工具）
│   ├── TodoWriteTool/      #   任务清单管理
│   └── ...
├── services/
│   ├── tools/              #   工具执行编排
│   ├── compact/            #   上下文压缩
│   └── api/                #   调 Claude API
├── commands/               # ~80 个 / 斜杠命令
└── hooks/                  # React hooks，包含 useCanUseTool（权限）
```

> 完整源码地图见 [源码地图](../%E9%99%84%E5%BD%95/%E6%BA%90%E7%A0%81%E5%9C%B0%E5%9B%BE.md)

---

## ⭐ 加深理解：为什么 Claude Code 选了"终端 + Bash"路线？

设想 3 种 Agent 形态：

| 形态 | 能力上限 | 安装难度 | 适用场景 |
|---|---|---|---|
| **浏览器扩展** | 只能操作网页 | 低 | 网页填表/抓取 |
| **IDE 插件** | 操作打开的文件 | 中 | 编辑器内编程 |
| **终端 CLI + Bash** | **能做命令行能做的一切** | 高（但开发者熟） | 复杂工程任务 |

Claude Code 选第三种，意味着：
- **能力上限最高**（甚至可以 ssh 到生产服务器）
- **必须配套强权限系统**（不然分分钟翻车）
- **目标用户是开发者**（习惯终端）

这个选择决定了 Claude Code 的所有后续设计——为什么有 Plan Mode、为什么 Bash 有那么多安全检查、为什么 prompt 里反复强调"破坏性操作要确认"。

---

## 🔗 延伸阅读

- 第 7 章 [00-章节总览](../07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) —— 看 Bash 怎么不让你翻车
- 第 3 章 [00-章节总览](../03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) —— 看 40 个工具怎么组织
- 上一节：[01-Agent是什么](01-Agent%E6%98%AF%E4%BB%80%E4%B9%88.md)
- 下一节：[03-最简核心循环](03-%E6%9C%80%E7%AE%80%E6%A0%B8%E5%BF%83%E5%BE%AA%E7%8E%AF.md) —— Claude Code 的核心 5 步

---

## ❓ 小测验

> 1. Cursor 和 Claude Code 都能"动手"，最大差别是什么？
> 2. 为什么 Claude Code 必须配套强权限系统？
> 3. `src/query.ts` 是干嘛的？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#12-claude-code-%E6%98%AF%E4%BB%80%E4%B9%88)

---

⬅ [01-Agent是什么](01-Agent%E6%98%AF%E4%BB%80%E4%B9%88.md)　|	➡ [03-最简核心循环](03-%E6%9C%80%E7%AE%80%E6%A0%B8%E5%BF%83%E5%BE%AA%E7%8E%AF.md)
