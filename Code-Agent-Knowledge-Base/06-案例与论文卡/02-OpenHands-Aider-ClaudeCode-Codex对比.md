---
tags: [Code-Agent, OpenHands, Aider, Claude-Code, Codex, 产品对比]
status: Active
source: "https://docs.openhands.dev/sdk; https://aider.chat/docs/; https://github.com/anthropics/claude-code; https://developers.openai.com/codex/cloud"
stage: cases
---

# 6.2 OpenHands / Aider / Claude Code / Codex 对比

⬅ [01-SWE-bench与SWE-agent](01-SWE-bench%E4%B8%8ESWE-agent.md) | ➡ [03-SWE-Gym与SWE-smith训练数据](03-SWE-Gym%E4%B8%8ESWE-smith%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE.md)

## 🎬 故事比喻：四种开发搭档

同样是 Code Agent，产品形态差别很大：

- **Aider** 像坐在你旁边的结对程序员：你告诉它要改哪些文件，它直接在本地 repo 里改。
- **Claude Code** 像住在终端里的工程师：会读仓库、跑命令、处理 git workflow，但需要权限和沙箱约束。
- **OpenHands** 像一个可搭建的开发机器人平台：你可以定义 agent、工具、运行环境，把它扩成自己的系统。
- **Codex cloud** 像云端后台工程师：你把任务派出去，它在自己的云环境里读、改、跑，完成后给你 diff。

## 🔧 技术拆解：四类产品形态

| 工具 / 平台 | 主要形态 | 强项 | 适合观察的能力 |
|---|---|---|---|
| Aider | 本地终端 pair programming | 多文件编辑、git repo 内协作 | 人机协作、文件选择、轻量 patch |
| Claude Code | 终端 agentic coding tool | long-horizon loop、工具、上下文、安全 | 工业级 agent loop 和权限系统 |
| OpenHands | 软件开发 agent 平台 / SDK | 可组合 agent、sandbox、云扩展 | 平台化 scaffold 与评测 |
| Codex cloud | 云端 coding agent | 后台并行任务、云环境执行 | 任务委派、异步开发流程 |

## 🧪 例子：同一个任务的产品路径

```text
任务: 修复一个 flaky test。

Aider:
  你把相关文件 /add 到上下文 -> 它给 patch -> 本地跑测试。

Claude Code:
  它自己搜索文件、运行命令、维护 todo、请求权限。

OpenHands:
  agent 在 sandbox 中规划、编辑、执行，可嵌入平台和评测。

Codex cloud:
  你把任务发到云端 -> 后台环境执行 -> 返回可应用 diff。
```

## ⚠️ 避坑

- 产品能跑任务，不代表模型已经被 RL 训练过；很多能力来自 scaffold。
- 本地 agent 要特别注意 secrets、未提交改动和破坏性命令。
- 云端 agent 要特别注意环境复现、权限边界和代码外发合规。
- “自动”不等于“无需 review”，Code Agent 输出仍需要工程师验收。

## 🔗 跨库链接

- Claude Code 源码学习：[00-总览](../../Claude-Code-Source/00-%E6%80%BB%E8%A7%88.md)
- Claude Code 权限：[00-章节总览](../../Claude-Code-Source/07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- Hello-Agents 工具系统：[05-工具系统](../../Hello-Agents/07-%E7%AC%AC7%E7%AB%A0-%E6%9E%84%E5%BB%BA%E4%BD%A0%E7%9A%84%E6%99%BA%E8%83%BD%E4%BD%93%E6%A1%86%E6%9E%B6/05-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F.md)
- 本库工程闭环：[01-Agent工程闭环总览](../04-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF/01-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF%E6%80%BB%E8%A7%88.md)

## 资料锚点

- [OpenHands SDK docs](https://docs.openhands.dev/sdk)
- [OpenHands GitHub](https://github.com/OpenHands/OpenHands)
- [Aider docs](https://aider.chat/docs/)
- [Claude Code GitHub](https://github.com/anthropics/claude-code)
- [Claude Code permissions](https://code.claude.com/docs/en/permissions)
- [OpenAI Codex cloud](https://developers.openai.com/codex/cloud)
- [Codex CLI cloud tasks](https://developers.openai.com/codex/cli/features)

## ❓ 自测题

- Aider 和 OpenHands 的产品定位有什么不同？
- Claude Code 的权限系统为什么是 Code Agent 学习重点？
- Codex cloud 为什么代表“异步软件工程 agent”形态？
