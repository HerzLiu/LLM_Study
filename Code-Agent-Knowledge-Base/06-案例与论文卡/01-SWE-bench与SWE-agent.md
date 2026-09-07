---
tags: [Code-Agent, SWE-bench, SWE-agent, ACI, 论文卡]
status: Active
source: "https://www.swebench.com/original.html; https://arxiv.org/abs/2405.15793; https://swe-agent.com/0.7/background/aci/"
stage: cases
---

# 6.1 SWE-bench 与 SWE-agent

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-OpenHands-Aider-ClaudeCode-Codex对比](02-OpenHands-Aider-ClaudeCode-Codex%E5%AF%B9%E6%AF%94.md)

## 🎬 故事比喻：考试场和工具台

**SWE-bench** 像真实维修考试：不是让你在纸上写“怎么修发动机”，而是给你一辆真车、一张故障工单、一套验收测试。你修完以后，检测仪说过就是过，说不过就是不过。

**SWE-agent** 像专门为新人工程师设计的工具台：螺丝刀、扳手、检测仪都重新摆放，说明标签更清楚，避免新人拿错工具。它的核心观点是：模型不是人类，它需要自己的电脑界面。

## 🔧 技术拆解：SWE-bench

SWE-bench 的任务来自真实 GitHub issue 和对应 PR。每个任务大致包含：

| 字段 | 含义 |
|---|---|
| issue / problem statement | 要解决的问题 |
| repo + base commit | 需要修改的代码库状态 |
| gold patch / test patch | 参考修复和测试变化 |
| FAIL_TO_PASS | 修复后应该从失败变通过的测试 |
| PASS_TO_PASS | 修复后必须保持通过的测试 |

这让模型必须做 repo-level 软件工程，而不是单函数生成。它要理解问题、定位文件、修改代码、运行测试，并避免回归。

## 🔧 技术拆解：SWE-agent

SWE-agent 的关键贡献是 **Agent-Computer Interface**。

ACI 的直觉：

```text
普通电脑界面是给人类用的。
Code Agent 需要给模型用的界面。
```

它会把搜索、查看、编辑、执行测试这类能力包装成更适合 LLM 的命令和反馈格式，降低模型在文件导航、编辑和测试执行上的操作难度。

## 🧪 例子：SWE-bench 任务在 agent 眼里是什么

```text
输入:
  repo: django/django
  issue: 某个 QuerySet 边界条件报错
  base_commit: abc123

agent 行动:
  search -> read -> edit -> pytest -> observe failure -> edit -> submit

验收:
  F2P 测试通过
  P2P 测试不回归
```

## ⚠️ 避坑

- SWE-bench 分数不是纯模型能力，也包含 scaffold、上下文选择、测试策略和工具设计。
- 只看 pass@1 容易忽略成本、轨迹长度和安全风险。
- 如果训练数据污染了 benchmark，resolved rate 会失真。

## 🔗 跨库链接

- Code Agent 定义：[01-Code-Agent与Coder模型边界](../01-Code-Agent%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/01-Code-Agent%E4%B8%8ECoder%E6%A8%A1%E5%9E%8B%E8%BE%B9%E7%95%8C.md)
- O/A/R 定义：[03-Observation-Action-Reward定义](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md)
- Claude Code 工具系统：[00-章节总览](../../Claude-Code-Source/03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- ReAct 范式：[02-ReAct范式](../../Hello-Agents/04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/02-ReAct%E8%8C%83%E5%BC%8F.md)

## 资料锚点

- [SWE-bench official](https://www.swebench.com/original.html)
- [SWE-bench GitHub](https://github.com/swe-bench/SWE-bench)
- [SWE-agent paper](https://arxiv.org/abs/2405.15793)
- [SWE-agent ACI docs](https://swe-agent.com/0.7/background/aci/)

## ❓ 自测题

- F2P 和 P2P 各自解决什么评测问题？
- ACI 为什么不是“工具越多越好”？
- SWE-bench 为什么会推动 Code Agent 从代码生成走向软件工程闭环？
