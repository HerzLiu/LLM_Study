---
tags: [Code-Agent, Agent工程, Scaffold, Sandbox, Trajectory]
status: Active
source: "用户提供：训练Coder模型全流程.md §五"
stage: engineering
---

# Agent 工程闭环总览

⬅ [04-RL](../03-%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5/04-RL.md) | ➡ [02-Scaffold-Sandbox-Proxy](02-Scaffold-Sandbox-Proxy.md)

## 🎬 故事比喻：软件开发流水线

Code Agent 像一个被安排在开发流水线上的工程师。

它不是在会议室里“建议你怎么修”，而是真的坐到工位上：打开仓库、读文件、改代码、跑测试、看日志、写提交说明。流水线旁边有门禁：哪些文件能碰，哪些命令要审批，哪些网络访问禁止，哪些测试必须通过。

## 闭环是什么

Code Agent 的训练不是“输入 prompt -> 输出答案”。

它是：

```text
observe repo / issue
  -> choose action
  -> execute tool
  -> observe result
  -> update state
  -> final patch
  -> run verifier
  -> reward
```

## 四个核心部件

| 部件 | 作用 |
|---|---|
| Scaffold | 管理 agent loop、工具协议、状态、提交 |
| Sandbox | 隔离执行 shell、编辑、测试 |
| Proxy | 把 scaffold 的 LLM 调用转发到待训模型 |
| Verifier | 把终态转换成 reward |

## 🔧 技术拆解：ACI 是给模型用的 IDE

SWE-agent 提出的 Agent-Computer Interface 可以理解为“专门给 LLM 用的 IDE”。人类喜欢 VS Code、断点、鼠标、文件树；模型更需要的是：

- 简洁稳定的命令空间。
- 可预测的文件查看格式。
- 不容易误用的编辑工具。
- 短而信息密度高的错误反馈。
- 明确的 submit 边界。

这和 [Claude Code Tool 接口设计](../../Claude-Code-Source/03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/02-Tool%E6%8E%A5%E5%8F%A3%E8%AE%BE%E8%AE%A1.md) 是同一个问题：工具不是越原始越好，而是要让模型少犯低级错。

## 🧪 例子：工具边界

| 工具 | 适合给 Agent 的形态 | 风险 |
|---|---|---|
| search | `rg` / symbol search / repo map | 搜索太泛导致上下文污染 |
| read | 带行号、窗口化读取 | 一次读太多造成 context rot |
| edit | patch / str_replace / AST edit | 全文件重写、格式漂移 |
| bash | 白名单命令 + 超时 | 删除文件、联网、跑飞 |
| browser | 截图、DOM、console | 视觉误判、资源泄漏 |
| git | diff / status / commit message | 覆盖用户改动 |

## ⚠️ 避坑

- 直接给 raw shell 不等于强大，可能只是把动作空间炸大。
- 工具返回越长越好是错的，反馈要压缩成可行动信息。
- sandbox 不是可选项，训练阶段尤其要可复现。

## 和 Claude Code 的对应

- loop：[02-queryLoop骨架](../../Claude-Code-Source/02-%E7%AC%AC2%E7%AB%A0-Agent-Loop%E4%B8%BB%E5%BE%AA%E7%8E%AF/02-queryLoop%E9%AA%A8%E6%9E%B6.md)
- tool：[00-章节总览](../../Claude-Code-Source/03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- context：[00-章节总览](../../Claude-Code-Source/04-%E7%AC%AC4%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- permission：[00-章节总览](../../Claude-Code-Source/07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

## 继续阅读

- [02-Scaffold-Sandbox-Proxy](02-Scaffold-Sandbox-Proxy.md)
- [03-工具调用与轨迹Schema](03-%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E4%B8%8E%E8%BD%A8%E8%BF%B9Schema.md)
