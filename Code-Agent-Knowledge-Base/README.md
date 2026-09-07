---
tags: [Code-Agent, README, 知识库, Coder-Model]
status: Active
source: "用户提供：训练Coder模型全流程.md"
stage: overview
---

# Code Agent Knowledge Base

这是一个专门落地 **Code Agent / Coder Model / Code Agentic RL** 的 Obsidian 知识库。

它不是把一篇长文原样堆进 vault，而是把「从通用 Base 训练到能修 bug 的 Coder Model」拆成可以双链、复习、扩展和对照已有知识库的笔记网络。

## 入口

- 主入口：[00-总览](00-%E6%80%BB%E8%A7%88.md)
- 落地路线：[Code-Agent落地路线](00-MOC/Code-Agent%E8%90%BD%E5%9C%B0%E8%B7%AF%E7%BA%BF.md)
- 训练全流程地图：[Coder模型训练全流程地图](00-MOC/Coder%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E5%85%A8%E6%B5%81%E7%A8%8B%E5%9C%B0%E5%9B%BE.md)
- 双库桥接：[双库桥接｜Agentic-RL到Code-Agent](90-%E9%99%84%E5%BD%95/%E5%8F%8C%E5%BA%93%E6%A1%A5%E6%8E%A5%EF%BD%9CAgentic-RL%E5%88%B0Code-Agent.md)
- 原始长文归档：[原始资料｜训练Coder模型全流程](90-%E9%99%84%E5%BD%95/%E5%8E%9F%E5%A7%8B%E8%B5%84%E6%96%99%EF%BD%9C%E8%AE%AD%E7%BB%83Coder%E6%A8%A1%E5%9E%8B%E5%85%A8%E6%B5%81%E7%A8%8B.md)

## 建议读法

```text
00-总览
  -> Code Agent 与 Coder 模型边界
  -> 四阶段数据总览
  -> Pre-train
  -> Mid-train
  -> SFT
  -> RL
  -> Reward Hacking 与 Sandbox 安全
```

## 和现有知识库的关系

- [Happy-LLM](../Happy-LLM/00-%E6%80%BB%E8%A7%88.md)：补 LLM、Transformer、预训练、SFT、RLHF 基础。
- [Hello-Agents](../Hello-Agents/00-%E6%80%BB%E8%A7%88.md)：补 Agent 范式、工具、上下文工程、Agentic RL。
- [Claude-Code-Source](../Claude-Code-Source/00-%E6%80%BB%E8%A7%88.md)：看工业级 Code Agent 的 loop、tool、context、permission。
- [agentic-rl-knowledge-base](../agentic-rl-knowledge-base/00-%E6%80%BB%E8%A7%88.md)：补 Code Agentic RL、SWE-bench、reward、sandbox。
- [双库桥接｜Agentic-RL到Code-Agent](90-%E9%99%84%E5%BD%95/%E5%8F%8C%E5%BA%93%E6%A1%A5%E6%8E%A5%EF%BD%9CAgentic-RL%E5%88%B0Code-Agent.md)：查两库之间的章节、概念和阅读路径映射。
