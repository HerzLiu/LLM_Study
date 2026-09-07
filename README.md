# LLM Study

从 **LLM 基础 → Agent 构建 → Code Agent 工程 → Agentic RL → Coder Model 训练** 的中文学习笔记与知识地图。

**[打开 LLM-Agent 学习地图](_LLM-Agent学习地图.md)** · [Coder Model 训练全流程](Code-Agent-Knowledge-Base/00-MOC/Coder模型训练全流程地图.md) · [Code Agentic RL 专题](agentic-rl-knowledge-base/04-第4章-Code-Agentic-RL专题/00-章节总览.md)

## 知识库导航

| 知识库 | 主要内容 | 文件数 |
| --- | --- | ---: |
| [Happy-LLM](Happy-LLM/00-总览.md) | NLP、Transformer、预训练、SFT / RLHF、LLM 应用 | 69 |
| [Hello-Agents](Hello-Agents/00-总览.md) | Agent 范式、工具、记忆、上下文、MCP、Agentic RL | 73 |
| [Claude-Code-Source](Claude-Code-Source/00-总览.md) | Agent Loop、工具系统、上下文压缩、SubAgent、安全与 MiniCC | 59 |
| [Agentic-RL](agentic-rl-knowledge-base/00-总览.md) | RL 基础、RLVR、GRPO、Code Agentic RL、动手项目 | 45 |
| [Code-Agent-Knowledge-Base](Code-Agent-Knowledge-Base/00-总览.md) | 数据工程、Pre-train / Mid-train / SFT / RL、reward、sandbox、案例 | 31 |
| [ML / DL Foundations](ml-dl-foundations/00-总览.md) | 数学、机器学习、深度学习及通向 LLM 的基础知识 | 50 |

前五项是学习地图中的核心知识库；ML / DL Foundations 补齐跨库引用的前置知识。共收录 **327 篇 Markdown 学习资料和 1 个 Obsidian Canvas**（含总地图，不计本 README）。

## 怎么读

- **从基础开始**：ML / DL Foundations → Happy-LLM → Hello-Agents。
- **关注 Agent 工程**：Hello-Agents → Claude-Code-Source → MiniCC 章节。
- **关注后训练与 Code Agent**：Happy-LLM 的训练章节 → Agentic-RL → Code-Agent-Knowledge-Base。
- **按主题查阅**：从总地图进入各库的 `00-总览`、章节总览与附录；章末保留跨库导航、自测题和延伸阅读。

可直接在 GitHub 点击笔记链接，也可下载仓库并作为 Obsidian Vault 打开。笔记间链接已转换为标准 Markdown 相对链接；Canvas 保留原格式，使用 Obsidian 查看。

## 内容说明

这是个人整理的学习笔记，保留原有技术解释、例子、代码块和参考资料。整理发布于 2026-09-07，源笔记快照截至 2026-06-14；其中模型、框架版本与论文数据反映笔记当时的资料，不代表当前最新结论。

Claude-Code-Source 是特定版本的源码阅读笔记，并非官方文档。MiniCC 的分段教学代码保留在第 8 章；原笔记提到的本地独立脚本和 Claude Code 源码目录不在源 Vault 中，因此本仓库不提供这些文件。代码示例未在本次整理中执行。

原课程、论文与项目的权利归各自作者所有，来源见各篇笔记。仓库未为第三方内容另行授予许可证。
