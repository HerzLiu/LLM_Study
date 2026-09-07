---
tags: [agentic-rl, 第3章, repo, TRL, verl, open-r1, agent-lightning, AgentBench, OpenRLHF, CleanRL, repo-map]
chapter: 3
section: 3.4
source: "agentic-rl-learning-map/03-repos.md (全文)"
---

# 3.4 Repo 地图与运行成本（5 必学 + 5 强推 + 5 可选）

⬅ [03-强推+可选论文索引](03-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95.md) | ➡ [05-6周通用路线](05-6%E5%91%A8%E9%80%9A%E7%94%A8%E8%B7%AF%E7%BA%BF.md)

---

## 🎬 故事比喻：训练框架 vs 评测环境 vs Agent 脚手架

```
┌────────────────────────────────────────────────┐
│ 训练框架（你用它写训练 loop）                    │
│ TRL  /  verl  /  open-r1  /  OpenRLHF          │
└────────────────────────────────────────────────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ 评测环境        │ │ Agent 脚手架    │ │ RL 教学/补课     │
│ AgentBench     │ │ Agent Lightning│ │ CleanRL         │
│ tau-bench      │ │ AutoGen         │ │ Gymnasium       │
└────────────────┘ └────────────────┘ └────────────────┘
```

> **不要把所有 repo 都当训练框架**。
> 看清楚每个 repo 的**角色**，再决定要不要装。

---

## 🏆 3.4.0 必学 Repo 5 个（核心表）

> **Stars / 更新时间为 2026-06-05 检索值**，会变。

来自原始资料 §"必学" 表（完整保留）：

| # | Repo | Stars | Forks | 更新 | 项目定位 | 适合学什么 | 初学者运行 | 环境 / 成本 |
|---|---|---:|---:|---|---|---|---|---|
| 1 | [TRL](https://github.com/huggingface/trl) | 18,547 | 2,767 | 2026-06-05 | Hugging Face 的 LLM RL / post-training 框架 | PPO、DPO、GRPO、SFT、reward modeling **最小实践入口** | **适合** | Python + Transformers，**单卡可跑小模型** |
| 2 | [verl](https://github.com/verl-project/verl) | 21,776 | 4,006 | 2026-06-05 | 灵活高效的 RL post-training 框架 | PPO / GRPO / DAPO / RLVR、distributed rollout/training | 先读 recipe，运行中高 | Python + Ray + vLLM/训练后端；建议**多 GPU** |
| 3 | [open-r1](https://github.com/huggingface/open-r1) | 26,033 | 2,423 | 2026-06-04 | **DeepSeek-R1 开放复现项目** | RLVR、reasoning data、GRPO/R1 recipe | 先读 recipe；完整复现成本高 | Python + HF stack + GPU；完整训练需多卡 |
| 4 | [agent-lightning](https://github.com/microsoft/agent-lightning) | 17,280 | 1,509 | 2026-06-04 | **任意 AI Agent 的 RL 训练框架** | 把 LangChain / OpenAI SDK / AutoGen 等 agent 轨迹接入 RL | 适合做 toy tool agent | Python + agent 框架 + API key；训练需 GPU 或小模型 |
| 5 | [AgentBench](https://github.com/THUDM/AgentBench) | 3,469 | 260 | 2026-06-04 | LLM agent **综合评测 benchmark** | 多环境 agent evaluation、任务成功率、交互评测 | 可运行部分环境 | Python + API key；不同环境依赖不同 |

### 速选指南

| 你的需求 | 选哪个 |
|---|---|
| 学 PPO/DPO/GRPO 的最快路径 | **TRL** |
| 想做大规模 RL 训练 | **verl** 或 OpenRLHF |
| 想复现 DeepSeek-R1 | **open-r1** |
| 把已有 agent 接 RL | **agent-lightning** |
| 评测 agent | **AgentBench** |

---

## 🌟 3.4.1 强烈推荐 Repo 5 个

| Repo | Stars | 项目定位 | 适合学什么 | 初学运行成本 |
|---|---:|---|---|---|
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 9,596 | Ray + vLLM 的可扩展 RLHF / Agentic RL 框架 | PPO、DAPO、REINFORCE++、异步 RL、分布式 | 中高（多 GPU 或云资源） |
| [verl-agent](https://github.com/langfengQ/verl-agent) | 1,975 | verl 的 LLM/VLM agent RL 扩展，**GiGPO 官方代码** | 长程 agent trajectory 训练、Group-in-Group 优化 | **高**（研究复现向） |
| [ArCHer](https://github.com/yifeizhou02/ArCHer) | 205 | ArCHer **官方研究代码** | multi-turn actor-critic、层次化 agent RL | 中高（适合读实现） |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) | 5,089 | 软件工程 agent benchmark | 真实 GitHub issue 修复评测、unit-test reward | 中（Docker 环境重） |
| [tau-bench](https://github.com/sierra-research/tau-bench) | 1,258 | Tool-Agent-User 多轮交互 benchmark | 用户模拟器、业务工具调用 | 中（API 成本可控） |

> ⚠️ **SWE-bench URL 不一致**：通用资料写 `SWE-bench/SWE-bench`，Code 专题资料写 `princeton-nlp/SWE-bench`。
> 详见 [02-资料不一致与待确认清单](../%E9%99%84%E5%BD%95/02-%E8%B5%84%E6%96%99%E4%B8%8D%E4%B8%80%E8%87%B4%E4%B8%8E%E5%BE%85%E7%A1%AE%E8%AE%A4%E6%B8%85%E5%8D%95.md) §A.2.2 不一致 #1。
> Code 专题（[07-Repo地图（SWE-agent-OpenHands等）](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/07-Repo%E5%9C%B0%E5%9B%BE%EF%BC%88SWE-agent-OpenHands%E7%AD%89%EF%BC%89.md)）统一用 `princeton-nlp`。

---

## 📦 3.4.2 可选拓展 Repo 5 个

| Repo | Stars | 项目定位 | 适合学什么 | 成本 |
|---|---:|---|---|---|
| [AppWorld](https://github.com/stonybrooknlp/appworld) | 436 | APP / function-calling agent 环境 | 可控 app 世界 | 中 |
| [PRM800K](https://github.com/openai/prm800k) | 2,140 | 80 万步级别数学推理 correctness labels | **PRM 数据格式 + 过程监督** | 低（读数据为主） |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | 9,904 | **单文件深度 RL 实现** | 用最少代码理解 PPO、DQN、SAC | **低**（适合补 RL 基础） |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | 12,000 | 单 agent RL 环境 API 标准 | 环境接口、step / reset / trajectory | **低** |
| [AutoGen](https://github.com/microsoft/autogen) | 58,696 | Agentic AI 编程框架 | 多 agent workflow（**不是 RL 框架**，但适合作 agent harness） | 低-中（要 API key） |

> 💡 **CleanRL 被严重低估**：单文件 PPO 实现，对 LLM 背景的人补 RL 基础**极其友好**。

---

## 🚦 3.4.3 推荐运行顺序（资料原文）

```
1. CleanRL PPO 或 Gymnasium     ← 最小环境理解 RL loop
2. TRL                          ← 跑 tiny PPO/DPO/GRPO 例子
3. AgentBench 或 tau-bench       ← 跑 agent evaluation
4. Agent Lightning              ← 把 toy tool-use agent 接成可训练轨迹
5. verl / open-r1               ← 阅读大规模 RLVR recipe；条件允许跑小模型实验
```

→ **不要跳级**。从 CleanRL 单文件开始，建立"看得懂 PPO 代码"的能力。

---

## 🏷 3.4.4 按角色分类（便于检索）

| 角色 | Repo |
|---|---|
| **LLM RL 训练框架** | TRL, verl, OpenRLHF, open-r1 |
| **Agent RL 训练框架（专门）** | agent-lightning, verl-agent, ArCHer |
| **Agent benchmark** | AgentBench, tau-bench, AppWorld, SWE-bench |
| **数据集（论文配套）** | PRM800K |
| **RL 教学 / 补课** | CleanRL, Gymnasium |
| **Agent 编排框架（非 RL）** | AutoGen |

---

## 💰 3.4.5 运行成本估算（粗略）

| Repo | 最小可玩成本 | 完整训练成本 |
|---|---|---|
| CleanRL | 笔记本 CPU 即可 | 单 GPU |
| Gymnasium | 笔记本 CPU 即可 | N/A |
| TRL（小模型） | 单卡 8GB | 单卡 24GB+ |
| AgentBench eval | API key + 单机 | N/A |
| Agent Lightning toy | 单卡 + API | 多卡 + LLM 训练 |
| OpenRLHF | 多 GPU 必备 | 8x H100+ |
| verl / open-r1 完整训练 | 多 GPU 必备 | 16+ H100 |

---

## ⚠️ 3.4.6 常见坑

1. **TRL 版本变动快**：API 经常变，看 README 时注意版本号
2. **verl 默认依赖 Ray + vLLM**：单卡试水时部分功能用不上
3. **open-r1 不等于"一键复现 DeepSeek-R1"**：是社区开放 recipe，**自己跑出 R1-级效果需要大算力**
4. **AgentBench 部分环境要 Docker**：装的时候踩坑多
5. **SWE-bench 单题 Docker build 慢**：第一次跑预留 30 分钟
6. **agent-lightning 还在快速迭代**：API 可能变化

---

## 📌 3.4 节要点

| 必装 Top 3 | 用法 |
|---|---|
| **TRL** | 学 PPO/DPO/GRPO 写代码 |
| **CleanRL** | 补 RL 基础（看单文件 PPO） |
| **AgentBench** 或 **tau-bench** | 跑 agent eval |

| 不要急着装 | 原因 |
|---|---|
| verl / open-r1 完整训练 | 需要多 GPU |
| OpenRLHF 完整跑 | 同上 |
| SWE-bench 全量 | Docker 慢 |

| Code 方向特别推荐看 | 第 4 章 Repo 地图 |
|---|---|
| [07-Repo地图（SWE-agent-OpenHands等）](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/07-Repo%E5%9C%B0%E5%9B%BE%EF%BC%88SWE-agent-OpenHands%E7%AD%89%EF%BC%89.md) | 5+5+7 个 Code-specific repo |

---

## 🔗 延伸阅读

- 下一节：[05-6周通用路线](05-6%E5%91%A8%E9%80%9A%E7%94%A8%E8%B7%AF%E7%BA%BF.md)
- Code 专题 Repo 地图：[07-Repo地图（SWE-agent-OpenHands等）](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/07-Repo%E5%9C%B0%E5%9B%BE%EF%BC%88SWE-agent-OpenHands%E7%AD%89%EF%BC%89.md)
- 原始资料：`03-repos.md`

---

⬅ [03-强推+可选论文索引](03-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95.md) | ➡ [05-6周通用路线](05-6%E5%91%A8%E9%80%9A%E7%94%A8%E8%B7%AF%E7%BA%BF.md)
