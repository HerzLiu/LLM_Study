---
tags: [agentic-rl, 第4章, papers, SWE-Gym, SWE-RL, SWE-smith, R2E-Gym, AlphaCode, LiveCodeBench, BigCodeBench, 论文索引]
chapter: 4
section: 4.6
source: "agentic-rl-learning-map/code-agentic-rl/02-papers.md §强烈推荐, §可选拓展"
---

# 4.6 强推 + 可选论文索引（SWE-Gym / SWE-RL 等）

⬅ [05-必读论文10篇精读卡（SWE-bench线）](05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md) | ➡ [07-Repo地图（SWE-agent-OpenHands等）](07-Repo%E5%9C%B0%E5%9B%BE%EF%BC%88SWE-agent-OpenHands%E7%AD%89%EF%BC%89.md)

---

## 🎬 故事比喻：从"评测"走向"训练"的转折

[4.5 必读](05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md) 那 10 篇大多在解决：**怎么评、怎么用 prompt loop 修 issue**。
本节论文是下一阶段：**怎么把 evaluation 变成 training environment？**

```
Eval  →  Training Environment  →  RL
─────    ─────────────────────    ──────
SWE-bench           SWE-Gym                  SWE-RL
(评测)             (训练数据+轨迹+verifier)   (真RL)
                     ↑
                  SWE-smith
                  (synthetic 数据生成)
                     ↑
                  R2E-Gym
                  (procedural 环境+hybrid verifier)
                     ↑
                  SWE-Dev
                  (可扩展测试构造)
```

→ 这是 2024-2025 Code Agentic RL 的**最前沿**。
→ 不是必精读，但**如果你做研究方向必须扫一遍**。

---

## 🌟 4.6.0 强烈推荐 8 篇（索引）

来自原始资料 §02-papers.md §强烈推荐。这里给"一句话定位 + 阅读时机"：

| # | 标题 | 年份 | 状态（资料原标注） | 一句话价值 | 难度 | 何时读 |
|---|---|---:|---|---|---|---|
| 1 | **SWE-Gym** | 2025 | ICML 2025；CCF-A **需确认** | 训练 SWE agents + verifiers 的环境和轨迹数据。**从 benchmark 走向训练环境** | 高 | 准备做 RL 实验前 |
| 2 | **SWE-RL** | 2025 | NeurIPS 2025；CCF-A **需确认** | 在真实软件演化数据上直接 RL 训练 SWE reasoning | 高 | 做 RL 方法时 |
| 3 | **SWE-smith** | 2025 | NeurIPS 2025 D&B Spotlight **需确认** | 大规模生成 SWE-agent 训练数据 | 中高 | 缺训练数据时 |
| 4 | **SWE-Dev** | 2025 | ACL Findings 2025 **需确认** | 可扩展测试构造 + training/inference scaling | 中高 | 关注数据/测试构造时 |
| 5 | **R2E-Gym** | 2025 | arXiv（机构 **需确认**） | 过程化 SWE 环境 + hybrid verifier | 高 | 研究 verifier 时 |
| 6 | AlphaCode | 2022 | Science 2022 | 不是 repo agent，但代表"采样 + 过滤 + 测试 + rerank"的 code scaling | 中 | 想理解 test-time scaling |
| 7 | LiveCodeBench | 2024/2025 | ICLR 2025 repo 标注 **需确认** | **抗污染**代码评测 | 中 | 担心 benchmark 泄漏时 |
| 8 | BigCodeBench | 2025 | ICLR 2025 repo 标注 **需确认** | 更真实的 **function-level** code generation benchmark | 中 | 补 code eval 维度时 |

> ⚠️ 这 8 篇里，论文状态都标了"需确认"。详细列在 [02-资料不一致与待确认清单](../%E9%99%84%E5%BD%95/02-%E8%B5%84%E6%96%99%E4%B8%8D%E4%B8%80%E8%87%B4%E4%B8%8E%E5%BE%85%E7%A1%AE%E8%AE%A4%E6%B8%85%E5%8D%95.md)。

---

## 🚦 4.6.0.1 强推 8 篇的读法

| 你的状态 | 怎么读 |
|---|---|
| 看完 4.5 必读，想往前走一步 | 只读 **SWE-Gym** + **SWE-RL** 摘要 + 方法图 |
| 准备做 toy bug-fix gym | 先读 **SWE-Gym** 的 reward / verifier 设计部分 |
| 想做数据合成 | 读 **SWE-smith** + **SWE-Dev** |
| 做 verifier 研究 | 读 **LEVER**（在 4.5）+ **R2E-Gym** hybrid verifier 部分 |
| 关注 eval 可靠性 | 读 **SWE-bench Verified**（在 4.5）+ **LiveCodeBench** |
| 想看更大尺度 | 读 **AlphaCode** Science 论文（采样/过滤/集成思想） |

---

## 📚 4.6.1 简要展开（高优先级 3 篇）

### SWE-Gym

| 项 | 内容 |
|---|---|
| 链接 | https://arxiv.org/abs/2412.21139 |
| 作者 | Pan / Wang / Neubig 等 |
| 一句话 | 把 SWE-bench-style 任务做成可批量 rollout 的 **gym 环境**，并提供 trajectory + verifier |
| 你需要看的 | 任务 schema / verifier 训练 / 轨迹格式 / sandbox |
| 应用价值 | 你写 Toy Bug-Fix Gym 时的**直接参考蓝本** |

### SWE-RL

| 项 | 内容 |
|---|---|
| 链接 | https://arxiv.org/abs/2502.18449 |
| 作者 | Meta / CMU 等 |
| 一句话 | 用 RL 在 **open software evolution 数据**（GitHub 真实 PR / issue / commit 序列）上训练 |
| 你需要看的 | rule-based reward、训练 pipeline、和 base model 对比的曲线 |
| 应用价值 | 让你看到"真 RL on code agent"的工程长啥样 |

### SWE-smith

| 项 | 内容 |
|---|---|
| 链接 | https://arxiv.org/abs/2504.21798 |
| 作者 | SWE-bench team 等 |
| 一句话 | **合成大规模 SWE 训练任务**（变异真实 repo 引入 bug，自动构造 failing test） |
| 你需要看的 | 任务合成方法、质量过滤、训练后效果 |
| 应用价值 | 解决"训练数据从哪来"的问题 |

---

## 📦 4.6.2 可选拓展 6 项（索引，不展开）

资料 §"可选拓展"。**只看一句话定位**，需要时再回查：

| 论文 / 项目 | 一句话 | 推荐时机 |
|---|---|---|
| **RepoBench** | repo-level code completion benchmark | 做 long-context code 研究时 |
| **HumanEvalFix / Debugging benchmarks** | bug fixing 小环境 | 做自修复 toy 实验时 |
| **APPS / MBPP / CodeContests** | 代码生成基础 benchmark | 补 code model 评测基础时 |
| **MetaGPT / ChatDev** | 多 agent 软件工程 workflow | 研究 multi-agent coding 时 |
| **Devin 公开材料** | 商业 autonomous SE 代表 | 做产品 / 系统形态调研时 |
| **Aider benchmark / blog** | 终端 pair-programming 实践 | 做工程工具对比时 |

---

## 🧭 4.6.3 整合：4.5 + 4.6 全图

```
                    Code Agent / RL 论文全景（22 篇）
                                │
        ┌───────────┬───────────┼──────────┬─────────────┐
        │           │           │          │             │
   【Benchmark】 【Scaffold】 【Verifier】 【RL/数据】  【对照/补充】
        │           │           │          │             │
    SWE-bench    SWE-agent   CodeT      CodeRL        AlphaCode
    Verified     OpenHands   LEVER      SWE-Gym       LiveCodeBench
    LiveCodeBench AutoCode-  Self-      SWE-RL        BigCodeBench
    BigCodeBench  Rover      Debugging  SWE-smith     RepoBench
    (R2E-Gym)    Agentless  (R2E-Gym)   SWE-Dev       HumanEvalFix
                                       (R2E-Gym)      MetaGPT/ChatDev
                                                      Devin / Aider
   ↑ 4.5 必读 10 篇 ←─────────────────────→ 4.6 强推 8 + 可选 6
```

---

## 📌 4.6 节要点

| 问题 | 一句话答案 |
|---|---|
| 4.5 vs 4.6 怎么分？ | 4.5 是"用 prompt loop 评测/修 issue"；4.6 是"从 eval 走到 training/RL" |
| 进入研究前必须看哪几篇？ | **SWE-Gym**、**SWE-RL**（两篇够撑起整个 RL 工程视角） |
| 数据从哪来？ | **SWE-smith**（合成）、**SWE-Dev**（可扩展测试） |
| 担心 benchmark 泄漏？ | **LiveCodeBench**（抗污染） |
| 想看 test-time scaling 思想？ | **AlphaCode**（虽然不是 repo agent） |

---

## ⚠️ 资料中需要进一步确认的状态（本节）

- SWE-Gym (ICML 2025) / SWE-RL (NeurIPS 2025) / SWE-smith (NeurIPS 2025 D&B) / SWE-Dev (ACL Findings 2025) / LiveCodeBench (ICLR 2025) / BigCodeBench (ICLR 2025) 的**会议状态与 CCF 归属**
- **R2E-Gym** 的**作者机构**（资料原文标"需要进一步确认"）

→ 全部细节见 [02-资料不一致与待确认清单](../%E9%99%84%E5%BD%95/02-%E8%B5%84%E6%96%99%E4%B8%8D%E4%B8%80%E8%87%B4%E4%B8%8E%E5%BE%85%E7%A1%AE%E8%AE%A4%E6%B8%85%E5%8D%95.md)

---

## 🔗 延伸阅读

- 上一节：[05-必读论文10篇精读卡（SWE-bench线）](05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md)
- 下一节：[07-Repo地图（SWE-agent-OpenHands等）](07-Repo%E5%9C%B0%E5%9B%BE%EF%BC%88SWE-agent-OpenHands%E7%AD%89%EF%BC%89.md)
- 原始资料：`code-agentic-rl/02-papers.md` §强烈推荐 + §可选拓展

---

⬅ [05-必读论文10篇精读卡（SWE-bench线）](05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md) | ➡ [07-Repo地图（SWE-agent-OpenHands等）](07-Repo%E5%9C%B0%E5%9B%BE%EF%BC%88SWE-agent-OpenHands%E7%AD%89%EF%BC%89.md)
