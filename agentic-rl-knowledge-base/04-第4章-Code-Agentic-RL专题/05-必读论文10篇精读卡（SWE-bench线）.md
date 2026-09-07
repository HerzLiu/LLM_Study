---
tags: [agentic-rl, 第4章, papers, SWE-bench, SWE-agent, Agentless, AutoCodeRover, Self-Debugging, CodeT, LEVER, CodeRL, OpenHands, 精读卡]
chapter: 4
section: 4.5
source: "agentic-rl-learning-map/code-agentic-rl/02-papers.md §必学, 07-shortest-path.md §10篇以内论文"
---

# 4.5 必读论文 10 篇精读卡（SWE-bench 主线）

⬅ [04-RL如何用在Code-Agent上](04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md) | ➡ [06-强推+可选论文索引（SWE-Gym-SWE-RL等）](06-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95%EF%BC%88SWE-Gym-SWE-RL%E7%AD%89%EF%BC%89.md)

---

## 🎬 故事比喻：一条 SWE-bench 时间线

```
2022 ────────────────────────── 2023 ────────────────────────── 2024 ────────────────────────── 2025
                                                                    │
                                                                    SWE-bench (Princeton)
                                                                    第一个真实 GitHub issue benchmark
                                                                    │
   CodeRL  ←─ 早期 code RL 范式                                   ↓
                                                              SWE-bench Verified (OpenAI)
   ↓                                                          人工筛 500 题，reward 可信
   Self-Debugging ←─ execution feedback ↓                        │
                                       OpenHands (CMU)        ↓
   CodeT ←─ generated tests/rerank      通用 SE agent 平台    SWE-agent (Princeton)
                                       ↓                       Agent-Computer Interface
   LEVER  ←─ execution verifier         Agentless              ↓
                                       不用 agent loop         AutoCodeRover (NUS)
                                       照样很强 baseline       结构感知 + program improvement
```

> 这 10 篇就是 Code Agent 方向的"主干道"。
> 读完它们，你打开任何 2025-2026 新论文都能秒定位。

---

## 📚 4.5.0 阅读顺序（资料推荐）

来自原始资料 §02-papers.md 末尾：

```
1. SWE-bench              ← 先建立任务概念
2. SWE-bench Verified     ← 修补 benchmark 噪声
3. SWE-agent              ← 经典 agent scaffold
4. Agentless              ← 强 baseline，破除"必须用 agent"迷信
5. AutoCodeRover          ← 结构检索 + localization
6. Self-Debugging         ← execution feedback 入门
7. CodeT                  ← generated tests + selection
8. CodeRL                 ← 进入 RL for code
9. OpenHands              ← 更完整的 SE agent 平台
10. LEVER                 ← execution-based verifier（资料归入"强推"，但 5 篇必读会跳，所以单独列）
```

> ⚠️ 原始资料的"必学 10 篇"和"最短路径 10 篇"略有差异。
> 本节按**必学 10 篇**（资料 §02-papers.md 必学表 9 项 + 推荐顺序里前 8）整合，OpenHands 来自必学表。
> LEVER 资料里在「必学」第 8 行（§02-papers.md row 8），所以一并保留。

---

## 🪪 必读卡 1 — SWE-bench

| 字段 | 内容 |
|---|---|
| 标题 | SWE-bench: Can Language Models Resolve Real-world Github Issues? |
| 年份 | 2024 |
| 作者 / 机构 | Carlos E. Jimenez et al., Princeton |
| 状态 | ICLR 2024（CCF-A **需进一步确认**） |
| 链接 | https://arxiv.org/abs/2310.06770 |
| 解决的问题 | 把"修真实 GitHub issue"做成可复现的 benchmark |
| 方法核心 | 收集 GitHub 上 issue + PR + 测试，构造 (repo, issue, FAIL_TO_PASS, PASS_TO_PASS) 任务 |
| 和 Code Agentic RL 关系 | **定义了 reward / evaluation 环境**。所有后续 RL 工作的目标函数源头 |
| 是否用 RL？ | ❌ 否，只是测试反馈用于 evaluation |
| 现在应不应该精读？ | ✅ **必精读**。读完整篇，懂 task 构造、harness、resolved rate |
| 阅读重点 | task 选择标准、FAIL_TO_PASS / PASS_TO_PASS 设计、Docker harness |

---

## 🪪 必读卡 2 — SWE-bench Verified

| 字段 | 内容 |
|---|---|
| 标题 | SWE-bench Verified |
| 年份 | 2024 |
| 作者 / 机构 | OpenAI + SWE-bench authors |
| 状态 | 技术 / 数据发布（不是论文） |
| 链接 | https://openai.com/index/introducing-swe-bench-verified/ |
| 解决的问题 | 原 SWE-bench 噪声大（测试覆盖不足、issue 描述歧义） |
| 方法核心 | 人工验证 500 题，明确 FAIL_TO_PASS / PASS_TO_PASS 集合 |
| 和 Code Agentic RL 关系 | **让 reward 更可信**，是训练 / 评测的首选子集 |
| 是否用 RL？ | 否，测试反馈 / verifier 视角 |
| 现在应不应该精读？ | ✅ **必读**（半小时即可，是 blog） |
| 阅读重点 | 为什么原 benchmark 有问题、人工验证流程 |

---

## 🪪 必读卡 3 — SWE-agent

| 字段 | 内容 |
|---|---|
| 标题 | SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering |
| 年份 | 2024 |
| 作者 / 机构 | John Yang et al., Princeton |
| 状态 | NeurIPS 2024（CCF-A **需进一步确认**） |
| 链接 | https://arxiv.org/abs/2405.15793 |
| 解决的问题 | 直接给 LLM raw bash → resolved rate 很低 |
| 方法核心 | 设计 **Agent-Computer Interface (ACI)**：为 LLM 优化的命令集（goto/scroll/edit/find_file 等） |
| 和 Code Agentic RL 关系 | **最经典的 SWE-bench agent scaffold**；action space 设计典范 |
| 是否用 RL？ | ❌ 否，纯 prompt loop |
| 现在应不应该精读？ | ✅ **必精读**。重点看 ACI 设计原则 |
| 阅读重点 | ACI 17 个命令、为什么 raw bash 不行、failure mode 分析 |

---

## 🪪 必读卡 4 — Agentless

| 字段 | 内容 |
|---|---|
| 标题 | Agentless: Demystifying LLM-based Software Engineering Agents |
| 年份 | 2024 |
| 作者 / 机构 | Xia et al. |
| 状态 | arXiv（**会议状态需进一步确认**） |
| 链接 | https://arxiv.org/abs/2407.01489 |
| 解决的问题 | "Agent loop 是不是必须？" |
| 方法核心 | 3 阶段 pipeline：**Localization → Repair → Patch Validation**（**无 agent loop**） |
| 和 Code Agentic RL 关系 | **强 baseline**。任何 agent loop 都应该证明自己比 Agentless 强 |
| 是否用 RL？ | ❌ 否，多阶段 pipeline |
| 现在应不应该精读？ | ✅ **必精读**。破除"必须 agent"迷信 |
| 阅读重点 | Localization 用了什么、采样 + 测试 rerank、降本对比 |

> 💡 **重要启示**：Agentless 上 SWE-bench Lite 的 resolved rate 一度**超过当时所有 agent 方法**。
> 教训：**不要为了用 agent 而用 agent**。

---

## 🪪 必读卡 5 — AutoCodeRover

| 字段 | 内容 |
|---|---|
| 标题 | AutoCodeRover: Autonomous Program Improvement |
| 年份 | 2024 |
| 作者 / 机构 | Zhang et al., NUS |
| 状态 | ISSTA 2024（**CCF 分类需确认**） |
| 链接 | https://arxiv.org/abs/2404.05427 |
| 解决的问题 | 普通 agent 不理解 repo 结构 |
| 方法核心 | **结构感知**：用 AST / class hierarchy / 调用图做 fault localization，再生成 patch |
| 和 Code Agentic RL 关系 | **Repo 检索 + localization 的代表**（[能力栈](02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md) 模块 1、2） |
| 是否用 RL？ | ❌ 否，执行 / 测试辅助 |
| 现在应不应该精读？ | ⚠️ **建议精读**。看它怎么用 AST/symbol search |
| 阅读重点 | structure-aware search API、context window 管理 |

---

## 🪪 必读卡 6 — Teaching Large Language Models to Self-Debug

| 字段 | 内容 |
|---|---|
| 标题 | Teaching Large Language Models to Self-Debug |
| 年份 | 2023 |
| 作者 / 机构 | Chen et al., Google |
| 状态 | arXiv |
| 链接 | https://arxiv.org/abs/2304.05128 |
| 解决的问题 | LLM 写错代码后不会"读自己的错误" |
| 方法核心 | **Self-debugging 模式**：执行 → 看输出 → 解释错误 → 修代码（rubber duck debugging） |
| 和 Code Agentic RL 关系 | **error recovery 的奠基论文**（[能力栈](02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md) 模块 7） |
| 是否用 RL？ | ❌ 否，prompt + execution feedback |
| 现在应不应该精读？ | ✅ **必精读**（短，易读，思路清晰） |
| 阅读重点 | rubber duck / explain-then-fix 模板、execution trace 怎么用 |

---

## 🪪 必读卡 7 — CodeT

| 字段 | 内容 |
|---|---|
| 标题 | CodeT: Code Generation with Generated Tests |
| 年份 | 2023 |
| 作者 / 机构 | Chen et al., Microsoft |
| 状态 | ICLR 2023（CCF-A **需进一步确认**） |
| 链接 | https://arxiv.org/abs/2207.10397 |
| 解决的问题 | 没有 ground-truth 测试时怎么选最好的代码？ |
| 方法核心 | LLM **生成测试 + 生成代码 → 自一致性筛选**（哪段代码通过最多生成的测试就选哪段） |
| 和 Code Agentic RL 关系 | **test-time verifier** 思路；为 RLVR 提供"假设测试"工具 |
| 是否用 RL？ | ❌ 否，generated tests + execution selection |
| 现在应不应该精读？ | ✅ **建议精读** |
| 阅读重点 | 生成测试的 prompt、self-consistency scoring |

---

## 🪪 必读卡 8 — CodeRL

| 字段 | 内容 |
|---|---|
| 标题 | CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning |
| 年份 | 2022 |
| 作者 / 机构 | Hung Le et al., Salesforce |
| 状态 | NeurIPS 2022（CCF-A **需进一步确认**） |
| 链接 | https://arxiv.org/abs/2207.01780 |
| 解决的问题 | 第一篇把 **functional correctness** 当 RL reward 训代码模型 |
| 方法核心 | actor-critic + 测试反馈作 reward，分 unit-test 通过粒度（PASS / FAIL / COMPILE_ERROR / RUNTIME_ERROR）打分 |
| 和 Code Agentic RL 关系 | **Code RL 鼻祖**。所有 RLVR-for-code 的概念原型 |
| 是否用 RL？ | ✅ 是 + critic + execution feedback |
| 现在应不应该精读？ | ⚠️ **建议精读**（思路重要，工程偏旧） |
| 阅读重点 | reward 粒度划分（4 类）、critic 设计 |

---

## 🪪 必读卡 9 — OpenHands

| 字段 | 内容 |
|---|---|
| 标题 | OpenHands: An Open Platform for AI Software Developers as Generalist Agents |
| 年份 | 2025 |
| 作者 / 机构 | Wang et al., CMU / Illinois 等 |
| 状态 | ICLR 2025（CCF-A **需进一步确认**） |
| 链接 | https://arxiv.org/abs/2407.16741 |
| 解决的问题 | 缺少一个**开放、模块化、完整**的 SE agent 平台 |
| 方法核心 | sandbox + 多 agent + 工具（shell / browser / IPython / edit）+ event stream 架构 |
| 和 Code Agentic RL 关系 | **通用 SE agent 平台**；是后续训练 / 评测的基础设施 |
| 是否用 RL？ | ❌ 否（核心论文），但平台支持 RL 接入 |
| 现在应不应该精读？ | ✅ **建议精读架构部分**，跳过具体 eval |
| 阅读重点 | event stream / 工具系统 / sandbox 设计 |

---

## 🪪 必读卡 10 — LEVER

| 字段 | 内容 |
|---|---|
| 标题 | LEVER: Learning to Verify Language-to-Code Generation with Execution |
| 年份 | 2023 |
| 作者 / 机构 | Ni et al. |
| 状态 | ICML 2023（CCF-A **需进一步确认**） |
| 链接 | https://arxiv.org/abs/2302.08468 |
| 解决的问题 | 测试不全 / 没测试时怎么验证代码？ |
| 方法核心 | 训练 **execution verifier**：用 execution trace + 代码 → 预测正确性 |
| 和 Code Agentic RL 关系 | **verifier 训练**的代表（[能力栈](02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md) 模块 10）；可做 reranker 或 RL reward |
| 是否用 RL？ | execution feedback + verifier（不是策略 RL） |
| 现在应不应该精读？ | ⚠️ **建议精读**。verifier 训练范式重要 |
| 阅读重点 | verifier 输入设计、训练数据来自哪 |

---

## 🧭 4.5.1 论文之间的关系

```
                           SWE-bench (任务定义)
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
       SWE-bench           Agent Scaffold          Pipeline 对照
        Verified           ───────────────         ─────────────
       (reward 可信)       SWE-agent (ACI)         Agentless
                          OpenHands (平台)
                          AutoCodeRover (结构感知)
                                  │
                                  ▼
                            执行反馈 / 验证
                            ─────────────
                            Self-Debugging (rubber duck)
                            CodeT (generated tests)
                            LEVER (learned verifier)
                                  │
                                  ▼
                           训练（RL）
                           ─────────
                           CodeRL (RL+critic)
                           → 后续 SWE-Gym / SWE-RL (见 4.6)
```

---

## 📌 4.5 节要点

| 论文 | 角色 | 一句话价值 |
|---|---|---|
| SWE-bench | benchmark | 定义 reward 来源 |
| SWE-bench Verified | benchmark | 让 reward 可信 |
| SWE-agent | scaffold | ACI 设计典范 |
| Agentless | pipeline | 强 baseline，破除迷信 |
| AutoCodeRover | pipeline | 结构感知 localization |
| Self-Debugging | execution | error recovery 奠基 |
| CodeT | verifier | generated tests + selection |
| CodeRL | RL | code RL 鼻祖，4 类粒度 reward |
| OpenHands | platform | 完整 SE agent 平台 |
| LEVER | verifier | learned execution verifier |

---

## ⚠️ 资料中需要进一步确认的状态

- SWE-bench / SWE-agent / OpenHands / CodeRL / CodeT / LEVER 的 **CCF-A 归属**
- Agentless 的 **会议状态**
- AutoCodeRover 的 **CCF 分类**

→ 完整清单见 [02-资料不一致与待确认清单](../%E9%99%84%E5%BD%95/02-%E8%B5%84%E6%96%99%E4%B8%8D%E4%B8%80%E8%87%B4%E4%B8%8E%E5%BE%85%E7%A1%AE%E8%AE%A4%E6%B8%85%E5%8D%95.md)

---

## 🔗 延伸阅读

- 下一节：[06-强推+可选论文索引（SWE-Gym-SWE-RL等）](06-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95%EF%BC%88SWE-Gym-SWE-RL%E7%AD%89%EF%BC%89.md)
- 能力栈对应：[02-Code-Agent能力栈10模块](02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md)
- 5 种 reward：[04-RL如何用在Code-Agent上](04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md)
- 原始资料：`code-agentic-rl/02-papers.md` §必学表

---

⬅ [04-RL如何用在Code-Agent上](04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md) | ➡ [06-强推+可选论文索引（SWE-Gym-SWE-RL等）](06-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95%EF%BC%88SWE-Gym-SWE-RL%E7%AD%89%EF%BC%89.md)
