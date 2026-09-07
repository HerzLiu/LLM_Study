---
tags: [agentic-rl, 第4章, code-agent, capability-stack, repo-navigation, patch, test-feedback]
chapter: 4
section: 4.2
source: "agentic-rl-learning-map/code-agentic-rl/04-capability-stack.md (全文)"
---

# 4.2 Code Agent 能力栈 10 模块

⬅ [01-Code-Agent定义与边界](01-Code-Agent%E5%AE%9A%E4%B9%89%E4%B8%8E%E8%BE%B9%E7%95%8C.md) | ➡ [03-Observation-Action-Reward定义](03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md)

---

## 🎬 故事比喻：你雇了一个初级工程师，他会崩在哪？

想象你雇了个新人开发，给他一个 GitHub issue。他可能在 10 个环节里任何一个崩：

```
1. 拿到 issue → 找不到相关文件 (Repo 理解)         💥
2. 找到了文件 → 但定位错根因 (Localization)        💥
3. 找对了根因 → patch 写得太多/太少 (Patch gen)    💥
4. 写了 patch → 不会跑测试 (Test execution)        💥
5. 跑了测试 → 不会用 git/编辑器 (Tool calling)     💥
6. 工具会用 → 不会拆任务 (Planning)                💥
7. 任务拆了 → 报错看不懂 (Error recovery)          💥
8. 报错能看 → 历史记忆混乱 (Context mgmt)          💥
9. 全做完了 → 你不知道他做得好不好 (Eval)          💥
10. 你想训他变好 → 不知道用什么 reward (Reward)    💥
```

> **每一个 💥 都是一篇论文 / 一个项目存在的理由**。
> 本章把这 10 个模块全列出来，每个给"解决什么 / 代表论文 / 代表 repo / RL 强相关度 / 初学练习"。

---

## 🏗 4.2.0 能力模块总览（核心表）

原始资料 §04-capability-stack.md 总表（**完整保留**）：

| # | 模块 | 解决什么问题 | 代表论文 | 代表 repo | RL 强相关度 | 初学者怎么学 |
|---|---|---|---|---|---|---|
| 1 | Repo 理解与检索 | 找相关文件、符号、调用链、配置和测试 | AutoCodeRover, Agentless, SWE-bench | AutoCodeRover, Agentless, Aider | 中 | 用 `rg`、AST/search 手工定位 3 个 bug |
| 2 | Issue / bug localization | 从 issue 或 failing test 定位根因区域 | Agentless, AutoCodeRover | Agentless, AutoCodeRover, SWE-bench | 中 | 把 SWE-bench task 拆成 issue→files→suspected symbols |
| 3 | Patch generation | 生成最小正确 diff，避免无关重写 | SWE-agent, OpenHands, Agentless | SWE-agent, OpenHands, Aider | 中高 | 固定定位信息，只让模型生成 patch |
| 4 | Test execution & feedback | 运行测试，读失败日志，判断是否修复 | SWE-bench Verified, Self-Debugging, CodeT | SWE-bench, SWE-ReX, Aider | **高** | 构造 failing tests，记录 fail→edit→rerun |
| 5 | Tool calling | 选择 shell、搜索、编辑、测试、git | SWE-agent, OpenHands | SWE-agent, OpenHands, SWE-ReX | **高** | 对比不同 agent-computer interface 的命令空间 |
| 6 | Planning & decomposition | 把复杂 issue 拆成可执行步骤 | OpenHands, SWE-agent | OpenHands, MetaGPT, ChatDev | 中 | 比较 plan-first 和 reactive loop 失败模式 |
| 7 | Error recovery | 从 traceback / lint / test failure 恢复 | Self-Debugging, SWE-agent | Aider, OpenHands, SWE-agent | **高** | 给模型错误日志，要它输出"原因+下一步" |
| 8 | Long-horizon context mgmt | 管理 repo、命令历史、patch、测试日志、假设 | SWE-bench, OpenHands | OpenHands, Aider, SWE-agent | 中高 | 设计 trajectory compression 模板 |
| 9 | Evaluation & benchmark | 稳定评估真实修复能力 | SWE-bench, SWE-Gym, LiveCodeBench | SWE-bench, SWE-Gym, LiveCodeBench | **高** | 学会 FAIL_TO_PASS、PASS_TO_PASS、resolved rate |
| 10 | RL / verifier / reward design | 把测试、执行、静态检查、AI judge 转成训练信号 | CodeRL, LEVER, SWE-Gym, SWE-RL | SWE-Gym, SWE-RL, CodeRL, LEVER | **最高** | 从 pass/fail reward 到 verifier rerank |

→ 每一行后面 §4.2.1 - §4.2.10 都有独立小节展开。

---

## 🧭 4.2.0.1 能力依赖图（必看）

```
                            最终: Issue Resolved
                                    ▲
                                    │
                             【10. RL/Reward】
                                    ▲
                                    │
                             【9. Evaluation】
                                    ▲
        ┌──────────┬────────────────┼──────────────┬─────────────┐
        │          │                │              │             │
   【1. Repo】 【2. Local】    【3. Patch】    【4. Test】    【7. Recovery】
        理解        定位           生成            反馈           错误恢复
                                                                  ▲
                                                                  │
                                                            【5. Tool】
                                                            工具调用
                                                                  ▲
                                                                  │
                                                            【6. Plan】
                                                            规划拆解
                                                                  ▲
                                                                  │
                                                          【8. Context】
                                                          长程上下文
```

→ **底层模块（5/6/7/8）任何一个崩，上层都白搭**。这就是为什么 OpenHands、SWE-agent 花大量篇幅在 agent-computer interface 设计上。

---

## 📚 4.2.1 Repo 理解与检索

**解决的问题**：

- repo 太大，无法全部放进上下文
- issue 描述通常**不会**直接告诉你要改哪个文件
- 正确 patch 往往依赖已有 API、风格和测试

**代表材料**：

- **AutoCodeRover**：结构感知搜索 + program improvement
- **Agentless**：用 localization → repair → rerank 拆解 SWE 任务
- **Aider**：适合体验"选上下文文件"的工程问题

**RL 关系**：中等。搜索动作可成为 policy action，但早期更适合先做启发式或 supervised baseline。

**初学练习**：选 3 个小 repo bug，只允许用 `rg`、`sed`、`pytest`，手工记录你如何定位文件。

---

## 📚 4.2.2 Issue / Bug Localization

**解决的问题**：

- 从自然语言 issue、失败日志或用户描述找根因
- 多文件任务中，错误定位决定后续 patch 是否有希望

**代表材料**：Agentless / AutoCodeRover / SWE-bench

**RL 关系**：中到高。可以用最终测试结果**反推** localization 质量，但 credit assignment 较难。

**初学练习**：给每个 task 输出 `suspected_files.json`：

```json
{
  "task_id": "task_001",
  "suspected_files": [
    {"path": "src/paths.py", "reason": "issue 提到 normalize_path", "confidence": 0.9},
    {"path": "tests/test_paths.py", "reason": "failing test 在这里", "confidence": 1.0}
  ]
}
```

---

## 📚 4.2.3 Patch Generation

**解决的问题**：

- 生成能通过测试的**最小**改动
- 避免全量重写、风格漂移、引入 regression

**代表材料**：SWE-agent / OpenHands / Agentless

**RL 关系**：**高**。patch 可用测试结果直接打分，是 **RLVR 的自然对象**。

**初学练习**：固定相关文件，让模型只做 patch；比较 ①单次生成 ②采样多个 patch ③测试 rerank 的成功率。

---

## 📚 4.2.4 Test Execution and Feedback

**解决的问题**：

- 测试失败日志很长，模型需要提取关键信号
- 一次 patch 失败后，需要决定：重跑？读新文件？回滚？改别处？

**代表材料**：SWE-bench Verified / Teaching LLMs to Self-Debug / CodeT

**RL 关系**：**很高**。测试结果是**最自然的 verifiable reward**。

**初学练习**：构造 20 个 Python failing tests，记录每轮 `patch → pytest → error summary → next action`。

---

## 📚 4.2.5 Tool Calling

**解决的问题**：

- 工具太多，动作空间大
- 错误命令、过度搜索、跑错测试都会浪费预算

**代表材料**：SWE-agent 的 **Agent-Computer Interface** / OpenHands 的 sandbox 和工具系统

**RL 关系**：**很高**。工具选择、参数、时机都可以训练。

**初学练习**：限制工具集合为 `rg / read / edit / pytest`，观察模型是否更稳定。

> 💡 SWE-agent 的核心 contribution 之一就是说明："**给 LLM 一个为 LLM 设计的工具界面**"（不是 raw bash），能显著提升 resolved rate。

---

## 📚 4.2.6 Planning and Task Decomposition

**解决的问题**：

- 长程 issue 不能一次性修完
- 需要维护假设、子目标、验证计划

**代表材料**：OpenHands / SWE-agent / MetaGPT / ChatDev（workflow 对照）

**RL 关系**：中等。计划本身难直接验证，但计划导致的最终成败可用于训练。

**初学练习**：对同一 issue 比较 ①"先写计划再行动" vs ②"直接 reactive loop" 的轨迹长度和成功率。

---

## 📚 4.2.7 Error Recovery

**解决的问题**：

- patch 经常导致**新错误**
- 需要读懂 traceback / assertion / type error / lint error

**代表材料**：Self-Debugging / SWE-agent

**RL 关系**：**高**。恢复动作有明确反馈，适合**过程奖励或局部 verifier**。

**初学练习**：给模型一组失败日志，让它**只输出"错误归因 + 下一步命令"，先不让它改代码**。

---

## 📚 4.2.8 Long-Horizon Context Management

**解决的问题**：

- 多轮行动中日志、文件、diff、假设会迅速膨胀
- 需要保留关键事实，丢弃噪声

**代表材料**：SWE-bench / OpenHands / SWE-Gym

**RL 关系**：中高。上下文选择影响成功率，但 reward 延迟。

**初学练习**：设计 `trajectory_summary.md` 模板：

```markdown
## Current Hypothesis
…
## Files Read
…
## Files Edited
…
## Test Results
…
## Next Step
…
```

---

## 📚 4.2.9 Evaluation and Benchmark

**解决的问题**：

- 必须知道 agent 是否**真的修复问题**，而不是只生成看似合理 patch
- benchmark 可能有噪声、污染和不稳定环境

**代表材料**：SWE-bench / SWE-bench Verified / SWE-Gym / LiveCodeBench / BigCodeBench

**RL 关系**：**高**。没有可靠评估，就没有可靠 reward。

**初学练习**：学会解释三个核心指标——

| 指标 | 含义 |
|---|---|
| `FAIL_TO_PASS` | 修复前 FAIL，修复后 PASS 的测试（你**必须**让它过） |
| `PASS_TO_PASS` | 修复前 PASS，修复后**仍要 PASS**（不能破坏旧功能） |
| `resolved rate` | 任务完全解决的比例 = 所有 FAIL_TO_PASS 过 + 所有 PASS_TO_PASS 仍过 |

---

## 📚 4.2.10 RL / Verifier / Reward Design ⭐ 本章核心中的核心

**解决的问题**：

- 最终测试 pass/fail **太稀疏**
- 需要从执行日志、部分测试、静态检查、patch 质量中提取训练信号

**代表材料**：CodeRL / LEVER / SWE-Gym / SWE-RL

**RL 关系**：**最高**。这是 Code Agentic RL 的核心研究问题。

**初学练习**：为 toy bug-fix 环境设计 **3 版 reward**：

1. **Binary**：全测过给 1，否则 0
2. **Dense**：编译成功 +0.2 / 部分测试通过 +0.2 / regression -0.2 …
3. **Verifier-rerank**：采样多个 patch，用 verifier/LLM judge 选最好

→ **详见 [04-RL如何用在Code-Agent上](04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md)**。

→ **每版可能的 reward hacking 见 [08-Reward-Hacking与Sandbox安全](08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)**。

---

## ⚠️ 学习避坑

1. **不要试图同时优化全部 10 个模块**：研究上一次只选 1-2 个突破。
2. **新手优先做 4.2.4（test feedback）+ 4.2.7（error recovery）**：信号最强、反馈最快。
3. **不要小看 4.2.5（tool calling）的工程量**：SWE-agent 的 ACI 设计就是整篇论文的 contribution。
4. **4.2.8（context mgmt）容易被忽视**：实战中长 trajectory 失败 70% 是上下文丢失，不是模型不行。

---

## 📌 4.2 节要点

| 模块 | 一句话定位 | 必读论文 |
|---|---|---|
| 1 Repo 理解 | "我连文件都找不对" | AutoCodeRover / Agentless |
| 2 Localization | "我找到了文件但根因错" | Agentless |
| 3 Patch | "我改太多 / 太少" | SWE-agent |
| 4 Test feedback | "我看不懂报错" | Self-Debugging |
| 5 Tool calling | "我命令打错" | SWE-agent (ACI) |
| 6 Planning | "我没拆任务就硬干" | OpenHands |
| 7 Error recovery | "我崩了不会重启" | Self-Debugging |
| 8 Context mgmt | "我忘了 5 步之前在做啥" | OpenHands |
| 9 Eval | "我不知道自己有没有真修好" | SWE-bench Verified |
| 10 Reward design | "我不知道用什么训它" | CodeRL / SWE-Gym |

---

## 🔗 延伸阅读

- 下一节：[03-Observation-Action-Reward定义](03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md)（把这 10 个模块写成 MDP）
- 论文卡：[05-必读论文10篇精读卡（SWE-bench线）](05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md)
- Repo 地图：[07-Repo地图（SWE-agent-OpenHands等）](07-Repo%E5%9C%B0%E5%9B%BE%EF%BC%88SWE-agent-OpenHands%E7%AD%89%EF%BC%89.md)
- 训练落地：[02-Code-Agent能力栈与训练目标](../../Code-Agent-Knowledge-Base/01-Code-Agent%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%88%E4%B8%8E%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87.md)
- 工程闭环：[01-Agent工程闭环总览](../../Code-Agent-Knowledge-Base/04-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF/01-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF%E6%80%BB%E8%A7%88.md)
- 原始资料：`code-agentic-rl/04-capability-stack.md`

---

⬅ [01-Code-Agent定义与边界](01-Code-Agent%E5%AE%9A%E4%B9%89%E4%B8%8E%E8%BE%B9%E7%95%8C.md) | ➡ [03-Observation-Action-Reward定义](03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md)
