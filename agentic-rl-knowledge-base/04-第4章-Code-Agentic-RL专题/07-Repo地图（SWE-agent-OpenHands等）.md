---
tags: [agentic-rl, 第4章, repo, SWE-agent, OpenHands, Aider, SWE-bench, SWE-Gym, swe-rex, repo-map]
chapter: 4
section: 4.7
source: "agentic-rl-learning-map/code-agentic-rl/03-repos.md (全文)"
---

# 4.7 Repo 地图（SWE-agent / OpenHands / Aider...）

⬅ [06-强推+可选论文索引（SWE-Gym-SWE-RL等）](06-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95%EF%BC%88SWE-Gym-SWE-RL%E7%AD%89%EF%BC%89.md) | ➡ [08-Reward-Hacking与Sandbox安全](08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)

---

## 🎬 故事比喻：从最便宜的 Aider 开始爬阶梯

```
                成本 / 复杂度
                     ▲
                     │
   SWE-Gym  ━━━━━━━━━┫  需要训练，GPU/集群
                     │
   OpenHands ━━━━━━━━┫  完整平台，Docker+UI+多工具，重
                     │
   SWE-agent ━━━━━━━━┫  Docker + API key + 单题十几分钟
                     │
   SWE-bench ━━━━━━━━┫  benchmark harness，Docker 重，单题运行 OK
                     │
   Aider ━━━━━━━━━━━━┫  ⭐ 最低成本入门：API key + 本地 pytest 即可
                     │
                     └─────────────────────────────────────► 时间
                     入门
```

> **不要跳级**。先跑 Aider 修一个 pytest bug，再爬阶梯。
> 这是原始资料 §推荐运行顺序明确建议的。

---

## 🏆 4.7.0 必学 Repo 5 个（核心表）

来自原始资料 §"必学" 表（**Stars / 更新时间为 2026-06-05 检索值**）：

| # | Repo | Stars | 项目定位 | RL 关系 | 适合初学 | 运行成本 | 优先级 | 建议 demo |
|---|---|---:|---|---|---|---|---|---|
| 1 | [SWE-bench](https://github.com/princeton-nlp/SWE-bench) | 5,089 | SWE-bench benchmark + evaluation harness | 定义 reward / eval 环境 | 中 | Docker 重，API/GPU 看模型 | 必学 | 跑一个已知 patch 的 evaluation |
| 2 | [SWE-agent](https://github.com/SWE-agent/SWE-agent) | 19,425 | GitHub issue → 自动修复 coding agent | 最经典 scaffold | 适合 | Docker + API key | 必学 | SWE-bench Lite / Verified 单题 |
| 3 | [OpenHands](https://github.com/All-Hands-AI/OpenHands) | 75,860 | AI software developer 平台 | 通用 SE agent 环境 | 中等 | Docker + API key + UI | 必学 | 在本地小 repo 修一个 bug |
| 4 | [Aider](https://github.com/Aider-AI/aider) | 45,775 | 终端 AI pair-programming | 最小 human-in-the-loop loop | **非常适合** | API key + 本地，无需 GPU | 必学 | 用 pytest 失败用例驱动修复 |
| 5 | [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | 684 | Training SE Agents + Verifiers | 从 eval 走向 training 的关键 | 研究向 | Docker + API + GPU 视实验 | 必学 | 先读 trajectory/verifier 数据 |

> ⚠️ **资料里两处 URL 不一致**：
> - 通用 `03-repos.md` 写 `https://github.com/SWE-bench/SWE-bench`
> - Code 专题 `03-repos.md` 写 `https://github.com/princeton-nlp/SWE-bench`
>
> **这两个 URL 当前**很可能指向同一个项目（迁移或镜像），但**资料未明确**。
> 我在本节采用 Code 专题写的 `princeton-nlp` 版本（更接近原作者来源）。
> 完整说明见 [02-资料不一致与待确认清单](../%E9%99%84%E5%BD%95/02-%E8%B5%84%E6%96%99%E4%B8%8D%E4%B8%80%E8%87%B4%E4%B8%8E%E5%BE%85%E7%A1%AE%E8%AE%A4%E6%B8%85%E5%8D%95.md)

---

## 🌟 4.7.1 强烈推荐 Repo 5 个

| Repo | Stars | 项目定位 | RL 关系 | 运行成本 | 建议 demo |
|---|---:|---|---|---|---|
| [Agentless](https://github.com/OpenAutoCoder/Agentless) | 2,062 | Agentless SWE-bench pipeline | 强 baseline：localization + repair + rerank | 中 | 跑 localization + repair pipeline |
| [AutoCodeRover](https://github.com/AutoCodeRoverSG/auto-code-rover) | 3,080 | 结构感知 program improvement | repo 结构理解 + fault localization | 中 | 单个 SWE-bench Lite issue |
| [SWE-RL](https://github.com/facebookresearch/swe-rl) | 696 | SWE-RL 官方代码 | **直接的 Code Agent RL** | 高 | **先读 reward/data pipeline 即可** |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | 668 | Scaling Data for SWE-agents | 训练数据合成 | 中 | 生成一小批 synthetic tasks |
| [swe-rex](https://github.com/SWE-agent/swe-rex) | 518 | **Sandboxed code execution** | code agent 安全执行 + 并行 rollout 基础设施 | 低-中 | 本地 sandbox execution demo |

> 💡 **swe-rex 被低估**：你做 toy gym 时如果要并行执行不可信代码，它的 sandbox 抽象很值得参考。

---

## 📦 4.7.2 可选拓展 Repo 7 个（索引）

| Repo | Stars | 用法 |
|---|---:|---|
| [CodeRL](https://github.com/salesforce/CodeRL) | 571 | 读 actor/critic 训练（思想重要，工程偏旧） |
| [LEVER](https://github.com/niansong1996/lever) | 90 | execution verifier / rerank |
| [CodeT](https://github.com/microsoft/CodeT) | 675 | generated tests / selection |
| [MetaGPT](https://github.com/geekan/MetaGPT) | 68,546 | 多 agent 软件公司（workflow，非 RL 主线） |
| [ChatDev](https://github.com/OpenBMB/ChatDev) | 33,313 | 多 agent 软件开发模拟 |
| [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | 880 | 抗污染代码评测 |
| [BigCodeBench](https://github.com/bigcode-project/bigcodebench) | 503 | function-level 代码生成 benchmark |

---

## 🚦 4.7.3 推荐运行顺序（原始资料原文）

```
1. Aider          ← ⭐ 最低成本体验"测试失败 → 修改 → 复测"loop
2. SWE-bench      ← 理解 benchmark harness 和 reward
3. SWE-agent      ← 跑一个真实 issue resolution agent
4. OpenHands      ← 体验更完整的软件开发 agent 平台
5. SWE-Gym/SWE-RL ← 进入训练、verifier 和 RL 研究
```

---

## 🛠 4.7.4 实操命令片段（来自资料 / 公开 README，未验证）

> ⚠️ 以下命令为概念示意，**不保证当前最新版本可直接跑**。运行前请回 repo 看最新 README。

### Aider（最低门槛）

```bash
# 安装
pip install aider-chat

# 进入一个 git repo
cd your-python-repo

# 写一条 failing test 后启动 aider
export OPENAI_API_KEY=sk-...
aider --model gpt-4o tests/test_foo.py src/foo.py

# 在 aider 里说："让 tests/test_foo.py 全过"
```

### SWE-bench（跑单题 evaluation）

```bash
git clone https://github.com/princeton-nlp/SWE-bench
cd SWE-bench
pip install -e .

# 用一个 dummy patch 跑 evaluation（仅 demo harness）
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path your_predictions.jsonl \
  --max_workers 1 \
  --run_id demo_run
```

### SWE-agent（修一题）

```bash
git clone https://github.com/SWE-agent/SWE-agent
cd SWE-agent
# 见官方 README 配置 Docker + LLM_KEY
sweagent run \
  --agent.model.name gpt-4o \
  --problem_statement.path issue.md \
  --env.repo.path . \
  --env.deployment.type docker
```

### OpenHands（最小启动）

```bash
docker pull ghcr.io/all-hands-ai/openhands:latest
docker run -it --rm \
  -e SANDBOX_RUNTIME_CONTAINER_IMAGE=ghcr.io/all-hands-ai/runtime:latest \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -p 3000:3000 \
  ghcr.io/all-hands-ai/openhands:latest
# 浏览器打开 http://localhost:3000
```

---

## 🏷 4.7.5 按角色分类（便于检索）

| 角色 | Repo |
|---|---|
| **Benchmark harness** | SWE-bench, LiveCodeBench, BigCodeBench |
| **Coding agent scaffold** | SWE-agent, OpenHands, Aider, Agentless, AutoCodeRover |
| **Training framework** | SWE-Gym, SWE-RL, CodeRL |
| **Verifier / rerank** | LEVER, CodeT |
| **Sandbox** | swe-rex（也可用 Docker、firecracker） |
| **Dataset generation** | SWE-smith |
| **Multi-agent workflow** | MetaGPT, ChatDev（**非 RL 主线**） |

---

## ⚠️ 注意事项

1. **Star 数和更新时间是 2026-06-05 检索值**，会变。
2. **不要一上来跑 OpenHands**：Docker 镜像几 GB，启动慢，先用 Aider 暖手。
3. **SWE-bench 单题至少几分钟**：依赖 Docker build，第一次跑要预留 30 分钟。
4. **SWE-agent 默认用 GPT-4o**：API 成本一题 \$0.5-\$5，先用 SWE-bench Lite 小批试水。
5. **SWE-Gym / SWE-RL 完整训练成本高**：先读 reward / data pipeline，不要急着跑训练。
6. **跑 agent rollout 前必看** [08-Reward-Hacking与Sandbox安全](08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)：sandbox 不到位会真的删你文件。

---

## 📌 4.7 节要点

| 入门优先级 | Repo |
|---|---|
| 1. 暖手 | **Aider**（成本最低） |
| 2. 理解评测 | **SWE-bench** 跑一题 evaluation |
| 3. 经典 scaffold | **SWE-agent** |
| 4. 完整平台 | **OpenHands** |
| 5. 进入训练 | **SWE-Gym**（先读，不一定跑） |

| 强 baseline 提醒 | Repo |
|---|---|
| 不一定要 agent loop | **Agentless** |
| 结构检索 | **AutoCodeRover** |
| 安全沙箱 | **swe-rex** |

---

## 🔗 延伸阅读

- 下一节：[08-Reward-Hacking与Sandbox安全](08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)
- 论文卡：[05-必读论文10篇精读卡（SWE-bench线）](05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md)
- 通用 Agentic RL Repo 地图：[04-Repo地图与运行成本](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/04-Repo%E5%9C%B0%E5%9B%BE%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%88%90%E6%9C%AC.md)
- 原始资料：`code-agentic-rl/03-repos.md`

---

⬅ [06-强推+可选论文索引（SWE-Gym-SWE-RL等）](06-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95%EF%BC%88SWE-Gym-SWE-RL%E7%AD%89%EF%BC%89.md) | ➡ [08-Reward-Hacking与Sandbox安全](08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)
