---
tags: [agentic-rl, 第4章, code-agent, MDP, observation, action, reward, trajectory]
chapter: 4
section: 4.3
source: "agentic-rl-learning-map/code-agentic-rl/01-boundaries.md, 04-capability-stack.md, 08-toy-bugfix-agent-gym.md"
---

# 4.3 Observation / Action / Reward 形式化定义

⬅ [02-Code-Agent能力栈10模块](02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md) | ➡ [04-RL如何用在Code-Agent上](04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md)

---

## 🎬 故事比喻：把 Code Agent 装进 MDP 的盒子

第 1 章你学了 MDP 五元组 `(S, A, P, R, γ)`。
当时是抽象的。本节把它**具体化到一个 SWE-bench task**：

```
你: agent，请修这个 issue 「normalize_path 在 trailing slash 上挂了」
                                  │
                          【Observation s_0】
                          issue 文本 + repo 树 + 当前 cwd
                                  │
              agent 想: "先 grep 看 normalize_path 在哪"
                                  │
                          【Action a_0】 search("normalize_path")
                                  │
              environment 返回: src/paths.py:42 def normalize_path(p)
                                  │
                          【Observation s_1】
                          上次结果 + repo 状态
                          【Reward r_0】 0.0 (中间步骤)
                                  │
              agent: "读这个文件"
                                  │
                          【Action a_1】 read("src/paths.py")
                                  │
                                 ...
                                  │
              最后:   【Action a_T】 run_tests → 全过
                                  │
                          【Reward r_T】 +1.0 ✅
                                  │
                          完整 trajectory τ:
                          (s_0,a_0,r_0,s_1,a_1,r_1,...,s_T,a_T,r_T)
```

> **本节的全部内容就是把上图的每个字段写清楚**，让你以后看任何 Code Agent 论文都能秒对应。

---

## 📐 4.3.1 MDP 五元组在 Code Agent 中的对应

来自原始资料 §`code-agentic-rl/08-toy-bugfix-agent-gym.md` 中 toy gym 的真实 schema 推广：

| MDP 元素 | 数学符号 | Code Agent 中的对应 |
|---|---|---|
| **State** | $s_t$ | issue 文本 + repo 当前 snapshot + 历史 (observation, action) + 测试结果 |
| **Observation** | $o_t$ | state 的可见子集（窗口截取，因为 state 太大放不进 context） |
| **Action** | $a_t$ | `{search / read / edit / run_tests / final}` 之一 |
| **Transition** | $P(s_{t+1}\|s_t, a_t)$ | 由 shell / pytest / 文件系统**确定性**决定（**几乎无随机性**） |
| **Reward** | $r_t$ | 中间步骤多为 0；最终 reward 来自 pytest 通过率（pass/fail）或 dense shaping |
| **Discount** | $\gamma$ | 常取 0.95-1.0；任务短可取 1.0 |
| **Horizon** | $T$ | 通常 10-50 步，OpenHands 等可到 100+ |

→ 几个关键观察：

1. **Code Agent 的 transition 几乎是确定的**（vs Atari 游戏里 transition 有大量随机性）
2. **observation ≠ state**：state 包含整个 repo + 所有历史，**observation 是你能塞进 context 的那部分**——这是 **POMDP**
3. **action space 是离散的高级动作**，不是 token-level（虽然 token 也可以看成更细粒度 action）

### 形式化定义（POMDP 七元组）

把 Code Agent 写成严格 **POMDP**：

$$\text{Code-Agent POMDP} = \big(\,S,\, A,\, O,\, P,\, R,\, \mathcal{O},\, \gamma\,\big)$$

**逐项定义**：

| 元素 | 数学形式 | Code Agent 中的具体类型 |
|---|---|---|
| 状态空间 $S$ | $S = \mathcal{R} \times \mathcal{H} \times \mathcal{T}$ | repo 文件树 × 历史命令 × 测试历史 |
| 动作空间 $A$ | $A = A_{\text{search}} \sqcup A_{\text{read}} \sqcup A_{\text{edit}} \sqcup A_{\text{run}} \sqcup \{a_{\text{final}}\}$ | 5 个离散动作集合的不交并 |
| 观察空间 $O$ | $O \subset \Sigma^{\le L}$，$L$ = context window 长度 | context 内可见 token 序列 |
| 转移 $P$ | $P(s' \mid s, a) \approx \delta_{s'\, =\, f_{\text{shell}}(s, a)}$ | **退化为确定函数** $f_{\text{shell}}$（pytest/edit 输出确定） |
| 奖励 $R$ | $R(s_t, a_t) = \mathbb{1}[t = T] \cdot V(\text{repo}_T, \text{tests}_T)$ | **仅终局非零**，由 verifier $V$ 决定 |
| 观察函数 $\mathcal{O}$ | $\mathcal{O}(o \mid s) = \text{summarize}_\Phi(s)$ | summarization 策略 $\Phi$（截断/总结） |
| 折扣 $\gamma$ | $\gamma \in [0.95, 1.0]$ | 任务短，常 1.0 |

**核心观察**：

- $P$ 几乎是 **deterministic delta function**：$P(s' \mid s, a) = 1$ 当且仅当 $s' = f_{\text{shell}}(s, a)$
- $R$ 是 **terminal-only**：$\sum_{t=0}^{T-1} r_t = 0$，$r_T = V(\cdot) \in \{0, 1\}$（最稀疏！）
- 真正"难"的不是 $P, R$，而是 **$\mathcal{O}$ 的设计**（怎么把巨大的 $s$ 压成小 $o$）

### 与单轮 RLHF 的形式化对比

| 量 | 单轮 RLHF（PBRFT） | Code Agent POMDP |
|---|---|---|
| $|S|$ | 1（只有 prompt） | 极大（repo 状态空间） |
| $|A|$ | $|V|^L$（一段文本） | 5 类高级动作 |
| Horizon $T$ | 1 | 10~100 |
| Reward 时机 | $t = 1$ | $t = T$（更稀疏） |
| Trajectory | $(s_0, a_0, r_0)$ | $(s_0, a_0, r_0, \dots, s_T, a_T, r_T)$ |
| 信用分配难度 | **无**（只一个 a） | **很难**（T 个 a，只有 1 个 r） |

→ 这就是为什么 [§1.5 sparse reward + credit assignment](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/05-Sparse-Reward%E4%B8%8E%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D.md) 是 Code Agent 的最大坑。

---

## 🎯 4.3.2 Observation 设计（核心）

**Observation 是 Code Agent 最容易崩的地方**。设计原则来自原始资料 §08-toy-bugfix-agent-gym.md 「Observation」段：

每一步 observation 应包含：

```yaml
observation:
  issue: "Function normalize_path fails on trailing slashes."
  current_hypothesis: "可能是 strip('/') 写法太粗暴"
  files_summary:
    - path: src/paths.py
      summary: "lines 1-80, key functions: normalize_path, join_path"
    - path: tests/test_paths.py
      summary: "test_trailing_slash failing"
  last_diff: |
    --- a/src/paths.py
    +++ b/src/paths.py
    @@ -42,3 +42,3 @@
    - return p.rstrip('/')
    + return p.rstrip('/') if p != '/' else p
  last_test_output: |
    tests/test_paths.py::test_trailing_slash PASSED
    tests/test_paths.py::test_root_path FAILED
  failing_tests:
    - tests/test_paths.py::test_root_path
  budget_remaining: 12  # 还能跑多少步
```

> 💡 **关键技巧**：用 `files_summary` 代替原文件，用 `last_diff` 代替全部 history。
> 这就是 OpenHands、SWE-agent 里大量的 "context compression" 工程。

---

## ⚙️ 4.3.3 Action Space 设计

来自原始资料 §08-toy-bugfix-agent-gym.md 「Agent 动作空间」：

| 动作 | 参数 | 返回 |
|---|---|---|
| `search(query)` | 查询字符串 | 匹配的文件/行号列表 |
| `read(path, [start, end])` | 文件路径，可选行号范围 | 文件内容 |
| `edit(path, patch)` | 路径 + unified diff 或 search/replace 块 | "applied" / 错误信息 |
| `run_tests(command)` | pytest 命令字符串 | 测试输出 |
| `final(answer)` | 最终 patch 摘要 | episode 结束 |

> ⚠️ **不要给 agent 完整 bash**。
> SWE-agent 论文最重要的发现之一：**raw bash → 限定工具集**，resolved rate 显著上升。
> 这就是 **Agent-Computer Interface (ACI)** 思想。

### 不同 Agent 的 action space 对比

| 项目 | Action space | 特点 |
|---|---|---|
| **Aider** | edit (search/replace 块) + git + 终端 | 极简，human-in-the-loop |
| **SWE-agent** | 自定义 ACI（17 个命令，包括 `goto`、`scroll`、`edit`） | 为 LLM 优化的命令 |
| **OpenHands** | sandbox shell + browser + IPython + edit | 最完整，最重 |
| **Agentless** | 不是 agent，是 pipeline：localization → repair → rerank | 强 baseline，对照组 |

---

## 🎁 4.3.4 Reward 设计（5 种类型先行预览）

详见 [04-RL如何用在Code-Agent上](04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md)，这里只给定义：

| Reward 类型 | 何时给 | 例子 |
|---|---|---|
| **Pass/fail (outcome)** | episode 结束 | 全部 FAIL_TO_PASS + PASS_TO_PASS 都过 → +1.0，否则 0 |
| **Dense (shaped)** | 每步 | 编译过 +0.2 / 错误数减少 +0.2 / regression -0.2 |
| **Process** | 每步 | 定位对 +0.3 / 命令选对 +0.1 / 错误归因对 +0.2 |
| **Verifier** | episode 结束 | learned verifier 给 patch 打 0-1 分 |
| **Rule** | episode 结束 | patch 与 gold diff 相似度 / diff size 惩罚 |

---

## 📜 4.3.5 Trajectory Schema（实际可用的 JSON）

直接照搬原始资料 §08-toy-bugfix-agent-gym.md 「Trajectory 格式」。**这是你写 toy gym 时可以直接用的 schema**：

```json
{
  "task_id": "task_001",
  "episode_id": "task_001_run_001",
  "steps": [
    {
      "t": 0,
      "observation": "Issue: normalize_path fails on trailing slashes...",
      "action_type": "read",
      "action": {"path": "src/paths.py"},
      "result": "file content summary",
      "reward": 0.0
    },
    {
      "t": 1,
      "observation": "Current hypothesis: strip too aggressive",
      "action_type": "edit",
      "action": {"path": "src/paths.py", "patch": "@@ -42,3 +42,3 @@..."},
      "result": "patch applied",
      "reward": 0.0
    },
    {
      "t": 2,
      "observation": "...",
      "action_type": "run_tests",
      "action": {"command": "pytest -q tests/test_paths.py"},
      "result": "1 passed, 1 failed",
      "reward": 0.2
    }
  ],
  "final_diff": "...",
  "tests_before": "1 passed, 2 failed",
  "tests_after": "3 passed, 0 failed",
  "success": true,
  "total_reward": 1.0,
  "failure_mode": null
}
```

> 📌 **保存 trajectory 是入门的第一步**。
> 没有 trajectory，你就无法做 SFT / DPO / verifier 训练。
> 没有 trajectory，你也无法分析失败模式。

---

## 💻 4.3.6 最小可跑代码片段（概念示意）

> ⚠️ **以下代码为概念示意，来自资料推导，不保证开箱即用**。真实使用请参考 Aider / SWE-agent repo。

```python
# minimal_code_agent_loop.py
# 一个 inference-time loop 的最简框架（无训练）

import json, subprocess
from pathlib import Path

def step_pytest(repo_dir, cmd="pytest -q"):
    r = subprocess.run(cmd, shell=True, cwd=repo_dir,
                       capture_output=True, text=True, timeout=60)
    return r.stdout + r.stderr

def parse_failing(test_output: str) -> list[str]:
    # 极简：找 FAILED 行
    return [l for l in test_output.splitlines() if "FAILED" in l]

def llm_act(observation: dict) -> dict:
    # 这里替换为你的 LLM 调用，返回 {"action_type":..., "action":...}
    raise NotImplementedError

def run_episode(task: dict, max_steps: int = 20) -> dict:
    repo = Path(task["repo_dir"])
    trajectory = {"task_id": task["id"], "steps": []}
    obs = {"issue": task["issue"], "tests": step_pytest(repo)}

    for t in range(max_steps):
        act = llm_act(obs)
        # 派发动作
        if act["action_type"] == "run_tests":
            result = step_pytest(repo, act["action"]["command"])
        elif act["action_type"] == "read":
            result = (repo / act["action"]["path"]).read_text()[:2000]
        elif act["action_type"] == "edit":
            apply_patch(repo, act["action"]["patch"])
            result = "applied"
        elif act["action_type"] == "final":
            break
        else:
            result = f"unknown action {act['action_type']}"

        step = {"t": t, "observation": obs, **act, "result": result, "reward": 0.0}
        trajectory["steps"].append(step)
        obs = {"issue": task["issue"], "tests": step_pytest(repo)}

    # 终局 reward (pass/fail)
    final_tests = step_pytest(repo)
    success = "FAILED" not in final_tests
    trajectory["success"] = success
    trajectory["total_reward"] = 1.0 if success else 0.0
    return trajectory

# def apply_patch(...): 用 unidiff 或 search/replace 实现
```

→ 这个 30 行的骨架就能跑出 [Toy Bug-Fix Agent Gym](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md) 的最小版本。

---

## ⚠️ 4.3.7 常见设计陷阱

1. **observation 里塞整个 repo** → context 立刻爆炸；用 summary
2. **action space 给 raw bash** → agent 会乱跑 `rm -rf`；用 ACI
3. **reward 只给最终 pass/fail** → trajectory 30 步只有 1 个非零 reward，学不动；考虑 dense
4. **trajectory 不保存 observation** → 没法做 SFT；保存全字段
5. **没有 budget_remaining** → agent 无限循环，**必须**给 step 预算
6. **没有 failure_mode 标注** → 失败分析全靠人肉，无法系统化改进

---

## 📌 4.3 节要点

| 字段 | Code Agent 里是什么 | 设计要点 |
|---|---|---|
| **State** | issue + repo snapshot + 历史 | 全部 state 拿不到，是 POMDP |
| **Observation** | summary + last diff + last test output | **必须压缩** |
| **Action** | search/read/edit/run_tests/final | **不要 raw bash**，用 ACI |
| **Reward** | pytest 通过率为主 | 见 [04-RL如何用在Code-Agent上](04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md) |
| **Trajectory** | JSON list of steps | **必须保存全字段** |
| **Horizon** | 10-50 步 | 给 `budget_remaining` |

---

## 🔗 延伸阅读

- 下一节：[04-RL如何用在Code-Agent上](04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md)
- MDP 基础回顾：[01-MDP与轨迹（用Agent语言讲RL）](../01-%E7%AC%AC1%E7%AB%A0-RL%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/01-MDP%E4%B8%8E%E8%BD%A8%E8%BF%B9%EF%BC%88%E7%94%A8Agent%E8%AF%AD%E8%A8%80%E8%AE%B2RL%EF%BC%89.md)
- Toy 实操：[02-Toy-Bug-Fix-Agent-Gym（核心）](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)
- 训练落地：[04-RL任务与验证器数据](../../Code-Agent-Knowledge-Base/02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/04-RL%E4%BB%BB%E5%8A%A1%E4%B8%8E%E9%AA%8C%E8%AF%81%E5%99%A8%E6%95%B0%E6%8D%AE.md)
- 轨迹 Schema：[03-工具调用与轨迹Schema](../../Code-Agent-Knowledge-Base/04-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF/03-%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E4%B8%8E%E8%BD%A8%E8%BF%B9Schema.md)
- 原始资料：`code-agentic-rl/08-toy-bugfix-agent-gym.md` Trajectory 格式段
- 跨链：[Hello-Agents 11.1.5 MDP 对比](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/01-%E4%BB%8ELLM%E8%AE%AD%E7%BB%83%E5%88%B0Agentic-RL.md#-1115-mdp-%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94%E5%BF%85%E8%83%8C)

---

⬅ [02-Code-Agent能力栈10模块](02-Code-Agent%E8%83%BD%E5%8A%9B%E6%A0%8810%E6%A8%A1%E5%9D%97.md) | ➡ [04-RL如何用在Code-Agent上](04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md)
