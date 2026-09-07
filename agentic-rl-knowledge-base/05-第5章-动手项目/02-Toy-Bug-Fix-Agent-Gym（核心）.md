---
tags: [agentic-rl, 第5章, toy-project, bug-fix-gym, code-agentic-rl, 核心, pytest, trajectory]
chapter: 5
section: 5.2
source: "agentic-rl-learning-map/code-agentic-rl/08-toy-bugfix-agent-gym.md (全文)"
---

# 5.2 Toy Bug-Fix Agent Gym（核心） ⭐⭐⭐⭐⭐

⬅ [01-通用Toy（Calculator-Tool-use-RL）](01-%E9%80%9A%E7%94%A8Toy%EF%BC%88Calculator-Tool-use-RL%EF%BC%89.md) | ➡ [00-章节总览](../06-%E7%AC%AC6%E7%AB%A0-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E5%BE%84%E9%80%9F%E6%9F%A5/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

## 🎬 故事比喻：你的 Code Agentic RL "毕业设计"

```
读了第 1-4 章的全部知识？
看过 SWE-bench / SWE-agent 论文？
理解了 GRPO / RLVR？

→ 现在动手做一个：
   30 个小 Python bug + pytest 验证 + 完整 trajectory + 3 版 reward + 失败模式分析
   = 你的 Code Agentic RL 入门毕业项目
```

> **这是本知识库的最重要单节之一**。
> 做完它，你才真正"懂"前面 4 章在说什么。
> **优先级 ⭐⭐⭐⭐⭐**。

---

## 🎯 5.2.1 项目目标

构建一个**本地最小闭环**，让 coding agent 完成：

```
读 bug 描述 → 读 repo → 修改代码 → 跑 pytest → 读取失败 → 再修改 → 最终通过
```

**并完成 4 件事**：

1. 记录完整 trajectory
2. 计算 reward
3. 统计失败模式
4. （进阶）做 verifier / rerank / DPO 数据

> 💡 **设计哲学**：
> - **不依赖 GPU**：能在 MacBook 上跑（baseline 部分）
> - **不依赖真实 SWE-bench Docker**：用本地 Python/pytest
> - **不依赖在线模型**：可换任意 LLM（OpenAI / Claude / Ollama）

---

## 📁 5.2.2 数据集结构

30 个小 Python bug，每个一个独立目录：

```
toy-bugfix-agent-gym/
├── tasks/
│   ├── task_001/
│   │   ├── README.md            ← 任务描述（人类可读）
│   │   ├── src/                 ← 含 bug 的代码
│   │   │   └── paths.py
│   │   ├── tests/               ← failing tests
│   │   │   └── test_paths.py
│   │   └── metadata.json        ← 机器可读元数据
│   ├── task_002/
│   │   └── ...
│   └── ...
├── runner/
│   ├── run_episode.py           ← 你写的 runner
│   ├── reward.py                ← 3 版 reward
│   └── trajectory_schema.py     ← JSON schema
├── baselines/
│   ├── baseline_notool.py
│   ├── baseline_react.py
│   └── baseline_test_feedback.py
├── analysis/
│   ├── failure_modes.py
│   └── compute_metrics.py
└── trajectories/
    └── *.jsonl
```

---

## 📋 5.2.3 metadata.json schema

每个任务的 `metadata.json`：

```json
{
  "id": "task_001",
  "issue": "Function normalize_path fails on trailing slashes.",
  "entrypoint": "pytest -q",
  "target_tests": ["tests/test_paths.py::test_trailing_slash"],
  "allowed_files": ["src/paths.py"],
  "gold_summary": "Strip trailing slashes except root.",
  "tags": ["string", "edge-case"]
}
```

| 字段 | 用途 |
|---|---|
| `issue` | agent 看到的 task 描述 |
| `entrypoint` | 跑测试的命令 |
| `target_tests` | 必须让它们 pass 的测试 |
| `allowed_files` | **agent 允许修改的文件**（防止改测试） |
| `gold_summary` | gold patch 的语义概括（用于分析） |
| `tags` | 任务类型，用于失败模式分类 |

---

## 🏷 5.2.4 推荐任务类型（资料原文）

至少覆盖：

| 类型 | 示例 |
|---|---|
| **off-by-one** | `range(n)` vs `range(n+1)` |
| **missing edge case** | 没处理空列表 / None / 0 |
| **wrong exception type** | raise ValueError 应该是 TypeError |
| **path/string parsing** | trailing slash / encoding |
| **datetime/unit conversion** | 小时→分钟 |
| **list/dict mutation** | 共享引用问题 |
| **simple class state bug** | __init__ 没初始化 |
| **regression after refactor** | 改 A 破坏 B |

---

## ⚙️ 5.2.5 Agent 动作空间（限定）

| 动作 | 说明 |
|---|---|
| `search(query)` | 在 repo 中搜索文本 |
| `read(path)` | 读取文件 |
| `edit(path, patch)` | 修改文件 |
| `run_tests(command)` | 运行 pytest 或指定测试 |
| `final(answer)` | 提交最终 patch summary |

> 💡 实现上可以**先不做真实 tool API**，用 prompt loop + 手工或脚本执行命令。
> 见 [4.3.6 30 行 loop](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md#-436-%E6%9C%80%E5%B0%8F%E5%8F%AF%E8%B7%91%E4%BB%A3%E7%A0%81%E7%89%87%E6%AE%B5%E6%A6%82%E5%BF%B5%E7%A4%BA%E6%84%8F) 骨架。

---

## 👁 5.2.6 Observation 设计

每一步 observation 含：

- issue 描述
- 当前已读文件摘要
- 最近一次 diff
- 最近一次测试输出
- 当前失败测试
- **剩余 step budget**（防死循环）

---

## 🏆 5.2.7 Reward 设计 — 3 版（核心）

### V1：Binary Reward

| 条件 | Reward |
|---|---:|
| 所有 `target_tests` 通过 | +1.0 |
| 否则 | 0.0 |

**优点**：简单、贴近 SWE-bench
**缺点**：稀疏，不利于学习

```python
def reward_v1_binary(test_result, task) -> float:
    return 1.0 if test_result.all_target_pass else 0.0
```

### V2：Dense Reward

| 条件 | Reward |
|---|---:|
| 所有 `target_tests` 通过 | +1.0 |
| 代码能 import/compile | +0.2 |
| 目标失败测试数量减少 | +0.2 |
| 无效命令 | -0.1 |
| 无效 patch | -0.1 |
| **破坏已有 passing tests** | **-0.2** |
| **修改测试文件** | **-0.5** |

```python
def reward_v2_dense(test_result, action_log, task) -> float:
    r = 0.0
    if test_result.all_target_pass:    r += 1.0
    if test_result.imports_ok:         r += 0.2
    if test_result.failures_reduced:   r += 0.2
    if action_log.invalid_commands:    r -= 0.1
    if action_log.invalid_patches:     r -= 0.1
    if test_result.regression_count > 0: r -= 0.2
    if action_log.edited_test_files:   r -= 0.5
    return r
```

**优点**：训练信号更密
**风险**：模型可能追求局部分数，**而不是完整修复** → [reward hacking](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md#a-reward-hacking-%EF%B8%8F-%E6%9C%80%E5%B8%B8%E8%A7%81%E6%9C%80%E5%8D%B1%E9%99%A9)

### V3：Verifier / Rerank Reward

对每个候选 patch 计算：

- 测试结果
- diff 大小
- 是否修改测试
- 是否引入明显 hard-code
- 是否符合已有风格

用规则或 LLM judge 得到 `patch_score`，用于 **rerank** 或构造 **preference pair**。

```python
def reward_v3_verifier(candidates: list[Patch], verifier) -> Patch:
    scored = [(verifier.score(p), p) for p in candidates]
    return max(scored, key=lambda x: x[0])[1]
```

---

## 📜 5.2.8 Trajectory 格式（保存到 .jsonl）

每条 trajectory 一行 JSON：

```json
{
  "task_id": "task_001",
  "episode_id": "task_001_run_001",
  "steps": [
    {
      "t": 0,
      "observation": "Issue: ...",
      "action_type": "read",
      "action": {"path": "src/paths.py"},
      "result": "file content summary",
      "reward": 0.0
    },
    {
      "t": 1,
      "observation": "Current hypothesis: ...",
      "action_type": "edit",
      "action": {"path": "src/paths.py", "patch": "..."},
      "result": "patch applied",
      "reward": 0.0
    }
  ],
  "final_diff": "...",
  "tests_before": "...",
  "tests_after": "...",
  "success": true,
  "total_reward": 1.0,
  "failure_mode": null
}
```

→ 详见 [4.3.5 完整 schema](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md#-435-trajectory-schema%E5%AE%9E%E9%99%85%E5%8F%AF%E7%94%A8%E7%9A%84-json)

---

## 🎯 5.2.9 Baselines（按强度递增）

资料原文 4 个 baseline：

1. **No-tool patch generation**：给 issue + 相关文件，**一次性**生成 patch
2. **ReAct-style tool loop**：允许 search/read/edit/run_tests
3. **Test-feedback loop**：**强制**每次 edit 后 run_tests，再根据日志修改
4. **Rerank baseline**：采样多个 patch，用 pytest/verifier 选

| Baseline | 优势 | 劣势 |
|---|---|---|
| No-tool | 最快、最便宜 | 多文件 bug 没用 |
| ReAct | 灵活 | trajectory 易混乱 |
| Test-feedback | 信号最强 | trajectory 最长 |
| Rerank | 用算力换正确率 | 推理成本 N 倍 |

---

## 📊 5.2.10 评估指标（必算）

| 指标 | 含义 |
|---|---|
| **success rate** | 最终全测通过比例 |
| **average steps** | 平均动作数 |
| **test runs per task** | 平均测试运行次数 |
| **invalid action rate** | 无效命令 / patch 比例 |
| **regression rate** | 破坏原通过测试比例 |
| **cost per success** | API token 或时间成本 |
| **failure modes** | 失败类型分布 |

---

## 🐛 5.2.11 失败模式 taxonomy（必标注）

至少标注 7 类（资料原文）：

| 失败类型 | 说明 |
|---|---|
| **localization failure** | 找错文件或函数 |
| **patch incorrect** | 定位对但修错 |
| **overfitting tests** | 只针对测试投机（reward hacking） |
| **regression** | 修了目标但破坏旧功能 |
| **tool misuse** | 命令、路径、patch 格式错误 |
| **context loss** | 忘记前面测试结果或假设 |
| **timeout / environment failure** | 环境问题（非模型问题） |

> 💡 标注失败模式是项目的**核心交付物**——证明你不是只跑了 baseline，而是**真正分析了 agent 行为**。

---

## 🚦 5.2.12 最小实施步骤（资料原文）

```
1. 创建 5 个任务先做 pilot，不一开始写满 30 个
2. 为每个任务写 failing test 和 gold fix
3. 写一个 runner：复制任务到临时目录，运行 pytest，收集结果
4. 用 Aider 或手写 ReAct prompt 跑 baseline
5. 保存 trajectory 和 final diff
6. 计算 reward 和指标
7. 扩展到 30 个任务
8. 构造 preference pairs：通过 patch > 未通过 patch
9. 做 verifier/rerank
10. 条件允许再接 DPO/GRPO/RLVR
```

> **遵循"5→30 渐进"**，不要一开始写满 30 个。

---

## 💻 5.2.13 Runner 代码骨架（来自 4.3.6，扩展版）

```python
# runner/run_episode.py
# 概念示意 - 不保证开箱即用

import json, subprocess, shutil, tempfile
from pathlib import Path

def setup_workdir(task_dir: Path) -> Path:
    """复制 task 到临时目录，避免污染原 repo"""
    tmp = Path(tempfile.mkdtemp(prefix=f"bugfix_{task_dir.name}_"))
    shutil.copytree(task_dir, tmp / "repo", dirs_exist_ok=True)
    return tmp / "repo"

def run_pytest(repo: Path, cmd="pytest -q --tb=short", timeout=60) -> dict:
    try:
        r = subprocess.run(cmd, shell=True, cwd=repo,
                           capture_output=True, text=True, timeout=timeout)
        passed = r.stdout.count(" passed")
        failed = r.stdout.count(" failed")
        return {"stdout": r.stdout, "stderr": r.stderr,
                "passed": passed, "failed": failed,
                "all_pass": (failed == 0 and passed > 0)}
    except subprocess.TimeoutExpired:
        return {"timeout": True, "all_pass": False}

def apply_patch(repo: Path, path: str, patch: str):
    """简化版：直接覆写或用 search/replace；生产用 unidiff"""
    target = repo / path
    target.write_text(patch)  # 示意

def reject_test_edit(allowed_files, path):
    """V2 reward 的核心防 hack 项：不允许改测试文件"""
    return ("test" in path.lower()) or (path not in allowed_files)

def run_episode(task_dir: Path, policy, max_steps=20):
    repo = setup_workdir(task_dir)
    meta = json.loads((task_dir / "metadata.json").read_text())
    trajectory = {"task_id": meta["id"], "steps": []}

    tests_before = run_pytest(repo, meta["entrypoint"])
    obs = {"issue": meta["issue"],
           "tests": tests_before["stdout"][:1500],
           "allowed_files": meta["allowed_files"]}

    for t in range(max_steps):
        act = policy.act(obs, trajectory["steps"])

        if act["action_type"] == "edit":
            if reject_test_edit(meta["allowed_files"], act["action"]["path"]):
                result = "REJECTED: cannot edit this file"
            else:
                apply_patch(repo, act["action"]["path"], act["action"]["patch"])
                result = "applied"
        elif act["action_type"] == "run_tests":
            result = run_pytest(repo, act["action"].get("command", meta["entrypoint"]))
        elif act["action_type"] == "read":
            result = (repo / act["action"]["path"]).read_text()[:2000]
        elif act["action_type"] == "final":
            break
        else:
            result = "unknown action"

        trajectory["steps"].append({"t": t, "obs": obs, **act,
                                    "result": str(result)[:1500],
                                    "reward": 0.0})
        # 更新 observation
        obs = {"issue": meta["issue"],
               "last_action": act["action_type"],
               "last_result": str(result)[:500],
               "budget_remaining": max_steps - t - 1}

    tests_after = run_pytest(repo, meta["entrypoint"])
    trajectory["tests_before"] = tests_before
    trajectory["tests_after"] = tests_after
    trajectory["success"] = tests_after.get("all_pass", False)
    trajectory["total_reward"] = 1.0 if trajectory["success"] else 0.0

    shutil.rmtree(repo.parent)  # 清理
    return trajectory

# === 主循环 ===
if __name__ == "__main__":
    from your_policy import make_react_policy  # 自己实现
    policy = make_react_policy(model="claude-3.5-sonnet")

    tasks = sorted(Path("tasks").iterdir())
    with open("trajectories/baseline_react.jsonl", "w") as f:
        for task_dir in tasks:
            traj = run_episode(task_dir, policy)
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
```

> ⚠️ 代码为概念示意，**未跑过**。真实使用请补全 `apply_patch`（用 `unidiff` 库）、policy 实现等。

---

## ✅ 5.2.14 验收标准

### 最低验收

- [ ] 至少 5 个可运行任务
- [ ] 每个任务有 failing test、gold summary、metadata
- [ ] 能运行 baseline 并保存 trajectory
- [ ] 能计算 success rate、invalid action rate、regression rate
- [ ] 能输出失败模式表

### 理想验收

- [ ] 30 个任务
- [ ] 至少 2 个 baseline 对比
- [ ] 有 verifier/rerank 结果
- [ ] 3 条「失败 → 成功」trajectory case study
- [ ] **reward hacking 风险分析**

---

## ⚠️ 5.2.15 Reward Hacking 防护（必做）

按 [4.8](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) 的 12 项清单挨条 check：

```
[ ] 测试文件 read-only（chmod 444）
[ ] reward 函数检测 pytest.skip / @skip
[ ] reward 函数检测大量 if/elif 分支
[ ] PASS_TO_PASS regression 已计算并惩罚
[ ] sandbox: Docker 或至少 tempdir + timeout
[ ] sandbox: --network=none（不需要联网）
[ ] trajectory 保存所有命令
[ ] 抽查 10% "成功" episode 看 patch 是否合理
[ ] 区分"环境失败"和"模型失败"
[ ] 用 hold-out 任务做 OOD 评测
```

---

## 🚀 5.2.16 进阶：接训练

完成 baseline + trajectory + 失败模式后，可继续：

| 进阶项 | 方法 | 跨链 |
|---|---|---|
| SFT 微调 | 用成功轨迹做 behavior cloning | [03-SFT训练实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/03-SFT%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md) |
| DPO 训练 | 构造 (success patch, fail patch) 偏好对 | [02-DPO（偏好直接优化）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/02-DPO%EF%BC%88%E5%81%8F%E5%A5%BD%E7%9B%B4%E6%8E%A5%E4%BC%98%E5%8C%96%EF%BC%89.md) |
| GRPO 训练 | 同 task sample K 个 trajectory + 组内相对 | [04-GRPO训练实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md) |
| Verifier 训练 | 训练 patch 评分模型，做 rerank | [LEVER (论文卡 10)](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/05-%E5%BF%85%E8%AF%BB%E8%AE%BA%E6%96%8710%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1%EF%BC%88SWE-bench%E7%BA%BF%EF%BC%89.md) |

---

## 📌 5.2 节要点

| 项 | 内容 |
|---|---|
| 任务 | 30 个 Python pytest bug |
| Reward | V1 binary / V2 dense / V3 verifier |
| Baseline | No-tool / ReAct / Test-feedback / Rerank |
| 必交付 | trajectory + metrics + 失败模式表 + reward hacking 分析 |
| 渐进 | 5 → 30 任务，先 baseline 再训练 |
| Reward hacking | **必须设防**（锁测试 / sandbox / 抽查） |

---

## 🎓 5.2.17 完成后你拥有什么

- ✅ 一个**可复现**的本地 Code Agentic RL 实验环境
- ✅ 一份**可用**的 trajectory schema 和 reward 函数
- ✅ 一组**真实**的失败模式 taxonomy（你以后看 SWE-Gym/SWE-RL 论文会更深）
- ✅ **入门 Code Agentic RL 的第一个可挂在简历/汇报上的项目**

> 💡 **这是面试讲 Code Agentic RL 时最有说服力的素材**。
> 比读 30 篇论文都管用。

---

## 🔗 延伸阅读

- 上一节：[01-通用Toy（Calculator-Tool-use-RL）](01-%E9%80%9A%E7%94%A8Toy%EF%BC%88Calculator-Tool-use-RL%EF%BC%89.md)
- 章末：[00-章节总览](../06-%E7%AC%AC6%E7%AB%A0-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E5%BE%84%E9%80%9F%E6%9F%A5/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- OAR 定义：[03-Observation-Action-Reward定义](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md)
- 5 种 reward：[04-RL如何用在Code-Agent上](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md)
- Reward hacking 防护：[08-Reward-Hacking与Sandbox安全](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)
- Code Agent 落地路线：[Code-Agent落地路线](../../Code-Agent-Knowledge-Base/00-MOC/Code-Agent%E8%90%BD%E5%9C%B0%E8%B7%AF%E7%BA%BF.md)
- Scaffold / Sandbox / Proxy：[02-Scaffold-Sandbox-Proxy](../../Code-Agent-Knowledge-Base/04-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF/02-Scaffold-Sandbox-Proxy.md)
- 真训练参考：[04-GRPO训练实战](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)
- 上一级参考：SWE-Gym（[06-强推+可选论文索引（SWE-Gym-SWE-RL等）](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/06-%E5%BC%BA%E6%8E%A8%2B%E5%8F%AF%E9%80%89%E8%AE%BA%E6%96%87%E7%B4%A2%E5%BC%95%EF%BC%88SWE-Gym-SWE-RL%E7%AD%89%EF%BC%89.md)）
- 原始资料：`code-agentic-rl/08-toy-bugfix-agent-gym.md`

---

⬅ [01-通用Toy（Calculator-Tool-use-RL）](01-%E9%80%9A%E7%94%A8Toy%EF%BC%88Calculator-Tool-use-RL%EF%BC%89.md) | ➡ [00-章节总览](../06-%E7%AC%AC6%E7%AB%A0-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E5%BE%84%E9%80%9F%E6%9F%A5/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
