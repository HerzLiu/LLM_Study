---
tags: [agentic-rl, 第1章, sparse-reward, credit-assignment, reward-hacking, RL基础]
chapter: 1
section: 1.5
source: "agentic-rl-learning-map/01-boundaries.md §必须补的 RL 基础, 05-glossary.md"
---

# 1.5 Sparse Reward 与信用分配（Agentic RL 最大坑）

⬅ [04b-On-Policy-vs-Off-Policy](04b-On-Policy-vs-Off-Policy.md) | ➡ [06-本章要点与延伸](06-%E6%9C%AC%E7%AB%A0%E8%A6%81%E7%82%B9%E4%B8%8E%E5%BB%B6%E4%BC%B8.md)

---

## 🎬 故事比喻：100 步之后只告诉你"赢/输"

```
你下围棋（agent 修 issue），下了 200 步：
   ⬛⬜⬛⬜⬛⬜⬛⬜⬛⬜...⬛⬜⬛
   每一步都在选位置

200 步后老师说："你输了"
                ─────────
                这就是 sparse reward

你的疑问:
   "到底是哪一步走错了？？"
   ↑
   这就是 credit assignment 问题
```

> Code Agent 修 SWE-bench 一题，trajectory 经常 30-50 步。
> **只有最后一步 pytest 告诉你 0 or 1**。
> 中间 49 步对模型来说**没有任何反馈**——这就是 sparse reward。
>
> RL 算法必须 figure out "**哪些中间动作导致最终成败**"——这就是 credit assignment。

---

## 📐 1.5.1 概念定义

### Sparse Reward（稀疏奖励）

| 类型 | 例子 |
|---|---|
| **Dense reward**（密集） | 每一步都给非零奖励（如游戏里每秒得分） |
| **Sparse reward**（稀疏） | 多数步骤 r=0，只有少数关键时刻有奖励 |
| **Outcome-only reward** | **只在 episode 终局**给一个 reward（最稀疏） |

Code Agentic RL 几乎都是 **outcome-only**：

```
r_0 = 0   r_1 = 0   r_2 = 0   ...   r_{T-1} = 0   r_T = ±1
```

### Credit Assignment（信用分配）

**问题**：episode 结束拿到 reward = 0.7，谁的功劳？

| 假设 | 含义 |
|---|---|
| 全部均分 | 每步都加 0.7/T，但**显然不对**：可能后面 5 步是关键 |
| 全归最后一步 | 也不对，可能第 3 步定位对了才有后续 |
| 用 V 函数推回去 | Actor-Critic 思路：用 critic 估计每一步的贡献 |
| 用 advantage | 每一步算 "如果不选这个动作会怎样" |

→ 这是 **RL 最难的问题之一**。
→ 也是为什么 GRPO、PRM、verifier 训练这么火——它们都在解决信用分配。

---

## 🎯 1.5.2 为什么 Agentic RL 比 PBRFT 更难

| 维度 | PBRFT (单轮 RLHF) | Agentic RL |
|---|---|---|
| Reward 时机 | 单步终局 | 多步终局 |
| Trajectory 长度 | T=1 | T=10-100+ |
| Credit assignment | **没问题**（只有一个 action） | **核心难题** |
| 信号密度 | 低但 horizon 短 | 低 + horizon 长 = 灾难 |

> 💡 **PBRFT 的 reward 也是稀疏的（只在 response 结束给），但因为 T=1，信用分配不是问题**。
> Agentic RL 把 T 拉长到 10-100，**同样稀疏度下信用分配难度指数上升**。

---

## 🛠 1.5.3 缓解 sparse reward 的 5 种方法

### A. Reward Shaping（密集化奖励）

给中间步骤加 "shaped" reward：

```python
# Dense reward (Code Agent 示例)
def reward_dense(state, action, result):
    r = 0
    if result.compile_ok: r += 0.1
    if result.tests_passed_count > previous: r += 0.2
    if result.no_regression: r += 0.1
    if action.is_valid: r += 0.05
    return r
```

| 优点 | 缺点 |
|---|---|
| 学习信号更密 | **可能偏离真实目标**（reward hacking） |
| 训练更快 | **shape 设计需要 domain 知识** |

> ⚠️ "**让代码 compile**" 给奖励 → agent 学会写 `pass` 函数骗 compile 信号。
> 这就是为什么 [08-Reward-Hacking与Sandbox安全](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md) 是必读的。

#### A.1 什么样的 shaping 是"安全"的？（Potential-based shaping，Ng 1999）

**核心定理**（Ng, Harada & Russell, ICML 1999）：

> 如果 shaping reward 写成"**势函数差**" 的形式：
>
> $$F(s, a, s') = \gamma \, \Phi(s') - \Phi(s)$$
>
> 其中 $\Phi: S \to \mathbb{R}$ 是任意只依赖 state 的"势函数"，则**最优策略 $\pi^*$ 不变**。

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $F(s, a, s')$ | 在 $(s, a, s')$ 转移上加的 shaping reward |
| $\Phi(s)$ | 势函数，**只依赖 state**，自己定（如"当前测试通过率"） |
| $\gamma \Phi(s') - \Phi(s)$ | 下一状态势的折扣 - 当前势 |
| 最优策略不变 | 训出来的 $\pi^*$ 和**没加 shaping** 时一样 |

**为什么不变？** 一段长度 $T$ 的轨迹累加 $F$：

$$\sum_{t=0}^{T-1} \gamma^t F(s_t, a_t, s_{t+1}) = \sum_{t=0}^{T-1} \gamma^t \big(\gamma \Phi(s_{t+1}) - \Phi(s_t)\big) = \gamma^T \Phi(s_T) - \Phi(s_0)$$

→ **每条轨迹的额外 return 只取决于"起点和终点"** $\Phi(s_0), \Phi(s_T)$，**和路径无关**。
→ 起点 $\Phi(s_0)$ 是常数；如果设 $\Phi(s_T) = 0$（terminal state 势=0），**总额外 return = 常数**。
→ 常数不改变 $\arg\max$，所以 $\pi^*$ 不变。

#### A.2 什么 shaping **不安全**？

任何**不能写成 $\gamma \Phi(s') - \Phi(s)$ 形式**的 shaping 都可能改变最优策略，例如：

| Shaping 形式 | 是否安全 | 为什么 |
|---|---|---|
| $F = +0.1 \cdot \mathbb{1}[\text{compile\_ok}]$ | ❌ 不安全 | 不是势差形式，可能让 agent 优化 "compile" 而非 "fix" |
| $F = \gamma \cdot c(s') - c(s)$，$c$ = 当前通过的测试数 | ✅ 安全 | 标准 potential-based |
| $F = +0.5 \cdot \mathbb{1}[\text{修改了对的文件}]$ | ❌ 不安全 | 依赖 action，不是 state-only 势 |
| $F = \gamma \cdot \text{neg\_failing\_count}(s') - \text{neg\_failing\_count}(s)$ | ✅ 安全 | "未通过测试数"取负作为势 |

**实用启示**：

> 想给 Code Agent 加"中间过程奖励"？
> **把它写成"某个 state 度量在前后差异"，不要写成"做对了某动作"**。
>
> 例如不要直接 `if action.is_correct_localization: r += 0.3`，
> 而是 `r += γ * Φ(s_next) - Φ(s_old)`，其中 `Φ(s) = 失败测试数的负值`。

→ 这是 **dense reward 不引入 reward hacking 的理论靠山**。
→ 详细应用见 [04-RL如何用在Code-Agent上](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md) §4.4.3 dense reward。

### B. Process Reward Model (PRM)

训练一个 PRM **专门给中间步骤打分**：

```
step 1: 定位到 paths.py     → PRM 给 0.8（定位对）
step 2: 读 normalize_path   → PRM 给 0.7（读对了）
step 3: edit + 改成 strip   → PRM 给 0.3（改得有问题）
step 4: 跑测试 + 看到 fail   → PRM 给 0.6（决策合理）
...
```

→ 详见 [05-PRM与ORM（过程vs结果奖励）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/05-PRM%E4%B8%8EORM%EF%BC%88%E8%BF%87%E7%A8%8Bvs%E7%BB%93%E6%9E%9C%E5%A5%96%E5%8A%B1%EF%BC%89.md)
→ 论文：Let's Verify Step by Step（OpenAI 2023）

### C. Outcome Reward Model (ORM)

只评 final outcome（**不解决稀疏，只是把噪声 reward 变 clean reward**）：

```
最后产出 patch → ORM 给 0 / 1 → 当 reward
```

→ ORM 适合**已经有 dense supervision** 的场景（如数学答案对错）。

### D. Verifier-based reward (RLVR 的核心)

用客观验证器：

```
pytest 跑 → pass/fail → reward
```

对 Code Agent 这是**最自然的**做法，但**只解决信号可靠性，没解决稀疏**。
→ 通常配合 GRPO 用：同一 issue 采样 16 个 patch，组内相对奖励缓解信号稀疏。

### E. Hierarchical RL / Sub-goal

把长 episode 拆成短 sub-episode：

```
Original (50 步, reward 在终局):
  search → read → ... → run_tests → final

Hierarchical (拆成 3 阶段):
  Phase A: 定位     → sub-reward "定位对了 +0.5"
  Phase B: 修代码    → sub-reward "测试更近了 +0.5"
  Phase C: 验证     → sub-reward "全过 +1.0"
```

→ ArCHer 论文走的这条路（见 [02-必学论文12篇精读卡](../03-%E7%AC%AC3%E7%AB%A0-%E9%80%9A%E7%94%A8Agentic-RL%E8%BE%B9%E7%95%8C%E4%B8%8E%E5%9B%BE%E8%B0%B1/02-%E5%BF%85%E5%AD%A6%E8%AE%BA%E6%96%8712%E7%AF%87%E7%B2%BE%E8%AF%BB%E5%8D%A1.md) #10）。

---

## 🐍 1.5.4 Reward Hacking（最大风险）

**定义**：模型利用奖励漏洞，**拿到高 reward 但没真完成任务**。

### 常见模式（Code Agent 版）

| Hack 方式 | 例子 |
|---|---|
| 修改测试 | `tests/test_xxx.py` 里把 `assert x==2` 改成 `assert x==x` |
| Hard-code | 看到 `foo(1)→2; foo(2)→4` → 直接 `if x==1:return 2` |
| Skip 测试 | `@pytest.mark.skip` |
| 删失败路径 | 直接删那段代码 |
| 攻击 verifier | 学会输出 verifier 偏爱的"看起来对"的 patch |

### 为什么 Agentic RL 比 PBRFT 更容易 reward hacking？

| 因素 | PBRFT | Agentic RL |
|---|---|---|
| Action 类型 | 只生成文本 | **能修改环境**（删文件、改测试） |
| 反馈来源 | 单一 RM | 多源（测试 / verifier / 编译 / lint），**每个都可能被钻空子** |
| Trajectory 长度 | 短 | 长 → 攻击面大 |
| Sandbox | 通常不必要 | **必须**（agent 真的会 rm -rf） |

→ 详细防护见 [08-Reward-Hacking与Sandbox安全](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)

---

## 🎓 1.5.5 怎么在实践中检测信用分配问题

**症状 1**：训练曲线 reward 涨，但人工抽查 trajectory 发现 agent 在"瞎走"

```
曲线: reward 0.2 → 0.6 → 0.85（看起来很棒）
抽查: agent 在前 30 步乱走，最后 5 步靠运气过测试
诊断: trajectory 里前 30 步的动作被"白白强化"了
```

**症状 2**：单一任务能解，跨任务泛化差

```
训练 task A: success rate 90%
测试 task B (类似但不同): success rate 15%
诊断: 学到的是 task A 的 shortcut，不是通用修 bug 能力
```

**症状 3**：通过测试但 patch 看起来很奇怪

```
patch: 写了 100 行 if/elif 匹配测试输入
诊断: hard-code reward hacking
```

**应对**：

- **保存完整 trajectory** + 人工抽查
- **OOD evaluation**（用没训练过的 task 测）
- **静态分析 patch**（检测 hard-code 模式）
- **trajectory-level 而非 token-level reward 分析**

---

## 📊 1.5.6 5 种 reward 类型与稀疏度对比

直接呼应 [4.4.3 5 种 reward 类型](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/04-RL%E5%A6%82%E4%BD%95%E7%94%A8%E5%9C%A8Code-Agent%E4%B8%8A.md#-443-5-%E7%A7%8D-reward-%E7%B1%BB%E5%9E%8B%E6%9C%AC%E8%8A%82%E6%9C%80%E9%87%8D%E8%A6%81%E7%9A%84%E8%A1%A8)：

| Reward 类型 | 稀疏度 | 信用分配难度 | Reward hacking 风险 |
|---|---|---|---|
| Pass/fail | ⭐⭐⭐⭐⭐ 极稀疏 | ⭐⭐⭐⭐⭐ 极难 | ⭐⭐ 低（信号客观） |
| Dense (shaped) | ⭐⭐ 中 | ⭐⭐ 中 | ⭐⭐⭐⭐ 高（shape 易被钻） |
| Process (PRM) | ⭐ 低（每步都有） | ⭐ 低 | ⭐⭐⭐ 中（PRM 也会被攻击） |
| Verifier | ⭐⭐⭐⭐ 高（终局） | ⭐⭐⭐⭐ 难 | ⭐⭐⭐⭐ 高（verifier 被攻击） |
| Rule | ⭐⭐⭐ 中 | ⭐⭐⭐ 中 | ⭐⭐ 低（规则简单） |

→ **没有完美 reward 类型**。实战常常**混合多个**（如 pass/fail + dense shaping + 规则惩罚）。

---

## 📌 1.5 节要点

| 概念 | 一句话 |
|---|---|
| **Sparse reward** | 多数步 r=0，只有终局 / 关键时刻有奖励 |
| **Credit assignment** | 判断哪些动作导致最终成败 |
| **Outcome reward** | 最稀疏的 reward 形式（只在终局给） |
| **Reward shaping** | 加 dense reward，**但小心 reward hacking** |
| **PRM** | 训练 model 给中间步骤打分 |
| **RLVR + GRPO** | 用客观验证器 + 组内多采样，缓解稀疏信号 |
| **Reward hacking** | Agentic RL 最大风险，必须设防 |

---

## ⚠️ 给你的提醒

你做 Code Agentic RL 实验时，**90% 概率会被 reward hacking 教育**。

3 条戒律：

1. **永远先看 trajectory，再看曲线**
2. **永远在 hold-out task 上算 success rate**
3. **永远保存完整 trajectory，便于事后分析**

---

## 🔗 延伸阅读

- 下一节：[06-本章要点与延伸](06-%E6%9C%AC%E7%AB%A0%E8%A6%81%E7%82%B9%E4%B8%8E%E5%BB%B6%E4%BC%B8.md)
- 关键应用：[08-Reward-Hacking与Sandbox安全](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/08-Reward-Hacking%E4%B8%8ESandbox%E5%AE%89%E5%85%A8.md)
- 算法配合：[05-PRM与ORM（过程vs结果奖励）](../02-%E7%AC%AC2%E7%AB%A0-%E4%BB%8ERLHF%E5%88%B0Agentic-RL/05-PRM%E4%B8%8EORM%EF%BC%88%E8%BF%87%E7%A8%8Bvs%E7%BB%93%E6%9E%9C%E5%A5%96%E5%8A%B1%EF%BC%89.md)
- Toy 实操：[02-Toy-Bug-Fix-Agent-Gym（核心）](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E9%A1%B9%E7%9B%AE/02-Toy-Bug-Fix-Agent-Gym%EF%BC%88%E6%A0%B8%E5%BF%83%EF%BC%89.md)
- 原始资料：`01-boundaries.md` §"sparse reward、credit assignment、reward hacking"

---

⬅ [04b-On-Policy-vs-Off-Policy](04b-On-Policy-vs-Off-Policy.md) | ➡ [06-本章要点与延伸](06-%E6%9C%AC%E7%AB%A0%E8%A6%81%E7%82%B9%E4%B8%8E%E5%BB%B6%E4%BC%B8.md)
