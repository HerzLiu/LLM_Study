---
tags: [agentic-rl, 第1章, MDP, POMDP, trajectory, RL基础]
chapter: 1
section: 1.1
source: "agentic-rl-learning-map/01-boundaries.md §必须补的 RL 基础, 05-glossary.md"
---

# 1.1 MDP 与轨迹（用 Agent 语言讲 RL）

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-Policy-Value-Advantage](02-Policy-Value-Advantage.md)

---

## 🎬 故事比喻：你已经在写 MDP 了，只是不知道

写一个 ReAct agent 修 bug：

```python
# 这是你熟悉的伪代码
while not done:
    obs = read_files() + last_test_output         # ← 你以为的"上下文"
    action = llm.act(obs)                          # ← 你以为的"LLM 输出 thought+action"
    result = env.execute(action)                   # ← 你以为的"工具执行"
    done = check_pytest_all_pass()                 # ← 你以为的"任务结束判定"
```

**RL 视角下的同一段代码**：

```python
while not done:
    o_t = observation(s_t)        # ← obs 其实是 observation o_t
    a_t = π(o_t)                  # ← LLM 是 policy π
    s_{t+1}, r_t = env(s_t, a_t)  # ← env 是 transition + reward
    done = is_terminal(s_{t+1})
```

> **唯一区别**：RL 视角下，每一步多了一个 **r_t（reward）**。
> 这个 r_t 就是你可以**反过来**训练 π 的"梯度信号"。

**学 RL = 学会把你已经在写的 agent loop 翻译成 (S, A, P, R, γ) 的语言。**

---

## 📐 1.1.1 MDP 五元组（必背）

**MDP** = Markov Decision Process（马尔可夫决策过程）

$$\text{MDP} = (S, A, P, R, \gamma)$$

| 符号 | 名字 | 在 Agent 任务里是什么 |
|---|---|---|
| $S$ | **状态空间** State space | 所有可能的"环境状态"集合（repo 当前所有文件 + 历史命令） |
| $A$ | **动作空间** Action space | agent 能选的所有动作（search/read/edit/run_tests/final） |
| $P(s'\|s,a)$ | **转移概率** Transition | 在 $s$ 选 $a$ 后转到 $s'$ 的概率（Code Agent 里几乎是确定的：pytest 输出几乎不变） |
| $R(s,a)$ | **奖励函数** Reward | 在 $s$ 选 $a$ 获得的即时奖励（中间步骤多为 0，终局看测试） |
| $\gamma$ | **折扣因子** Discount | 未来奖励的权重，0 ≤ γ ≤ 1（短任务用 1.0，长任务用 0.95-0.99） |

### 转移概率 $P$ 的细节理解

$$P: S \times A \times S \to [0, 1]$$

这是一个**三元函数**，但更常用条件概率写法 $P(s' \mid s, a)$。

- **类型**：对每个固定的 $(s, a)$，$P(\cdot \mid s, a)$ 是 $S$ 上的**概率分布**（积分为 1）
- **维度**：若 $|S| = N$、$|A| = K$，$P$ 可看成 $N \times K \times N$ 的张量
- **必须满足**：$\sum_{s' \in S} P(s' \mid s, a) = 1$，$\forall (s, a)$
- **Code Agent 特殊性**：几乎确定性，即 $P(s' \mid s, a) \in \{0, 1\}$——`pytest` 跑一次输出几乎一样
- **意义**：环境对"你选 $a$ 之后 world 会怎样"的回答

### 奖励函数 $R$ 的细节理解

资料里常见三种等价写法：

| 写法 | 含义 | 何时用 |
|---|---|---|
| $R(s, a)$ | 在 $s$ 选 $a$ 获得的**期望**即时奖励（标量） | 理论分析最常用 |
| $R(s, a, s')$ | 同上但带下一状态（用于奖励依赖结果时） | 推导 Bellman 时用 |
| $r_t$ | 第 $t$ 步**实际拿到**的 reward（一次采样的实现） | 训练日志、trajectory 里用 |

→ 三者关系：$R(s_t, a_t) = \mathbb{E}[r_t \mid s_t, a_t]$。
→ 我们**采样的是 $r_t$**，**理论里讨论的是 $R$**。

### 折扣因子 $\gamma$ 的细节理解

$\gamma \in [0, 1]$ 控制"未来 reward 的衰减"：

| $\gamma$ | 第 $k$ 步未来 reward 权重 $\gamma^k$ | 直觉 |
|---|---|---|
| 0 | $0^k = 0$（仅 $k=0$ 时为 1） | **只看眼前**，退化为单步决策（bandit） |
| 0.5 | $0.5, 0.25, 0.125, \dots$ | 远期奖励**指数衰减**，10 步后只剩 0.1% |
| 0.9 | $0.9, 0.81, 0.73, \dots$ | 10 步后剩 35% |
| 0.99 | $0.99, 0.98, 0.97, \dots$ | 100 步后还剩 37%（**LLM RL 常用**） |
| 1.0 | $1, 1, 1, \dots$ | 完全不打折，只在 episode **有限**时才用 |

**为什么需要 $\gamma$？**

1. **数学**：$\gamma < 1$ 保证 $\sum_{t=0}^{\infty} \gamma^t r_t$ 在 $r_t$ 有界时**收敛**
2. **建模**：现实里"今天 100 块" > "10 年后 100 块"
3. **算法**：$\gamma$ 控制 critic 学习的"视野"，太大方差大、太小近视

→ Code Agent 多数任务有限步（30-50），$\gamma = 0.99$ 或 $1.0$ 都常见。

### 马尔可夫性质（Markov Property）

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t)$$

**逐项解读这个等式**：

- **左边**：$s_{t+1}$ 的条件概率，**条件**是从最初到现在的**全部**历史
- **右边**：$s_{t+1}$ 的条件概率，**条件**只有**当前** $(s_t, a_t)$
- **等号**：两者**相等** = "历史里除了当前 state 之外的信息**全是冗余**"

**人话**："**未来只和现在的 state 有关，和过去无关**"。

**重要边界条件**：

- ✅ "马尔可夫"**不等于**"无记忆"——它是说**所有有用的历史信息**都已经压缩进 $s_t$ 了
- ✅ 你可以让 $s_t$ 包含"过去 10 步的总结"，这样仍然满足马尔可夫
- ❌ 如果 $s_t$ 只是"当前文件名"而**没有**包含历史命令，那就**不满足**马尔可夫

→ **Code Agent 严格来说不满足马尔可夫**——下一步行动依赖你前面读过哪些文件。
→ 所以 Code Agent 实际是 **POMDP**（见 §1.1.3）。
→ 工程上的解决：把"历史压缩"作为 observation 的一部分（[§4.3](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md)）。

---

## 🌍 1.1.2 用 Code Agent 任务填这张表

| 元素 | SWE-bench task 里的具体值 |
|---|---|
| $S$ | (issue 文本, repo 所有文件内容, 所有 shell 历史, 所有测试历史) 的笛卡尔积 |
| $A$ | `{search(q), read(p), read(p,start,end), edit(p,patch), run_tests(cmd), final()}` |
| $P$ | pytest / shell / 文件系统操作的**确定性**输出（几乎无随机） |
| $R$ | 中间步骤 0；终局：全部 FAIL_TO_PASS 过 + PASS_TO_PASS 不破 → +1，否则 0 |
| $\gamma$ | 通常 1.0（任务短）或 0.99（长 trajectory） |

**注意**：实际 $S$ **极大**——一个真实 repo 状态空间是天文数字。
但**这没关系**——RL 不要求遍历 $S$，只要 sample 和 update 就行。

---

## 👁 1.1.3 POMDP：你看到的不是全部

Code Agent 的真实情况是：

> **state 是整个 repo + 完整历史，但 agent 只能看到 context window 里的一小部分。**

这就是 **POMDP** = Partially Observable MDP（部分可观测马尔可夫决策过程）。

```
真实 state s_t   ──→  生成 observation o_t  ──→  agent 看到 o_t
(整个 repo +              (你塞进 context        (做决策)
 历史 + 假设)              的那部分)
       ↑                                              │
       │                                              ▼
       └──────── 环境根据 a_t 更新 ←─── policy π(o_t) ─┘
```

| 概念 | 含义 |
|---|---|
| State $s_t$ | 客观存在，环境维护 |
| Observation $o_t$ | agent 视角下的"输入"（context window 内容） |
| Observation function $O(o\|s)$ | 状态到观察的映射（你的 summarization 策略） |

> 💡 **关键启示**：observation 设计（怎么压缩 / 截断 / 总结）是 Code Agent 工程的核心。
> 这就是为什么 OpenHands、SWE-agent 花大量篇幅做 context management。

---

## 🛤 1.1.4 Trajectory（轨迹）：Agentic RL 优化的对象

一次完整的 episode 写成：

$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T, a_T, r_T)$$

**逐项解释**：

- $s_0$：**初始状态**，由 `env.reset()` 给出（Code Agent 里 = 拿到 issue 和 repo 那一刻的全部信息）
- $a_t$：在 $s_t$ 下 agent **采样**得到的动作，$a_t \sim \pi(\cdot \mid s_t)$
- $r_t$：环境根据 $(s_t, a_t)$ **返回**的即时奖励，$r_t \sim R(s_t, a_t)$
- $s_{t+1}$：环境根据 $(s_t, a_t)$ **转移**到的下一状态，$s_{t+1} \sim P(\cdot \mid s_t, a_t)$
- $T$：**终止时间步**（有限 horizon）或 $\infty$（持续任务）

**注意时序**：$s_t \to a_t \to r_t \to s_{t+1}$，即"reward 在转移**之前**给出"（这是 Sutton & Barto 的约定，OpenAI 等也用此约定）。

或更紧凑（用 observation 替代 state，因为 POMDP）：

$$\tau = (o_t, a_t, r_t)_{t=0}^{T}$$

→ 这就是你保存到 `trajectories.jsonl` 的那个东西（见 [4.3.5 Trajectory Schema](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md#-435-trajectory-schema%E5%AE%9E%E9%99%85%E5%8F%AF%E7%94%A8%E7%9A%84-json)）。

**$\tau$ 是随机变量**：由于 $\pi$、$P$、$R$ 都有随机性，每次 rollout 得到的 $\tau$ 都不同。
$\tau \sim \pi$ 这个记号表示"在策略 $\pi$ 下采样一条轨迹"。

### Return（回报）：trajectory 的"总奖励"

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

**逐项解读这个求和**：

- **下标 $k$**：从 0 加到 $T-t$（即从"当前步"到"episode 结束"）
- **$\gamma^k$**：第 $k$ 步**之后**的 reward 权重，$k=0$ 时权重 1（眼前 reward 满权）
- **$r_{t+k}$**：第 $t+k$ 步实际拿到的 reward
- **$G_t$**：从时刻 $t$ 看到的"未来累积奖励"（注意：**不包含**过去）

**递归形式（核心，后面 Bellman 全靠它）**：

$$G_t = r_t + \gamma \cdot G_{t+1}$$

**推导**：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$$
$$\quad = r_t + \gamma \underbrace{(r_{t+1} + \gamma r_{t+2} + \dots)}_{= G_{t+1}}$$
$$\quad = r_t + \gamma G_{t+1}$$

→ 这个等式**告诉你**：要算 $G_t$，**不需要重头加**，只需 $r_t + \gamma \times$（下一步的 $G$）。
→ 这是后面 Bellman 方程的根。

**数值小例子**（Code Agent 一次成功修 bug 的轨迹，$\gamma = 0.99$，$T = 4$）：

```
t=0: r_0 = 0    (search)
t=1: r_1 = 0    (read)
t=2: r_2 = 0    (edit)
t=3: r_3 = 0    (run_tests)
t=4: r_4 = 1.0  (final, 全测通过)
```

倒推：

$$G_4 = r_4 = 1.0$$
$$G_3 = r_3 + 0.99 \cdot G_4 = 0 + 0.99 \cdot 1.0 = 0.99$$
$$G_2 = r_2 + 0.99 \cdot G_3 = 0 + 0.99 \cdot 0.99 = 0.9801$$
$$G_1 = r_1 + 0.99 \cdot G_2 = 0 + 0.99 \cdot 0.9801 = 0.9703$$
$$G_0 = r_0 + 0.99 \cdot G_1 = 0 + 0.99 \cdot 0.9703 = 0.9606$$

→ **越早的步骤 $G$ 越小**（同一个 reward 被多次乘 $\gamma$）。
→ 这就是"延迟奖励被打折"在数值上的样子。

| γ 值 | 直觉 |
|---|---|
| γ = 1.0 | 远期和近期奖励**一样重要**（$G_0 = G_T = 1.0$） |
| γ = 0.99 | 远期略打折，常用（5 步前的 1.0 → 0.95） |
| γ = 0.95 | 远期明显折扣（5 步前的 1.0 → 0.77） |
| γ = 0 | **只看眼前**（退化为 bandit，$G_t = r_t$） |

### Agentic RL 的目标函数

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[ \sum_{t=0}^{T} \gamma^t r_t \right] = \mathbb{E}_{\tau \sim \pi}[G_0]$$

$$\pi^* = \arg\max_\pi J(\pi)$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $J(\pi)$ | 策略 $\pi$ 的**评分**（标量），是我们要最大化的目标 |
| $\mathbb{E}_{\tau \sim \pi}[\cdot]$ | 对 $\pi$ 采样的**所有可能轨迹**求**期望**（即"平均跑很多次") |
| $\sum_{t=0}^{T} \gamma^t r_t$ | 单条轨迹的总折扣回报 = $G_0$ |
| $\arg\max_\pi$ | 找到**让 $J$ 最大**的那个 $\pi$ |
| $\pi^*$ | 最优策略 |

**为什么是期望？**

环境和策略都有随机性，**单次 rollout 的 $G_0$ 是随机变量**。
我们关心的不是"侥幸跑出好结果"，是**平均**能拿到多高的 return。

**展开期望（看是怎么"加权平均"的）**：

$$J(\pi) = \sum_{\tau} P(\tau \mid \pi) \cdot G_0(\tau)$$

其中：

$$P(\tau \mid \pi) = \underbrace{P(s_0)}_{\text{初始状态分布}} \cdot \prod_{t=0}^{T} \underbrace{\pi(a_t \mid s_t)}_{\text{策略采样}} \cdot \underbrace{P(s_{t+1} \mid s_t, a_t)}_{\text{环境转移}}$$

→ 一条轨迹的**概率**由 3 部分相乘：初始分布 × 每步策略 × 每步转移。
→ **梯度优化时**（[§1.3](03-Policy-Gradient%E4%B8%8EActor-Critic.md)），只有 $\pi(a_t \mid s_t)$ 这一项含参数 $\theta$，其它都是环境给的常数——这是 policy gradient 能算出来的关键。

**人话总结**：**调 $\pi$，让"很多次跑下来平均的累积折扣奖励"最大**。

---

## 🆚 1.1.5 PBRFT vs Agentic-RL（核心对比，跨链 Hello-Agents）

跨链：[Hello-Agents 11.1.5](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/01-%E4%BB%8ELLM%E8%AE%AD%E7%BB%83%E5%88%B0Agentic-RL.md#-1115-mdp-%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94%E5%BF%85%E8%83%8C) 已经讲了这张表，这里再用一次：

### PBRFT（单轮，相当于 RLHF 的标准用法）

| 维度 | 值 |
|---|---|
| State $S$ | 仅用户 prompt |
| Action $A$ | 仅文本生成（输出整段 response） |
| Transition $P$ | 无（单步） |
| Reward $R$ | 单步，仅任务结束时 |
| 时间 $T$ | 1 |
| 目标 | $\max \mathbb{E}[R(s, a)]$ |

### Agentic-RL（多步）

| 维度 | 值 |
|---|---|
| State $S$ | 历史观察 + 上下文 |
| Action $A$ | 文本生成 + 工具调用 + 环境操作 |
| Transition $P$ | state 根据 action **动态变化** |
| Reward $R$ | **多步**，可中间步骤给奖励 |
| 时间 $T$ | $> 1$（多步） |
| 目标 | $\max \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$ |

> 📌 **关键差异：trajectory 是 Agentic RL 的优化对象，不是单个 (s, a, r)**。

---

## 🧪 1.1.6 把你熟悉的 ReAct 翻译成 MDP

ReAct 的典型 trajectory：

```
Thought: 我应该先看 normalize_path 在哪
Action: search("normalize_path")
Observation: src/paths.py:42

Thought: 读这个文件
Action: read("src/paths.py")
Observation: <文件内容>

Thought: 看起来 rstrip('/') 太粗暴
Action: edit("src/paths.py", patch)
Observation: applied

Thought: 跑测试
Action: run_tests()
Observation: 1 passed, 1 failed

...

Final Answer: <patch>
```

**MDP 翻译**：

| ReAct 字段 | MDP 元素 |
|---|---|
| 整个 trajectory（thought+action+observation 序列） | $\tau$ |
| 当前看到的 issue+history | $o_t$（observation） |
| Action（包括 thought + 工具调用） | $a_t$ |
| 每步 Observation | env 返回的 $s_{t+1}$ 部分可见 |
| Final 时算 reward | $r_T$（其它 $r_t = 0$） |

> 💡 **重要**：thought 也是 action 的一部分（也是 token 输出，policy 也在 sample 它们）。
> 这就是为什么 RL 训练能**间接**优化"模型怎么想"，而不只是"模型怎么调工具"。

---

## 💻 1.1.7 最小 RL loop 伪代码

```python
# 概念示意，来自资料归纳，不是可跑代码
def rl_loop(env, policy, num_episodes):
    for episode in range(num_episodes):
        # 1. 重置环境
        s = env.reset()
        trajectory = []

        # 2. rollout 一个 episode
        done = False
        while not done:
            o = observe(s)              # 部分可观测
            a = policy.sample(o)         # 从 π(a|o) 采样
            s_next, r, done = env.step(a)
            trajectory.append((o, a, r))
            s = s_next

        # 3. 计算 return
        returns = compute_returns(trajectory, gamma=0.99)

        # 4. 用 trajectory 更新 policy
        policy.update(trajectory, returns)
```

→ 这 15 行就是所有 RL 算法的**骨架**。
→ PPO / GRPO / RLVR 只是第 4 步 `policy.update(...)` 的不同实现。

---

## ⚠️ 常见误解

| 误解 | 真相 |
|---|---|
| "MDP 要求转移是随机的" | 不要求。Code Agent 的 transition 几乎是确定的，照样是 MDP |
| "state 必须能完全表达系统" | POMDP 处理"看不全"的情况 |
| "Agentic RL 一定多步" | 是。单步 (T=1) 是 PBRFT / 普通 RLHF，不算 Agentic |
| "γ 必须 < 1" | 任务短可以 γ=1，但理论分析常用 γ<1 保收敛 |
| "Trajectory 只用来训练" | 也是**失败分析、debug、可视化**的核心数据 |

---

## 📌 1.1 节要点

| 概念 | 一句话 |
|---|---|
| MDP | (S, A, P, R, γ) 五元组，描述"做决策"的数学框架 |
| POMDP | state 看不全的 MDP，**Code Agent 实际是 POMDP** |
| Trajectory τ | 一次 episode 的 (o, a, r) 序列，**Agentic RL 优化的对象** |
| Return G | 累积折扣奖励 $\sum \gamma^t r_t$ |
| 目标 | $\max_\pi \mathbb{E}_\tau[\sum r_t]$ |
| PBRFT vs Agentic | 单步 vs 多步，最大差异在 trajectory |

---

## 🔗 延伸阅读

- 下一节：[02-Policy-Value-Advantage](02-Policy-Value-Advantage.md)
- 跨链：[Hello-Agents 11.1.5 MDP 对比](../../Hello-Agents/11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/01-%E4%BB%8ELLM%E8%AE%AD%E7%BB%83%E5%88%B0Agentic-RL.md#-1115-mdp-%E6%A1%86%E6%9E%B6%E5%AF%B9%E6%AF%94%E5%BF%85%E8%83%8C)
- 应用：[03-Observation-Action-Reward定义](../04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/03-Observation-Action-Reward%E5%AE%9A%E4%B9%89.md)
- 原始资料：`01-boundaries.md` §必须补的 RL 基础
- 经典教材：Sutton & Barto, "Reinforcement Learning: An Introduction" Ch 3

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-Policy-Value-Advantage](02-Policy-Value-Advantage.md)
