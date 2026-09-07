---
tags: [ml-dl-foundations, 第4章, 优化器, SGD, Adam, AdamW]
chapter: 4
section: 4.6
---

# 4.6 优化器进化：SGD → AdamW

⬅ [05-梯度下降-反向传播完整推导](05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md) | ➡ [07-学习率调度](07-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6.md)

---

## 🎬 故事比喻：下山方式的进化

```
你要下山（最小化 loss）。优化器 = "怎么走"的策略。

  SGD:    每步只看当前坡度，往下迈。简单但磕磕碰碰。
  Momentum: 加上"惯性"——上一步往哪走，这步还想往那走一点。
            像滚下山的球，能冲过小山包。
  AdaGrad: 走过多次的方向放慢脚步（学习率自适应）。
  RMSProp: AdaGrad 改进版，不让学习率衰到 0。
  Adam:    Momentum + RMSProp 合体。万金油。
  AdamW:   Adam + 修正过的 weight decay。LLM 训练首选。
```

> 这一节解释 Happy-LLM SFT 训练代码里 `torch.optim.AdamW(model.parameters(), lr=5e-5)` 内部到底在干什么。

---

## 📐 4.6.1 SGD（Stochastic Gradient Descent，随机梯度下降）

最朴素的优化器（[§4.5.1](05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md)）：

$$\theta_{t+1} = \theta_t - \eta \cdot g_t, \quad g_t = \nabla_\theta \mathcal{L}(\theta_t)$$

**"Stochastic"（随机）什么意思？**

- **批量梯度下降 (BGD)**：每步用**全部数据**算梯度。慢、稳。
- **随机梯度下降 (SGD)**：每步用**一个样本**算梯度。快、噪声大。
- **小批量梯度下降 (Mini-batch SGD)**：每步用**一小批样本**（如 batch_size=32）算梯度。**实际用的**。

→ 所谓"SGD"在 PyTorch 里默认是 **mini-batch**，不是真"每次一个"。

**优点**：
- 实现简单
- 内存小
- 调好了能收敛到很好结果

**缺点**：
- **学习率敏感**——太小慢、太大震荡
- 在"沟壑形" loss landscape 上**反复横跳**

---

## 📐 4.6.2 Momentum（动量法）

**思想**：加上"惯性"——保留一部分上一步的方向。

$$
\begin{aligned}
v_{t+1} &= \mu \cdot v_t + g_t \\
\theta_{t+1} &= \theta_t - \eta \cdot v_{t+1}
\end{aligned}
$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $v_t$ | "速度"，累积的梯度方向 |
| $\mu$ | 动量系数，常 0.9 |
| $\mu \cdot v_t$ | 保留 90% 上一步速度 |
| $+ g_t$ | 加上当前梯度 |

**直觉**：

```
SGD:        每步只看当下，掉沟里反复横跳

Momentum:   像滚下山的球
            - 持续往一个方向走 → 速度加快（v 增加）
            - 突然变方向 → 惯性让你冲过小坑
            - 在沟壑里上下震荡的分量被抵消
```

**效果**：
- ✅ 收敛快 2-10 倍
- ✅ 能冲过局部最小值（局部坑）

---

## 📐 4.6.3 AdaGrad（自适应学习率，2011）

**思想**：**对每个参数**用不同学习率——出现频繁的参数放慢学习率，少出现的加快。

$$
\begin{aligned}
G_t &= G_{t-1} + g_t^2 \quad \text{（累积梯度平方）}\\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t
\end{aligned}
$$

**直觉**：梯度大且频繁的参数 → $G$ 大 → 实际学习率 $\eta/\sqrt{G}$ 小，**走慢**。

**问题**：$G$ **只增不减**，学习率最终趋近 0 → 后期完全学不动。

---

## 📐 4.6.4 RMSProp（AdaGrad 改进，2012）

**思想**：用**指数移动平均**（EMA）代替累加，让旧梯度"遗忘"。

$$
\begin{aligned}
G_t &= \beta G_{t-1} + (1-\beta) g_t^2 \quad \text{（EMA，常 } \beta = 0.99\text{）}\\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t
\end{aligned}
$$

**关键变化**：$G_t$ 是**滑动平均**，不会无限增长，学习率不会衰到 0。

---

## 📐 4.6.5 Adam（Adaptive Moment Estimation，2014）⭐ 神级算法

**思想**：**Momentum + RMSProp 合体**。

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{（一阶矩：动量）}\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{（二阶矩：RMSProp）}\\
\hat{m}_t &= m_t / (1 - \beta_1^t) \quad \text{（偏差修正）}\\
\hat{v}_t &= v_t / (1 - \beta_2^t) \quad \text{（偏差修正）}\\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
\end{aligned}
$$

**逐项解读**：

| 符号 | 含义 | 默认 |
|---|---|---|
| $m_t$ | 梯度 EMA（一阶矩，方向） | - |
| $v_t$ | 梯度平方 EMA（二阶矩，幅度） | - |
| $\beta_1$ | 一阶矩 EMA 衰减 | **0.9** |
| $\beta_2$ | 二阶矩 EMA 衰减 | **0.999** |
| $\epsilon$ | 数值稳定项 | 1e-8 |
| $\hat{m}, \hat{v}$ | 修正偏差（因为初始 $m_0=v_0=0$，前几步会偏小） | - |

**为什么 Adam 这么强？**
- ✅ **自动调每个参数的学习率**（二阶矩）
- ✅ **有动量**（一阶矩）
- ✅ **对超参数不敏感**（默认值通常够好）
- ✅ 实践中几乎"开箱即用"

**几乎成为 DL 默认优化器**。

---

## 📐 4.6.6 AdamW（修正版 Adam，2017）⭐ LLM 训练首选

**问题**：Adam 加 L2 正则化（weight decay）的方式**有问题**。

**原始 Adam 做正则化**：

$$g_t \leftarrow g_t + \lambda \theta_t \quad \text{（梯度加上 weight decay）}$$

→ 这个 $\lambda \theta_t$ 会被 $\sqrt{\hat{v}_t}$ 缩放——**大梯度参数的正则化变弱**。

**AdamW 的修正**：把 weight decay 从梯度里**拿出来**，直接乘到参数上：

$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ | 标准 Adam 更新方向 |
| $\lambda \theta_t$ | **独立的** weight decay |
| $\lambda$ | weight decay 系数，常 0.01 |

**优势**：
- ✅ Weight decay **正确生效**（防过拟合）
- ✅ 训练更稳定
- ✅ 泛化更好

**用在哪**：
- ✅ **几乎所有 LLM 训练**（BERT/GPT/LLaMA/Qwen/...）
- ✅ Hugging Face Transformers 默认优化器

→ 这就是为什么 Happy-LLM 训练代码用 `AdamW`：
```python
optimizer = optim.AdamW(model.parameters(), lr=5e-5)  # ← 这就是它
```

---

## 📊 4.6.7 全家对比表

| 优化器 | 年份 | 核心 | 优点 | 适合场景 |
|---|---|---|---|---|
| **SGD** | 1950s | $\theta -= \eta g$ | 简单稳定 | 经典 ML、小模型 |
| **SGD + Momentum** | 1960s | + 动量 | 收敛快 | CNN（如 ResNet） |
| **AdaGrad** | 2011 | 自适应 lr | 稀疏特征 | NLP 早期 |
| **RMSProp** | 2012 | + EMA | 不让 lr 衰到 0 | RNN |
| **Adam** | 2014 | Momentum + RMSProp | 万金油 | DL 默认 |
| **AdamW** | 2017 | Adam + 修正 weight decay | **泛化更好** | **LLM 训练首选** |
| **Lion** | 2023 | 只用一阶矩 + sign | 省显存 | 大模型新尝试 |

---

## 🧪 4.6.8 一图看进化

```
                  SGD
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
   + Momentum             + 自适应 lr
   (惯性)                 (每个参数不同 lr)
       │                       │
       │                  ┌────┴────┐
       │                  ▼         ▼
       │              AdaGrad  RMSProp
       │              (累加)   (EMA)
       │                  │
       └──────────┬───────┘
                  ▼
                Adam
                  │
                  ▼
                AdamW (修正 weight decay)
                  │
                  ▼
            (LLM 训练标配)
```

---

## 💰 4.6.9 优化器的"显存代价"

不同优化器**需要存的额外状态**不同——这对训练大模型**关键**：

| 优化器 | 每个参数额外存的状态 | 显存代价 |
|---|---|---|
| SGD | 无 | **1×** |
| SGD + Momentum | $v$ | **2×** |
| Adam / AdamW | $m, v$ | **3×** |
| Adam + fp32 master weights | $m, v, \theta_{fp32}$ | **4×** |

**举例**：训练 7B 模型用 AdamW + fp32 master weights：
- 模型本身（fp16）：14 GB
- 梯度（fp16）：14 GB
- 优化器状态（fp32 m + v + master）：**84 GB**
- 合计：> 112 GB

→ 这就是为什么 LLM 训练 OOM——**优化器状态占大头**。
→ 解决：DeepSpeed ZeRO（[§6.3](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/03-%E5%88%86%E5%B8%83%E5%BC%8F%E4%B8%8E%E6%98%BE%E5%AD%98%E4%BC%98%E5%8C%96.md)）把这些拆到多卡。

---

## 💻 4.6.10 PyTorch 代码（概念示意，未严格验证）

```python
import torch
import torch.optim as optim

model = MyModel()

# === 各种优化器 ===
opt_sgd      = optim.SGD(model.parameters(), lr=0.01)
opt_sgd_mom  = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
opt_adagrad  = optim.Adagrad(model.parameters(), lr=0.01)
opt_rmsprop  = optim.RMSprop(model.parameters(), lr=0.001)
opt_adam     = optim.Adam(model.parameters(), lr=0.001)
opt_adamw    = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)  # LLM 用这个

# === 训练循环里用法都一样 ===
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

---

## ⚠️ 4.6.11 常见误解

| 误解 | 真相 |
|---|---|
| "Adam 一定比 SGD 好" | **不一定**。CV 任务里 SGD+Momentum 调好了可能更强（更好泛化） |
| "AdamW 和 Adam 区别不大" | LLM 训练上**差异显著**——AdamW 必选 |
| "lr=5e-5 是 AdamW 的固定值" | 不是。是 LLM SFT 的**经验值**。不同任务要试 |
| "weight_decay 越大越防过拟合" | 太大会**欠拟合**——典型 0.01-0.1 |
| "Adam 不需要 lr 调度" | **需要**——见 §4.7 |
| "优化器 = 学习率调度器" | **不是**——两个独立组件，配合用 |

---

## 📌 4.6 节要点

| 时代 | 标配优化器 |
|---|---|
| 经典 ML | SGD |
| 早期 DL (CNN) | SGD + Momentum |
| 现代 DL | Adam |
| **LLM 训练** | **AdamW** |
| 资源紧张 | Lion / Adafactor |

**3 个核心思想**：

1. **动量**：保留上一步方向，加速收敛、冲过坑
2. **自适应学习率**：每个参数有自己的"实际 lr"，根据历史梯度调
3. **Weight decay**：正则化项，防过拟合（AdamW 修正了 Adam 的实现 bug）

**显存代价**：Adam/AdamW 比 SGD 贵 **3 倍**，LLM 训练 OOM 主因。

---

## 🔗 延伸阅读

- 上一节：[05-梯度下降-反向传播完整推导](05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md)
- 下一节：[07-学习率调度](07-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6.md)（lr 怎么调）
- 显存优化：[03-分布式与显存优化](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/03-%E5%88%86%E5%B8%83%E5%BC%8F%E4%B8%8E%E6%98%BE%E5%AD%98%E4%BC%98%E5%8C%96.md)
- 已有锚点：[10-SFT训练](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/10-SFT%E8%AE%AD%E7%BB%83.md)（你跑过 AdamW）
- 已有锚点：[09-预训练循环](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/09-%E9%A2%84%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)（AMP + AdamW 一起用）
- 桥接章：[03-从SGD到PPO-GRPO](../07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/03-%E4%BB%8ESGD%E5%88%B0PPO-GRPO.md)（SGD 怎么变成 PPO）

---

⬅ [05-梯度下降-反向传播完整推导](05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md) | ➡ [07-学习率调度](07-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6.md)
