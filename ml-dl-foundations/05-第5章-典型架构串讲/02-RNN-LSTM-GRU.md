---
tags: [ml-dl-foundations, 第5章, RNN, LSTM, GRU, 序列模型]
chapter: 5
section: 5.2
---

# 5.2 RNN / LSTM / GRU

⬅ [01-CNN卷积神经网络](01-CNN%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md) | ➡ [03-Attention与Transformer回顾](03-Attention%E4%B8%8ETransformer%E5%9B%9E%E9%A1%BE.md)

---

## 🎬 故事比喻：边读边记笔记

```
你读一句话:
  "The dog that I bought yesterday is very cute."

你脑子里是这样处理的:
  - 读到 "The dog" → 记住"主语是 dog"
  - 读到 "that I bought yesterday" → 这是修饰
  - 读到 "is very cute" → 主语 dog 在描述 "cute"
                              ↑
                       你必须记得前面的 "dog"

这就是 RNN：
  - 每次只读一个词
  - 维护一个"hidden state"（笔记）
  - 把笔记和新词结合，更新笔记
  - 处理下一个词
```

> **理解 RNN 是理解 Transformer 为啥赢的前提**。
> Transformer 抛弃 RNN 的"必须顺序处理"，**并行**算所有位置——这才是革命。

---

## 📐 5.2.1 RNN（Recurrent Neural Network）

### 数学定义

输入序列 $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T$，hidden state $\mathbf{h}_t$：

$$\mathbf{h}_t = \tanh(\mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{b})$$

$$\mathbf{y}_t = \mathbf{W}_y \mathbf{h}_t + \mathbf{b}_y$$

**逐项解读**：

| 符号 | 含义 |
|---|---|
| $\mathbf{x}_t$ | 时刻 t 的输入 |
| $\mathbf{h}_t$ | 时刻 t 的 hidden state（"记忆"） |
| $\mathbf{h}_{t-1}$ | 上一时刻的 hidden state |
| $\mathbf{W}_x, \mathbf{W}_h, \mathbf{W}_y$ | 三组**共享权重**（所有时刻一样） |

**展开图**：

```
   x_1        x_2        x_3        x_4
    │          │          │          │
    ▼          ▼          ▼          ▼
   ┌───┐ h_1  ┌───┐ h_2  ┌───┐ h_3  ┌───┐
h0 │ R │─────►│ R │─────►│ R │─────►│ R │── h_4
   └───┘      └───┘      └───┘      └───┘
    │          │          │          │
    ▼          ▼          ▼          ▼
   y_1        y_2        y_3        y_4

R = 同一个 RNN cell, 权重共享
```

### 关键特性

| 特性 | 含义 |
|---|---|
| **顺序处理** | 必须算完 $\mathbf{h}_{t-1}$ 才能算 $\mathbf{h}_t$ |
| **参数共享** | 所有时刻用同一组 W |
| **长度任意** | 可处理任意长序列 |
| **有"记忆"** | $\mathbf{h}_t$ 含历史信息 |

---

## 🪤 5.2.2 RNN 的致命问题：长程依赖

**问题**：如果句子很长（如 50 个词），早期 token 的信息**经过 50 次矩阵乘**才传到末尾。

数学上：反向传播时梯度连乘 → **梯度消失**或**梯度爆炸**。

**例子**：
```
"The cat that we adopted last year from the shelter we visited
in California after a long road trip was actually very ___"

要预测 ___ (cat 的描述)，必须记住开头 70 个词外的"cat"。
RNN 很难做到。
```

→ 这就是 RNN 的根本缺陷。

---

## 🛡 5.2.3 LSTM（Long Short-Term Memory，1997）

**思路**：加**门控机制**，让网络**有选择地记忆/遗忘**。

### 核心：3 个门 + 1 个 cell state

```
                    ┌────────────────┐
                    │   cell state c │ ← "长期记忆"，类似传送带
                    └────────────────┘
                            │
       ┌─────┐    ┌───┐    │    ┌───┐    ┌───┐
  ─→  │遗忘 │ → │ × │ ──→ + ── │输入 │── │输出 │ ── h_t
       └─────┘    └───┘    │    └───┘    └───┘
       Forget Gate          ↑    Input    Output
                       Cell State 更新
```

**3 个门**（每个都是 sigmoid，0-1 控制开关）：

1. **遗忘门 (forget gate)**：决定丢掉多少旧 cell state
2. **输入门 (input gate)**：决定加多少新信息
3. **输出门 (output gate)**：决定输出多少给 hidden state

### 数学（看公式认就行）

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t]) \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t]) \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t]) \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t]) \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{aligned}
$$

**关键**：**cell state $\mathbf{c}_t$ 是个"传送带"**——信息可以**几乎无衰减**地穿过多步（解决长程依赖）。

### LSTM vs RNN

| 维度 | RNN | LSTM |
|---|---|---|
| 长程依赖 | 弱 | **强**（门控保护） |
| 参数 | 少 | 4 倍（3 门 + cell） |
| 速度 | 快 | 慢 |
| 用在哪 | 短序列 | **NLP 主流（2014-2017）** |

---

## 🚪 5.2.4 GRU（Gated Recurrent Unit，2014）

LSTM 的**简化版**：

- 只 2 个门（合并 forget+input 为 update gate）
- 没有独立 cell state（hidden state 兼任）

**优势**：参数少，训练快，效果接近 LSTM。

→ 现在 RNN 用 LSTM 还是 GRU，**看品味**——基本平手。

---

## 🆚 5.2.5 RNN 家族对比

| 模型 | 长程依赖 | 参数 | 速度 | 现在用？ |
|---|---|---|---|---|
| 原始 RNN | ❌ | 少 | 快 | 几乎不用 |
| **LSTM** | ✅ | 4× | 慢 | 历史用得多，**逐渐被淘汰** |
| **GRU** | ✅ | 3× | 中 | 历史用得多，**逐渐被淘汰** |
| **Transformer** | ✅ | 多 | **并行快** | **现在主流** |

---

## 🔥 5.2.6 为什么 RNN 全家被 Transformer 干掉？

### 1. 训练慢（顺序处理）

RNN 必须按时间步**顺序**算：

```
h_1 → h_2 → h_3 → ... → h_T  (T 步必须等)
```

GPU 喜欢并行，RNN 这种依赖**让 GPU 用不起来**。

### 2. 仍有长程依赖问题

LSTM 缓解了，但**没根本解决**——50+ 步外的依赖仍弱。

### 3. Transformer 全方位胜出

| 特性 | RNN/LSTM | Transformer |
|---|---|---|
| 并行训练 | ❌ 顺序 | ✅ 全位置并行 |
| 长程依赖 | 弱-中 | ✅ 一层即全图 |
| 训练速度 | 慢 | 快 N 倍 |
| 准确率 | 一般 | **高很多** |
| 推理速度 | 慢 | KV-Cache 后也快 |

→ **2017 Transformer 出来后，NLP 几乎全转 Transformer**。

---

## 💻 5.2.7 PyTorch 代码（速览）

```python
import torch
import torch.nn as nn

# 原始 RNN
rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)

# LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)

# GRU
gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)

x = torch.randn(32, 50, 10)        # [batch=32, seq_len=50, input_dim=10]
output, hidden = lstm(x)
print(output.shape)                 # torch.Size([32, 50, 20])
```

→ D2L §8-§9 有详细 RNN/LSTM 教程。

---

## 🔑 5.2.8 你应该带走的 3 件事

1. **RNN 是 NLP 的史前生物**——LSTM/GRU 主导 2014-2017
2. **长程依赖 + 不能并行** 是 RNN 的两个致命伤
3. **Transformer 同时解决了这两个问题**——所以全面胜出

→ 这就是为什么你 2025 学 LLM 而不是 LSTM。

---

## ⚠️ 5.2.9 常见误解

| 误解 | 真相 |
|---|---|
| "RNN 完全死了" | 还有用在小设备、流式音频等场景；但 NLP 主流没了 |
| "LSTM 比 GRU 一定好" | 平手——看任务和数据 |
| "Transformer 完全替代 RNN" | NLP 是。但时序预测 / 强化学习仍有用 |
| "RNN 不能并行" | 训练时不能（顺序依赖），**推理时多 batch 可并行** |
| "Mamba 是 RNN 的复兴" | Mamba (2023+) 是新型状态空间模型，**比 RNN 强**，**和 Transformer 竞争中**——值得关注 |

---

## 📌 5.2 节要点

| 模型 | 一句话 |
|---|---|
| RNN | "边读边记笔记"，但记不远 |
| LSTM | 加门控保护"传送带"，记得远 |
| GRU | LSTM 简化版，平手 |
| **Transformer** | **抛弃顺序处理，全位置并行 attention** |

**最重要的一句**：
> **理解 RNN 的痛点** = **理解 Transformer 的设计动机**。

---

## 🔗 延伸阅读

- 上一节：[01-CNN卷积神经网络](01-CNN%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
- 下一节：[03-Attention与Transformer回顾](03-Attention%E4%B8%8ETransformer%E5%9B%9E%E9%A1%BE.md)
- 已有锚点：[01-为什么需要注意力](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E6%B3%A8%E6%84%8F%E5%8A%9B.md)（接续讲解）
- 外部：D2L §8 https://zh.d2l.ai/chapter_recurrent-neural-networks/index.html

---

⬅ [01-CNN卷积神经网络](01-CNN%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md) | ➡ [03-Attention与Transformer回顾](03-Attention%E4%B8%8ETransformer%E5%9B%9E%E9%A1%BE.md)
