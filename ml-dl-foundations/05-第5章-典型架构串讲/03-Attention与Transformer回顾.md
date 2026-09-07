---
tags: [ml-dl-foundations, 第5章, Attention, Transformer, 回顾]
chapter: 5
section: 5.3
---

# 5.3 Attention 与 Transformer 回顾

⬅ [02-RNN-LSTM-GRU](02-RNN-LSTM-GRU.md) | ➡ [04-架构发展时间轴](04-%E6%9E%B6%E6%9E%84%E5%8F%91%E5%B1%95%E6%97%B6%E9%97%B4%E8%BD%B4.md)

---

## 🎬 故事比喻：从"按顺序读"到"全文一眼看"

```
RNN 读句子:
  "The   dog   that   I   bought   is   cute"
   ↓     ↓     ↓     ↓     ↓        ↓    ↓
  必须  一个  一个   顺序  处理     ...

Transformer:
  "The   dog   that   I   bought   is   cute"
   ↑     ↑     ↑     ↑     ↑        ↑    ↑
   所有词同时, 每个词"看"其它所有词
   ↑
  Attention 让每个 token 直接关联到全句任意位置
```

> 这一节是 [Happy-LLM 第 2 章](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) 的**总结锚点**。
> 你已经详细学过 Transformer，本节只是从"补完角度"重新串起来——**让你知道前面 §4-§5 学的所有东西都是为它服务的**。

---

## 🧠 5.3.1 Attention 一句话定义

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

**逐项解读**（你已学，复习）：

| 符号 | 形状 | 含义 |
|---|---|---|
| $\mathbf{Q}$ | $[T, d_k]$ | Query：每个 token 想问什么 |
| $\mathbf{K}$ | $[T, d_k]$ | Key：每个 token 能提供什么 |
| $\mathbf{V}$ | $[T, d_v]$ | Value：每个 token 实际信息 |
| $\mathbf{Q}\mathbf{K}^T$ | $[T, T]$ | 每对 token 的"相关度" |
| $\text{softmax}(\cdot/\sqrt{d_k})$ | $[T, T]$ | 归一化为权重 |
| 最终输出 | $[T, d_v]$ | 加权混合的 value |

**人话**：「**每个 token 用 Q 去问，K 给答案，V 是真实信息；按相关度加权得到输出**」

→ 完整推导见 [02-注意力机制QKV](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/02-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6QKV.md)。

---

## 🏗 5.3.2 Transformer Block（你已学的全景）

```
       Input  x [B, T, dim]
         │
         ▼
   ┌──────────────────────────┐
   │  LayerNorm (Pre-Norm)    │ ← §4.8 LN
   └──────────────────────────┘
         │
         ▼
   ┌──────────────────────────┐
   │  Multi-Head Attention    │ ← §5.3 + Happy-LLM §2.5
   └──────────────────────────┘
         │                  │
         ▼                  │
       ＋ ← ──────────────── (残差) ← §5.1 ResNet 借来
         │
         ▼
   ┌──────────────────────────┐
   │  LayerNorm               │
   └──────────────────────────┘
         │
         ▼
   ┌──────────────────────────┐
   │  FFN / MLP (SwiGLU)      │ ← §4.2 MLP + §4.3 SwiGLU
   └──────────────────────────┘
         │                  │
         ▼                  │
       ＋ ← ──────────────── (残差)
         │
         ▼
       Output [B, T, dim]
```

→ **Transformer block 用上了本书学的所有积木**：MLP（§4.2）+ 激活（§4.3）+ Norm（§4.8）+ 残差（§5.1）+ Attention。

---

## 🔁 5.3.3 用本书第 4 章重新理解 Transformer

| Transformer 组件 | 本书第 4 章对应 | 你之前在 Happy-LLM 看 |
|---|---|---|
| Multi-Head Attention | 多个 Linear 层 + softmax + 矩阵乘 | §2.5 多头 |
| FFN/MLP | §4.2 MLP（2-3 层 + SwiGLU） | §2.6 / §5.4 |
| LayerNorm | §4.8.2 | §2.7 |
| 残差 | §5.1 ResNet 思想 | §2.8 |
| 位置编码 | 不算第 4 章基础 | §2.9 |
| AdamW 训练 | §4.6 | §5.9, §5.10 |
| Cross-Entropy loss | §4.4 | §4.6, §6.3 |
| 反向传播 | §4.5 | （隐式） |

**结论**：
> **Transformer = 第 4 章基础积木 + Attention**。
> 你已学的 Happy-LLM 都被"打开"了——每一层都能从第一性原理理解。

---

## 🆚 5.3.4 Attention vs RNN vs CNN 对比

| 维度 | RNN | CNN | Attention |
|---|---|---|---|
| 长程依赖 | 弱-中 | 弱（要堆层） | **强**（一层即可） |
| 并行 | ❌ 顺序 | ✅ | ✅ |
| 局部归纳偏置 | 中（按时序） | 强（局部连接） | 弱（全位置等价） |
| 参数量 | 少 | 中 | 多 |
| 计算复杂度 | $O(T \cdot d^2)$ | $O(T \cdot d \cdot k)$ | $O(T^2 \cdot d)$ ❌ 长序列贵 |
| 适合 | 短序列 | 图像 | 中长序列、海量数据 |

→ Attention 的最大代价是 **$O(T^2)$**——这就是为什么有 Flash Attention、Linear Attention、Mamba 等优化。

---

## 📊 5.3.5 三大 Transformer 流派

| 流派 | 代表 | 用法 |
|---|---|---|
| **Encoder-only** | BERT | 理解任务（分类、NER） |
| **Decoder-only** | **GPT / LLaMA / Qwen** | **生成任务（你学的 LLM）** |
| **Encoder-Decoder** | T5 / BART / 翻译模型 | seq2seq（翻译、摘要） |

→ 详见 [00-章节总览](../../Happy-LLM/03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)。

→ 现代 LLM 几乎全是 **Decoder-only**（GPT 系列）。

---

## 🚀 5.3.6 现代 Transformer 改进

Happy-LLM §5 你跑过的代码里有这些**现代 LLM 改进**：

| 改进 | 解决什么 | 用在 |
|---|---|---|
| **RMSNorm** | 比 LayerNorm 快 | LLaMA / Qwen |
| **Pre-Norm** | 训练更稳 | 所有现代 LLM |
| **RoPE** 旋转位置编码 | 更好的位置编码 | LLaMA / Qwen |
| **SwiGLU** FFN | 比 ReLU FFN 强 | LLaMA / Qwen |
| **GQA / MQA** | 减少 KV-Cache 内存 | LLaMA-2/3 |
| **Flash Attention** | 加速 attention 计算 | 几乎所有 |

→ 详见 [00-章节总览](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)。

---

## 🔑 5.3.7 你应该带走的 3 件事

1. **Transformer = MLP + Attention + Norm + 残差** —— 你学过的所有东西的组合
2. **Attention 之所以革命**：并行 + 长程依赖一次解决
3. **现代 LLM 几乎全是 Decoder-only Transformer** —— 这是你已学的世界

---

## ⚠️ 5.3.8 常见误解

| 误解 | 真相 |
|---|---|
| "Transformer 取代了所有 DL 架构" | 不！CV 上 CNN 仍流行；时序预测 RNN/Mamba 仍有用 |
| "Attention is all you need" | 论文标题。**实际上 FFN 也很关键**（占 LLM 参数 40%+） |
| "Transformer 必须 encoder + decoder" | 不！现代 LLM 是 decoder-only |
| "BERT 是 LLM" | BERT 严格说是 PLM（预训练语言模型），不是 LLM（generate 任务） |
| "Attention 永远 $O(T^2)$" | Flash / Sparse / Linear Attention 等优化能降到接近 $O(T)$ |
| "T5 已淘汰" | encoder-decoder 在翻译/摘要仍 SOTA |

---

## 📌 5.3 节要点

| 概念 | 一句话 |
|---|---|
| Attention | $\text{softmax}(QK^T/\sqrt{d_k}) V$ |
| Transformer Block | MHA + FFN + 残差 + Norm |
| Encoder-only | BERT，理解任务 |
| Decoder-only | **GPT/LLaMA**，生成任务 |
| 现代 LLM | Decoder-only + RMSNorm + RoPE + SwiGLU + GQA |

**最重要的一句**：
> **你已学过 Transformer，这一节是把它"塞回"DL 基础体系**。
> Transformer = 第 4 章 + 第 5 章的所有积木 + Attention 这个新机制。

---

## 🔗 延伸阅读

- 上一节：[02-RNN-LSTM-GRU](02-RNN-LSTM-GRU.md)
- 下一节：[04-架构发展时间轴](04-%E6%9E%B6%E6%9E%84%E5%8F%91%E5%B1%95%E6%97%B6%E9%97%B4%E8%BD%B4.md)
- 已有锚点（核心）：[00-章节总览](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 已有锚点：[00-章节总览](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 桥接：[01-从MLP到Transformer的桥梁](../07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/01-%E4%BB%8EMLP%E5%88%B0Transformer%E7%9A%84%E6%A1%A5%E6%A2%81.md)

---

⬅ [02-RNN-LSTM-GRU](02-RNN-LSTM-GRU.md) | ➡ [04-架构发展时间轴](04-%E6%9E%B6%E6%9E%84%E5%8F%91%E5%B1%95%E6%97%B6%E9%97%B4%E8%BD%B4.md)
