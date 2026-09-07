---
tags: [Hello-Agents, 第3章, LLM, 架构, 速览]
chapter: 3
section: 3.1
---

# 3.1 LLM 架构速览(N-gram → Transformer → Decoder-Only)

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)	|	➡ [02-与LLM交互](02-%E4%B8%8ELLM%E4%BA%A4%E4%BA%92.md)

> ⚠️ **本节是"速览"** —— 想深入学每一步的原理,**直接去 [Happy-LLM](../../Happy-LLM/00-%E6%80%BB%E8%A7%88.md)**。
> 这里只给**做 Agent 必备的关键骨架**。

---

## 🎬 故事比喻:语言模型的三次进化

让计算机理解人话,经过了三大进化:

| 阶段 | 比喻 | 核心思想 |
|---|---|---|
| **N-gram**(统计时代) | **背词典**的人 | 数"哪些词常一起出现" |
| **RNN / LSTM**(神经网络早期) | **传话筒**的人 | 边读边记,但记不远 |
| **Transformer**(2017+) | **会议室开会**的人 | 所有词同时互相看(注意力) |
| **Decoder-Only**(GPT 系列) | **写小说**的人 | 看前文,预测下一个词 |

---

## 📚 3.1.1 N-gram 到 RNN(已在 Happy-LLM 详讲)

### 🚀 N-gram(统计语言模型)

核心:**一个词的概率只看它前面 N-1 个词**。
公式:`P(word_n | word_1...word_{n-1}) ≈ P(word_n | word_{n-N+1}...word_{n-1})`

致命缺陷:
- **数据稀疏**(没见过的组合概率为 0)
- **泛化差**(无法理解"agent" 和 "robot" 在语义上类似)

→ **完整讲解** + 代码示例:[Happy-LLM 1.4 N-gram](../../Happy-LLM/01-%E7%AC%AC1%E7%AB%A0-NLP%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/04-%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%E6%BC%94%E8%BF%9B.md#2-n-gram-%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-ngram)

### 🧠 词嵌入(Word Embedding)革命

**Bengio 2003**:用**连续向量**表示词,让语义相近的词向量也相近。
→ 经典例子:`vec(国王) - vec(男) + vec(女) ≈ vec(女王)`

→ **完整讲解**(含可视化代码):[Happy-LLM Word2Vec](../../Happy-LLM/01-%E7%AC%AC1%E7%AB%A0-NLP%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/04-%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%E6%BC%94%E8%BF%9B.md#3-word2vec2013mikolov)

### 🔁 RNN / LSTM

**核心创新**:引入**隐藏状态**(hidden state)作为"短期记忆"
**致命问题**:**长距离依赖** + **梯度消失/爆炸** + **无法并行**

LSTM 通过**门控机制**(遗忘门 / 输入门 / 输出门)缓解了长依赖,但**没解决并行问题**。

→ **完整讲解**:[Happy-LLM 2.1.1](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E6%B3%A8%E6%84%8F%E5%8A%9B.md)

---

## ⚡ 3.1.2 Transformer 架构(深度跳 Happy-LLM)

### 一句话理解 Transformer

> **彻底抛弃循环结构,完全用注意力 + 并行计算**。

### 整体架构(Encoder + Decoder)

```
   输入 ─→ Encoder 块×N ─→ 编码后的表示
                            ↓
   输出 ←─ Decoder 块×N ←── 与上面交叉注意力
```

### 5 大核心组件(完整讲解都在 Happy-LLM 第 2 章)

| 组件 | 作用 | Happy-LLM 详解 |
|---|---|---|
| **多头自注意力**(MHA) | 并行从多视角理解序列 | [2.1.5](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/05-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md) |
| **位置编码**(PE) | 注入位置信息 | [2.3.2](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/09-%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md) |
| **前馈网络**(FFN) | 提取高阶特征 | [2.2](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/06-Encoder-Decoder.md) |
| **残差连接 + LayerNorm** | 稳定训练 | [2.2.4](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md) |
| **掩码自注意力** | Decoder 防"偷看未来" | [2.1.4](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/04-%E6%8E%A9%E7%A0%81%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md) |

### ⭐ 最关键的注意力公式(必背)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**直觉**:Query 跟所有 Key 算"相关度",然后按相关度**加权聚合** Value。

→ 详细推导 + 代码:[Happy-LLM 2.1.2](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/02-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6QKV.md)

---

## 🎯 3.1.3 Decoder-Only(现代 LLM 的统治架构)

### 关键转折:GPT 的简化

> **原版 Transformer**:Encoder + Decoder(2 套权重 + 交叉注意力,复杂)
> **GPT (Decoder-Only)**:**只保留 Decoder**,**抛弃 Encoder**

### 核心思想:**自回归(Autoregressive)**

```
Step 1: 输入 "Datawhale Agent is"
        ↓ 模型预测
Step 2: 输出 "a"
        ↓ 加到输入
Step 3: 输入 "Datawhale Agent is a"
        ↓ 模型预测
Step 4: 输出 "powerful"
        ↓ ......循环
```

→ **就像"文字接龙"**,每次预测下一个最可能的词。

### 为什么 Decoder-Only 赢了?

| 优势 | 解释 |
|---|---|
| **训练目标统一** | 永远是"预测下一个词",适合海量无标注训练 |
| **结构简单** | 易规模化,千亿参数也能扛 |
| **天然适合生成** | 对话/写作/代码/推理都是"生成"的形式 |

→ 更深刻的"为什么 Decoder-Only 赢了":[Happy-LLM 3.0.4](../../Happy-LLM/03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/01-PLM%E4%B8%89%E5%A4%A7%E6%B5%81%E6%B4%BE%E5%AF%B9%E6%AF%94.md#-%E5%8A%A0%E6%B7%B1%E7%90%86%E8%A7%A3%E4%B8%BA%E4%BB%80%E4%B9%88-decoder-only-%E6%9C%80%E7%BB%88%E8%B5%A2%E4%BA%86)

### Decoder-Only 的"灵魂"——掩码自注意力

为什么不会"偷看"未来?
→ 在 softmax 前,**把未来位置的分数置为负无穷**,softmax 后就是 0。

→ 详细代码:[Happy-LLM 2.1.4](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/04-%E6%8E%A9%E7%A0%81%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md)

---

## 🤔 做 Agent 该懂多少 LLM 架构?

| 你想干啥                       | 该懂的程度                                        |
| -------------------------- | -------------------------------------------- |
| **用 Agent 框架做应用**          | 知道 LLM 是 Decoder-Only Transformer 就够         |
| **微调 LLM 适配 Agent 场景**     | 必懂 attention + Decoder 架构(看 Happy-LLM 第 2 章) |
| **自己造 LLM**                | 看完整本 Happy-LLM,手撕 LLaMA2                     |
| **训 Agentic-RL**(本书第 11 章) | 至少懂 Decoder-Only + RLHF                      |

---

## ⚠️ 小白避坑

1. **别陷入数学细节**
   - 做 Agent 不需要懂反向传播
   - 但**必须懂 Decoder 自回归 + 掩码机制**
2. **架构不是越复杂越好**
   - Decoder-Only **大道至简**
   - **简单架构 + 海量数据 + 大规模算力** = 现代 LLM
3. **N-gram 没死**
   - 输入法、拼写纠错仍在用
   - **学了就懂为什么"快速但浅薄"**

---

## 📌 3.1 节要点(必背)

| 知识点 | 一句话 |
|---|---|
| N-gram | 统计模型,**稀疏 + 泛化差** |
| Transformer | **注意力 + 并行计算**,2017 年革命 |
| 注意力公式 | `softmax(QK^T/√d_k) V` |
| Decoder-Only | **GPT 的简化** —— 只保留 Decoder,自回归生成 |
| 掩码自注意力 | **防止偷看未来**,Decoder-Only 的灵魂 |

---

## 🔗 延伸阅读

- 上一节:[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一节:[02-与LLM交互](02-%E4%B8%8ELLM%E4%BA%A4%E4%BA%92.md) —— **Agent 强相关,精读**
- **完整 Transformer 学习**:[Happy-LLM 第 2 章](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- **完整预训练模型学习**:[Happy-LLM 第 3 章](../../Happy-LLM/03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)	|	➡ [02-与LLM交互](02-%E4%B8%8ELLM%E4%BA%A4%E4%BA%92.md)
