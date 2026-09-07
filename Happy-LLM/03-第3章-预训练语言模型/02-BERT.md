---
tags: [Happy-LLM, 第3章, PLM, BERT, Encoder-only]
chapter: 3
section: 3.1.1
---

# 3.1.1 BERT —— NLU 之王

⬅ [01-PLM三大流派对比](01-PLM%E4%B8%89%E5%A4%A7%E6%B5%81%E6%B4%BE%E5%AF%B9%E6%AF%94.md)　|	➡ [03-RoBERTa](03-RoBERTa.md)

> 📅 Google 2018 年发布
> 📄 论文：《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》
> 🏆 一发布就在 11 个 NLP 任务上拿 SOTA，**预训练-微调范式正式确立**

---

## 🎬 故事比喻：完形填空大师

之前的语言模型（如 GPT-1）训练时是「**给前文，预测下一个词**」 —— 像让你写续作。
BERT 反其道而行 —— 「**给你一段挖了空的文章，请填空**」 —— 这就是高考语文的完形填空！

**为什么完形填空更牛？**
- 写续作只能看前面（**单向**）
- 完形填空既能看左边又能看右边（**双向**） → 上下文理解更深

> 这就是 BERT 名字里 **B**idirectional（双向）的含义。

---

## 🏗 BERT 三大思想沿承

| 来源 | 沿承的思想 |
|---|---|
| [Transformer](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/06-Encoder-Decoder.md) | 全注意力架构，但**只取 Encoder** |
| [ELMo](../01-%E7%AC%AC1%E7%AB%A0-NLP%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/04-%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%E6%BC%94%E8%BF%9B.md#%E6%96%B9%E6%A1%88-c%E5%8A%A8%E6%80%81%E7%94%BB%E5%83%8Felmo) | 预训练 + 微调范式、双向上下文 |
| 自己的创新 | **MLM + NSP** 预训练任务 |

---

## 🔧 模型架构：Encoder-Only

### 整体结构

```
输入文本 ──→ Tokenizer ──→ input_ids
                              │
                              ▼
                        Embedding 层
                              │
                              ▼
                ┌─────────────────────────┐
                │ Encoder Layer × N        │
                │  ┌────────────────────┐  │
                │  │ Multi-Head Attn   │  │
                │  │ + Add & Norm      │  │
                │  │ Feed Forward      │  │
                │  │ + Add & Norm      │  │
                │  └────────────────────┘  │
                └─────────────────────────┘
                              │
                              ▼
                      prediction_heads
                       （任务特定头）
                              │
                              ▼
                       任务输出（如分类标签）
```

### 两种规格

| 版本 | 层数 | 隐藏维度 | 头数 | 参数量 |
|---|---|---|---|---|
| **BERT-base** | 12 | 768 | 12 | **110M** |
| **BERT-large** | 24 | 1024 | 16 | **340M** |

### 关键工程细节

- **Tokenizer**：用 [WordPiece](../01-%E7%AC%AC1%E7%AB%A0-NLP%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/03-NLP%E4%B9%9D%E5%A4%A7%E4%BB%BB%E5%8A%A1.md#%E2%91%A1-%E5%AD%90%E8%AF%8D%E5%88%87%E5%88%86subword--%E9%87%8D%E8%A6%81)（中文按字切分）
- **激活函数**：**GELU**（高斯误差线性单元）—— BERT 让 GELU 火了起来
  - GELU(x) = x · Φ(x)，把"是否激活"建模成概率
- **位置编码**：用**可学习**的位置 embedding（不是 [sin/cos](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/09-%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)）
  - 优点：能拟合更丰富的位置关系
  - 缺点：**最大长度被限死 512**，无法外推
- **特殊 token**：`[CLS]` 在句首，`[SEP]` 分隔句子，`[MASK]` 用于遮蔽

---

## 🎯 预训练任务（核心创新）

### 任务 1：MLM（Masked Language Model）—— 完形填空

#### 思路
随机遮蔽部分 token，让模型根据上下文预测。

```
输入：I [MASK] you because you are [MASK]
输出：[MASK] → love;  [MASK] → wonderful
```

#### ⭐ 加深理解：为什么不用传统 LM？

传统 LM（如 GPT 用的 CLM）"看前预测后"，只能单向。
但 NLU 任务（如理解、分类）需要**理解整句**，应该既看左又看右。
→ MLM 让模型**双向编码**信息，理解能力大增。

#### 关键技巧：15% 遮蔽 + 80/10/10 策略

随机选 **15%** 的 token 来遮蔽，但**不全部替换为 [MASK]**：

| 比例  | 操作            | 动机                             |
| --- | ------------- | ------------------------------ |
| 80% | 替换成 [MASK]    | 主要训练信号                         |
| 10% | 替换成随机其他 token | 强迫模型对每个位置都保持注意（不能只盯 [MASK]）    |
| 10% | 保持原 token 不变  | 缓解"训练有 [MASK]、推理无 [MASK]" 的不一致 |

> ⚠️ **小白避坑**：MLM 的最大缺陷是**预训练-微调不一致**（推理时根本没有 [MASK]）。后来的 Decoder-only 模型用 CLM 就完全没这个问题。

### 任务 2：NSP（Next Sentence Prediction）—— 下一句预测

#### 思路
判断两个句子是否是连续的上下文。

```
正例：
  A: I love you.
  B: Because you are wonderful.
  Label: 1（是上下文）

负例：
  A: I love you.
  B: Because today's dinner is so nice.
  Label: 0（不是上下文）
```

#### 动机
MLM 是 **token 级**任务，但很多下游任务是 **句子级**（问答匹配、自然语言推理）。
NSP 强迫模型学**句间关系**。

#### ⚠️ NSP 的争议
- 后来 RoBERTa 实验证明 NSP **作用不大甚至有害**
- ALBERT 改进为 SOP（句子顺序预测）
- → 见 [03-RoBERTa](03-RoBERTa.md) 和 [04-ALBERT](04-ALBERT.md)

---

## 📊 预训练数据与计算量

| 项 | 数值 |
|---|---|
| 数据 | BooksCorpus（800M 词）+ 英文维基百科（2500M 词），共 **13GB / 3.3B token** |
| Batch size | 256 |
| 训练步数 | 1M 步（约 40 个 epoch） |
| 序列长度 | 90% 用 128，最后 10% 用 512 |
| 硬件 | BERT-base：16 块 TPU；BERT-large：64 块 TPU |
| 训练时长 | 约 4 天 |

> 💡 **对比**：现代 LLM 一般只训 **1 个 epoch**，但用 **远大于 256 的 batch size** 和 **数万亿 token** 的数据。这反映了一个趋势：**数据规模 >> 训练轮次**。

---

## 🎯 下游任务微调

### 微调范式

> **预训练（一次性，烧钱）→ 微调（每个任务一次，轻量）**

针对不同下游任务，只需替换/微调最顶层的 `prediction_heads`：

| 任务 | 怎么用 BERT |
|---|---|
| **文本分类** | 取 `[CLS]` 的输出 → 接分类头 |
| **NER（序列标注）** | 取每个 token 的输出 → 接分类头 |
| **句子相似度** | 取两个句子的 `[CLS]` → 算余弦相似度 |
| **问答（SQuAD）** | 预测答案的起止位置 |

### `[CLS]` 的妙用
- BERT 在每个输入开头加 `[CLS]` 特殊 token
- 经过 N 层 Encoder 后，`[CLS]` 位置的向量 = **整句的语义表示**
- → 句级任务（如分类）直接用它

---

## 📈 BERT 的历史意义

1. **正式确立预训练-微调范式** —— 之后所有 PLM 都遵循
2. **NLU 任务王者** —— 11 个 SOTA 是降维打击
3. **开启 Encoder-only 流派** —— 后续 RoBERTa、ALBERT、ERNIE、DeBERTa 都是其变种
4. **直到 LLM 时代才被部分替代** —— 但在 embedding / 检索 / 分类等场景仍然不可或缺

---

## ⚠️ 小白避坑

1. **BERT 不能生成文本**
   - Encoder-only 架构没有自回归能力
   - 让 BERT 写作文是不行的（虽然有人魔改尝试过）
2. **BERT 的最大输入长度是 512**
   - 因为用了可学习位置编码
   - 处理长文档要分段
3. **MLM 是"完形填空"，CLM 是"写续作"**
   - 这是 BERT 和 GPT 的本质区别
   - 一句话理解：BERT 像高考语文，GPT 像写小说

---

## 🔗 延伸阅读

- 上一节：[01-PLM三大流派对比](01-PLM%E4%B8%89%E5%A4%A7%E6%B5%81%E6%B4%BE%E5%AF%B9%E6%AF%94.md)
- 下一节：[03-RoBERTa](03-RoBERTa.md) —— 把 BERT 训得更狠
- 对比对象：[06-GPT](06-GPT.md) —— Decoder-only 派
- 横向对比：[09-横向对比与选型](09-%E6%A8%AA%E5%90%91%E5%AF%B9%E6%AF%94%E4%B8%8E%E9%80%89%E5%9E%8B.md)

---

⬅ [01-PLM三大流派对比](01-PLM%E4%B8%89%E5%A4%A7%E6%B5%81%E6%B4%BE%E5%AF%B9%E6%AF%94.md)　|	➡ [03-RoBERTa](03-RoBERTa.md)
