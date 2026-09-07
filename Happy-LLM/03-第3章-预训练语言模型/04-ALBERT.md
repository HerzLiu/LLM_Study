---
tags: [Happy-LLM, 第3章, PLM, ALBERT, BERT变种]
chapter: 3
section: 3.1.3
---

# 3.1.3 ALBERT —— 把 BERT 压得更小

⬅ [03-RoBERTa](03-RoBERTa.md)　|	➡ [05-T5](05-T5.md)

> 📅 Google 2020 年发布
> 🎯 核心信息：**BERT 太胖了！** 减少参数还能涨点吗？

---

## 🎬 故事比喻：BERT 是大胖子，ALBERT 健身减肥

[RoBERTa](03-RoBERTa.md) 走的是"BERT 没训够，给我使劲喂"路线（更大更狠）；
**ALBERT 反方向走** —— "BERT 太重了，能不能瘦身但不掉性能？"

→ 三招减肥术：
1. **Embedding 参数分解**（瘦词嵌入）
2. **跨层参数共享**（瘦 Encoder）
3. **SOP 任务**（顺带改了下 NSP）

最终 ALBERT-xlarge 用 **59M 参数**（不到 BERT-large 的 1/6）就超过了 BERT-large（340M）。

---

## 🔧 三大优化详解

### 优化 1：Embedding 参数分解（Factorized Embedding）

#### 问题观察

BERT 的 Embedding 层参数 = `V × H`
- `V` = 词表大小 = 30K
- `H` = 隐藏层维度 = 1024（BERT-large）
- → Embedding 参数 = **30M**

而且：**`H` 越大，Embedding 越爆炸**。
比如想做 `H=2048` 的更宽模型，Embedding 直接 **61M**。

#### 关键洞察

> 来自 [Word2Vec 的经验](../01-%E7%AC%AC1%E7%AB%A0-NLP%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/04-%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%E6%BC%94%E8%BF%9B.md#3-word2vec2013mikolov)：
> **词向量根本不需要这么大维度**。Word2Vec 100 维就效果很好。

那为什么要把 Embedding 维度和隐藏层维度绑在一起？

#### 解法：把 Embedding 拆成两步

**原方案**：`V × H` 一个大矩阵
```
[V, H]: 词 id → H 维向量（直接到隐藏层维度）
参数量: V × H
```

**ALBERT 方案**：拆成 `V × E` + `E × H` 两步（E << H）
```
Step 1: [V, E]: 词 id → E 维（小的）
Step 2: [E, H]: E 维 → H 维（线性升维）
参数量: V × E + E × H
```

#### 参数对比

| 设置 | V | E | H | 参数 |
|---|---|---|---|---|
| BERT-large 原版 | 30K | – | 1024 | **30.7M** |
| ALBERT（E=128） | 30K | 128 | 1024 | **3.97M** |
| 压缩比 | – | – | – | **缩到 13%** |

→ 当 `E << H` 时，参数大幅下降。

### 优化 2：跨层参数共享（Cross-layer Parameter Sharing）

#### 问题观察

BERT 有 24 层 Encoder Layer，**每一层都有独立的权重**。
但 ALBERT 团队发现：**各层的参数其实高度一致**（学到的东西很像）！

#### 解法

**只初始化 1 层 Encoder**，但在 forward 时**循环用 24 次**。

```
传统 BERT:
  Layer 1（独立权重）→ Layer 2（独立权重）→ ... → Layer 24（独立权重）

ALBERT:
  Layer 1（共享权重）→ Layer 1（同一组权重）→ ... → Layer 1（同一组权重）
                      （24 次循环）
```

#### 效果

| 模型 | 层数 | 隐藏维度 | 参数量 | 性能 |
|---|---|---|---|---|
| BERT-large | 24 | 1024 | 340M | 基线 |
| **ALBERT-xlarge** | 24（共享） | **2048** | **59M** | 反超 BERT-large |

→ 用瘦身省下的参数预算，**把模型变得更"宽"**（隐藏维度 2048）。

#### ⚠️ 关键缺陷：训练/推理速度没快

- 参数少 ≠ 计算少
- 24 次 Encoder 计算照样要做
- 训练和推理速度只比 BERT 略快一点
- → **这是 ALBERT 没能取代 BERT 的主要原因**

### 优化 3：SOP 任务（Sentence Order Prediction）

#### NSP 的问题（沿袭 RoBERTa 的发现）

| 任务 | 正例 | 负例 | 难度 |
|---|---|---|---|
| **NSP** | 两个连续句子 | 两个**不同文档**的随机句子 | 太简单（主题都不一样） |

模型很容易通过"看主题相似性"就能判断，根本没学到句子关系。

#### ALBERT 的改进：SOP

不直接去掉 NSP，而是**加大难度**：

| 任务 | 正例 | 负例 | 难度 |
|---|---|---|---|
| **SOP** | 两个连续句子 A → B | **同样两句但顺序反过来 B → A** | 难（主题一样，必须真懂句间逻辑） |

#### 例子

```
正例：
  A: I love you.
  B: Because you are wonderful.
  Label: 1

负例（SOP）：
  A: Because you are wonderful.
  B: I love you.
  Label: 0  ← 顺序错了
```

#### 实验结论

> `MLM + SOP` > `MLM` 单独 > `MLM + NSP`

→ **SOP 真的有用**，验证了"句间关系是值得学的，只是 NSP 设计太弱"。

---

## 🆚 BERT vs RoBERTa vs ALBERT 三派对比

| 维度 | BERT | **RoBERTa**（更狠） | **ALBERT**（更瘦） |
|---|---|---|---|
| 架构改动 | – | 无 | Embedding 分解 + 参数共享 |
| 预训练 | MLM + NSP | 只用 MLM | MLM + SOP |
| 数据规模 | 13 GB | 160 GB | 与 BERT 相近 |
| 参数 | 340M | 355M | **59M / 233M** |
| 训练速度 | 基线 | 类似 | ⚠️ 没快 |
| 性能 | 基线 | **更好** | **更好（用更小参数）** |

---

## 📈 ALBERT 的历史意义

1. **证明了"参数共享"可行** —— 后来在很多轻量化场景被借鉴
2. **SOP 思想** —— 对后续设计预训练任务有启发
3. **更宽模型的探索** —— 不只是堆深，也可以堆宽
4. **缺陷暴露：参数少 ≠ 跑得快** —— 这教训提醒后人：**衡量模型成本要看 FLOPs，不是只看参数**

---

## ⚠️ 小白避坑

1. **ALBERT 没能取代 BERT** —— 主要因为"参数减少但速度没快"
2. **参数共享 ≠ 效果完全等价** —— 共享会损失一些表达能力，但 ALBERT 通过加宽弥补
3. **SOP 任务的难度设计哲学** —— 预训练任务**要够难**才能逼模型学到东西
   - 太简单（NSP）：学不到
   - 太难（如随机预测）：学不会
   - SOP 的"主题相同但顺序不同"，难度刚好

---

## 📌 BERT 系小结

[BERT](02-BERT.md) → [RoBERTa](03-RoBERTa.md)（更狠） → ALBERT（更瘦） 是 Encoder-only 派的三大代表作。

但接下来世界发生了变化 ——
**Decoder-only 派开始反攻**，最终在 LLM 时代统治一切。

下一站，让我们走出 BERT 的世界，看看 **T5** 是如何用"大一统"思想想做平衡的：

→ [05-T5](05-T5.md)

---

## 🔗 延伸阅读

- 上一节：[03-RoBERTa](03-RoBERTa.md)
- 下一节：[05-T5](05-T5.md) —— Encoder-Decoder 派的代表
- BERT 系对比：[09-横向对比与选型](09-%E6%A8%AA%E5%90%91%E5%AF%B9%E6%AF%94%E4%B8%8E%E9%80%89%E5%9E%8B.md)

---

⬅ [03-RoBERTa](03-RoBERTa.md)　|	➡ [05-T5](05-T5.md)
