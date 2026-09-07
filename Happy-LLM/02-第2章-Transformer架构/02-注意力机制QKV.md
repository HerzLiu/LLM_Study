---
tags: [Happy-LLM, 第2章, Transformer, 注意力, 核心概念]
chapter: 2
section: 2.1.2
---

# 2.1.2 注意力机制：QKV 三件套

⬅ [01-为什么需要注意力](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E6%B3%A8%E6%84%8F%E5%8A%9B.md)　|	➡ [03-自注意力](03-%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md)

> ⭐ 这是全章最绕但最重要的一节，请慢慢读。**理解了 QKV，就理解了 Transformer**。

---

## 🎬 故事比喻：图书馆查资料

你去图书馆，想找"水果相关的书"。

- **Query（查询）**：你要找的东西 —— "水果"
- **Key（键）**：每本书的标题/标签 —— "苹果"、"香蕉"、"椅子"
- **Value（值）**：每本书的实际内容 —— 苹果资料、香蕉资料、椅子资料

**你怎么找？**
1. 拿着 Query "水果"，去对照每本书的 Key
2. 算出**相关度**：
   - 水果 ↔ 苹果 = 高分
   - 水果 ↔ 香蕉 = 高分
   - 水果 ↔ 椅子 = 0 分
3. 按相关度**加权**取出 Value：60% 苹果资料 + 40% 香蕉资料 + 0% 椅子资料
4. 这就是你最终得到的"水果"答案

**这就是注意力机制的本质**：
> **Query 和 Key 算相似度，按相似度加权取 Value。**

---

## 🔧 技术拆解：从字典到注意力公式（6 步推导）

### Step 1：精确字典（不用注意力）

```python
dict = {"apple": 10, "banana": 5, "chair": 2}
dict["apple"]  # 直接返回 10
```

只能精确匹配，没有"模糊查找"的能力。

### Step 2：模糊查询（注意力雏形）

`Query = "fruit"`，没有精确匹配怎么办？
**人工赋权重**：
- apple → 0.6
- banana → 0.4
- chair → 0.0

答案 = `0.6 × 10 + 0.4 × 5 + 0.0 × 2 = 8`

### Step 3：自动算权重

怎么让模型**自动**算出"fruit 和 apple 相关度高、和 chair 相关度低"？
→ **用词向量点积**衡量相似度。

> 来自 [词向量](../01-%E7%AC%AC1%E7%AB%A0-NLP%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/04-%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%E6%BC%94%E8%BF%9B.md#%E6%96%B9%E6%A1%88-b%E6%80%A7%E6%A0%BC%E7%94%BB%E5%83%8Fword2vec)的性质：
> 语义相近的词，向量点积大；语义远的，点积小（甚至负数）。

$$x = q \cdot K^T$$

- `q` 是 fruit 的词向量
- `K` 是所有 Key 词向量堆成的矩阵
- `x` 就是 fruit 和每个 Key 的相似度

### Step 4：归一化成概率

点积值是**任意实数**，但我们要的是"权重"（相加 = 1）。
→ 过一个 **Softmax**：

$$\text{weights} = \text{softmax}(x) = \text{softmax}(q K^T)$$

### Step 5：加权 Value

$$\text{Attention}(q, K, V) = \text{softmax}(q K^T) \cdot V$$

### Step 6：批量化 + 缩放（最终公式）

一次查多个 Query（堆成矩阵 `Q`），并除以 `√d_k` 防梯度爆炸：

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V}$$

---

## ⭐ 加深理解：为什么要除以 √d_k？

> 这是面试高频题，必懂。

- `d_k` 是 Key 向量的维度（比如 64）
- 当 `d_k` 大时，`QK^T` 的值会**非常大**（统计上方差正比于 `d_k`）
- Softmax 输入一旦很大，输出会变成"几乎独热"（大的接近 1，小的接近 0）
- **这种情况下 Softmax 的梯度几乎为 0**（饱和区），模型学不动

**结论**：
> 除以 √d_k 把方差拉回 1 附近，让 Softmax 工作在敏感区，**梯度健康**。

### 数学直觉

假设 Q、K 各分量独立、均值 0、方差 1。
则 `q · k = Σ q_i × k_i`，方差 = `d_k`（独立和的方差等于方差和）。
除以 `√d_k` 后，方差回到 1。

---

## 🐍 代码实现

```python
import torch
import math

def attention(query, key, value, dropout=None):
    """
    注意力计算函数
    args:
        query: 查询矩阵 Q, shape = (..., seq_len_q, d_k)
        key:   键矩阵 K,  shape = (..., seq_len_k, d_k)
        value: 值矩阵 V,  shape = (..., seq_len_k, d_v)
    """
    # 1. 取出 key 的最后一维（每个 key 向量的维度 d_k）
    d_k = query.size(-1)

    # 2. 计算 Q · K^T，再除以 √d_k
    #    key.transpose(-2, -1) 把最后两维交换 → (..., d_k, seq_len_k)
    #    matmul 后 shape = (..., seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 3. Softmax 沿最后一维 → 每行加起来 = 1，得到注意力权重
    p_attn = scores.softmax(dim=-1)

    # 4. 训练时随机扔掉一些权重，防止过拟合
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 5. 用权重对 V 加权求和 → 最终输出
    #    shape = (..., seq_len_q, d_v)
    return torch.matmul(p_attn, value), p_attn
```

### 📝 代码解读（中等深度）

**关键行 1：`d_k = query.size(-1)`**
- `.size(-1)` 取最后一维大小
- 用 -1 而不写死是因为同一个函数要适配 2D / 3D / 4D tensor

**关键行 2：`key.transpose(-2, -1)`**
- 交换最后两个维度，把 `(B, L, D)` 变成 `(B, D, L)`
- 这样和 query 做矩阵乘法时维度对得上
- ⚠️ 不要用 `.T`！`.T` 只对 2D 矩阵安全

**关键行 3：`scores.softmax(dim=-1)`**
- `dim=-1` 表示沿最后一维做 softmax
- 输出 `p_attn` 的 shape = `(B, seq_q, seq_k)`
- 第 `(i, j)` 个值代表"query 第 i 个位置应给 key 第 j 个位置多少权重"

**关键行 4：`torch.matmul(p_attn, value)`**
- 用权重加权 V
- shape 变化：`(B, seq_q, seq_k) × (B, seq_k, d_v) → (B, seq_q, d_v)`

---

## ⚠️ 小白避坑

1. **QKV 不是固定的数据**
   - 它们是**从同一个输入算出来的**
   - 在[自注意力](03-%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md)里：Q = K = V = 同一输入 × 不同的权重矩阵
2. **注意力不是"看哪里"，是"加权融合"**
   - 输出的不是位置坐标，是一个新向量
3. **`d_k` 通常等于 `d_v`**（虽然理论可以不等）

---

## 🔗 延伸阅读

- 上一节：[01-为什么需要注意力](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E6%B3%A8%E6%84%8F%E5%8A%9B.md)
- 下一节：[03-自注意力](03-%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md) —— Q=K=V 的特殊情况
- 完整应用：[05-多头注意力](05-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md)、[99-代码-MultiHeadAttention](99-%E4%BB%A3%E7%A0%81-MultiHeadAttention.md)

---

⬅ [01-为什么需要注意力](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E6%B3%A8%E6%84%8F%E5%8A%9B.md)　|	➡ [03-自注意力](03-%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md)
