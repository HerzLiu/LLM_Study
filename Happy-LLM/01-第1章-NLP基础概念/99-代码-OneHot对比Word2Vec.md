---
tags: [Happy-LLM, 第1章, 代码, Python]
chapter: 1
section: 1.4-code
---

# 💻 代码 Demo：One-Hot vs Word2Vec

⬅ [04-文本表示演进](04-%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%E6%BC%94%E8%BF%9B.md)　|	➡ [进入第 2 章](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

> 🎯 **目标**：用最小可运行 demo，让你**亲眼看到** One-Hot 和 Embedding 在"语义相似度"上的差别。

---

## 🐍 完整代码

```python
# ============ 准备工作 ============
# numpy 是科学计算库，处理向量/矩阵的标配
import numpy as np

# ============ 1. One-Hot 表示 ============
# 假设"词汇表"只有 5 个词（实际可能有几万个）
vocab = ["猫", "狗", "汽车", "飞机", "跑步"]

# 给每个词建立索引（word -> id）
word2id = {w: i for i, w in enumerate(vocab)}
# 此时 word2id = {"猫":0, "狗":1, "汽车":2, "飞机":3, "跑步":4}

def one_hot(word):
    """把词转成 One-Hot 向量"""
    vec = np.zeros(len(vocab))   # 先造一个全 0 的 5 维向量
    vec[word2id[word]] = 1       # 把该词对应位置改成 1
    return vec

print("猫的 One-Hot:", one_hot("猫"))   # [1. 0. 0. 0. 0.]
print("狗的 One-Hot:", one_hot("狗"))   # [0. 1. 0. 0. 0.]

# ⚠️ 关键观察：用余弦相似度看"猫"和"狗"像不像
def cos_sim(a, b):
    """余弦相似度：1=完全一样，0=完全无关，-1=相反"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("One-Hot 下 猫vs狗:",   cos_sim(one_hot("猫"), one_hot("狗")))   # 0.0 !
print("One-Hot 下 猫vs汽车:", cos_sim(one_hot("猫"), one_hot("汽车"))) # 0.0 !
# 🚨 问题：猫狗、猫车，相似度都是 0，模型看不出"猫狗都是动物"

# ============ 2. Word2Vec 风格的稠密向量（手动模拟）============
# 真实场景下，这些向量是 Word2Vec 从海量语料学出来的
# 这里手动给一组"假装学好"的向量（3 维）
embedding = {
    "猫":   np.array([0.8, 0.1, 0.0]),   # 动物维度高、交通维度低
    "狗":   np.array([0.7, 0.2, 0.0]),   # 和猫接近
    "汽车": np.array([0.0, 0.1, 0.9]),   # 交通维度高
    "飞机": np.array([0.1, 0.0, 0.85]),  # 和汽车接近
    "跑步": np.array([0.4, 0.6, 0.1]),   # 动作类
}

print("\nEmbedding 下 猫vs狗:",   cos_sim(embedding["猫"], embedding["狗"]))
# 结果接近 1，说明猫和狗"语义相似"
print("Embedding 下 猫vs汽车:", cos_sim(embedding["猫"], embedding["汽车"]))
# 结果接近 0，说明猫和汽车不像
```

---

## 📝 代码解读（中等深度）

### 关键行 1：字典推导式
```python
word2id = {w: i for i, w in enumerate(vocab)}
```
- 这是 **Python 字典推导式**，等价于：
  ```python
  word2id = {}
  for i, w in enumerate(vocab):
      word2id[w] = i
  ```
- `enumerate` 同时给出**索引**和**元素**，非常常用，**一定要记住**。

### 关键行 2：创建零向量
```python
vec = np.zeros(len(vocab))
```
- `np.zeros(N)` 创建一个长度为 N、全是 0 的数组。
- 这是构造 One-Hot 的标准套路。

### 关键行 3：余弦相似度公式

$$\text{cos\_sim}(\vec{a},\vec{b}) = \frac{\vec{a}\cdot\vec{b}}{\|\vec{a}\|\cdot\|\vec{b}\|}$$

- `np.dot(a, b)`：向量内积
- `np.linalg.norm(a)`：向量长度（L2 范数）

**为什么用余弦而不用欧氏距离？**
→ 余弦**只关心方向、不关心长度**，更符合"语义相似"的直觉。

---

## ⭐ 关键观察：为什么 One-Hot 下相似度永远是 0？

```
猫:   [1, 0, 0, 0, 0]
狗:   [0, 1, 0, 0, 0]
```

两个向量的内积 = `1×0 + 0×1 + 0×0 + 0×0 + 0×0 = 0`

→ **任意两个不同词的 One-Hot 相似度必然为 0**，因为它们的"1"在不同位置。

而 Embedding 下：
```
猫:   [0.8, 0.1, 0.0]
狗:   [0.7, 0.2, 0.0]
```
内积 = `0.8×0.7 + 0.1×0.2 + 0 = 0.58`（不为 0，体现了相似性）

**这就是 Word2Vec 革命性的地方** ↑

---

## ⚠️ 小白容易踩的坑

1. **`np.array` vs `list`**
   - 列表 `[1,2,3]` **不能**直接做向量运算
   - 必须先 `np.array([1,2,3])` 转成 numpy 数组
2. **维度对不齐**
   - `np.dot` 要求两个向量长度一样，否则报错
3. **真实 Word2Vec 不是手写的**
   - 用 `gensim` 库训练：
     ```python
     from gensim.models import Word2Vec
     model = Word2Vec(
         sentences=[["猫","抓","老鼠"], ["狗","看","家"]],
         vector_size=100
     )
     print(model.wv["猫"])  # 100 维向量
     ```

---

## 🔬 进阶练习（推荐）

跑完上面代码后，试试：

1. 把 `vocab` 扩到 20 个词，验证 One-Hot 维度爆炸
2. 用 `gensim` 在中文小语料（如《三国演义》前 10 章）上训练 Word2Vec，看 "刘备" 和 "曹操" 的相似度
3. 用 `gensim` 的 `model.wv.most_similar("刘备")` 看最接近的词

---

## 🔗 延伸阅读

- 理论篇：[04-文本表示演进](04-%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%E6%BC%94%E8%BF%9B.md)
- 章节总览：[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一章：[00-章节总览](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

⬅ [04-文本表示演进](04-%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%E6%BC%94%E8%BF%9B.md)　|	➡ [进入第 2 章](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
