---
tags: [Happy-LLM, 第2章, Transformer, 归一化]
chapter: 2
section: 2.2.3
---

# 2.2.3 Layer Norm vs Batch Norm

⬅ [06-Encoder-Decoder](06-Encoder-Decoder.md)　|	➡ [08-残差连接](08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md)

---

## 🎬 故事比喻：班级考试后，按班归一化分数

不同班难度不同，直接比绝对分数不公平。
**做法**：每个班把成绩归一化到均值 0、方差 1，再比较。

神经网络层数一多，每层输出的数值分布会"漂移"（**internal covariate shift**），训练不稳定。
**归一化就是把它们拉回标准分布。**

---

## 🆚 LayerNorm vs BatchNorm 核心对比

| 对比项 | BatchNorm | **LayerNorm** |
|---|---|---|
| **归一化维度** | 同一特征，跨样本 | 同一样本，跨特征 |
| **依赖 batch 大小** | 是（batch 小则不稳） | ❌ 否 |
| **适合 RNN/Transformer？** | ❌ 不适合（变长序列） | ✅ 适合 |
| **训练/推理是否一致？** | 不一致（需保存统计量） | ✅ 一致 |

### 直观差别

假设有 4 个样本，每个样本是 3 维向量：

```
样本1: [1.0, 2.0, 3.0]
样本2: [2.0, 4.0, 6.0]
样本3: [0.5, 1.5, 2.5]
样本4: [3.0, 1.0, 5.0]
```

- **BatchNorm**：对每一列（同一特征，跨样本）求均值方差归一化
  ```
  对第 1 列 [1.0, 2.0, 0.5, 3.0] 求 μ 和 σ
  ```
- **LayerNorm**：对每一行（同一样本，跨特征）求均值方差归一化
  ```
  对样本 1 [1.0, 2.0, 3.0] 求 μ 和 σ
  ```

---

## ⭐ 为什么 NLP 选 LayerNorm？

### BatchNorm 在 NLP 的死穴

1. **batch 小时不稳**：显存有限时 mini-batch 只有几个样本，统计量不准
2. **变长序列**：句子有长有短，不同位置参与统计的样本数不同
3. **训练/推理不一致**：训练时用 batch 统计量，推理时用滑动平均，可能差异大
4. **每 step 都要存统计量**：耗时耗内存

### LayerNorm 完美适配

- ✅ 和 batch 大小无关
- ✅ 和序列长度无关
- ✅ 训练推理完全一致
- ✅ 简单高效

---

## 🔧 LayerNorm 公式

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

| 符号 | 含义 |
|---|---|
| `μ, σ` | 当前样本所有特征的均值/标准差 |
| `γ, β` | **可学习**的缩放和偏移参数（让模型自己决定要不要保留归一化） |
| `ε` | 极小数（如 1e-6）防止除 0 |

---

## 🐍 代码实现

```python
class LayerNorm(nn.Module):
    """Layer Norm 层"""
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # 可学习参数 γ 和 β
        self.a_2 = nn.Parameter(torch.ones(features))   # γ 初始化为 1
        self.b_2 = nn.Parameter(torch.zeros(features))  # β 初始化为 0
        self.eps = eps

    def forward(self, x):
        # 在最后一维（特征维）求均值方差
        # keepdim=True：保留维度，方便后面广播
        mean = x.mean(-1, keepdim=True)   # shape: (..., 1)
        std = x.std(-1, keepdim=True)     # shape: (..., 1)
        # 归一化 + 仿射变换
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

### 📝 代码解读

- **`nn.Parameter`**：包起来的张量会被自动加入模型的**可学习参数列表**
- **`keepdim=True`**：求均值时一定要加，否则维度对不齐没法广播
- **`γ, β` 的设计哲学**：归一化把所有维度强行拉到同一分布，可能丢失有用信息。`γ` 和 `β` 让模型自己学"要不要还原一部分"

---

## 🔬 进阶：RMSNorm（LLaMA 用的简化版）

> 现代 LLM（LLaMA、Qwen 等）实际用的是 **RMSNorm**，是 LayerNorm 的简化：

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}}$$

**区别**：
- 去掉了减均值（`x - μ`）
- 只用 RMS（均方根）做归一化
- 速度更快，效果几乎一样

→ 第 5 章实现 LLaMA2 时会用到。

---

## ⚠️ 小白避坑

1. **不要把 LayerNorm 放在错误的维度上**
   - 对 `(B, L, D)` 的输入，应该在 `D` 维（最后一维）做归一化
2. **`γ` 初始化为 1，`β` 初始化为 0**
   - 这样初始时 LayerNorm 等价于恒等映射，训练更稳
3. **LayerNorm 和 Dropout 顺序**：Dropout 通常在 LayerNorm **之后**

---

## 🔗 延伸阅读

- 上一节：[06-Encoder-Decoder](06-Encoder-Decoder.md)
- 下一节：[08-残差连接](08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md) —— 另一个"训练救星"
- Pre-Norm vs Post-Norm：[08-残差连接](08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md#-pre-norm-vs-post-norm)

---

⬅ [06-Encoder-Decoder](06-Encoder-Decoder.md)　|	➡ [08-残差连接](08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md)
