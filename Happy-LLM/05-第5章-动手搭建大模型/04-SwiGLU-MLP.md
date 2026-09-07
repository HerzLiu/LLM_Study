---
tags: [Happy-LLM, 第5章, 实战, SwiGLU, MLP, FFN]
chapter: 5
section: 5.1.4
---

# 5.1.4 构建 LLaMA2 MLP（SwiGLU）

⬅ [03-GQA与RoPE](03-GQA%E4%B8%8ERoPE.md)	|	➡ [05-DecoderLayer组装](05-DecoderLayer%E7%BB%84%E8%A3%85.md)

---

## 🎬 故事比喻：双通道带"开关"的 FFN

普通 FFN（[GPT/BERT 用的](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/06-Encoder-Decoder.md#-%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9Cffn)）：
- 升维 → 激活 → 降维
- 一条直路

**SwiGLU** 是带"门控"的版本：
- 升维有**两条路**：一条过 SiLU 激活、一条不过
- 两条路**相乘** = 门控（一条像"开关"控制另一条的强度）
- 降维输出

直觉上：
> 第一条像水龙头 —— 决定**水量多大**
> 第二条像阀门 —— 决定**水开还是关**
> 两者结合 = **更精细的信息流控制**

---

## 🔧 数学公式

普通 FFN：
$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x)$$

**SwiGLU**：
$$\text{SwiGLU}(x) = W_2 \cdot \big(\text{SiLU}(W_1 x) \odot (W_3 x)\big)$$

其中：
- `⊙` 是逐元素相乘
- $\text{SiLU}(x) = x \cdot \sigma(x)$（也叫 Swish）

→ **多了一组 `W_3` 矩阵 + 一个元素乘法**

---

## 🐍 完整代码

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没指定隐藏维度，按 LLaMA 公式自动计算
        if hidden_dim is None:
            hidden_dim = 4 * dim                                # 升 4 倍
            hidden_dim = int(2 * hidden_dim / 3)                # 缩 2/3（SwiGLU 经验）
            hidden_dim = multiple_of * (
                (hidden_dim + multiple_of - 1) // multiple_of   # 对齐到 multiple_of 倍数
            )

        # 三个线性变换（注意：没有 bias）
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # 升维 1（接 SiLU）
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # 降维
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # 升维 2（门控）

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SiLU(W1 x) ⊙ (W3 x) → W2 → Dropout
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

---

## 📝 代码深度解读

### 关键 1：三个矩阵 `w1, w2, w3`

| 矩阵 | shape | 作用 |
|---|---|---|
| `w1` | (dim, hidden_dim) | 升维 + 后接 SiLU 激活 |
| `w3` | (dim, hidden_dim) | 升维 + 不过激活，作为门控信号 |
| `w2` | (hidden_dim, dim) | 降维回到 dim |

比起普通 FFN（只有 `w1, w2`），**SwiGLU 多了一个 `w3`**。

### 关键 2：`hidden_dim` 为什么要 ×4 再 ×2/3？

```python
hidden_dim = 4 * dim                  # 经验：FFN 中间层一般是 4×dim
hidden_dim = int(2 * hidden_dim / 3)  # SwiGLU 多了一个 w3，所以缩小 2/3 保持总参数不变
hidden_dim = multiple_of * (...)      # 对齐 GPU 友好的倍数
```

**为什么是 2/3？**
- 普通 FFN 参数：`2 × dim × hidden_dim` （w1 + w2）
- SwiGLU 参数：`3 × dim × hidden_dim` （w1 + w2 + w3）
- 要让 SwiGLU 总参数 = 普通 FFN，需要 `hidden_dim_swiglu = (2/3) × hidden_dim_relu`
- → 这是 LLaMA 论文的细节

**举例**（`dim=768`）：
```
4 × 768 = 3072
× 2/3 = 2048
对齐 64 = 2048
```

### 关键 3：`F.silu`（也叫 Swish）

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1+e^{-x}}$$

- 看起来像 ReLU 但**平滑**
- 负数区域**不完全截断**（保留一点点信息）
- 在大模型上效果优于 ReLU 和 GELU

```python
SiLU(2)   ≈ 1.76    (接近 2)
SiLU(0.5) ≈ 0.31
SiLU(0)   = 0
SiLU(-1)  ≈ -0.27   (不像 ReLU 完全归 0)
SiLU(-5)  ≈ -0.03   (大负数才接近 0)
```

### 关键 4：核心一行

```python
F.silu(self.w1(x)) * self.w3(x)
```

- `F.silu(self.w1(x))`：升维 + 激活 → **信息流**
- `self.w3(x)`：升维不激活 → **门控**
- `*`：逐元素相乘 → **门控信息流**

直觉上：门控 `w3(x)` 决定 SiLU 输出的每一位"被放大多少"。
- 如果 `w3(x)_i = 0`：那个位置被关闭
- 如果 `w3(x)_i = 2`：那个位置被放大 2 倍

### 关键 5：所有线性层 `bias=False`

- LLaMA 系列**不用 bias**
- 大模型时代发现 bias 几乎没用
- 省参数 + 训练更稳

---

## 🆚 SwiGLU vs 普通 FFN 对比

| 项    | 普通 FFN       | **SwiGLU**                   |
| ---- | ------------ | ---------------------------- |
| 线性层数 | 2 个          | **3 个**                      |
| 激活函数 | ReLU/GELU    | **SiLU + 门控**                |
| 表达能力 | 普通           | **更强**（门控更细粒度）               |
| 参数量  | 2×dim×hidden | 3×dim×hidden（但 hidden 缩 2/3） |
| 计算量  | 100%         | ~150%                        |
| 训练效果 | 基线           | **略好**                       |

---

## 🔬 测试

```python
mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
x = torch.randn(1, 50, args.dim)
output = mlp(x)
print(output.shape)  # torch.Size([1, 50, 768])
```

✅ 输入输出 shape 一致，可以叠入 DecoderLayer。

---

## 🧠 进阶：为什么门控这么有用？

门控机制（gating）的本质是 **让模型对不同位置的信息有不同的"信任度"**：

| 任务场景 | 门控的作用 |
|---|---|
| 数学计算 | 门控可以"放大数字、关闭语气词" |
| 代码生成 | 门控可以"关注语法符号、关闭注释" |
| 对话理解 | 门控可以"重点关注关键词、忽略助词" |

这种"软选择"机制比硬 ReLU（要么全开、要么全关）灵活得多。

→ 现代 LLM 几乎都用 SwiGLU 或类似门控（GeGLU、ReGLU）。

---

## ⚠️ 小白避坑

1. **`*` 是逐元素乘，不是矩阵乘**
   - 用 `*`（element-wise），不要写成 `@`（matmul）
2. **三个矩阵的 shape 必须严格对齐**
   - `w1` 和 `w3` 输入都是 dim，输出都是 hidden_dim
   - 这样它们的输出才能相乘
3. **`bias=False` 是 LLaMA 的设计**
   - 改成 True 不影响功能，但偏离 LLaMA 风格
4. **`hidden_dim` 不要乱填**
   - 必须能被 `multiple_of` 整除，否则 GPU 利用率低

---

## 🔗 延伸阅读

- 上一节：[03-GQA与RoPE](03-GQA%E4%B8%8ERoPE.md)
- 下一节：[05-DecoderLayer组装](05-DecoderLayer%E7%BB%84%E8%A3%85.md)
- 理论篇：[普通 FFN](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/06-Encoder-Decoder.md#-%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9Cffn)、[SwiGLU 起源](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07-LLaMA.md#%E6%94%B9%E8%BF%9B-3swiglu-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0)

---

⬅ [03-GQA与RoPE](03-GQA%E4%B8%8ERoPE.md)	|	➡ [05-DecoderLayer组装](05-DecoderLayer%E7%BB%84%E8%A3%85.md)
