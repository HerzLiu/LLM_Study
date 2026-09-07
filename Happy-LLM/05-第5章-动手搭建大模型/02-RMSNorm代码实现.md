---
tags: [Happy-LLM, 第5章, 实战, RMSNorm]
chapter: 5
section: 5.1.2
---

# 5.1.2 构建 RMSNorm

⬅ [01-定义超参数](01-%E5%AE%9A%E4%B9%89%E8%B6%85%E5%8F%82%E6%95%B0.md)	|	➡ [03-GQA与RoPE](03-GQA%E4%B8%8ERoPE.md)

---

## 🎬 故事比喻：体检版的"轻量归一化"

[LayerNorm](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/07-LayerNorm%E4%B8%8EBatchNorm.md) 像**体检**：
- 量身高（减均值）
- 量体重（除标准差）
- 然后做仿射变换（缩放 + 偏移）

**RMSNorm 像快速体检**：
- 跳过量身高
- 只看体重（RMS 均方根）
- 再做缩放
- → **省一步、更快、效果几乎一样**

---

## 🔧 数学公式

$$\text{RMSNorm}(x_i) = \gamma_i \cdot \frac{x_i}{\sqrt{\frac{1}{n}\sum_{j=1}^{n} x_j^2 + \epsilon}}$$

| 符号 | 含义 |
|---|---|
| `x_i` | 输入向量第 i 维 |
| `γ_i` | 可学习的缩放参数（对应代码中的 `self.weight`） |
| `n` | 向量维度 |
| `ε` | 数值稳定项（防止除 0） |

### ⭐ 与 LayerNorm 的对比

| 操作 | LayerNorm | **RMSNorm** |
|---|---|---|
| 减均值 | ✅ | ❌ |
| 除标准差 | ✅ | ❌（改用 RMS） |
| 仿射变换 | γ + β | **只有 γ** |
| 计算量 | 100% | **~70%** |
| 效果 | 基线 | 几乎一致 |

→ LLaMA、Qwen、DeepSeek 等现代 LLM 都用 RMSNorm。

---

## 🐍 完整代码

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps 是为了防止除以 0
        self.eps = eps
        # weight 是可学习的缩放参数，全部初始化为 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """RMSNorm 的核心计算"""
        # x.pow(2).mean(-1, keepdim=True) 计算输入 x 的平方的均值
        # torch.rsqrt 是平方根的倒数（reciprocal square root）
        # 加上 eps 防止分母为 0
        # 最后乘以 x，得到 RMSNorm 的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 1. 转 float 计算（保证数值稳定性，尤其在 bf16/fp16 训练时）
        # 2. RMSNorm 归一化
        # 3. 转回原数据类型
        # 4. 乘以可学习的缩放因子 weight
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

---

## 📝 代码深度解读

### 关键 1：`torch.rsqrt`

```python
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

`rsqrt(z) = 1 / sqrt(z)` —— **平方根的倒数**

为什么用 `rsqrt` 而不是 `/ sqrt`？
- `rsqrt` 是 GPU 上的**原生指令**（一个浮点操作）
- `1.0 / sqrt(x)` 是两个操作（先开方再除）
- → **快约 20%**

### 关键 2：`mean(-1, keepdim=True)`

```
输入  x.shape:  (batch, seq_len, dim)
mean 后 shape: (batch, seq_len, 1)
```

`keepdim=True` 保留了最后一维（变成 1），方便后续广播。

### 关键 3：先 `.float()` 再 `.type_as(x)`

```python
output = self._norm(x.float()).type_as(x)
```

为什么？
- 训练时常用 **bf16 / fp16** 混合精度
- 但 RMSNorm 内部计算（平方、求和、开方）在低精度下**容易溢出或不精准**
- 所以**临时升到 fp32 计算**，算完再转回原类型
- → **数值稳定性 + 训练速度**双赢

### 关键 4：`nn.Parameter(torch.ones(dim))`

```python
self.weight = nn.Parameter(torch.ones(dim))
```

- `nn.Parameter` 包起来 → 自动加入模型参数列表，会被训练
- 初始化为全 1 → 初始时 `RMSNorm` 等价于"只归一化、不缩放"，训练稳定

### 关键 5：没有 β（偏移参数）

LayerNorm 有 γ 和 β 两个学习参数，RMSNorm **只保留 γ**。
- β 在大模型实验中**贡献很小**
- 去掉省参数、省计算

---

## 🔬 测试代码

```python
norm = RMSNorm(args.dim, args.norm_eps)
x = torch.randn(1, 50, args.dim)
output = norm(x)
print(output.shape)
```

输出：
```
torch.Size([1, 50, 768])
```

✅ 输入输出 shape 完全一致 —— 归一化**不改变形状**，只改变数值分布。

---

## 🧠 进阶：为什么不减均值也可以？

LayerNorm：$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$
RMSNorm： $\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x)}$

**论文 [Root Mean Square Layer Normalization, 2019] 的发现**：
- LayerNorm 的"减均值"步骤对效果**几乎没贡献**
- 真正起作用的是"**归一化幅度**"（divide by scale）
- 所以减均值步骤可以去掉

实验证明：
- 训练速度提升 **7%~64%**
- 模型性能几乎不变
- 在大模型上甚至略好

→ **现代 LLM 标配 RMSNorm**。

---

## ⚠️ 小白避坑

1. **`keepdim=True` 不能漏**
   - 漏了维度对不齐，无法广播
2. **必须先 `.float()` 再算**
   - bf16 下平方和直接算可能 NaN
3. **`weight` 初始化为 1 而不是 0**
   - 0 会让输出永远是 0，无法学习
4. **`eps` 不要太大**
   - 太大会让归一化失去意义
   - 一般 1e-5 或 1e-6

---

## 🔗 延伸阅读

- 上一节：[01-定义超参数](01-%E5%AE%9A%E4%B9%89%E8%B6%85%E5%8F%82%E6%95%B0.md)
- 下一节：[03-GQA与RoPE](03-GQA%E4%B8%8ERoPE.md) —— 全章最难的代码
- 理论篇：[07-LayerNorm与BatchNorm](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/07-LayerNorm%E4%B8%8EBatchNorm.md)

---

⬅ [01-定义超参数](01-%E5%AE%9A%E4%B9%89%E8%B6%85%E5%8F%82%E6%95%B0.md)	|	➡ [03-GQA与RoPE](03-GQA%E4%B8%8ERoPE.md)
