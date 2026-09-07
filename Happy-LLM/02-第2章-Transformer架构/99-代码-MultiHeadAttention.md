---
tags: [Happy-LLM, 第2章, 代码, Python, 多头注意力]
chapter: 2
section: 2.1.6-code
---

# 💻 代码：MultiHeadAttention 完整实现

⬅ [10-完整Transformer拼装](10-%E5%AE%8C%E6%95%B4Transformer%E6%8B%BC%E8%A3%85.md)　|	➡ [进入第 3 章](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

> 🎯 这是整个 Transformer 最核心的代码。**这一段读懂了，整个 LLM 实现就懂一大半**。

---

## 🐍 完整代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头自注意力模块"""

    def __init__(self, args, is_causal=False):
        super().__init__()
        # 模型维度必须能被头数整除（要不然没法平分）
        assert args.dim % args.n_heads == 0

        # 每个头的维度
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.is_causal = is_causal

        # 三个线性投影：把输入投到 Q/K/V 空间
        # 工程技巧：把 h 个头的小权重 (dim, head_dim) 拼成一个大矩阵 (dim, dim)
        # 这样一次大矩阵乘 >> h 次小矩阵乘
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)

        # 输出投影
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        # 如果是掩码注意力，预先创建 mask 矩阵
        # 避免每次 forward 重新创建，提速
        if is_causal:
            mask = torch.full(
                (1, 1, args.max_seq_len, args.max_seq_len),
                float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)
            # register_buffer：mask 跟随模型一起 save/load 和移设备，但不参与梯度
            self.register_buffer("mask", mask)

    def forward(self, q, k, v):
        # 输入 shape: (batch_size, seq_len, dim)
        bsz, seqlen, _ = q.shape

        # ===== Step 1: 线性投影得到 Q/K/V =====
        # shape: (B, L, dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # ===== Step 2: 拆头 =====
        # (B, L, dim) → (B, L, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # 交换维度：把 head 维提到 batch 后面
        # (B, L, n_heads, head_dim) → (B, n_heads, L, head_dim)
        # 为啥要换？因为后面的 matmul 是在最后两维进行，head 要被当做"batch 维"
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # ===== Step 3: 注意力计算 =====
        # QK^T / √d_k
        # (B, nh, L, hd) × (B, nh, hd, L) → (B, nh, L, L)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 如果是掩码注意力，加上 mask（屏蔽未来位置）
        if self.is_causal:
            scores = scores + self.mask[:, :, :seqlen, :seqlen]

        # Softmax 归一化
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)

        # 加权 V
        # (B, nh, L, L) × (B, nh, L, hd) → (B, nh, L, hd)
        output = torch.matmul(scores, xv)

        # ===== Step 4: 拼头 =====
        # (B, nh, L, hd) → (B, L, nh, hd) → (B, L, dim)
        # contiguous() 是为了让内存连续，view 才能成功
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # ===== Step 5: 输出投影 =====
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```

---

## 📝 代码深度解读

### 关键点 1：为什么 wq/wk/wv 都是 `dim × dim`？

**直觉错误版**：
```python
# 错误做法：每个头单独一个矩阵
self.wq = nn.ModuleList([nn.Linear(dim, head_dim) for _ in range(n_heads)])
# 然后循环 n_heads 次
```

**正确做法（工程优化）**：
```python
self.wq = nn.Linear(dim, dim, bias=False)  # 一个大矩阵
# 算完后 reshape 成多头
```

**为什么？**
- 数学上等价：一个 `(dim, dim)` 矩阵 = `h` 个 `(dim, head_dim)` 矩阵的拼接
- GPU 上：一次大矩阵乘 >> `h` 次小矩阵乘（kernel launch 开销）
- 速度差距可能 5~10 倍

### 关键点 2：`view + transpose` 拆头

```python
xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)   # 拆
xq = xq.transpose(1, 2)                                    # 调维度顺序
```

这是 PyTorch 实现多头的**标准套路**，背下来。

**`view` vs `transpose` 的区别**：
- `view` 只改变 shape，不改变数据顺序（不复制内存）
- `transpose` 交换维度，也不复制（但底层 stride 变了）

**为什么后面要 `contiguous()`？**
```python
output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
```
- `transpose` 后内存不连续了
- 直接 `view` 会报错
- `contiguous()` 强制开辟新内存让数据连续

### 关键点 3：`register_buffer`

- **普通 attribute**：不会跟模型一起 save/load，也不会自动 `to(device)`
- **`register_buffer`** 注册的张量：会，但不参与梯度更新
- 非常适合 **mask** 这种"常量但需要跟模型走"的东西

### 关键点 4：Tensor shape 全程跟踪（**必背**）

```
输入:                  (B, L, dim)
QKV after wq/wk/wv:    (B, L, dim)
拆头后:                 (B, n_heads, L, head_dim)
scores:                (B, n_heads, L, L)
softmax 后:             (B, n_heads, L, L)
加权 V 后:               (B, n_heads, L, head_dim)
拼头后:                 (B, L, dim)
输出（经 wo）:           (B, L, dim)
```

> ⭐ 输入输出 shape **完全一样** → 可以无限堆叠

---

## 🔬 调用示例

```python
from types import SimpleNamespace

# 模拟配置
args = SimpleNamespace(
    dim=512,
    n_heads=8,
    max_seq_len=128,
    dropout=0.1,
)

# 创建模块
mha = MultiHeadAttention(args, is_causal=False)

# 模拟输入：batch=2, seq_len=10, dim=512
x = torch.randn(2, 10, 512)

# 自注意力调用：q=k=v=x
output = mha(x, x, x)
print(output.shape)  # torch.Size([2, 10, 512])
```

---

## 🆚 三种调用方式

```python
# 1. 自注意力（Encoder 用）
mha = MultiHeadAttention(args, is_causal=False)
output = mha(x, x, x)

# 2. 掩码自注意力（Decoder 第 1 段用）
mha_causal = MultiHeadAttention(args, is_causal=True)
output = mha_causal(x, x, x)

# 3. 交叉注意力（Decoder 第 2 段用）
mha_cross = MultiHeadAttention(args, is_causal=False)
output = mha_cross(decoder_x, encoder_out, encoder_out)
#                  ↑ Q          ↑ K/V 来自 Encoder
```

---

## ⚠️ 常见 Bug 排查

| 报错 | 原因 | 解法 |
|---|---|---|
| `RuntimeError: view size is not compatible` | `transpose` 后没 `contiguous` 就 `view` | 加 `.contiguous()` |
| `RuntimeError: matmul shape mismatch` | 头数不能整除 dim | 检查 `dim % n_heads == 0` |
| `IndexError: index out of range` | mask 大小不够 | 检查 `max_seq_len` |
| Loss 不下降 | 忘了除 √d_k 或 mask 用错 | 加打印检查 scores 分布 |

---

## 🔗 延伸阅读

- 理论篇：[05-多头注意力](05-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md)
- 公式来源：[02-注意力机制QKV](02-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6QKV.md)
- 应用场景：[06-Encoder-Decoder](06-Encoder-Decoder.md)
- 完整模型：[10-完整Transformer拼装](10-%E5%AE%8C%E6%95%B4Transformer%E6%8B%BC%E8%A3%85.md)

---

⬅ [10-完整Transformer拼装](10-%E5%AE%8C%E6%95%B4Transformer%E6%8B%BC%E8%A3%85.md)　|	➡ [进入第 3 章](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
