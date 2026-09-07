---
tags: [Happy-LLM, 第5章, 实战, GQA, RoPE, 注意力]
chapter: 5
section: 5.1.3
---

# 5.1.3 构建 LLaMA2 Attention（GQA + RoPE）

⬅ [02-RMSNorm代码实现](02-RMSNorm%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0.md)	|	➡ [04-SwiGLU-MLP](04-SwiGLU-MLP.md)

> ⚠️ **本章最难的一节**，建议至少读两遍。代码量大，但每一段都是现代 LLM 的核心。

---

## 🎬 故事比喻：GQA = 多个秘书共享一份资料库

[标准多头注意力（MHA）](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/05-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md)：
- 8 个秘书（Q 头）
- 每个秘书有**自己的资料库**（独立 K/V）
- 8 套资料库占内存大

**MQA（多查询注意力）**：
- 8 个秘书
- **所有人共用 1 套资料库**
- 省内存但**信息丢失**

**GQA（分组查询注意力）⭐ LLaMA-2 采用**：
- 8 个秘书分成 4 组，每组 2 人
- **每组共用 1 套资料库**（共 4 套）
- **平衡内存和效果**

```
MHA   : Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7    →  8 套 K/V
MQA   : Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7    →  1 套 K/V
GQA-4 : [Q0,Q1] [Q2,Q3] [Q4,Q5] [Q6,Q7]  →  4 套 K/V
```

---

## 🎯 本节代码三大块

| 模块                                          | 作用                                 |
| ------------------------------------------- | ---------------------------------- |
| `repeat_kv`                                 | 把 K/V 复制 N 份适配 Q（让 GQA 能复用注意力计算逻辑） |
| `precompute_freqs_cis` + `apply_rotary_emb` | **RoPE 旋转位置编码**                    |
| `Attention` 类                               | 组装最终的注意力模块                         |

---

## 🐍 模块 1：`repeat_kv`

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    把 K/V 张量在头维度上重复 n_rep 次
    用于让 GQA 的 KV 头数与 Q 头数对齐
    """
    bs, slen, n_kv_heads, head_dim = x.shape

    # 如果不需要重复，直接返回
    if n_rep == 1:
        return x

    # 在第 3 维（n_kv_heads 后）插入新维度 → expand 复制 → reshape 合并
    return (
        x[:, :, :, None, :]
         .expand(bs, slen, n_kv_heads, n_rep, head_dim)
         .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
```

### 📝 解读

输入 K 的 shape：`(B, L, n_kv_heads=4, head_dim)`
GQA-4 下 `n_rep = n_heads / n_kv_heads = 8/4 = 2`
经过 repeat_kv：`(B, L, 4*2=8, head_dim)` —— 和 Q 的 shape 对齐

### shape 变化

```
原始 K:           (B, L, 4, D)
[:, :, :, None]:  (B, L, 4, 1, D)   ← 插入新维度
.expand(...):     (B, L, 4, 2, D)   ← 沿新维度复制 2 次（不实际复制内存）
.reshape(...):    (B, L, 8, D)      ← 合并维度
```

> ⭐ `expand` 是"虚拟复制"（不占额外内存），`reshape` 后才真正展开。

---

## 🐍 模块 2：RoPE（旋转位置编码）

### 数学原理（速览）

把每对维度 `(x_{2i}, x_{2i+1})` 视为复数 `x_{2i} + i·x_{2i+1}`，按位置 `m` 旋转角度 `mθ_i`：

$$\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}$$

其中频率 $\theta_i = 10000^{-2i/d}$

→ 这样 Q·K 的点积**自然包含了相对位置信息**。

### 代码 1：预计算 sin/cos

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预先算好所有位置的 sin/cos 值
    dim: 每个 head 的维度（不是模型 dim）
    end: 最大序列长度
    """
    # 生成频率序列 θ_i = 1 / (10000^(2i/d))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # shape: (dim/2,)

    # 生成位置序列 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # shape: (end,)

    # 外积：t × freqs，得到每个位置 × 每个频率
    freqs = torch.outer(t, freqs).float()
    # shape: (end, dim/2)，每个元素是 m·θ_i

    # 算 cos 和 sin
    freqs_cos = torch.cos(freqs)   # 实部
    freqs_sin = torch.sin(freqs)   # 虚部
    return freqs_cos, freqs_sin
```

#### 📝 解读

- `torch.arange(0, dim, 2)` → `[0, 2, 4, ..., dim-2]`，对应 `2i`
- 频率随 `i` 增大而减小（高维 = 低频，刻画粗粒度位置）
- `torch.outer(t, freqs)` → 每个位置 `m` × 每个频率 `θ_i`
- 输出 `(end, dim/2)` 矩阵，**只算一次，永久使用**

### 代码 2：reshape 用于广播

```python
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """把 freqs_cis 调整成能和 x 广播的 shape"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # 除了第 1 维（seq_len）和最后一维（head_dim/2），其他都是 1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)
```

输入 `x.shape = (B, L, H, D/2)` → 输出 `freqs_cis.shape = (1, L, 1, D/2)`
→ 可以和 x 广播。

### 代码 3：应用 RoPE

```python
def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    """
    给 Q 和 K 应用旋转位置编码
    """
    # 把最后一维拆成 "实部, 虚部" 两半
    # xq.shape: (B, L, H, D) → (B, L, H, D/2, 2) → unbind 后两个 (B, L, H, D/2)
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 调整 cos/sin 形状以广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 旋转公式：
    #   实部' = 实部·cos - 虚部·sin
    #   虚部' = 实部·sin + 虚部·cos
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 拼回原始形状: 实部/虚部交错排列
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

#### 📝 解读核心 4 行

```python
xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin   # 复数乘法的实部
xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos   # 复数乘法的虚部
```

这就是**二维旋转矩阵**乘以向量 `(x_r, x_i)`。

**为什么这样能编码位置？**
- 位置 `m` 的旋转角度 `mθ_i`
- 位置 `n` 的旋转角度 `nθ_i`
- 它们的点积 = 函数 `f(m - n)` —— **只依赖相对位置**！
- → 这就是 RoPE 神奇的地方：**用绝对位置的旋转，自然实现了相对位置编码**

---

## 🐍 模块 3：Attention 类

```python
class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # 决定 KV 头数：默认与 Q 头数相同（MHA），可指定为 GQA
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0

        # 多 GPU 时的并行处理，单卡为 1
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size

        # 关键：n_rep = Q 头数 / KV 头数（即每组 KV 被多少个 Q 共享）
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # 权重矩阵：注意 wk/wv 的输出维度比 wq 小（GQA 关键）
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # Flash Attention 检查（PyTorch 2.0+）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self, x, freqs_cos, freqs_sin):
        bsz, seqlen, _ = x.shape

        # Step 1: 算 Q/K/V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Step 2: 拆头
        # Q: (B, L, n_heads, head_dim)
        # K/V: (B, L, n_kv_heads, head_dim)  ← GQA 这里头数少
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Step 3: 应用 RoPE（注意只对 Q 和 K，不对 V）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Step 4: 把 K/V 重复 n_rep 次，让头数对齐 Q
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Step 5: 把 head 维度提前，方便 matmul
        xq = xq.transpose(1, 2)  # (B, n_heads, L, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Step 6: 注意力计算
        if self.flash:
            # PyTorch 2.0+ 内置 Flash Attention（更快、更省显存）
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True   # 自动应用因果 mask
            )
        else:
            # 手动实现
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # Step 7: 拼头 + 输出投影
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```

---

## 📝 代码深度解读

### 关键 1：GQA 体现在 `wk` / `wv` 的输出维度

```python
self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)        # n_heads × head_dim
self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)     # n_kv_heads × head_dim ← 小！
self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)     # n_kv_heads × head_dim ← 小！
```

**参数量对比**（`dim=1024, n_heads=16, n_kv_heads=8`）：
| | MHA (n_kv=16) | **GQA (n_kv=8)** | 节省 |
|---|---|---|---|
| `wk` | 1024×1024 = 1M | 1024×512 = 512K | **50%** |
| `wv` | 1024×1024 = 1M | 1024×512 = 512K | **50%** |
| `wq` 和 `wo` | 不变 | 不变 | 0 |

### 关键 2：Flash Attention 加速

```python
output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, ...)
```

PyTorch 2.0+ 内置的 **Flash Attention** 实现：
- **更省显存**：避免存中间的 (B, H, L, L) 注意力矩阵
- **更快**：底层用 CUDA kernel 优化
- **结果等价**：和手动实现数值上一致

**性能差距**：
| 序列长度 | 标准注意力显存 | Flash Attention 显存 |
|---|---|---|
| 1024 | ~4GB | ~0.5GB |
| 8192 | ~256GB（爆炸） | ~4GB |

### 关键 3：`is_causal=True`

让 Flash Attention 自动应用因果 mask（屏蔽未来位置），不需要手动构造上三角矩阵。

### 关键 4：shape 全程跟踪（**必记**）

```
输入:                (B, L, dim)
xq after wq:         (B, L, n_heads × head_dim)
xq view+reshape:     (B, L, n_heads, head_dim)
xk view+reshape:     (B, L, n_kv_heads, head_dim)    ← GQA 这里小
RoPE 后:              shape 不变
repeat_kv 后 xk:      (B, L, n_heads, head_dim)       ← 扩展到 n_heads
transpose:           (B, n_heads, L, head_dim)
注意力输出:           (B, n_heads, L, head_dim)
拼回:                 (B, L, n_heads × head_dim)
wo 输出:              (B, L, dim)
```

---

## 🔬 测试

```python
attention_model = Attention(args)
x = torch.rand(1, 50, args.dim)  # (B=1, L=50, dim=768)
freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, 50)
output = attention_model(x, freqs_cos, freqs_sin)
print("Output shape:", output.shape)  # torch.Size([1, 50, 768])
```

✅ 输入输出 shape 一致，可以放心叠层了。

---

## ⚠️ 小白避坑

1. **RoPE 只对 Q 和 K，不对 V**
   - V 不需要位置信息
2. **`repeat_kv` 必须在 RoPE 之后**
   - 否则会重复算多余的 RoPE
3. **`is_causal=True` 必须配合 Flash Attention 用**
   - 老版本 PyTorch 没这个，需要手动 mask
4. **`head_dim = dim / n_heads`，不是 `dim / n_kv_heads`**
   - 注意区分
5. **`contiguous()` 别忘了**
   - `transpose` 后内存不连续，`view` 会报错

---

## 🔗 延伸阅读

- 上一节：[02-RMSNorm代码实现](02-RMSNorm%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0.md)
- 下一节：[04-SwiGLU-MLP](04-SwiGLU-MLP.md)
- 理论篇：[07-LLaMA](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07-LLaMA.md)、[05-多头注意力](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/05-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md)

---

⬅ [02-RMSNorm代码实现](02-RMSNorm%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0.md)	|	➡ [04-SwiGLU-MLP](04-SwiGLU-MLP.md)
