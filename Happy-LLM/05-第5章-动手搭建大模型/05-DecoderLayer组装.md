---
tags: [Happy-LLM, 第5章, 实战, DecoderLayer]
chapter: 5
section: 5.1.5
---

# 5.1.5 DecoderLayer 组装

⬅ [04-SwiGLU-MLP](04-SwiGLU-MLP.md)	|	➡ [06-完整LLaMA2模型](06-%E5%AE%8C%E6%95%B4LLaMA2%E6%A8%A1%E5%9E%8B.md)

> ✅ 前面几节我们已经准备好了所有"零件"：
> - [RMSNorm](02-RMSNorm%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0.md)
> - [Attention（GQA + RoPE）](03-GQA%E4%B8%8ERoPE.md)
> - [MLP（SwiGLU）](04-SwiGLU-MLP.md)
>
> 这一节把它们拼成 **DecoderLayer**（一个完整的 Transformer 层）。

---

## 🎬 故事比喻：组装一台机器

| 零件 | 作用 |
|---|---|
| RMSNorm × 2 | 两个归一化器（在 Attention 前 + MLP 前） |
| Attention | 注意力模块（信息交流） |
| MLP | 前馈模块（独立加工） |
| 残差连接 ×2 | 两条电梯（防梯度消失） |

→ 组装成一个 **完整的 Transformer Layer**。

---

## 🏗 数据流图

```
   x ─────────────┬───────────────────────────┬───────────────► out
                  │                            │
                  ▼                            ▼
              RMSNorm                       RMSNorm
                  │                            │
                  ▼                            ▼
              Attention                       MLP
                  │                            │
                  +───► h ─────────────────────+
                  ↑                            ↑
              第 1 个残差                   第 2 个残差
```

→ 这就是 [Pre-Norm](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md#-pre-norm-vs-post-norm) 风格的 Transformer Layer。

---

## 🐍 完整代码

```python
class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        # 注意力模块
        self.attention = Attention(args)

        # MLP 模块
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )

        self.layer_id = layer_id

        # 两个 RMSNorm
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # 第 1 个子层：归一化 → 注意力 → 残差
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)

        # 第 2 个子层：归一化 → MLP → 残差
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out
```

---

## 📝 代码深度解读

### 关键 1：两个 RMSNorm 是独立的

```python
self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
```

虽然名字都叫 RMSNorm，但**两个是独立的实例**，各自有自己的 `weight`（γ 参数）。

为什么不共享？
- 不同位置的归一化目标不同（一个面向 Attention 输入、一个面向 MLP 输入）
- 让模型自己学不同的缩放参数，效果更灵活

### 关键 2：Pre-Norm 顺序

```python
h = x + self.attention.forward(self.attention_norm(x), ...)
```

**注意顺序**：
1. `self.attention_norm(x)`：**先归一化**
2. `self.attention.forward(...)`：然后过 Attention
3. `x + ...`：最后**残差加回去**

这是 **Pre-Norm**（归一化在残差之前）。

**对比 Post-Norm**（原论文，现已淘汰）：
```python
h = self.attention_norm(x + self.attention.forward(x, ...))   # ❌ Post-Norm
```

Pre-Norm 更稳定，所以现代 LLM 全用它。
→ 详见 [Pre-Norm vs Post-Norm](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md#-pre-norm-vs-post-norm)

### 关键 3：残差连接的 `+ x`

```python
h = x + self.attention.forward(...)
```

这个 `+ x` 至关重要：
- 让信息可以**直接跨层传递**
- 反向传播时梯度 `∂out/∂x = 1 + ...`
- → **避免梯度消失**，让深层网络可训练

详见 [残差与梯度](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md#-%E5%8A%A0%E6%B7%B1%E7%90%86%E8%A7%A3%E4%B8%BA%E4%BB%80%E4%B9%88%E6%AE%8B%E5%B7%AE%E8%83%BD%E8%A7%A3%E5%86%B3%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1)

### 关键 4：`freqs_cos, freqs_sin` 通过参数传递

```python
def forward(self, x, freqs_cos, freqs_sin):
    ...
    self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
```

- RoPE 的 cos/sin 频率是**所有层共用的**
- 在外层（Transformer 类）**预计算一次**，作为 buffer 保存
- 每层 forward 时传进来给 Attention 用
- → 节省计算（不用每层重复算）

### 关键 5：`layer_id` 参数

```python
self.layer_id = layer_id
```

虽然这里没用，但**保存层 ID 是好习惯**：
- 方便调试（打印来自第几层）
- 某些技巧（如分层学习率、梯度检查点）需要用
- 后续可能用于**层归一化的特殊初始化**

---

## 🔬 测试

```python
decoderlayer = DecoderLayer(0, args)
x = torch.randn(1, 50, args.dim)
freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, 50)
out = decoderlayer(x, freqs_cos, freqs_sin)
print(out.shape)  # torch.Size([1, 50, 768])
```

✅ 输入输出 shape 一致 —— 可以**无限堆叠**。

---

## 🎯 完整 LLaMA2 Layer 结构总结

```
                 输入 x (B, L, dim)
                       │
        ┌──────────────┴──────────────┐
        │                              │
        ▼                              ▼
   attention_norm                  (保留 x)
   (RMSNorm)                            │
        │                              │
        ▼                              │
   Attention                            │
   (GQA + RoPE +                        │
    Flash Attn +                        │
    causal mask)                        │
        │                              │
        └─────────► +  ◄────────────────┘  ← 第 1 个残差
                    │
                    │ h
                    │
        ┌──────────┴──────────┐
        │                      │
        ▼                      ▼
    ffn_norm                (保留 h)
    (RMSNorm)                  │
        │                      │
        ▼                      │
    MLP                        │
    (SwiGLU)                   │
        │                      │
        └─────► +  ◄───────────┘  ← 第 2 个残差
                │
                ▼ out (B, L, dim)
```

---

## ⚠️ 小白避坑

1. **两个 RMSNorm 必须独立**
   - 不能写 `self.norm = RMSNorm(...)` 然后两处都用同一个
2. **Pre-Norm 顺序别搞反**
   - 先 norm 再 sublayer，最后残差
3. **传 `freqs_cos / freqs_sin` 时长度要够**
   - 否则 RoPE 会越界
4. **`forward` 别忘了 `self.feed_forward.forward(...)`**
   - 直接写 `self.feed_forward(...)` 也行（更 pytonic）

---

## 📌 阶段性总结

到这里，我们已经实现了：

- ✅ **RMSNorm**（[02-RMSNorm代码实现](02-RMSNorm%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0.md)）
- ✅ **Attention with GQA + RoPE + Flash Attention**（[03-GQA与RoPE](03-GQA%E4%B8%8ERoPE.md)）
- ✅ **MLP with SwiGLU**（[04-SwiGLU-MLP](04-SwiGLU-MLP.md)）
- ✅ **DecoderLayer**（本节）

下一节我们把 N 个 DecoderLayer 堆起来，组装完整的 LLaMA2 模型。

---

## 🔗 延伸阅读

- 上一节：[04-SwiGLU-MLP](04-SwiGLU-MLP.md)
- 下一节：[06-完整LLaMA2模型](06-%E5%AE%8C%E6%95%B4LLaMA2%E6%A8%A1%E5%9E%8B.md)
- 理论篇：[06-Encoder-Decoder](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/06-Encoder-Decoder.md)、[08-残差连接](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md)

---

⬅ [04-SwiGLU-MLP](04-SwiGLU-MLP.md)	|	➡ [06-完整LLaMA2模型](06-%E5%AE%8C%E6%95%B4LLaMA2%E6%A8%A1%E5%9E%8B.md)
