---
tags: [ml-dl-foundations, 第7章, 桥接, MLP, Transformer]
chapter: 7
section: 7.1
---

# 7.1 从 MLP 到 Transformer 的桥梁

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-从交叉熵到next-token-loss](02-%E4%BB%8E%E4%BA%A4%E5%8F%89%E7%86%B5%E5%88%B0next-token-loss.md)

---

## 🎬 故事比喻：积木重组

```
你前 4 章学到的积木:
  - 神经元（§4.1）
  - MLP（§4.2）
  - 激活函数（§4.3）
  - LayerNorm（§4.8）
  - 残差连接（§5.1 从 CNN 借）

把这些积木重新组合 + 加一个新积木 (Attention)
= Transformer

你以前在 Happy-LLM 看 Transformer 觉得"复杂"
读完本节会发现: 它就是"积木的特定组合"。
```

> 这一节让你**用第 4-5 章的语言重新解读 Transformer**。
> 不讲新东西，只串起来。

---

## 🧩 7.1.1 Transformer Block 拆解

```
       x (输入)
         │
         ▼
   ┌──────────────────────┐
   │ LayerNorm (§4.8.2)   │ ← 你已学
   └──────────────────────┘
         │
         ▼
   ┌──────────────────────┐
   │ Multi-Head Attention │ ← 唯一新东西
   └──────────────────────┘
         │
         ▼
       ＋ ← ─── (残差) ← §5.1 ResNet 思想
         │
         ▼
   ┌──────────────────────┐
   │ LayerNorm            │
   └──────────────────────┘
         │
         ▼
   ┌──────────────────────┐
   │ MLP / FFN (§4.2)     │ ← 你已学
   │ Linear → SwiGLU      │ ← §4.3 SwiGLU 你已学
   │      → Linear        │
   └──────────────────────┘
         │
         ▼
       ＋ ← ─── (残差)
         │
         ▼
       y (输出)
```

→ **Transformer Block = 4 个第 4-5 章组件 + 1 个新组件（Attention）**。

---

## 🔍 7.1.2 Multi-Head Attention 详细拆解

```
   Input x [B, T, dim]
         │
         ▼
   ┌─────────────────────────────┐
   │ 3 个 Linear 层 (§4.1 神经元) │
   │ Q = nn.Linear(dim, dim)(x)  │
   │ K = nn.Linear(dim, dim)(x)  │
   │ V = nn.Linear(dim, dim)(x)  │
   └─────────────────────────────┘
         │
         ▼ reshape 成 [B, h, T, d_k]
         │
   ┌─────────────────────────────┐
   │ Scores = Q @ K^T            │ ← §2.1 矩阵乘
   │ Scores /= sqrt(d_k)         │ ← 数值稳定
   │ Mask（如果 causal）         │
   │ Attention = softmax(Scores) │ ← §2.3 softmax
   │ Output = Attention @ V       │ ← §2.1 矩阵乘
   └─────────────────────────────┘
         │
         ▼ reshape 回 [B, T, dim]
         │
   ┌─────────────────────────────┐
   │ Output Linear (§4.1)        │
   └─────────────────────────────┘
         │
         ▼
       Output
```

→ **Multi-Head Attention = 4 个 Linear 层 + 矩阵乘 + softmax**。
→ 完全由你已学的零件构成！

---

## 🆚 7.1.3 用第 4 章语言重新定义 Transformer

| Transformer 概念 | 第 4-5 章语言 |
|---|---|
| **Token Embedding** | 一个查找表 = 神经元的特殊形式 |
| **Q/K/V 投影** | 3 个 `nn.Linear` = 3 组神经元 |
| **Attention 加权** | softmax 归一化 + 加权混合（§2.3 + §2.1） |
| **MHA 输出投影** | 又一个 `nn.Linear` |
| **FFN/MLP** | 2-3 层 MLP（§4.2） + SwiGLU 激活（§4.3） |
| **LayerNorm** | §4.8.2，每个 sublayer 前 |
| **残差连接** | $y = x + F(x)$（§5.1 借自 ResNet） |
| **位置编码** | 加到 Embedding 上（绝对位置）或修改 attention（RoPE） |
| **输出层** | `nn.Linear(dim, vocab_size)` + softmax（§4.4 CE loss） |

→ **每一行你都已经在第 4-5 章学过**。

---

## 🎯 7.1.4 看 Happy-LLM §5 代码"瞬间懂"

回去看 Happy-LLM 第 5 章的 Transformer 代码，每一段你现在能解释：

```python
class TransformerBlock(nn.Module):
    def __init__(self, args):
        # 第 4.8: LayerNorm
        self.attention_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

        # 唯一新东西: Attention
        self.attention = Attention(args)

        # 第 4.2: MLP / FFN，第 4.3: SwiGLU
        self.feed_forward = MLP(args)

    def forward(self, x, freqs_cos, freqs_sin):
        # Pre-Norm + 残差（§4.8 + §5.1）
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

→ 8 行代码 = 第 4-5 章所学的所有积木的拼装。

→ 详见 [05-DecoderLayer组装](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/05-DecoderLayer%E7%BB%84%E8%A3%85.md)。

---

## 🚀 7.1.5 LLM = Transformer × N + 词嵌入 + 输出层

```
Token IDs [B, T]
   ↓
Embedding Layer       (vocab_size → dim)        ← 也是一种 nn.Linear
   ↓ + 位置编码
TransformerBlock × N   (32 层 LLaMA-7B / 80 层 LLaMA-70B)
   ↓
RMSNorm
   ↓
LM Head               (dim → vocab_size)        ← nn.Linear
   ↓ softmax
下一个 token 的概率分布
```

→ **整个 LLM = 几十层 Transformer Block + 输入嵌入 + 输出投影**。
→ **参数大头**：Embedding + 每层的 MLP（§4.2）+ Attention 投影（§4.1）。

---

## 💡 7.1.6 为什么 Attention 是革命性的？

回顾 [02-RNN-LSTM-GRU](../05-%E7%AC%AC5%E7%AB%A0-%E5%85%B8%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%B2%E8%AE%B2/02-RNN-LSTM-GRU.md) § 5.2.6 "为什么 RNN 全家被 Transformer 干掉"：

| 维度 | RNN 痛点 | Attention 解 |
|---|---|---|
| 顺序处理 | 必须 h_t-1 → h_t | **全位置并行**（一次矩阵乘） |
| 长程依赖 | 信息要传 T 步 | **一层即全图**（每个 token 直接看其它） |
| 难训练 | LSTM 4× 参数仍弱 | Transformer 训练稳定且强 |

**核心数学**：

```
RNN:   h_t = f(h_{t-1}, x_t)        ← 串行
Attn:  h_t = softmax(QK^T)V         ← 并行（一次矩阵乘搞定所有 t）
```

→ **Attention 是用矩阵乘换掉了 RNN 的序列依赖**。
→ GPU 喜欢矩阵乘，所以 Transformer 训得快 N 倍。

---

## 🎓 7.1.7 你应该能解释的事

学完前 6 章 + 本节，你应该能向新人解释：

- [ ] Transformer 的每个组件来自哪里
- [ ] 为什么 Attention 能并行而 RNN 不行
- [ ] FFN 在 Transformer 里占多少参数（40%+）
- [ ] 为什么 LLM 用 Pre-Norm 而不是 Post-Norm
- [ ] 为什么 LLM 用 RMSNorm 而不是 LayerNorm
- [ ] 为什么 LLM 用 SwiGLU 而不是 ReLU
- [ ] 为什么 LLM 用 RoPE 而不是绝对位置编码

→ 如果有不能答的，回 Happy-LLM §2-§5 对照看，每点都能找到。

---

## 📌 7.1 节要点

| 概念 | 来自本书哪一节 |
|---|---|
| Q/K/V Linear | §4.1 神经元 |
| MLP/FFN | §4.2 MLP |
| SwiGLU | §4.3 激活函数 |
| LayerNorm/RMSNorm | §4.8 归一化 |
| 残差连接 | §5.1 CNN 借来 |
| Attention 矩阵乘 | §2.1 矩阵乘 |
| Softmax | §2.3 softmax |

**结论**：
> **Transformer 是 90% 已学的 + 10% 新的（Attention）**。
> 你之前觉得"复杂"是因为基础没补——现在你知道每个零件了。

---

## 🔗 延伸阅读

- 上一节：[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一节：[02-从交叉熵到next-token-loss](02-%E4%BB%8E%E4%BA%A4%E5%8F%89%E7%86%B5%E5%88%B0next-token-loss.md)
- 详细 Transformer：[00-章节总览](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 详细代码：[00-章节总览](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 回顾 RNN 痛点：[02-RNN-LSTM-GRU](../05-%E7%AC%AC5%E7%AB%A0-%E5%85%B8%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%B2%E8%AE%B2/02-RNN-LSTM-GRU.md)

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-从交叉熵到next-token-loss](02-%E4%BB%8E%E4%BA%A4%E5%8F%89%E7%86%B5%E5%88%B0next-token-loss.md)
