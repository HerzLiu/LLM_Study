---
tags: [Happy-LLM, 第2章, Transformer, Encoder, Decoder]
chapter: 2
section: 2.2
---

# 2.2 Encoder-Decoder：把注意力堆成大楼

⬅ [05-多头注意力](05-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md)　|	➡ [07-LayerNorm与BatchNorm](07-LayerNorm%E4%B8%8EBatchNorm.md)

> [注意力](05-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md) 是"砖块"，Encoder 和 Decoder 是用砖块砌的"楼层"，[Transformer](10-%E5%AE%8C%E6%95%B4Transformer%E6%8B%BC%E8%A3%85.md) 是整座大楼。

---

## 🎬 故事：翻译的工厂流水线

机器翻译 = **Seq2Seq** 任务（序列到序列）
- 输入：中文序列 "今天天气真好"
- 输出：英文序列 "Today is a good day"

工厂里有两条流水线：
- **Encoder（编码器）**：把中文"压缩"成一个语义向量（更复杂的词向量表示）
- **Decoder（解码器）**：根据这个语义向量"展开"成英文

---

## 🏗 Transformer 整体架构

```
        输入序列                           目标序列
           │                                  │
        Embedding                         Embedding
           +                                  +
       位置编码                            位置编码
           │                                  │
   ┌───────▼────────┐               ┌────────▼────────┐
   │  EncoderLayer  │ ×N            │   DecoderLayer  │ ×N
   │  ┌──────────┐  │               │  ┌────────────┐ │
   │  │自注意力   │  │               │  │掩码自注意力 │ │
   │  └────┬─────┘  │               │  └─────┬──────┘ │
   │       │        │               │        │        │
   │  ┌────▼─────┐  │               │  ┌─────▼──────┐ │
   │  │  FFN     │  │               │  │交叉注意力  │◄┼──┐
   │  └──────────┘  │               │  └─────┬──────┘ │  │
   └────────┬───────┘               │        │        │  │
            │                       │  ┌─────▼──────┐ │  │
            │                       │  │  FFN       │ │  │
            │                       │  └────────────┘ │  │
            └───────────────────────►─────────────────┘  │
                                            │           │
                                       enc_out 传给 Decoder
```

---

## 🔧 Encoder Layer 拆解

一个 EncoderLayer = **自注意力 + FFN**（中间夹 LayerNorm 和残差）

```python
class EncoderLayer(nn.Module):
    """Encoder 中的一层"""
    def __init__(self, args):
        super().__init__()
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder 用普通自注意力，不需要掩码
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args)

    def forward(self, x):
        # Pre-Norm + 残差（现代 LLM 标准做法）
        norm_x = self.attention_norm(x)
        h = x + self.attention(norm_x, norm_x, norm_x)   # 自注意力: q=k=v=x
        out = h + self.feed_forward(self.fnn_norm(h))
        return out


class Encoder(nn.Module):
    """N 层 EncoderLayer 堆叠"""
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(args) for _ in range(args.n_layer)]
        )
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

---

## 🔧 Decoder Layer 拆解

DecoderLayer 比 EncoderLayer 多一个注意力子层，**共三段**：

| 段 | 内容 | 作用 |
|---|---|---|
| 1 | **掩码自注意力**（[is_causal=True](04-%E6%8E%A9%E7%A0%81%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md)） | 看自己生成过的 token，不许偷看未来 |
| 2 | **交叉注意力**（is_causal=False） | Q 来自 Decoder，K/V 来自 Encoder 输出 |
| 3 | **FFN** | 独立思考 |

```python
class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # 第一段: 掩码自注意力
        self.mask_attention = MultiHeadAttention(args, is_causal=True)

        self.attention_norm_2 = LayerNorm(args.n_embd)
        # 第二段: 交叉注意力（is_causal=False）
        self.attention = MultiHeadAttention(args, is_causal=False)

        self.ffn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args)

    def forward(self, x, enc_out):
        # x: Decoder 自己的输入
        # enc_out: Encoder 的输出（来自源语言编码）

        # 1. 掩码自注意力（q=k=v=x）
        norm_x = self.attention_norm_1(x)
        x = x + self.mask_attention(norm_x, norm_x, norm_x)

        # 2. 交叉注意力（q=x, k=v=enc_out）—— Decoder 和 Encoder 沟通的桥梁
        norm_x = self.attention_norm_2(x)
        h = x + self.attention(norm_x, enc_out, enc_out)

        # 3. FFN
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

---

## ⭐ 加深理解：三种注意力的位置

| 位置 | 用什么 |
|---|---|
| **Encoder 自注意力** | 普通[自注意力](03-%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md) |
| **Decoder 第 1 段** | [掩码自注意力](04-%E6%8E%A9%E7%A0%81%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md) |
| **Decoder 第 2 段** | **交叉注意力**（Q 来自 Decoder，K/V 来自 Encoder） |

**核心机制**：
- 掩码自注意力让 Decoder 看自己生成过的内容
- **交叉注意力让 Decoder "查阅" Encoder 编码好的源信息**
- 这是机器翻译中"Decoder 一边生成、一边参考源语言"的实现

---

## 🔧 前馈神经网络（FFN）

### 🎬 故事：开完会后，每人各自消化

[注意力机制](02-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6QKV.md)让所有位置互相交流（开会）。
但光开会不行，每个位置还需要"独立思考一下"。
**FFN 就是给每个位置一个独立的小脑袋，对开会内容做加工。**

### 结构

$$\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$$

- 一个**升维**线性层（如 512 → 2048）
- 一个 ReLU 激活
- 一个**降维**线性层（2048 → 512）
- 加 Dropout 防过拟合

### ⭐ 为什么要升维再降维？

- **升到 4 倍宽**：给模型更多"思考空间"
- **再压回原维度**：方便后面继续叠层
- 这种"**瓶颈结构**"是深度学习常见招数

### 代码

```python
class MLP(nn.Module):
    """前馈神经网络（FFN/MLP）"""
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # 升维: 512 → 2048
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # 降维: 2048 → 512
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 升维 → ReLU → 降维 → Dropout
        return self.dropout(self.w2(F.relu(self.w1(x))))
```

---

## 📌 几个关键归一化与连接

- **LayerNorm**：让每层输出分布稳定 → 详见 [07-LayerNorm与BatchNorm](07-LayerNorm%E4%B8%8EBatchNorm.md)
- **残差连接**：让深层网络可训练 → 详见 [08-残差连接](08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md)
- **Pre-Norm vs Post-Norm**：现代 LLM 选 Pre-Norm → 见 [08-残差连接](08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md#-pre-norm-vs-post-norm)

---

## ⚠️ 小白避坑

1. **Encoder 和 Decoder 的 N 层数量通常一样**（如各 6 层）
2. **Decoder 多了"交叉注意力"是和 Encoder 唯一的结构差异**
3. **后续衍生模型只用其中一部分**：
   - 只用 Encoder：BERT 类（理解任务）
   - 只用 Decoder：GPT、LLaMA 类（生成任务）
   - 完整 Encoder-Decoder：T5、BART（序列转换任务）
   → 详见 [第 3 章预训练模型](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

## 🔗 延伸阅读

- 上一节：[05-多头注意力](05-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md)
- 子组件：[07-LayerNorm与BatchNorm](07-LayerNorm%E4%B8%8EBatchNorm.md)、[08-残差连接](08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md)
- 下一节：[07-LayerNorm与BatchNorm](07-LayerNorm%E4%B8%8EBatchNorm.md)
- 拼装总览：[10-完整Transformer拼装](10-%E5%AE%8C%E6%95%B4Transformer%E6%8B%BC%E8%A3%85.md)

---

⬅ [05-多头注意力](05-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md)　|	➡ [07-LayerNorm与BatchNorm](07-LayerNorm%E4%B8%8EBatchNorm.md)
