---
tags: [Happy-LLM, 第2章, Transformer, 完整代码]
chapter: 2
section: 2.3.3
---

# 2.3.3 完整 Transformer 拼装

⬅ [09-位置编码](09-%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)　|	➡ [99-代码-MultiHeadAttention](99-%E4%BB%A3%E7%A0%81-MultiHeadAttention.md)

---

## 🧩 组件清单（前面都讲过了）

| 组件 | 文档 |
|---|---|
| 注意力机制 | [02-注意力机制QKV](02-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6QKV.md) |
| 自注意力 / 掩码自注意力 | [03-自注意力](03-%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md) / [04-掩码自注意力](04-%E6%8E%A9%E7%A0%81%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md) |
| 多头注意力 | [05-多头注意力](05-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md) |
| Encoder / Decoder | [06-Encoder-Decoder](06-Encoder-Decoder.md) |
| LayerNorm | [07-LayerNorm与BatchNorm](07-LayerNorm%E4%B8%8EBatchNorm.md) |
| 残差连接 | [08-残差连接](08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md) |
| 位置编码 | [09-位置编码](09-%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md) |
| **Embedding 层** | 本节 |

---

## 🔧 Embedding 层：把 token id 变成向量

### 🎬 故事：图书馆的索取号 → 实际的书

[tokenizer](../01-%E7%AC%AC1%E7%AB%A0-NLP%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/03-NLP%E4%B9%9D%E5%A4%A7%E4%BB%BB%E5%8A%A1.md#%E2%91%A1-%E5%AD%90%E8%AF%8D%E5%88%87%E5%88%86subword--%E9%87%8D%E8%A6%81) 把"我喜欢你"切成 `[我, 喜欢, 你]`，再查词表得到 id `[0, 1, 2]`。
但 id 是离散整数，没法做数学运算。

**Embedding 层是一张超大的查找表**：每个 id 对应一行连续向量。

```python
self.tok_embeddings = nn.Embedding(vocab_size, embedding_dim)
# 例如 vocab_size=10000, embedding_dim=512
# 内部是一个 (10000, 512) 的可训练矩阵
# 输入 id=5 → 取第 5 行 → 输出 512 维向量
```

### Shape 变化

- 输入：`(batch_size, seq_len)`，存的是 token id
- 输出：`(batch_size, seq_len, embedding_dim)`

---

## 🐍 完整 Transformer 类

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    """完整的 Transformer 模型"""

    def __init__(self, args):
        super().__init__()
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args

        # 用 ModuleDict 组织子模块，方便按名字访问
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),  # 词嵌入
            wpe = PositionalEncoding(args),                     # 位置编码
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),
        ))

        # 最后的输出层：n_embd → vocab_size，预测下一个 token
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有权重
        self.apply(self._init_weights)

        # 打印参数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=False):
        """统计模型参数量"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        """权重初始化：线性层和 Embedding 用正态分布"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     输入 token id 序列, shape = (batch, seq_len)
        targets: 目标序列（训练时用于算 loss）, shape = (batch, seq_len)
        """
        b, t = idx.size()
        assert t <= self.args.block_size, \
            f"序列长度 {t} 超过最大 {self.args.block_size}"

        # 1. 词嵌入 + 位置编码
        tok_emb = self.transformer.wte(idx)        # (B, L, n_embd)
        pos_emb = self.transformer.wpe(tok_emb)    # 加上位置编码
        x = self.transformer.drop(pos_emb)

        # 2. Encoder → Decoder
        enc_out = self.transformer.encoder(x)
        x = self.transformer.decoder(x, enc_out)

        # 3. 输出
        if targets is not None:
            # 训练：所有位置都要预测，算交叉熵
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # 推理：只取最后一个位置预测下一个 token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
```

---

## 📝 代码解读（重点）

### 关键 1：`nn.ModuleDict` vs 普通 dict

- `ModuleDict` 会让 PyTorch **自动识别**里面的子模块（计入参数、可 `to(device)`）
- 普通 dict 不会，模型 save/load 会出问题

### 关键 2：`apply(self._init_weights)`

- 递归地对所有子模块调用 `_init_weights`
- 这是 PyTorch **自定义初始化**的标准套路

### 关键 3：`logits.view(-1, vocab_size)` + `targets.view(-1)`

- 把 `(B, L, V)` 拉平成 `(B*L, V)`
- targets 拉平成 `(B*L,)`
- 这样交叉熵可以**一次性算 B*L 个位置的 loss**

### 关键 4：推理时 `x[:, [-1], :]`

- 推理时只关心**最后一个位置**预测下一个 token
- 用 `[-1]` 加方括号是为了**保留维度**（结果还是 3D），方便后续操作

---

## 🎨 整体数据流可视化

```
输入 idx: (B, L)
      │
      ├──→ wte (Embedding) ──→ (B, L, n_embd)
      │
      ├──→ wpe (位置编码) ──→ (B, L, n_embd)
      │
      ├──→ drop (Dropout) ──→ (B, L, n_embd)
      │
      ├──→ Encoder × N ──→ enc_out: (B, L, n_embd)
      │           │
      │           ▼
      ├──→ Decoder × N（用 enc_out 做交叉注意力）──→ (B, L, n_embd)
      │
      └──→ lm_head ──→ logits: (B, L, vocab_size)
                          │
                          ├──→ 训练: 算 cross_entropy
                          └──→ 推理: 取最后一个位置 softmax 采样
```

---

## ⚠️ 小白避坑

1. **原论文用 Post-Norm，但实际源码和现代实现都是 Pre-Norm**
   → 见 [08-残差连接](08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md#-pre-norm-vs-post-norm)
2. **Encoder-Decoder 是机器翻译的经典架构**，但**现代 LLM 都不用了**
   - GPT/LLaMA 只用 Decoder
   - BERT 只用 Encoder
3. **vocab_size 通常很大**（几万到十几万），`lm_head` 是参数大头
4. **forward 里的 `assert` 是新手友好提示**，生产代码可以去掉

---

## 📝 本章一句话总结

**注意力 + 多头 + LayerNorm + 残差 + 位置编码 + Embedding + Encoder/Decoder = Transformer = 现代 LLM 的发动机**。

---

## ❓ 本章自测题（共 8 题）

→ 见 [自测题汇总-第 2 章](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#%E7%AC%AC-2-%E7%AB%A0)

---

## 📌 下一章预告

**第 3 章 预训练语言模型** —— 三大架构流派：
- **Encoder-only**：BERT、RoBERTa、ALBERT（理解任务）
- **Encoder-Decoder**：T5（万能转换）
- **Decoder-only**：GPT、LLaMA、GLM（生成任务，**现代 LLM 主流**）

看 LLM 是怎么从 Transformer 演化过来的。

---

## 🔗 延伸阅读

- 上一节：[09-位置编码](09-%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)
- 多头代码细节：[99-代码-MultiHeadAttention](99-%E4%BB%A3%E7%A0%81-MultiHeadAttention.md)
- 章节总览：[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 进入下一章：[00-章节总览](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

⬅ [09-位置编码](09-%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)　|	➡ [99-代码-MultiHeadAttention](99-%E4%BB%A3%E7%A0%81-MultiHeadAttention.md)
