---
tags: [Happy-LLM, 第5章, 实战, Transformer, 完整模型]
chapter: 5
section: 5.1.6
---

# 5.1.6 构建完整 LLaMA2 模型

⬅ [05-DecoderLayer组装](05-DecoderLayer%E7%BB%84%E8%A3%85.md)	|	➡ [07-训练Tokenizer](07-%E8%AE%AD%E7%BB%83Tokenizer.md)

> 🎉 终于到了**激动人心的一刻** —— 把所有零件拼成一个真正的 LLaMA2 模型！

---

## 🏗 整体架构

```
输入 tokens (B, L)
       │
       ▼
  tok_embeddings      ← 词嵌入
       │
       ▼
   Dropout
       │
       ▼
  ┌─────────────────┐
  │ DecoderLayer 1  │  ← 用预算好的 freqs_cos/sin 做 RoPE
  │ DecoderLayer 2  │
  │      ...        │
  │ DecoderLayer N  │
  └─────────────────┘
       │
       ▼
   RMSNorm           ← 最后一次归一化
       │
       ▼
   output (Linear)    ← 投影到词表（与 tok_embeddings 权重共享）
       │
       ▼
   logits (B, L, vocab_size)
       │
       ├──→ 训练：算 cross_entropy loss
       └──→ 推理：取最后位置 softmax 采样下一个 token
```

---

## 🐍 完整代码

```python
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional

class Transformer(PreTrainedModel):
    config_class = ModelConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)

        # N 层 DecoderLayer
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))

        # 最后一次归一化
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # 输出投影：dim → vocab_size
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # ⭐ 权重共享：词嵌入和输出层共享权重
        self.tok_embeddings.weight = self.output.weight

        # 预计算 RoPE 的 cos/sin（一次性算好，所有层共用）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)

        # 对残差投影做特殊缩放初始化（GPT-2 论文的技巧）
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=0.02 / math.sqrt(2 * args.n_layers)
                )

        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        """权重初始化：线性层和 Embedding 用正态分布"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None, **keyargs):
        # 兼容 HF 风格的参数
        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']

        _bsz, seqlen = tokens.shape

        # 1. 词嵌入 + Dropout
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        # 2. 取出本次需要的 RoPE 频率
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 3. 过 N 层 DecoderLayer
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        # 4. 最后归一化
        h = self.norm(h)

        # 5. 输出
        if targets is not None:
            # 训练：所有位置都要预测
            logits = self.output(h)
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,       # 0 是 padding，不算 loss
                reduction='none'      # 不求平均，外面用 loss_mask 算
            )
        else:
            # 推理：只取最后一个位置算 logits
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        # 包装成 HF 风格输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256,
                 temperature=1.0, top_k=None):
        """
        自回归生成 token
        idx: 起始 token 序列 (B, L)
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 若超过最大长度，截断（不用 KV cache 的简单版）
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len \
                       else idx[:, -self.args.max_seq_len:]

            # 前向得到最后位置的 logits
            logits = self(idx_cond).logits
            logits = logits[:, -1, :]   # (B, vocab_size)

            if temperature == 0.0:
                # 贪心：选最大概率
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 温度采样
                logits = logits / temperature
                if top_k is not None:
                    # top-k 过滤：只保留 top-k，其他设为 -inf
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # 遇到终止符就停
            if idx_next == stop_id:
                break

            # 把新 token 接到序列后
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:]   # 只返回新生成的部分
```

---

## 📝 代码深度解读

### 关键 1：权重共享（Weight Tying）

```python
self.tok_embeddings.weight = self.output.weight
```

⭐ 这是个**重要技巧**：让词嵌入层和输出层**共享同一份权重矩阵**。

**为什么？**
- 词嵌入：`vocab_size × dim`（id → 向量）
- 输出层：`dim × vocab_size`（向量 → 词表分数）
- 这两个矩阵**形状互为转置**，本质上是同一个东西
- 共享后**参数量减半**，效果还更好

**节省的参数量**：
- `vocab_size=6144, dim=1024` → 6.3M 参数（占小模型的 ~10%）

### 关键 2：RoPE 频率预计算 + register_buffer

```python
freqs_cos, freqs_sin = precompute_freqs_cis(
    self.args.dim // self.args.n_heads,
    self.args.max_seq_len
)
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

- **预先算好所有位置的 cos/sin**，后续直接查表
- `register_buffer`：跟随模型 save/load，但**不参与梯度更新**
- `persistent=False`：不保存到 state_dict（节省存储，因为可以重算）

### 关键 3：残差投影的特殊初始化

```python
for pn, p in self.named_parameters():
    if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers))
```

- 对 MLP 的 `w3` 和 Attention 的输出投影 `wo` 用**更小的初始化方差**
- 防止深层网络初始时**累积过大的梯度**

这是 GPT-2 论文的**经典稳定训练技巧**，缩放因子是 `1/√(2N)`，N 是层数。

### 关键 4：训练 vs 推理的 forward 不同

```python
if targets is not None:
    # 训练
    logits = self.output(h)              # 所有位置的 logits
    self.last_loss = F.cross_entropy(...)
else:
    # 推理
    logits = self.output(h[:, [-1], :])  # 只算最后一个位置
    self.last_loss = None
```

**推理时为什么只取最后位置？**
- 自回归生成：每次只需要预测下一个 token
- 前面位置的 logits 已经用过了
- → **大幅节省计算**

### 关键 5：`ignore_index=0`

```python
F.cross_entropy(..., ignore_index=0, reduction='none')
```

- 训练数据里 padding 用 `0` 填充
- `ignore_index=0` 告诉交叉熵**忽略 padding 位置的 loss**
- `reduction='none'`：返回每个位置的 loss，**让外面用 loss_mask 做精细控制**

### 关键 6：`generate` 的采样策略

```python
if temperature == 0.0:
    _, idx_next = torch.topk(logits, k=1, dim=-1)   # 贪心
else:
    logits = logits / temperature                    # 温度缩放
    if top_k is not None:
        ...                                          # top-k 过滤
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)  # 多项式采样
```

| 策略 | 效果 |
|---|---|
| **temperature=0** | 贪心：完全确定，但容易重复 |
| **temperature=0.7** | 平衡（常用） |
| **temperature=1.5** | 更随机，更有创造性 |
| **top_k=50** | 只在 top-50 候选里采样，避免低概率 token |

→ 这就是 ChatGPT 后端的**采样原理**。

---

## 🔬 测试

```python
x = torch.randint(0, 6144, (1, 50))   # (batch=1, seq_len=50)
model = Transformer(args=args)

num_params = sum(p.numel() for p in model.parameters())
print('Number of parameters:', num_params)

out = model(x)
print(out.logits.shape)
```

输出：
```
Number of parameters: 82594560
torch.Size([1, 1, 6144])
```

✅ **8259 万参数的 LLaMA2 模型**搭建完成！
（输出 shape 是 `(1, 1, 6144)` 是因为推理模式只取最后一个位置）

---

## 🆚 模型规模对比

通过修改 `ModelConfig` 就能缩放：

| 配置 | dim | n_layers | 参数量 |
|---|---|---|---|
| 默认 | 768 | 12 | **~80M** |
| 实战版（5.3 节用） | 1024 | 18 | **~215M** |
| LLaMA-7B | 4096 | 32 | 7B |
| LLaMA-13B | 5120 | 40 | 13B |

→ **同一份代码框架**，缩放参数即可适配不同规模。

---

## 🎯 完整模型回顾（13 行代码概括）

```python
# 1. 词嵌入
self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
# 2. N 层 Decoder
self.layers = nn.ModuleList([DecoderLayer(i, args) for i in range(args.n_layers)])
# 3. 最后归一化
self.norm = RMSNorm(args.dim, eps=args.norm_eps)
# 4. 输出层
self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
# 5. 权重共享
self.tok_embeddings.weight = self.output.weight
# 6. RoPE 频率
freqs_cos, freqs_sin = precompute_freqs_cis(...)

# Forward:
h = self.dropout(self.tok_embeddings(tokens))
for layer in self.layers:
    h = layer(h, freqs_cos, freqs_sin)
h = self.norm(h)
logits = self.output(h)
```

→ **就这 13 行**，构成了一个能跑通的 LLaMA2 模型主干。

---

## ⚠️ 小白避坑

1. **权重共享必须在初始化所有权重之后做**
   - 否则共享关系会被覆盖
2. **`generate` 不要用普通 `torch.no_grad()`**
   - 用 `@torch.inference_mode()` 更快（更彻底地关 autograd）
3. **`temperature=0` 时不能用 softmax**
   - 直接用 topk，避免数值问题
4. **`vocab_size` 一旦确定就不能改**
   - 否则旧权重无法加载

---

## 📌 5.1 节小结

到此为止，5.1 节的所有内容已经完成 —— 我们**从零实现了一个完整的 LLaMA2 模型**！

| 文件 | 内容 |
|---|---|
| [01-定义超参数](01-%E5%AE%9A%E4%B9%89%E8%B6%85%E5%8F%82%E6%95%B0.md) | ModelConfig |
| [02-RMSNorm代码实现](02-RMSNorm%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0.md) | RMSNorm |
| [03-GQA与RoPE](03-GQA%E4%B8%8ERoPE.md) | Attention（含 GQA + RoPE） |
| [04-SwiGLU-MLP](04-SwiGLU-MLP.md) | MLP（SwiGLU） |
| [05-DecoderLayer组装](05-DecoderLayer%E7%BB%84%E8%A3%85.md) | DecoderLayer |
| **本节** | **Transformer（完整模型）** |

接下来 5.2 节我们要训练自己的 **Tokenizer** —— 把汉字变成模型能吃的 id。

---

## 🔗 延伸阅读

- 上一节：[05-DecoderLayer组装](05-DecoderLayer%E7%BB%84%E8%A3%85.md)
- 下一节：[07-训练Tokenizer](07-%E8%AE%AD%E7%BB%83Tokenizer.md)
- 理论基础：[07-LLaMA](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07-LLaMA.md)

---

⬅ [05-DecoderLayer组装](05-DecoderLayer%E7%BB%84%E8%A3%85.md)	|	➡ [07-训练Tokenizer](07-%E8%AE%AD%E7%BB%83Tokenizer.md)
