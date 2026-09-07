---
tags: [ml-dl-foundations, 第4章, 初始化, LayerNorm, BatchNorm, Dropout, 正则化]
chapter: 4
section: 4.8
---

# 4.8 初始化 / 归一化 / Dropout（深度学习"工程三件套"）

⬅ [07-学习率调度](07-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6.md) | ➡ [09-PyTorch最小训练循环](09-PyTorch%E6%9C%80%E5%B0%8F%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)

---

## 🎬 故事比喻：盖楼三件套

```
你要盖一栋摩天大楼（深度神经网络）：

  1. 初始化 = 打地基。地基歪了，楼盖不起来
     - 全 0 初始化 → 楼直接塌
     - 太大初始化 → 楼一阵风就倒
     - 合适初始化 → 稳稳盖

  2. 归一化 = 每层加抗震结构
     - 没归一化 → 楼摇得越来越厉害
     - LayerNorm/BatchNorm → 每层都"重置稳定性"

  3. Dropout = 多重保险
     - 训练时随机断电几个神经元 → 防止过度依赖个别神经元
     - 测试时全开 → 模型抗扰动能力强
```

> 这一节解释 Happy-LLM 第 2 章学过但没深入的 LayerNorm、Pre-Norm、Dropout，以及一直没讲过的**权重初始化**。

---

## 🏗 4.8.1 权重初始化（Weight Initialization）

### 为什么不能全 0 初始化？

如果所有 $\mathbf{W}^{(i)} = 0, \mathbf{b}^{(i)} = 0$：
- 第 1 层所有神经元输出都一样
- 反向传播时所有梯度都一样
- 所有神经元**永远保持相同** → 等价于 1 个神经元

→ 称为"**对称性问题**"。

### 为什么不能太大？

如果初始权重很大（如 $W \sim N(0, 1)$ 但 dim=4096）：
- 第 1 层输出 $\mathbf{W} \mathbf{x}$ 数值很大
- 经过 sigmoid → 全部饱和到 0 或 1
- 梯度 ≈ 0 → 学不动

→ "**梯度消失**"。

### Xavier / Glorot 初始化（2010）

适合 **sigmoid / tanh** 激活：

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)$$

**直觉**：让每层输入输出**方差大致相同**，避免越深越爆/越衰。

### Kaiming / He 初始化（2015）

适合 **ReLU** 激活（ReLU 把负半轴砍掉，输入方差减半，需要补回来）：

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}}}\right)$$

→ **ResNet/CNN 标配**。

### LLM 用的初始化

- 大多数 LLM 用 **truncated normal**：$W \sim \mathcal{N}(0, \sigma^2)$ 截断到 ±2σ
- $\sigma$ 通常 0.02 或 $\sqrt{2/(5 \cdot d_{\text{model}})}$
- 偏置 $b$ 初始化为 0

→ 这就是为什么 Happy-LLM 代码里你看到 `nn.init.normal_(p, mean=0.0, std=0.02)`。

---

## 📊 4.8.2 归一化（Normalization）

**目标**：让每层的输入"分布稳定"（均值 0、方差 1 附近）。

**为什么重要**：
- 训练过程中分布会**漂移**（covariate shift）
- 后层的输入不断变化，难以收敛
- 容易梯度消失/爆炸

### BatchNorm（2015）

对**一个 batch 内**的**同一特征维度**归一化：

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

其中 $\mu_B, \sigma_B$ 是**当前 batch** 在这个特征上的均值和方差，$\gamma, \beta$ 是可学习的缩放和偏移。

**问题**（NLP 场景）：
- 依赖 batch 内统计 → batch 小时不稳
- 序列长度不固定时尴尬
- 推理时还要维护 running mean/var

→ **NLP/LLM 几乎不用 BatchNorm**。

### LayerNorm（2016）⭐ Transformer/LLM 标配

对**单个样本**的**所有特征**归一化：

$$\hat{x}_i = \frac{x_i - \mu_x}{\sqrt{\sigma_x^2 + \epsilon}}, \quad y_i = \gamma_i \hat{x}_i + \beta_i$$

其中 $\mu_x, \sigma_x$ 是**这一个样本**所有特征的均值方差。

**优势**：
- ✅ 和 batch 无关 → batch=1 也能用
- ✅ 序列长度无关
- ✅ 推理简单

### BatchNorm vs LayerNorm（一图）

```
设 batch 是 [batch_size=4, features=3] 的矩阵：

      f1  f2  f3
sample1 [a, b, c]
sample2 [d, e, f]
sample3 [g, h, i]
sample4 [j, k, l]

BatchNorm: 沿"列方向"归一化
  - f1 列 {a, d, g, j} 减均值除标准差
  - f2 列 {b, e, h, k} 同样
  - f3 列 {c, f, i, l} 同样
  → 4 个样本"配对"算

LayerNorm: 沿"行方向"归一化
  - 第 1 行 {a, b, c} 减均值除标准差
  - 第 2 行 {d, e, f} 同样
  - ...
  → 每个样本独立
```

→ Happy-LLM §2.7 已讲过这张图。

### RMSNorm（2019）⭐ LLaMA 用的

LayerNorm 的简化版——**只除标准差，不减均值**：

$$y_i = \frac{x_i}{\sqrt{\frac{1}{d} \sum_j x_j^2 + \epsilon}} \cdot \gamma_i$$

**优势**：
- ✅ 比 LayerNorm 快 10-50%
- ✅ 效果几乎一样
- ✅ LLaMA / Qwen / DeepSeek 都用

### Pre-Norm vs Post-Norm（重要）

**Post-Norm（原始 Transformer 2017）**：

$$\mathbf{y} = \text{LN}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$$

**Pre-Norm（现代 LLM 标准）**：

$$\mathbf{y} = \mathbf{x} + \text{Sublayer}(\text{LN}(\mathbf{x}))$$

**关键**：Pre-Norm 把 LN 移到残差之前。
**优势**：训练**更稳定**——梯度有"直通通道"（残差 +1 不被 LN 缩放）。
**所有现代 LLM** 都用 Pre-Norm。

→ Happy-LLM §2.8 已详讲。

---

## 💧 4.8.3 Dropout（防过拟合）

**思想**：训练时**随机把一部分神经元的输出置 0**，强迫网络不依赖某个神经元。

```
正常前向:
  h = [0.3, 0.7, 0.1, 0.9, 0.5]

Dropout(p=0.4) 后:
  h = [0.5, 1.17, 0,   1.5, 0  ]
       ↑     ↑    ↑    ↑    ↑
       保留  保留  drop 保留  drop
       并放大 (除以 0.6，保持期望不变)
```

**数学**：

$$y_i = \begin{cases}
x_i / (1-p) & \text{以概率 } 1-p \\
0 & \text{以概率 } p
\end{cases}$$

**关键**：训练时 dropout + scale，**推理时关闭 dropout**（用全部神经元）。

**为什么有效**？

1. **集成视角**：每次训练等于在"子网络"上训，等价于训了 $2^N$ 个模型的集成
2. **正则化**：强迫每个神经元独立有用，不能"摸鱼依赖别人"

**用在哪**：
- ✅ FFN 层之间
- ✅ Attention 输出之后
- ⚠️ **LayerNorm 之后**不建议
- ⚠️ 现代 LLM 训练**不太用 dropout**（数据足够多，正则化效果不明显）

### 典型 p 值

| 场景 | p |
|---|---|
| 小模型（CNN） | 0.5 |
| BERT | 0.1 |
| GPT-2 | 0.1 |
| 现代 LLM 训练 | 0（不用） |
| LLM 微调（防过拟合小数据） | 0.1 |

---

## 🛡 4.8.4 别的正则化技术

### L1 / L2 正则化（Weight Decay）

在 loss 上加上参数大小的惩罚：

$$\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda \|\theta\|_2^2 \quad \text{(L2)}$$

**直觉**："参数太大有罪"——逼模型用小参数解释数据。

→ **AdamW 的 weight_decay 参数**就是这个。详见 [§4.6.6](06-%E4%BC%98%E5%8C%96%E5%99%A8%E8%BF%9B%E5%8C%96-SGD%E5%88%B0AdamW.md)。

### Early Stopping（早停）

监控验证集 loss，**连续 N 步不降就停**。

### Data Augmentation（数据增强）

CV：旋转/裁剪/翻转图片
NLP：同义词替换/back-translation/mask

→ **LLM 几乎不用**（数据本来就海量）。

---

## 💻 4.8.5 PyTorch 代码（概念示意，未严格验证）

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Pre-Norm 风格
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

# 初始化
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

model = MyModel()
model.apply(init_weights)

# 训练 vs 推理
model.train()  # dropout 启用
loss = model(x)

model.eval()   # dropout 关闭
with torch.no_grad():
    output = model(x)
```

---

## ⚠️ 4.8.6 常见误解

| 误解 | 真相 |
|---|---|
| "初始化随便填就行" | 错！全 0 / 太大都会让训练失败 |
| "BatchNorm 比 LayerNorm 强" | CV 是，NLP **不是**——LayerNorm 完胜 |
| "Dropout 越大越好" | 太大 (>0.5) 会欠拟合 |
| "推理时也开 Dropout" | **不！** 推理时关闭（`model.eval()` 自动） |
| "AdamW 的 weight_decay 和 Dropout 一回事" | 都是正则化，但**机制不同**——前者改 loss，后者改激活 |
| "LayerNorm 后面要加 ReLU" | 不需要（LayerNorm 已经"归一化"了） |
| "Pre-Norm 和 Post-Norm 性能一样" | Pre-Norm **训练更稳**，是现代标准 |

---

## 📌 4.8 节要点

| 工程三件套 | 解决 |
|---|---|
| **权重初始化** | 防止对称性、梯度消失/爆炸（用 Xavier/Kaiming/截断正态） |
| **归一化** | 让每层输入分布稳定（NLP 用 LayerNorm/RMSNorm） |
| **Dropout** | 防过拟合（LLM 训练已少用） |

**LLM 训练标准配置**：
- 初始化：`std=0.02` 的截断正态
- 归一化：**RMSNorm + Pre-Norm**
- Dropout：训练时 0，微调时 0.1
- 正则化：靠 AdamW 的 weight_decay（不靠 Dropout）

---

## 🔗 延伸阅读

- 上一节：[07-学习率调度](07-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6.md)
- 下一节：[09-PyTorch最小训练循环](09-PyTorch%E6%9C%80%E5%B0%8F%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)
- 已有锚点：[07-LayerNorm与BatchNorm](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/07-LayerNorm%E4%B8%8EBatchNorm.md)（深度讲解）
- 已有锚点：[08-残差连接](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md)（Pre-Norm vs Post-Norm）
- 已有锚点：[术语表](../../Happy-LLM/%E9%99%84%E5%BD%95/%E6%9C%AF%E8%AF%AD%E8%A1%A8.md) § Dropout
- 外部：D2L §5.5 https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html

---

⬅ [07-学习率调度](07-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6.md) | ➡ [09-PyTorch最小训练循环](09-PyTorch%E6%9C%80%E5%B0%8F%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)
