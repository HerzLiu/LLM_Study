---
tags: [ml-dl-foundations, 第5章, CNN, 卷积, 速览]
chapter: 5
section: 5.1
---

# 5.1 CNN 卷积神经网络（速览）

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-RNN-LSTM-GRU](02-RNN-LSTM-GRU.md)

---

## 🎬 故事比喻：用同一个"模板"扫描整张图

```
你看一张猫的图，找"猫耳朵":
  - 不用每个像素位置都用不同的"找猫耳朵眼睛"
  - 用同一个"猫耳朵模板"，从左上扫到右下，每个位置匹配一下
  - 哪里匹配度高 → 那里有猫耳朵

这就是卷积 (convolution):
  - 共享同一组权重（卷积核 kernel）
  - 在输入上滑动
  - 每个位置算一次内积
```

> 你的目标是 LLM/Agent，**CNN 用得少**，本节速览。
> 但需要懂"为什么 ImageNet 2012 引爆 DL"——这是史观必备。

---

## 🧩 5.1.1 CNN 的核心组件

### 卷积层 (Convolutional Layer)

**输入**：一张图 $H \times W \times C$（高 × 宽 × 通道，如 224×224×3）

**卷积核 (kernel/filter)**：小矩阵 $k \times k \times C$（如 3×3×3）

**操作**：

```
输入图:                  卷积核 K (3×3):
┌──────────────────┐     ┌─────────┐
│ . . . . . . . . │     │ a b c   │
│ . . . . . . . . │     │ d e f   │
│ . . . X X X . . │ ⊗   │ g h i   │
│ . . . X X X . . │     └─────────┘
│ . . . X X X . . │
│ . . . . . . . . │
└──────────────────┘
   滑动 K 到每个位置, 算内积
   →  得到一张"激活图" (feature map)
```

**计算每个位置**：

$$\text{output}[i, j] = \sum_{u, v, c} K[u, v, c] \cdot \text{input}[i+u, j+v, c]$$

**关键性质**：
- **参数共享**：同一个 K 用在所有位置（vs FFN 每个位置一组参数）
- **局部连接**：每个输出只依赖输入的一小块（vs FFN 全连接）
- **平移不变**：图里物体在哪都能识别

### 池化层 (Pooling)

**作用**：降采样（缩小空间维度）。

**Max Pooling 2×2**：每 2×2 区域取最大值。

### 完整 CNN（如 VGG）

```
输入图 224×224×3
   ↓
[Conv 3×3, 64] + ReLU + [Conv 3×3, 64] + ReLU + MaxPool
   ↓ 112×112×64
[Conv 3×3, 128] + ... + MaxPool
   ↓ 56×56×128
[Conv 3×3, 256] + ... + MaxPool
   ↓ 28×28×256
[Conv 3×3, 512] + ... + MaxPool
   ↓ 14×14×512
[Conv 3×3, 512] + ... + MaxPool
   ↓ 7×7×512
Flatten + FFN → 1000 类输出
```

---

## 🏆 5.1.2 CNN 经典网络（一图速看）

| 网络 | 年份 | 创新 | 错误率 (ImageNet) |
|---|---|---|---|
| **LeNet-5** | 1998 | 第一个 CNN | — (MNIST) |
| **AlexNet** | 2012 | ReLU + Dropout + GPU | **16%** ⭐ 引爆 DL |
| **VGG** | 2014 | 全部 3×3 卷积，19 层 | 7.3% |
| **GoogLeNet (Inception)** | 2014 | 多尺度并行 | 6.7% |
| **ResNet** | 2015 | **残差连接**，152 层 | 3.6% ⭐ |
| **EfficientNet** | 2019 | 自动搜索架构 | 2.5% |
| **ViT (Vision Transformer)** | 2020 | **用 Transformer 做视觉** | 击败 CNN |

→ 2020+ ViT 出现后，CV 也开始转向 Transformer。

---

## ⭐ 5.1.3 ResNet 残差连接（最重要的 CNN 概念）

**问题**：网络越深越好，但 100+ 层时反向传播梯度消失，**深网反而效果差**。

**ResNet 解法**：加 **shortcut（跳过连接）**：

$$\mathbf{y} = \mathbf{x} + F(\mathbf{x})$$

**图示**：

```
       x ──────────────┐
       │                │
       ▼                │
       Conv             │
       │                │
       ▼                ▼
       ReLU             加
       │                │
       ▼                │
       Conv             │
       │                │
       └────────────────┘
              ↓
              y
```

**为什么有效？** 反向传播时 $\partial y / \partial x = 1 + \partial F / \partial x$，**+1 保证梯度不消失**。

→ **现代 Transformer 也用残差连接**（Happy-LLM §2.8 详讲）。
→ 这是从 CNN 借来的最重要思想之一。

---

## 🆚 5.1.4 CNN vs Transformer（CV 视角）

| 维度 | CNN | Transformer (ViT) |
|---|---|---|
| 归纳偏置 | 强（局部性 + 平移不变） | 弱（全 attention） |
| 数据需求 | 中等 | **海量** |
| 长程依赖 | 弱（要堆很多层） | **强**（一层即全图） |
| 计算 | 卷积，省 | Attention $O(N^2)$ |
| 现状 | 小数据 + 移动端仍主流 | **大数据上击败 CNN** |

→ **2020+ ViT 等让 Transformer 进军 CV**。
→ 多模态 LLM（GPT-4V / Claude Vision）也用 Transformer 处理图像。

---

## 🔑 5.1.5 对你最重要的 3 个 take-away

1. **CNN 引爆了 DL**（AlexNet 2012）—— 历史必知
2. **残差连接**是 CNN 借给 Transformer 的最大遗产
3. **CV 现在也在 Transformer 化**（ViT、SAM、DiT）—— 多模态 LLM 时代到了

→ 你**不需要会设计 CNN**，但要知道**CV 的 DL 体系是怎么走过来的**。

---

## 💻 5.1.6 PyTorch 一行调用（速览）

```python
import torch
import torchvision.models as models

# 直接加载预训练 ResNet
model = models.resnet50(pretrained=True)

# 或自己写一个小 CNN
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)
```

→ D2L §6-§7 有完整 CNN 教程。

---

## ⚠️ 5.1.7 常见误解

| 误解 | 真相 |
|---|---|
| "卷积只能用于图像" | 也可用 1D 卷积处理时序（如 WaveNet 音频） |
| "CNN 比 Transformer 老所以差" | 小数据 + 移动端 CNN 还是首选 |
| "ResNet 是 CNN" | ResNet 是 CNN 架构，但**残差连接被 Transformer 借用** |
| "ViT 一定击败 CNN" | 在**大数据**上才行；小数据 CNN 仍领先 |
| "多模态 LLM 用 CNN" | 现代多模态 LLM **几乎都用 ViT 风格** |

---

## 📌 5.1 节要点

| 概念 | 一句话 |
|---|---|
| 卷积 | 共享权重的"滑动模板匹配" |
| 池化 | 降采样 |
| ResNet 残差 | $y = x + F(x)$，解决深网梯度消失 |
| ViT | Transformer 进军 CV，2020+ 趋势 |
| 对你 | 知道 CNN 史观 + 残差思想就够 |

---

## 🔗 延伸阅读

- 上一节：[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一节：[02-RNN-LSTM-GRU](02-RNN-LSTM-GRU.md)
- 残差在 Transformer：[08-残差连接](../../Happy-LLM/02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md)
- 转折历史：[05-传统ML到DL的转折](../03-%E7%AC%AC3%E7%AB%A0-%E7%BB%8F%E5%85%B8%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8/05-%E4%BC%A0%E7%BB%9FML%E5%88%B0DL%E7%9A%84%E8%BD%AC%E6%8A%98.md)
- 外部：D2L §6-§7 CNN https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) | ➡ [02-RNN-LSTM-GRU](02-RNN-LSTM-GRU.md)
