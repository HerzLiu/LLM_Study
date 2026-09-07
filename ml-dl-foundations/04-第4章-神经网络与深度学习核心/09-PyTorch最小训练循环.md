---
tags: [ml-dl-foundations, 第4章, PyTorch, 训练循环, MNIST, 实战]
chapter: 4
section: 4.9
---

# 4.9 PyTorch 最小训练循环（30 行训 MNIST）

⬅ [08-初始化-归一化-Dropout](08-%E5%88%9D%E5%A7%8B%E5%8C%96-%E5%BD%92%E4%B8%80%E5%8C%96-Dropout.md) | ➡ [00-章节总览](../05-%E7%AC%AC5%E7%AB%A0-%E5%85%B8%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%B2%E8%AE%B2/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

## 🎬 故事比喻：把 §4.1-§4.8 拼起来

```
§4.1 神经元      ─┐
§4.2 MLP         ─┤
§4.3 激活函数    ─┤
§4.4 损失函数    ─┼──→ 30 行 PyTorch 代码 ──→ 训出能识别 MNIST 的模型
§4.5 反向传播    ─┤
§4.6 优化器      ─┤
§4.7 学习率调度  ─┤
§4.8 初始化/Norm ─┘
```

> 这一节是第 4 章的**毕业作品**。
> 把前 8 节学的东西拼成一个能跑的完整 PyTorch 程序。
> **看懂这 30 行 = 看懂所有后续 DL 代码**（包括 Happy-LLM 的 LLM 训练）。

---

## 🎯 4.9.1 任务：识别手写数字

```
输入:           输出:
[28×28 灰度图]  [0-9 的概率分布]
   ┌────┐         [0.01, 0.02, ..., 0.9, ..., 0.01]
   │  3 │  →                              ↑
   │    │                            "3" 概率 0.9
   └────┘
```

**MNIST**：60000 张手写数字训练 + 10000 张测试。**ML 界的 "Hello World"**。

---

## 🏗 4.9.2 模型设计

最简单的 MLP：

```
输入: 28×28 = 784 维
   ↓
隐藏层 1: 128 神经元 + ReLU
   ↓
隐藏层 2: 64 神经元 + ReLU
   ↓
输出层: 10 神经元（10 个类别）
   ↓
Softmax (cross_entropy 内部自带)
   ↓
loss
```

---

## 💻 4.9.3 完整代码（概念示意，未严格验证可运行）

```python
"""
PyTorch 最小训练循环：MLP 识别 MNIST
依赖: pip install torch torchvision
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# 1. 数据 (Dataset + DataLoader)
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),                          # PIL → Tensor [0,1]
    transforms.Normalize((0.1307,), (0.3081,)),     # 标准化
])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=64, shuffle=False)

# ============================================================
# 2. 模型 (nn.Module)
# ============================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)     # §4.2 第 1 层
        self.fc2 = nn.Linear(128, 64)           # §4.2 第 2 层
        self.fc3 = nn.Linear(64, 10)            # §4.2 输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)                 # flatten 图片
        x = F.relu(self.fc1(x))                 # §4.3 ReLU
        x = F.relu(self.fc2(x))                 # §4.3 ReLU
        x = self.fc3(x)                         # 输出 logits（不激活）
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)

# ============================================================
# 3. 损失函数 + 优化器 (§4.4 + §4.6)
# ============================================================
loss_fn = nn.CrossEntropyLoss()                 # §4.4 多分类 CE
optimizer = optim.AdamW(model.parameters(), lr=1e-3)  # §4.6 AdamW

# ============================================================
# 4. 训练循环 (§4.5 5 步)
# ============================================================
def train_epoch(epoch):
    model.train()                               # §4.8 启用 dropout/BN
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # ─── 核心 5 步 ───
        optimizer.zero_grad()                   # 1. 清零旧梯度（§4.5）
        output = model(data)                    # 2. 前向（§4.5）
        loss = loss_fn(output, target)          # 3. 算 loss（§4.4）
        loss.backward()                         # 4. 反向传播（§4.5）
        optimizer.step()                        # 5. 更新参数（§4.6）

        total_loss += loss.item()
    print(f"Epoch {epoch}: train loss = {total_loss / len(train_loader):.4f}")

# ============================================================
# 5. 评估循环
# ============================================================
def evaluate():
    model.eval()                                # §4.8 关 dropout/BN
    correct = 0
    with torch.no_grad():                       # 推理不要梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)         # 选最高概率的类
            correct += (pred == target).sum().item()
    acc = correct / len(test_set)
    print(f"Test accuracy: {acc:.4f}")

# ============================================================
# 6. 跑！
# ============================================================
for epoch in range(1, 6):
    train_epoch(epoch)
    evaluate()
```

**预期输出**：

```
Epoch 1: train loss = 0.30xx     Test accuracy: 0.95xx
Epoch 2: train loss = 0.12xx     Test accuracy: 0.96xx
Epoch 3: train loss = 0.08xx     Test accuracy: 0.97xx
Epoch 4: train loss = 0.06xx     Test accuracy: 0.97xx
Epoch 5: train loss = 0.04xx     Test accuracy: 0.98xx
```

→ **5 个 epoch，98% 准确率**——你的第一个 DL 模型成功了。

---

## 🧭 4.9.4 这 30 行代码用到了第 4 章哪些知识？

| 代码 | 对应节 |
|---|---|
| `nn.Linear(in, out)` | §4.1 神经元 / §4.2 MLP |
| `F.relu(...)` | §4.3 激活函数 |
| `nn.CrossEntropyLoss()` | §4.4 损失函数 |
| `loss.backward()` | §4.5 反向传播 |
| `optim.AdamW(..., lr=1e-3)` | §4.6 优化器 |
| `model.train() / model.eval()` | §4.8 Dropout/BN 训练/推理切换 |
| `optimizer.zero_grad()` → `optimizer.step()` 5 步 | §4.5 完整训练步 |

---

## 🆚 4.9.5 和 Happy-LLM SFT 训练对比

把上面 30 行**和 Happy-LLM 第 5 章 SFT 训练核心代码对照**：

| 概念 | MNIST MLP (本节) | Happy-LLM SFT |
|---|---|---|
| 模型 | 3 层 MLP | LLaMA-style Transformer |
| 输入 | 28×28 灰度图 | (input_ids, attention_mask) 序列 |
| 输出 | 10 类概率 | 词表 V 维概率 |
| Loss | CrossEntropyLoss | CrossEntropyLoss（带 -100 mask） |
| Optimizer | AdamW lr=1e-3 | AdamW lr=5e-5 |
| 训练步 | zero_grad → forward → backward → step | **完全一样** |
| 评估 | accuracy | perplexity / 验证 loss |

→ **核心 5 步完全一样**！LLM 训练 = 同样 5 步 + 更大模型 + 更复杂数据 + 工程优化（AMP、ZeRO、梯度累积）。

→ 后续看 LLM 训练代码，**你能秒认每一步**。

---

## 🚀 4.9.6 进一步可做的实验（推荐你跑一下）

| 实验 | 改什么 | 预期 |
|---|---|---|
| 减小学习率 | `lr=1e-5` | 收敛慢 10 倍 |
| 增大学习率 | `lr=1` | 训练发散，loss=NaN |
| 不用 ReLU | 删掉 `F.relu` | 准确率从 98% → 92%（多层折叠为一层） |
| 加 Dropout | `nn.Dropout(0.5)` 在 fc1/fc2 后 | 训练慢但泛化稍好 |
| 加 LayerNorm | `nn.LayerNorm(128)` | 训练更稳 |
| 换 SGD | `optim.SGD(lr=0.01)` | 收敛慢，需要更多 epoch |

→ **每个实验 = 第 4 章某一节的"消融测试"**。
→ 跑一遍，第 4 章的每个概念都"亲手验证"了。

---

## 📓 4.9.7 配套阅读资源

**D2L 第 3-5 章对应内容**：

- D2L §3 线性神经网络 https://zh.d2l.ai/chapter_linear-networks/index.html
- D2L §4 多层感知机 https://zh.d2l.ai/chapter_multilayer-perceptrons/index.html
- D2L §5 深度学习计算 https://zh.d2l.ai/chapter_deep-learning-computation/index.html

→ 这些章节用 PyTorch 实际跑通你刚学的所有概念。
→ **强烈建议跟着 D2L 跑一遍**，比看 100 篇博客有用。

---

## ⚠️ 4.9.8 常见坑

| 坑 | 解决 |
|---|---|
| `RuntimeError: Trying to backward through the graph a second time` | 多个 `loss.backward()` 没清梯度。加 `optimizer.zero_grad()` |
| `Tensor not in CUDA` | 模型在 GPU 但数据在 CPU。两边都 `.to(device)` |
| Test acc 比 train acc 高很多 | 数据有问题（如 test 简单），或 bug |
| Train loss 下降但 test acc 没涨 | **过拟合**（§1.4） |
| `nan` loss | lr 太大、初始化不好、梯度爆炸 |
| `out of memory` | batch_size 太大，减半 |

→ 训练遇到问题速查 [04-梯度问题诊断](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/04-%E6%A2%AF%E5%BA%A6%E9%97%AE%E9%A2%98%E8%AF%8A%E6%96%AD.md) + [05-训练诊断手册](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/05-%E8%AE%AD%E7%BB%83%E8%AF%8A%E6%96%AD%E6%89%8B%E5%86%8C.md)

---

## 📌 4.9 节要点

| 项 | 内容 |
|---|---|
| **训练流程 5 步** | `zero_grad → forward → loss → backward → step` |
| **训练 / 评估切换** | `model.train()` / `model.eval()` + `with torch.no_grad()` |
| **核心组件** | DataLoader / nn.Module / Loss / Optimizer |
| **MNIST baseline** | 3 层 MLP + ReLU + AdamW → 5 epoch 达到 98% |
| **和 LLM SFT 一致** | 同样 5 步骨架，更大模型 + 更多工程 |

---

## 🎓 4.9.9 第 4 章总评

**读到这里你应该能**：

- [ ] 解释 LLM 训练代码每一行在干嘛
- [ ] 改 5e-5、batch_size、epoch 知道有什么后果
- [ ] 看到 loss 曲线异常能定位问题
- [ ] 看到任何 DL 论文公式不再发懵

**还不会的**：

- 想自己设计新架构（这要看 D2L §6-§10）
- 想做大规模训练（这要第 6 章工程篇）
- 想做 RL 训练（看 agentic-rl-knowledge-base）

→ **本章已完成最关键的部分**。
→ 接下来读第 5 章把 CNN/RNN/Transformer 串起来，第 6 章工程优化，第 7 章桥接到你已学的世界。

---

## 🔗 延伸阅读

- 上一节：[08-初始化-归一化-Dropout](08-%E5%88%9D%E5%A7%8B%E5%8C%96-%E5%BD%92%E4%B8%80%E5%8C%96-Dropout.md)
- 下一章：[00-章节总览](../05-%E7%AC%AC5%E7%AB%A0-%E5%85%B8%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%B2%E8%AE%B2/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 进阶：[00-章节总览](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 桥接到你已学：[00-章节总览](../07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) ⭐
- 已有锚点：[10-SFT训练](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/10-SFT%E8%AE%AD%E7%BB%83.md)（对比看）
- 外部：D2L 全本 https://zh.d2l.ai

---

⬅ [08-初始化-归一化-Dropout](08-%E5%88%9D%E5%A7%8B%E5%8C%96-%E5%BD%92%E4%B8%80%E5%8C%96-Dropout.md) | ➡ [00-章节总览](../05-%E7%AC%AC5%E7%AB%A0-%E5%85%B8%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%B2%E8%AE%B2/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
