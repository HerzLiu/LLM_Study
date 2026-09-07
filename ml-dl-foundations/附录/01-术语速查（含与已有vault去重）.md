---
tags: [ml-dl-foundations, 附录, 术语速查, glossary]
chapter: 附录
section: A.1
---

# 附录 A.1 术语速查（与已有 vault 去重）

⬅ [05-你现在掌握的全景图](../07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/05-%E4%BD%A0%E7%8E%B0%E5%9C%A8%E6%8E%8C%E6%8F%A1%E7%9A%84%E5%85%A8%E6%99%AF%E5%9B%BE.md) | ➡ [02-PyTorch-API速查表](02-PyTorch-API%E9%80%9F%E6%9F%A5%E8%A1%A8.md)

---

## 🎬 用法

只列**本书覆盖的 ML/DL 基础术语**，**和 vault 已有术语表去重**：

- LLM/Transformer 术语 → 见 [术语表](../../Happy-LLM/%E9%99%84%E5%BD%95/%E6%9C%AF%E8%AF%AD%E8%A1%A8.md)
- RL/Agentic RL 术语 → 见 [01-术语速查（含常见误区）](../../agentic-rl-knowledge-base/%E9%99%84%E5%BD%95/01-%E6%9C%AF%E8%AF%AD%E9%80%9F%E6%9F%A5%EF%BC%88%E5%90%AB%E5%B8%B8%E8%A7%81%E8%AF%AF%E5%8C%BA%EF%BC%89.md)
- 公式速查 → [04-公式速查表](../../agentic-rl-knowledge-base/%E9%99%84%E5%BD%95/04-%E5%85%AC%E5%BC%8F%E9%80%9F%E6%9F%A5%E8%A1%A8.md)

---

## 🔤 ML/DL 基础术语

### A

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Activation Function** | 激活函数 | 给神经元加非线性 | [03-激活函数全家](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/03-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%85%A8%E5%AE%B6.md) |
| **Adam / AdamW** | – | 自适应学习率优化器 | [06-优化器进化-SGD到AdamW](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/06-%E4%BC%98%E5%8C%96%E5%99%A8%E8%BF%9B%E5%8C%96-SGD%E5%88%B0AdamW.md) |
| **AMP** | 自动混合精度 | bf16/fp16 + 自动管理精度 | [02-混合精度训练-fp16-bf16](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/02-%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83-fp16-bf16.md) |
| **AutoGrad** | 自动微分 | PyTorch 自动算梯度 | [05-梯度下降-反向传播完整推导](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md) |
| **Autoregressive** | 自回归 | 用前 t 个预测第 t+1 个 | [02-从交叉熵到next-token-loss](../07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/02-%E4%BB%8E%E4%BA%A4%E5%8F%89%E7%86%B5%E5%88%B0next-token-loss.md) |

### B

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Backpropagation** | 反向传播 | 链式法则的高效实现 | [05-梯度下降-反向传播完整推导](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md) |
| **Batch / Batch Size** | – | 一次喂模型多少样本 | [01-超参数调优体系](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/01-%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E4%BC%98%E4%BD%93%E7%B3%BB.md) |
| **BCE** | 二分类交叉熵 | $-y\log\hat{p} - (1-y)\log(1-\hat{p})$ | [04-损失函数大全](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/04-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%A4%A7%E5%85%A8.md) |
| **Bias** | 偏置 | 神经元的"门槛" | [01-神经元-从生物到数学](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/01-%E7%A5%9E%E7%BB%8F%E5%85%83-%E4%BB%8E%E7%94%9F%E7%89%A9%E5%88%B0%E6%95%B0%E5%AD%A6.md) |
| **bf16** | brain float 16 | 16 位浮点，范围 = fp32 | [02-混合精度训练-fp16-bf16](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/02-%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83-fp16-bf16.md) |
| **Bias-Variance Tradeoff** | 偏差方差权衡 | 欠拟合 vs 过拟合 | [04-过拟合欠拟合与正则化](../01-%E7%AC%AC1%E7%AB%A0-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/04-%E8%BF%87%E6%8B%9F%E5%90%88%E6%AC%A0%E6%8B%9F%E5%90%88%E4%B8%8E%E6%AD%A3%E5%88%99%E5%8C%96.md) |

### C

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **CE / CrossEntropy** | 交叉熵 | 多分类标配 loss | [04-损失函数大全](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/04-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%A4%A7%E5%85%A8.md) |
| **CNN** | 卷积神经网络 | 共享权重的滑动模板 | [01-CNN卷积神经网络](../05-%E7%AC%AC5%E7%AB%A0-%E5%85%B8%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%B2%E8%AE%B2/01-CNN%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md) |
| **Computational Graph** | 计算图 | 张量记录"来自哪" | [05-梯度下降-反向传播完整推导](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md) |

### D

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **DDP** | 分布式数据并行 | PyTorch 多卡训练 | [03-分布式与显存优化](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/03-%E5%88%86%E5%B8%83%E5%BC%8F%E4%B8%8E%E6%98%BE%E5%AD%98%E4%BC%98%E5%8C%96.md) |
| **DeepSpeed** | – | 微软分布式训练框架 | 同上 |
| **Derivative** | 导数 | 函数变化率 | [02-微积分3件套-导数偏导链式法则](../02-%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/02-%E5%BE%AE%E7%A7%AF%E5%88%863%E4%BB%B6%E5%A5%97-%E5%AF%BC%E6%95%B0%E5%81%8F%E5%AF%BC%E9%93%BE%E5%BC%8F%E6%B3%95%E5%88%99.md) |
| **Dropout** | – | 随机丢神经元防过拟合 | [08-初始化-归一化-Dropout](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/08-%E5%88%9D%E5%A7%8B%E5%8C%96-%E5%BD%92%E4%B8%80%E5%8C%96-Dropout.md) |

### E

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Early Stopping** | 早停 | val loss 不降就停 | [04-过拟合欠拟合与正则化](../01-%E7%AC%AC1%E7%AB%A0-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/04-%E8%BF%87%E6%8B%9F%E5%90%88%E6%AC%A0%E6%8B%9F%E5%90%88%E4%B8%8E%E6%AD%A3%E5%88%99%E5%8C%96.md) |
| **Epoch** | – | 数据集过完一遍 | [01-超参数调优体系](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/01-%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E4%BC%98%E4%BD%93%E7%B3%BB.md) |
| **Embedding** | 嵌入 | id → 稠密向量 | [术语表](../../Happy-LLM/%E9%99%84%E5%BD%95/%E6%9C%AF%E8%AF%AD%E8%A1%A8.md)（已有详讲） |

### F

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Forward** | 前向传播 | 模型从输入到输出 | [05-梯度下降-反向传播完整推导](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md) |
| **fp16 / fp32** | – | 浮点精度 | [02-混合精度训练-fp16-bf16](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/02-%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83-fp16-bf16.md) |
| **FSDP** | 全分片数据并行 | PyTorch 原生 ZeRO-3 | [03-分布式与显存优化](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/03-%E5%88%86%E5%B8%83%E5%BC%8F%E4%B8%8E%E6%98%BE%E5%AD%98%E4%BC%98%E5%8C%96.md) |

### G

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **GELU** | – | 平滑版 ReLU（BERT/GPT 用） | [03-激活函数全家](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/03-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%85%A8%E5%AE%B6.md) |
| **GPU** | – | 图形处理器，DL 的加速器 | – |
| **Gradient** | 梯度 | 偏导向量 | [02-微积分3件套-导数偏导链式法则](../02-%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/02-%E5%BE%AE%E7%A7%AF%E5%88%863%E4%BB%B6%E5%A5%97-%E5%AF%BC%E6%95%B0%E5%81%8F%E5%AF%BC%E9%93%BE%E5%BC%8F%E6%B3%95%E5%88%99.md) |
| **Gradient Accumulation** | 梯度累积 | 模拟大 batch 的技巧 | [01-超参数调优体系](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/01-%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E4%BC%98%E4%BD%93%E7%B3%BB.md) |
| **Gradient Checkpointing** | 梯度检查点 | 时间换显存 | [03-分布式与显存优化](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/03-%E5%88%86%E5%B8%83%E5%BC%8F%E4%B8%8E%E6%98%BE%E5%AD%98%E4%BC%98%E5%8C%96.md) |
| **Gradient Clipping** | 梯度裁剪 | 防梯度爆炸 | [04-梯度问题诊断](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/04-%E6%A2%AF%E5%BA%A6%E9%97%AE%E9%A2%98%E8%AF%8A%E6%96%AD.md) |
| **Gradient Descent** | 梯度下降 | $\theta -= \eta \nabla$ | [05-梯度下降-反向传播完整推导](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%AE%8C%E6%95%B4%E6%8E%A8%E5%AF%BC.md) |
| **GradScaler** | – | fp16 训练防梯度下溢 | [02-混合精度训练-fp16-bf16](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/02-%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83-fp16-bf16.md) |

### H

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Hidden Layer** | 隐藏层 | 输入输出之间的层 | [02-MLP前馈神经网络](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/02-MLP%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md) |
| **Hyperparameter** | 超参数 | 不靠梯度学的参数（lr/batch 等） | [01-超参数调优体系](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/01-%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E4%BC%98%E4%BD%93%E7%B3%BB.md) |

### I

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Initialization** | 初始化 | 权重起始值 | [08-初始化-归一化-Dropout](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/08-%E5%88%9D%E5%A7%8B%E5%8C%96-%E5%BD%92%E4%B8%80%E5%8C%96-Dropout.md) |
| **Inner Product** | 内积 | $\mathbf{a} \cdot \mathbf{b}$，标量 | [01-线性代数4件套-向量矩阵](../02-%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/01-%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B04%E4%BB%B6%E5%A5%97-%E5%90%91%E9%87%8F%E7%9F%A9%E9%98%B5.md) |

### K

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Kaiming Init** | – | ReLU 用的初始化 | [08-初始化-归一化-Dropout](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/08-%E5%88%9D%E5%A7%8B%E5%8C%96-%E5%BD%92%E4%B8%80%E5%8C%96-Dropout.md) |
| **KL Divergence** | KL 散度 | 两分布的差距 | [03-概率5件套-分布期望KL](../02-%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/03-%E6%A6%82%E7%8E%875%E4%BB%B6%E5%A5%97-%E5%88%86%E5%B8%83%E6%9C%9F%E6%9C%9BKL.md) |

### L

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **LayerNorm** | 层归一化 | 单样本所有特征归一化（NLP 标配） | [08-初始化-归一化-Dropout](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/08-%E5%88%9D%E5%A7%8B%E5%8C%96-%E5%BD%92%E4%B8%80%E5%8C%96-Dropout.md) |
| **Learning Rate** | 学习率 | 更新步长 | [07-学习率调度](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/07-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6.md) |
| **Linear Regression** | 线性回归 | $\hat{y} = \mathbf{w}^T\mathbf{x} + b$ | [01-线性回归从最小二乘到梯度下降](../03-%E7%AC%AC3%E7%AB%A0-%E7%BB%8F%E5%85%B8%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8/01-%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E4%BB%8E%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E5%88%B0%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md) |
| **Logistic Regression** | 逻辑回归 | 线性回归 + sigmoid，二分类 | [02-逻辑回归与交叉熵](../03-%E7%AC%AC3%E7%AB%A0-%E7%BB%8F%E5%85%B8%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8/02-%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E4%B8%8E%E4%BA%A4%E5%8F%89%E7%86%B5.md) |
| **LoRA** | 低秩适配 | 冻结模型 + 训小矩阵 | [术语表](../../Happy-LLM/%E9%99%84%E5%BD%95/%E6%9C%AF%E8%AF%AD%E8%A1%A8.md)（已详） |
| **Loss** | 损失 | 模型预测 vs 真实的差距 | [04-损失函数大全](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/04-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%A4%A7%E5%85%A8.md) |
| **LSTM** | 长短时记忆 | RNN 改进版 | [02-RNN-LSTM-GRU](../05-%E7%AC%AC5%E7%AB%A0-%E5%85%B8%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%B2%E8%AE%B2/02-RNN-LSTM-GRU.md) |

### M

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **MAE** | 平均绝对误差 | $\lvert \hat{y}-y \rvert$ | [04-损失函数大全](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/04-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%A4%A7%E5%85%A8.md) |
| **Matrix Multiplication** | 矩阵乘 | $[m,k] \times [k,n] = [m,n]$ | [01-线性代数4件套-向量矩阵](../02-%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/01-%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B04%E4%BB%B6%E5%A5%97-%E5%90%91%E9%87%8F%E7%9F%A9%E9%98%B5.md) |
| **MLE** | 最大似然估计 | 选参数让数据概率最大 | [03-概率5件套-分布期望KL](../02-%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/03-%E6%A6%82%E7%8E%875%E4%BB%B6%E5%A5%97-%E5%88%86%E5%B8%83%E6%9C%9F%E6%9C%9BKL.md) |
| **MLP** | 多层感知机 | = FFN，多层神经元堆叠 | [02-MLP前馈神经网络](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/02-MLP%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md) |
| **Momentum** | 动量 | 优化器加"惯性" | [06-优化器进化-SGD到AdamW](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/06-%E4%BC%98%E5%8C%96%E5%99%A8%E8%BF%9B%E5%8C%96-SGD%E5%88%B0AdamW.md) |
| **MSE** | 均方误差 | $(\hat{y}-y)^2$ | [04-损失函数大全](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/04-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%A4%A7%E5%85%A8.md) |

### N

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **NaN** | 非数 | 训练崩的典型表现 | [04-梯度问题诊断](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/04-%E6%A2%AF%E5%BA%A6%E9%97%AE%E9%A2%98%E8%AF%8A%E6%96%AD.md) |
| **Neuron** | 神经元 | $y = f(\mathbf{w}^T\mathbf{x}+b)$ | [01-神经元-从生物到数学](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/01-%E7%A5%9E%E7%BB%8F%E5%85%83-%E4%BB%8E%E7%94%9F%E7%89%A9%E5%88%B0%E6%95%B0%E5%AD%A6.md) |
| **NLL** | 负对数似然 | = CE（不同名字） | [04-损失函数大全](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/04-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%A4%A7%E5%85%A8.md) |
| **Normalization** | 归一化 | 让输入/激活稳定 | [08-初始化-归一化-Dropout](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/08-%E5%88%9D%E5%A7%8B%E5%8C%96-%E5%BD%92%E4%B8%80%E5%8C%96-Dropout.md) |

### O

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Optimizer** | 优化器 | 决定怎么更新参数（SGD/Adam） | [06-优化器进化-SGD到AdamW](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/06-%E4%BC%98%E5%8C%96%E5%99%A8%E8%BF%9B%E5%8C%96-SGD%E5%88%B0AdamW.md) |
| **Overfitting** | 过拟合 | 训练好但 val 差 | [04-过拟合欠拟合与正则化](../01-%E7%AC%AC1%E7%AB%A0-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/04-%E8%BF%87%E6%8B%9F%E5%90%88%E6%AC%A0%E6%8B%9F%E5%90%88%E4%B8%8E%E6%AD%A3%E5%88%99%E5%8C%96.md) |

### P

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Parameter** | 参数 | 模型权重和偏置（可学） | [01-神经元-从生物到数学](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/01-%E7%A5%9E%E7%BB%8F%E5%85%83-%E4%BB%8E%E7%94%9F%E7%89%A9%E5%88%B0%E6%95%B0%E5%AD%A6.md) |
| **Partial Derivative** | 偏导 | 多元函数对一个变量求导 | [02-微积分3件套-导数偏导链式法则](../02-%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/02-%E5%BE%AE%E7%A7%AF%E5%88%863%E4%BB%B6%E5%A5%97-%E5%AF%BC%E6%95%B0%E5%81%8F%E5%AF%BC%E9%93%BE%E5%BC%8F%E6%B3%95%E5%88%99.md) |
| **Perplexity** | 困惑度 | LLM 评估，$e^{\text{loss}}$ | [05-评估指标全家桶](../01-%E7%AC%AC1%E7%AB%A0-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/05-%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87%E5%85%A8%E5%AE%B6%E6%A1%B6.md) |
| **Pre-Norm / Post-Norm** | – | LN 放残差前/后 | [08-初始化-归一化-Dropout](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/08-%E5%88%9D%E5%A7%8B%E5%8C%96-%E5%BD%92%E4%B8%80%E5%8C%96-Dropout.md) |

### R

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **ReLU** | – | $\max(0, x)$ | [03-激活函数全家](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/03-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%85%A8%E5%AE%B6.md) |
| **Regularization** | 正则化 | 防过拟合 | [04-过拟合欠拟合与正则化](../01-%E7%AC%AC1%E7%AB%A0-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/04-%E8%BF%87%E6%8B%9F%E5%90%88%E6%AC%A0%E6%8B%9F%E5%90%88%E4%B8%8E%E6%AD%A3%E5%88%99%E5%8C%96.md) |
| **Residual Connection** | 残差连接 | $y = x + F(x)$ | [01-CNN卷积神经网络](../05-%E7%AC%AC5%E7%AB%A0-%E5%85%B8%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%B2%E8%AE%B2/01-CNN%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md) |
| **RMSNorm** | – | LayerNorm 简化版 | [08-初始化-归一化-Dropout](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/08-%E5%88%9D%E5%A7%8B%E5%8C%96-%E5%BD%92%E4%B8%80%E5%8C%96-Dropout.md) |
| **RNN** | 循环神经网络 | 处理序列的旧架构 | [02-RNN-LSTM-GRU](../05-%E7%AC%AC5%E7%AB%A0-%E5%85%B8%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%B2%E8%AE%B2/02-RNN-LSTM-GRU.md) |

### S

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **SGD** | 随机梯度下降 | 最朴素优化器 | [06-优化器进化-SGD到AdamW](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/06-%E4%BC%98%E5%8C%96%E5%99%A8%E8%BF%9B%E5%8C%96-SGD%E5%88%B0AdamW.md) |
| **Sigmoid** | – | $1/(1+e^{-x})$，输出 0-1 | [03-激活函数全家](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/03-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%85%A8%E5%AE%B6.md) |
| **Softmax** | – | 实数 → 概率分布 | [03-概率5件套-分布期望KL](../02-%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/03-%E6%A6%82%E7%8E%875%E4%BB%B6%E5%A5%97-%E5%88%86%E5%B8%83%E6%9C%9F%E6%9C%9BKL.md) |
| **Supervised Learning** | 监督学习 | 有 (x, y) 对 | [02-监督-无监督-强化学习三大范式](../01-%E7%AC%AC1%E7%AB%A0-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/02-%E7%9B%91%E7%9D%A3-%E6%97%A0%E7%9B%91%E7%9D%A3-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%89%E5%A4%A7%E8%8C%83%E5%BC%8F.md) |
| **SwiGLU** | – | LLM 用的门控 FFN | [03-激活函数全家](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/03-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%85%A8%E5%AE%B6.md) |

### T

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Tanh** | 双曲正切 | $(-1, 1)$ 激活函数 | [03-激活函数全家](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/03-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%85%A8%E5%AE%B6.md) |
| **Tensor** | 张量 | ≥3 维数组 | [01-线性代数4件套-向量矩阵](../02-%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/01-%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B04%E4%BB%B6%E5%A5%97-%E5%90%91%E9%87%8F%E7%9F%A9%E9%98%B5.md) |
| **Test Set** | 测试集 | 最终评估用（只看一次） | [03-训练验证测试集](../01-%E7%AC%AC1%E7%AB%A0-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/03-%E8%AE%AD%E7%BB%83%E9%AA%8C%E8%AF%81%E6%B5%8B%E8%AF%95%E9%9B%86.md) |
| **Train Set** | 训练集 | 训参数用 | 同上 |
| **Transpose** | 转置 | $A^T$ | [01-线性代数4件套-向量矩阵](../02-%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/01-%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B04%E4%BB%B6%E5%A5%97-%E5%90%91%E9%87%8F%E7%9F%A9%E9%98%B5.md) |

### U

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Underfitting** | 欠拟合 | 模型学不动 | [04-过拟合欠拟合与正则化](../01-%E7%AC%AC1%E7%AB%A0-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/04-%E8%BF%87%E6%8B%9F%E5%90%88%E6%AC%A0%E6%8B%9F%E5%90%88%E4%B8%8E%E6%AD%A3%E5%88%99%E5%8C%96.md) |
| **Unsupervised Learning** | 无监督学习 | 只有 x，没 y | [02-监督-无监督-强化学习三大范式](../01-%E7%AC%AC1%E7%AB%A0-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/02-%E7%9B%91%E7%9D%A3-%E6%97%A0%E7%9B%91%E7%9D%A3-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%89%E5%A4%A7%E8%8C%83%E5%BC%8F.md) |

### V

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Validation Set** | 验证集 | 调超参用 | [03-训练验证测试集](../01-%E7%AC%AC1%E7%AB%A0-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BF%83%E6%99%BA%E6%A8%A1%E5%9E%8B/03-%E8%AE%AD%E7%BB%83%E9%AA%8C%E8%AF%81%E6%B5%8B%E8%AF%95%E9%9B%86.md) |
| **Vector** | 向量 | $\mathbb{R}^n$ | [01-线性代数4件套-向量矩阵](../02-%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E6%9C%80%E5%B0%8F%E5%BF%85%E8%A6%81%E5%9F%BA%E7%A1%80/01-%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B04%E4%BB%B6%E5%A5%97-%E5%90%91%E9%87%8F%E7%9F%A9%E9%98%B5.md) |

### W

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Warmup** | – | lr 从 0 慢慢涨 | [07-学习率调度](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/07-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6.md) |
| **Weight** | 权重 | 神经元参数 | [01-神经元-从生物到数学](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/01-%E7%A5%9E%E7%BB%8F%E5%85%83-%E4%BB%8E%E7%94%9F%E7%89%A9%E5%88%B0%E6%95%B0%E5%AD%A6.md) |
| **Weight Decay** | 权重衰减 | L2 正则化 | [06-优化器进化-SGD到AdamW](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/06-%E4%BC%98%E5%8C%96%E5%99%A8%E8%BF%9B%E5%8C%96-SGD%E5%88%B0AdamW.md) |

### X / Z

| 术语 | 中文 | 一句话 | 详讲 |
|---|---|---|---|
| **Xavier Init** | – | sigmoid/tanh 用的初始化 | [08-初始化-归一化-Dropout](../04-%E7%AC%AC4%E7%AB%A0-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83/08-%E5%88%9D%E5%A7%8B%E5%8C%96-%E5%BD%92%E4%B8%80%E5%8C%96-Dropout.md) |
| **XGBoost** | – | 表格数据王者（boosting） | [03-决策树与集成（速览）](../03-%E7%AC%AC3%E7%AB%A0-%E7%BB%8F%E5%85%B8%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8/03-%E5%86%B3%E7%AD%96%E6%A0%91%E4%B8%8E%E9%9B%86%E6%88%90%EF%BC%88%E9%80%9F%E8%A7%88%EF%BC%89.md) |
| **ZeRO** | 零冗余优化器 | DeepSpeed 的分片技术 | [03-分布式与显存优化](../06-%E7%AC%AC6%E7%AB%A0-%E8%AE%AD%E7%BB%83%E5%AE%9E%E8%B7%B5%E4%B8%8E%E5%B7%A5%E7%A8%8B/03-%E5%88%86%E5%B8%83%E5%BC%8F%E4%B8%8E%E6%98%BE%E5%AD%98%E4%BC%98%E5%8C%96.md) |

---

## 🔗 跨 vault 术语表索引

| 主题 | 去哪查 |
|---|---|
| Transformer 内部组件（QKV / GQA / RoPE / ...） | [术语表](../../Happy-LLM/%E9%99%84%E5%BD%95/%E6%9C%AF%E8%AF%AD%E8%A1%A8.md) |
| LLM 训练流程（Pretrain / SFT / RLHF / LoRA / DeepSpeed / ...） | [术语表](../../Happy-LLM/%E9%99%84%E5%BD%95/%E6%9C%AF%E8%AF%AD%E8%A1%A8.md) |
| Agent 概念（ReAct / Memory / Tool / Planning / ...） | [00-总览](../../Hello-Agents/00-%E6%80%BB%E8%A7%88.md)（各章节内嵌） |
| RL 算法（PG / Actor-Critic / PPO / GRPO / RLVR / PRM / ...） | [01-术语速查（含常见误区）](../../agentic-rl-knowledge-base/%E9%99%84%E5%BD%95/01-%E6%9C%AF%E8%AF%AD%E9%80%9F%E6%9F%A5%EF%BC%88%E5%90%AB%E5%B8%B8%E8%A7%81%E8%AF%AF%E5%8C%BA%EF%BC%89.md) |
| 公式速查 | [04-公式速查表](../../agentic-rl-knowledge-base/%E9%99%84%E5%BD%95/04-%E5%85%AC%E5%BC%8F%E9%80%9F%E6%9F%A5%E8%A1%A8.md) |

---

## 🔗 延伸阅读

- 上一节：[05-你现在掌握的全景图](../07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/05-%E4%BD%A0%E7%8E%B0%E5%9C%A8%E6%8E%8C%E6%8F%A1%E7%9A%84%E5%85%A8%E6%99%AF%E5%9B%BE.md)
- 下一节：[02-PyTorch-API速查表](02-PyTorch-API%E9%80%9F%E6%9F%A5%E8%A1%A8.md)

---

⬅ [05-你现在掌握的全景图](../07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/05-%E4%BD%A0%E7%8E%B0%E5%9C%A8%E6%8E%8C%E6%8F%A1%E7%9A%84%E5%85%A8%E6%99%AF%E5%9B%BE.md) | ➡ [02-PyTorch-API速查表](02-PyTorch-API%E9%80%9F%E6%9F%A5%E8%A1%A8.md)
