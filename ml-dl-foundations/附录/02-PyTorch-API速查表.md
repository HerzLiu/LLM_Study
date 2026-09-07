---
tags: [ml-dl-foundations, 附录, PyTorch, API, 速查]
chapter: 附录
section: A.2
---

# 附录 A.2 PyTorch API 速查表

⬅ [01-术语速查（含与已有vault去重）](01-%E6%9C%AF%E8%AF%AD%E9%80%9F%E6%9F%A5%EF%BC%88%E5%90%AB%E4%B8%8E%E5%B7%B2%E6%9C%89vault%E5%8E%BB%E9%87%8D%EF%BC%89.md) | ➡ [03-资源外部链接表](03-%E8%B5%84%E6%BA%90%E5%A4%96%E9%83%A8%E9%93%BE%E6%8E%A5%E8%A1%A8.md)

---

## 🎬 用法

写训练代码时回这里查。**只覆盖 LLM/Agent 场景常用 API**，不全。

---

## 📦 1. Tensor 基础

```python
import torch

# 创建
torch.tensor([1, 2, 3])              # 从 list
torch.zeros(3, 4)                     # 全 0
torch.ones(3, 4)                      # 全 1
torch.randn(3, 4)                     # N(0, 1) 随机
torch.arange(10)                      # 0-9
torch.empty(3, 4)                     # 未初始化（快但要小心）

# 形状
x.shape                               # torch.Size([3, 4])
x.size()                              # 同上
x.view(2, 6)                          # reshape（必须连续内存）
x.reshape(2, 6)                       # reshape（自动处理）
x.transpose(0, 1)                     # 交换两个维度
x.permute(1, 0, 2)                    # 任意维度顺序
x.unsqueeze(0)                        # 加一维 [3,4] → [1,3,4]
x.squeeze()                           # 去掉为 1 的维

# 设备
x.to('cuda')                          # 移到 GPU
x.to(device)                          # 移到指定 device
x.cpu()                               # 移回 CPU

# 数据类型
x.float()                             # → fp32
x.bfloat16()                          # → bf16
x.long()                              # → int64（label 用）
x.detach()                            # 断开计算图
```

---

## 🧮 2. 数学运算

```python
# 逐元素
a + b, a - b, a * b, a / b
torch.exp(x), torch.log(x), torch.sqrt(x)
torch.sin(x), torch.tanh(x)

# 矩阵运算
a @ b                                 # 矩阵乘
torch.matmul(a, b)                    # 同上
torch.bmm(a, b)                       # batch 矩阵乘
torch.einsum('bij,bjk->bik', a, b)    # 一般张量运算

# 聚合
x.sum()                               # 全部求和
x.sum(dim=-1)                         # 某维度求和
x.mean(dim=0)                         # 某维度求平均
x.max(dim=-1)                         # 返回 (values, indices)
x.argmax(dim=-1)                      # 只要索引

# 范数
torch.norm(x)                         # L2 范数
x.norm(p=2, dim=-1)                   # 某维度 L2
```

---

## 🧱 3. 神经网络层（nn.Module）

```python
import torch.nn as nn
import torch.nn.functional as F

# 全连接（神经元）
linear = nn.Linear(in_features=10, out_features=5)
y = linear(x)                         # x.shape=[B,10], y.shape=[B,5]
linear.weight                         # [5, 10] 可训权重
linear.bias                           # [5] 可训偏置

# Embedding
emb = nn.Embedding(num_embeddings=50000, embedding_dim=512)
y = emb(token_ids)                    # ids.shape=[B,T], y.shape=[B,T,512]

# Norm
ln = nn.LayerNorm(normalized_shape=512)
bn = nn.BatchNorm1d(num_features=64)

# Dropout
drop = nn.Dropout(p=0.1)              # 训练时启用，推理时关

# 激活
F.relu(x)
F.gelu(x)
F.silu(x)                              # = swish
F.softmax(x, dim=-1)
F.log_softmax(x, dim=-1)

# Attention（PyTorch 内置）
F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

---

## 🎨 4. 自定义模型

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = MyModel().to(device)

# 看参数
for name, p in model.named_parameters():
    print(name, p.shape, p.requires_grad)

# 参数总数
total = sum(p.numel() for p in model.parameters())

# 冻结/解冻
for p in model.fc1.parameters():
    p.requires_grad = False
```

---

## 🎯 5. 损失函数

```python
import torch.nn.functional as F

# 回归
F.mse_loss(y_pred, y)
F.l1_loss(y_pred, y)                  # = MAE

# 分类（最常用）
F.cross_entropy(logits, labels)                       # 多分类（输入 logits）
F.cross_entropy(logits, labels, ignore_index=-100)    # SFT 必用
F.binary_cross_entropy_with_logits(logits, labels)    # 二分类（输入 logits，推荐）

# KL 散度
F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')

# 用 nn.Module 形式（等价，可统一管理）
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fn(logits, labels)
```

---

## ⚙️ 6. 优化器和学习率调度

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# 优化器
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
opt = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.95))

# lr 调度
scheduler = CosineAnnealingLR(opt, T_max=10000)

# HF 一键 warmup+cosine
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=200, num_training_steps=10000)

# 用法
opt.zero_grad()
loss.backward()
opt.step()
scheduler.step()                      # ← 注意 step() 顺序
```

---

## 🔄 7. 训练循环（核心 5 步）

```python
model.train()                          # 训练模式（启用 dropout）
for batch in dataloader:
    x, y = batch
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()              # 1. 清零旧梯度
    pred = model(x)                    # 2. 前向
    loss = loss_fn(pred, y)            # 3. 算 loss
    loss.backward()                    # 4. 反向传播

    # 可选: 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()                   # 5. 更新参数
    scheduler.step()                   # 6. 调 lr
```

---

## 📊 8. 评估循环

```python
model.eval()                           # 评估模式（关 dropout）
correct = 0
with torch.no_grad():                  # 不要梯度（省显存 + 快）
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
acc = correct / len(test_set)
```

---

## 💾 9. 保存 / 加载

```python
# 保存
torch.save(model.state_dict(), 'model.pt')

# 加载
model = MyModel()
model.load_state_dict(torch.load('model.pt'))
model.to(device)
model.eval()

# 保存训练 checkpoint（含 optimizer 状态）
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')
```

---

## 🚀 10. AMP（混合精度）

```python
from torch.cuda.amp import autocast, GradScaler

# fp16 + GradScaler（老办法）
scaler = GradScaler()
for batch in dataloader:
    optimizer.zero_grad()
    with autocast(dtype=torch.float16):
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# bf16（推荐）
for batch in dataloader:
    optimizer.zero_grad()
    with autocast(dtype=torch.bfloat16):
        loss = model(batch)
    loss.backward()                    # 不需要 scaler
    optimizer.step()
```

---

## 📦 11. 数据加载

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

loader = DataLoader(
    MyDataset(data),
    batch_size=32,
    shuffle=True,
    num_workers=4,                     # 加速加载
    pin_memory=True,                   # GPU 训练加速
)
```

---

## 🤗 12. Hugging Face 常用

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16)

# Tokenize
inputs = tokenizer("Hello", return_tensors="pt")

# 推理
outputs = model.generate(**inputs, max_new_tokens=50)
text = tokenizer.decode(outputs[0])

# 训练
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./out",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    bf16=True,
    gradient_accumulation_steps=8,
    warmup_steps=200,
    lr_scheduler_type="cosine",
)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
trainer.train()
```

---

## 🔧 13. 常用 debug 技巧

```python
# 看张量形状
print(f"x shape: {x.shape}")

# 看是否有 NaN
print(torch.isnan(x).any())

# 看梯度
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name}: grad norm = {p.grad.norm():.4f}")

# 看显存
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 看模型结构
print(model)

# 看参数总数
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
```

---

## ⚠️ 14. 常见坑

| 坑 | 解 |
|---|---|
| `loss.backward()` 报错 "graph traversed twice" | 没 `optimizer.zero_grad()` |
| Tensor 在 CPU 但模型在 GPU | 都 `.to(device)` |
| `view` 报错 | 改用 `reshape` |
| `nn.CrossEntropyLoss` 输入概率 | **输入 logits**！它内部 softmax |
| 推理 OOM | 用 `with torch.no_grad():` |
| `optimizer.step()` 没效果 | 看是否 `zero_grad` + `backward` 顺序 |
| `model.eval()` 后还在 dropout | 检查是不是 `model.train()` 又开了 |

---

## 🔗 延伸阅读

- 上一节：[01-术语速查（含与已有vault去重）](01-%E6%9C%AF%E8%AF%AD%E9%80%9F%E6%9F%A5%EF%BC%88%E5%90%AB%E4%B8%8E%E5%B7%B2%E6%9C%89vault%E5%8E%BB%E9%87%8D%EF%BC%89.md)
- 下一节：[03-资源外部链接表](03-%E8%B5%84%E6%BA%90%E5%A4%96%E9%83%A8%E9%93%BE%E6%8E%A5%E8%A1%A8.md)
- 主教材 D2L PyTorch：https://zh.d2l.ai
- 官方文档：https://pytorch.org/docs/stable/

---

⬅ [01-术语速查（含与已有vault去重）](01-%E6%9C%AF%E8%AF%AD%E9%80%9F%E6%9F%A5%EF%BC%88%E5%90%AB%E4%B8%8E%E5%B7%B2%E6%9C%89vault%E5%8E%BB%E9%87%8D%EF%BC%89.md) | ➡ [03-资源外部链接表](03-%E8%B5%84%E6%BA%90%E5%A4%96%E9%83%A8%E9%93%BE%E6%8E%A5%E8%A1%A8.md)
