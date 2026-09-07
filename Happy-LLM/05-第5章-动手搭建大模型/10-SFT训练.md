---
tags: [Happy-LLM, 第5章, 实战, SFT]
chapter: 5
section: 5.3.5
---

# 5.3.5 SFT 训练

⬅ [09-预训练循环](09-%E9%A2%84%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)	|	➡ [11-生成文本](11-%E7%94%9F%E6%88%90%E6%96%87%E6%9C%AC.md)

> 预训练好的模型是个"博览群书的书呆子"，**SFT 让它学会对话**。

---

## 🎬 故事比喻：把书呆子培训成助理

```
预训练完的模型：博览群书但不会对话
        ↓ SFT 训练
教他理解指令 + 给出回复
        ↓
得到 Chat / Instruct 模型
```

---

## 🆚 SFT 训练 vs 预训练 —— 几乎一样！

SFT 训练流程**和预训练 95% 相同**，只有 3 处区别：

| 项       | Pretrain                      | **SFT**                      |
| ------- | ----------------------------- | ---------------------------- |
| Dataset | [PretrainDataset](08-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%9E%84%E9%80%A0.md) | **[SFTDataset](08-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%9E%84%E9%80%A0.md)** |
| 初始权重    | 从零开始                          | **加载预训练权重**                  |
| 优化器     | Adam                          | **AdamW**（常用）                |
| 其余训练循环  | ✅ 一样                          | ✅ 一样                         |

---

## 🐍 SFT 完整代码

```python
import os, time, math, warnings
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer
from k_model import ModelConfig, Transformer
from dataset import SFTDataset
import swanlab

warnings.filterwarnings('ignore')


def get_lr(it, all):
    """Warmup + Cosine（和预训练一样）"""
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch):
    """训练一个 epoch（和预训练几乎一样）"""
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X, Y, loss_mask = X.to(args.device), Y.to(args.device), loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            print(f'Epoch:[{epoch+1}/{args.epochs}] ({step}/{iter_per_epoch}) '
                  f'loss:{loss.item() * args.accumulation_steps:.3f} lr:{lr:.7f}')

        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}.pth'
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) \
                         else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model():
    """初始化模型（关键：加载预训练权重）"""
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')
    model = Transformer(lm_config)

    # ⭐ 关键：加载预训练权重
    ckp = './base_model_215M/pretrain_1024_18_6144.pth'
    state_dict = torch.load(ckp, map_location=args.device)

    # 兼容性处理：去掉 torch.compile 加的前缀
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)

    # 多卡处理
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(args.device)
    return model, tokenizer


if __name__ == "__main__":
    # ... argparse 解析（和预训练一样）
    # 主要参数：
    # --data_path BelleGroup_sft.jsonl
    # --learning_rate 2e-4
    # --batch_size 64
    # --accumulation_steps 8

    lm_config = ModelConfig(dim=1024, n_layers=18)
    max_seq_len = lm_config.max_seq_len

    # 初始化
    model, tokenizer = init_model()

    # ⭐ 用 SFTDataset 而不是 PretrainDataset
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, pin_memory=True,
        drop_last=False, shuffle=True, num_workers=args.num_workers
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    # ⭐ 用 AdamW（带 weight decay 的 Adam，SFT 常用）
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)
```

---

## 📝 关键差异详解

### 差异 1：加载预训练权重

```python
ckp = './base_model_215M/pretrain_1024_18_6144.pth'
state_dict = torch.load(ckp, map_location=args.device)

# 兼容性处理：torch.compile 可能加 _orig_mod. 前缀
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict, strict=False)
```

**关键点**：
- `strict=False` 允许部分权重不匹配（比如新加的层）
- 前缀处理是为了**兼容用 torch.compile 训练保存的权重**
- 不加载预训练权重的话，SFT **就是从零训练**，效果会很差

### 差异 2：用 SFTDataset

```python
train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
```

SFTDataset 内部用了 [generate_loss_mask](08-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%9E%84%E9%80%A0.md#-533-sftdataset%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D)，只对 assistant 部分算 loss。

### 差异 3：AdamW 优化器

```python
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

| 优化器 | 区别 |
|---|---|
| Adam | 经典 |
| **AdamW** | 加了**权重衰减**（weight decay）= L2 正则化 |

AdamW 在 SFT/微调场景更常用，**有助于防止过拟合**。

---

## 🆚 完整对比：Pretrain vs SFT 代码

```diff
- # Pretrain
- model = Transformer(lm_config)
- # 不加载权重，从零训练
- train_ds = PretrainDataset(...)
- optimizer = optim.Adam(...)

+ # SFT
+ model = Transformer(lm_config)
+ state_dict = torch.load('pretrain_xxx.pth')   # 加载预训练
+ model.load_state_dict(state_dict, strict=False)
+ train_ds = SFTDataset(...)                     # 用 SFTDataset
+ optimizer = optim.AdamW(...)                    # 用 AdamW
```

→ **就这 4 处不同**！

---

## 🛠 启动 SFT

```bash
python sft.py \
    --data_path BelleGroup_sft.jsonl \
    --gpus 0,1,2,3 \
    --batch_size 64 \
    --accumulation_steps 8 \
    --learning_rate 2e-4 \
    --epochs 1 \
    --use_swanlab
```

---

## 📊 SFT 训练资源

| 项 | 数值 |
|---|---|
| 模型参数 | 215M |
| SFT 数据 | 3.5M 条对话 |
| GPU | 4 × A100 (80G) |
| 训练时间 | **约 2~5 小时**（比预训练快很多） |

→ SFT 通常**比预训练快很多**（数据少、轮数少）。

---

## ⭐ SFT 关键技巧总结

### 1. 学习率比预训练小

| 阶段 | 典型 lr |
|---|---|
| Pretrain | 1e-4 ~ 5e-4 |
| **SFT** | **5e-6 ~ 5e-5** |

为什么？预训练已经有了好基础，SFT 只是**微调**，大 lr 会破坏预训练知识。

> 注：本书 demo 为了演示用了 2e-4，实际生产可调小。

### 2. 数据质量比数量重要

- 几千条高质量人工标注 >> 几百万条低质量自动生成
- 优先用 ChatGPT/GPT-4 生成的数据 + 人工校对

### 3. 多任务混合训练

把各种类型数据混在一起（翻译、摘要、问答、编程、闲聊…）→ 模型获得**泛化指令遵循能力**。

### 4. 多轮对话训练

如本章 [SFTDataset](08-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%9E%84%E9%80%A0.md#-533-sftdataset%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D) 设计 —— **方式 3**（一次性预测多轮）最高效。

### 5. 别 SFT 太久

- SFT 1~3 epoch 通常足够
- 太久会**过拟合**指令风格，丧失通用能力（**灾难性遗忘**）

---

## ⚠️ 小白避坑

1. **SFT 前必须有预训练权重**
   - 否则就是"从零学对话"，效果差
2. **检查 loss_mask 是否正确**
   - print 几个样本，确认只有 assistant 部分 = 1
3. **学习率别和预训练一样大**
   - 实际生产建议 5e-6 ~ 5e-5
4. **数据格式要和 chat_template 一致**
   - 否则 generate_loss_mask 找不到 assistant 区间
5. **SFT 后模型可能"啰嗦"**
   - 因为标注员喜欢长回复
   - 这是 ChatGPT 早期"过度啰嗦"的根源

---

## 🔗 延伸阅读

- 上一节：[09-预训练循环](09-%E9%A2%84%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)
- 下一节：[11-生成文本](11-%E7%94%9F%E6%88%90%E6%96%87%E6%9C%AC.md)
- 理论：[06-SFT有监督微调](../04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-SFT%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md)

---

⬅ [09-预训练循环](09-%E9%A2%84%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)	|	➡ [11-生成文本](11-%E7%94%9F%E6%88%90%E6%96%87%E6%9C%AC.md)
