---
tags: [Happy-LLM, 第6章, 实战速览, Trainer, DeepSpeed, ZeRO]
chapter: 6
section: 6.1.4-6.1.5
---

# 6.1.4-6.1.5 Trainer + DeepSpeed 实战要点

⬅ [01-从手工作坊到工业流水线](01-%E4%BB%8E%E6%89%8B%E5%B7%A5%E4%BD%9C%E5%9D%8A%E5%88%B0%E5%B7%A5%E4%B8%9A%E6%B5%81%E6%B0%B4%E7%BA%BF.md)	|	➡ [03-SFT实战要点](03-SFT%E5%AE%9E%E6%88%98%E8%A6%81%E7%82%B9.md)

> ⚡ 本节是**实战速览**，重点理解"DeepSpeed 为什么这么用"和"ZeRO 配置文件的关键字段"。完整代码请看 Happy-LLM 仓库 `code/pretrain.py`。

---

## 🎯 Trainer 类做了什么？

把第 5 章的训练循环**全部封装**：

| 第 5 章手写 | **Trainer 封装** |
|---|---|
| 写 `for epoch in ...` | ✅ |
| 写 `for step in ...` | ✅ |
| 写 `forward + backward` | ✅ |
| 写 GradScaler 混合精度 | ✅ |
| 写梯度累积 | ✅（参数 `gradient_accumulation_steps`） |
| 写 LR 调度 | ✅（参数 `lr_scheduler_type`） |
| 写日志记录 | ✅（集成 wandb / swanlab / tensorboard） |
| 写 checkpoint 保存 | ✅（参数 `save_steps`） |
| 写多卡分布式 | ✅（DeepSpeed/DDP/FSDP 一行切换） |

→ **你只需关心：模型、数据、超参**。

---

## 🐍 TrainingArguments 核心超参

```python
training_args = TrainingArguments(
    output_dir="output",                  # 保存路径
    per_device_train_batch_size=4,        # 单卡 batch
    gradient_accumulation_steps=4,        # 梯度累积步数 → effective_batch = 4×4×n_gpus
    num_train_epochs=1,
    learning_rate=1e-4,
    warmup_steps=200,
    bf16=True,                            # bf16 训练
    gradient_checkpointing=True,          # 用时间换显存
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,                   # 最多保留 1 个 checkpoint
    deepspeed="./ds_config_zero2.json",   # ⭐ DeepSpeed 配置文件
    report_to="swanlab",                   # 实验追踪
)
```

⭐ **`gradient_checkpointing`** 是省显存利器：
- 不保存中间激活，反向传播时**重新前向计算**
- 显存省 30%~50%，速度慢 20%~30%
- 训大模型时**必开**

---

## ⚡ 为什么需要 DeepSpeed？

[第 4 章](../04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05-Pretrain%E9%A2%84%E8%AE%AD%E7%BB%83.md#-zero-%E4%B8%8E-deepspeed) 讲过原理：
> 7B 模型用 Adam 优化器，**单卡显存装不下完整训练状态**。
> 解法：**ZeRO 把模型状态切分到多卡**。

|训练状态| 模型参数 | 模型梯度 | Adam 状态 | **总占用** |
|---|---|---|---|---|
| 1B 模型 (bf16) | 2GB | 2GB | 12GB | **16GB** |
| 7B 模型 (bf16) | 14GB | 14GB | 84GB | **112GB** |

→ 单卡 80G A100 都装不下 7B 训练，必须用 ZeRO 切分。

---

## 🐍 DeepSpeed 配置文件（`ds_config_zero2.json`）

> 不需要写 Python 代码，**配置文件 + `--deepspeed config.json` 一行启动**。

### 关键字段

```json
{
    "bf16": {
        "enabled": "auto"                  // 自动用 bf16
    },
    "optimizer": {
        "type": "AdamW",                    // 优化器
        "params": {
            "lr": "auto",                   // 从 TrainingArguments 同步
            "betas": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,                         // ⭐ ZeRO 阶段：1 / 2 / 3
        "offload_optimizer": {
            "device": "none",               // 不 offload；可选 "cpu" 进一步省显存
            "pin_memory": true
        },
        "overlap_comm": true,               // 计算和通信重叠
        "reduce_scatter": true,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### ⭐ 选哪个 ZeRO Stage？

| Stage | 切分内容 | 显存 | 速度 | 推荐场景 |
|---|---|---|---|---|
| **ZeRO-1** | 只切优化器状态 | 中 | 快 | 7B 以下小模型 |
| **ZeRO-2** ⭐ | + 梯度 | 小 | 中 | **7B~13B（最常用）** |
| **ZeRO-3** | + 模型参数 | 极小 | 慢 | 70B+ 超大模型 |
| ZeRO-3 + CPU offload | + 状态扔到内存 | 巨小 | 极慢 | 用小显存训巨大模型 |

→ 详见 [ZeRO 三级优化](../04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05-Pretrain%E9%A2%84%E8%AE%AD%E7%BB%83.md#-zero-%E4%B8%8E-deepspeed)。

---

## 🛠 启动训练（bash 脚本）

```bash
CUDA_VISIBLE_DEVICES=0,1

deepspeed pretrain.py \
    --config_name Qwen/Qwen2.5-1.5B \
    --tokenizer_name Qwen/Qwen2.5-1.5B \
    --train_files data/pretrain.jsonl \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_steps 5 \
    --save_steps 100 \
    --block_size 2048 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero2.json \
    --output_dir output/pretrain \
    --report_to swanlab
```

→ `deepspeed` 命令会自动启动多卡训练（替代 `python`）。

---

## 📝 工程化注意事项（速览）

### 1. 用 `bash` 脚本而不是 Jupyter

> 长时间训练，Jupyter 容易**中断丢失进度**，所以用 `.sh` + `.py` 脚本。

### 2. 用 `logging` 而不是 `print`

> 大规模训练日志量大，`print` 容易刷屏丢失信息。用 Python `logging` 库分级别记录。

```python
import logging
logger = logging.getLogger(__name__)
logger.info("开始训练")
logger.warning("显存不足")
```

### 3. checkpoint 恢复机制

```python
from transformers.trainer_utils import get_last_checkpoint

last_checkpoint = get_last_checkpoint(training_args.output_dir)
if last_checkpoint is not None:
    logger.info(f"从 {last_checkpoint} 恢复训练")
trainer.train(resume_from_checkpoint=last_checkpoint)
```

→ 训练崩了**自动从最后一个 checkpoint 续训**。

### 4. SwanLab / WandB 监控

```python
import swanlab
swanlab.init(project="pretrain", experiment_name="qwen-1.5b")

# TrainingArguments 里加：
# report_to="swanlab"
```

→ 浏览器实时看 loss、lr、显存利用率。

---

## ⚠️ 小白避坑

1. **DeepSpeed config 里的 `"auto"` 字段会自动从 `TrainingArguments` 同步**
   - 不要手动写死值，否则会冲突
2. **ZeRO-3 速度慢很多**
   - 不到必要别开
   - 7B 以下用 ZeRO-2 通常足够
3. **多卡训练用 `deepspeed` 启动，不要用 `python`**
   - `deepspeed pretrain.py ...` 才会调用多卡
4. **save_total_limit 设小一点**
   - 否则 checkpoint 把硬盘塞满
5. **不开 `gradient_checkpointing` 训不动 7B+**
   - 显存爆炸

---

## 📌 6.1 节要点回顾

- HuggingFace **Transformers + Trainer** 封装了 90% 的训练样板代码
- **DeepSpeed** 通过 ZeRO 分布式让多卡训练大模型成为可能
- 训练全流程 = **配置文件 + 几行 Python**，工业级简洁

---

## 🔗 延伸阅读

- 上一节：[01-从手工作坊到工业流水线](01-%E4%BB%8E%E6%89%8B%E5%B7%A5%E4%BD%9C%E5%9D%8A%E5%88%B0%E5%B7%A5%E4%B8%9A%E6%B5%81%E6%B0%B4%E7%BA%BF.md)
- 下一节：[03-SFT实战要点](03-SFT%E5%AE%9E%E6%88%98%E8%A6%81%E7%82%B9.md)
- 理论：[05-Pretrain预训练](../04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05-Pretrain%E9%A2%84%E8%AE%AD%E7%BB%83.md)

---

⬅ [01-从手工作坊到工业流水线](01-%E4%BB%8E%E6%89%8B%E5%B7%A5%E4%BD%9C%E5%9D%8A%E5%88%B0%E5%B7%A5%E4%B8%9A%E6%B5%81%E6%B0%B4%E7%BA%BF.md)	|	➡ [03-SFT实战要点](03-SFT%E5%AE%9E%E6%88%98%E8%A6%81%E7%82%B9.md)
