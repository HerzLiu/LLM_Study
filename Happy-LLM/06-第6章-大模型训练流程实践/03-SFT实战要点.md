---
tags: [Happy-LLM, 第6章, 实战速览, SFT, ChatTemplate]
chapter: 6
section: 6.2
---

# 6.2 SFT 实战要点

⬅ [02-Trainer与DeepSpeed实战](02-Trainer%E4%B8%8EDeepSpeed%E5%AE%9E%E6%88%98.md)	|	➡ [04-高效微调三大流派](04-%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E4%B8%89%E5%A4%A7%E6%B5%81%E6%B4%BE.md)

> ⚡ **本节速览**：SFT 训练**和 Pretrain 几乎一样**，只改数据处理。完整代码见 Happy-LLM `code/finetune.py`。

---

## 🎯 SFT vs Pretrain：唯一区别

[第 4 章](../04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-SFT%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md) 讲过：
> SFT 和 Pretrain 都用 CLM 任务，**唯一区别在数据处理 + loss 计算**：
> - **Pretrain**：对整段文本算 loss
> - **SFT**：**只对 assistant 回复算 loss**（user/system 部分 mask 掉）

→ Trainer / DeepSpeed / 超参 **完全可以复用 Pretrain 那一套**。

---

## 🔑 SFT 数据处理 3 个核心要点

### 要点 1：定义特殊 token

```python
# BOS / EOS
im_start = tokenizer("<|im_start|>").input_ids
im_end = tokenizer("<|im_end|>").input_ids
# 换行符
nl_tokens = tokenizer('\n').input_ids
# 角色标识
_system = tokenizer('system').input_ids + nl_tokens
_user = tokenizer('human').input_ids + nl_tokens   # Belle 数据用 "human"
_assistant = tokenizer('assistant').input_ids + nl_tokens
# ⭐ Loss 忽略标记
IGNORE_TOKEN_ID = -100   # PyTorch 标准 ignore_index
```

> ⚠️ **`IGNORE_TOKEN_ID = -100`** 是 PyTorch `CrossEntropyLoss` 的默认 ignore 值，所有标记为 -100 的位置都**不会算 loss**。

### 要点 2：拼接多轮对话 + 构造 labels

核心思路（伪代码）：

```python
input_ids, labels = [], []

# System: <|im_start|>system\nYou are helpful<|im_end|>\n
input_ids += im_start + _system + tokenizer(system_msg) + im_end + nl_tokens
labels    += [IGNORE_TOKEN_ID] * len(...)   # ← system 不算 loss

# 每一轮对话
for sentence in conversation:
    role = sentence["from"]   # "human" 或 "assistant"

    # User: <|im_start|>human\n...<|im_end|>\n
    if role == "human":
        input_ids += im_start + _user + tokenizer(text) + im_end + nl_tokens
        labels    += [IGNORE_TOKEN_ID] * len(...)   # ← user 不算 loss

    # Assistant: <|im_start|>assistant\n...<|im_end|>\n
    elif role == "assistant":
        input_ids += im_start + _assistant + tokenizer(text) + im_end + nl_tokens
        labels    += [IGNORE_TOKEN_ID] * len("<|im_start|>assistant\n") \
                   + tokenizer(text)        # ← assistant 文本算 loss
                   + im_end + nl_tokens
```

#### 📝 关键设计

| 位置 | input_ids | **labels** | 含义 |
|---|---|---|---|
| `<|im_start|>system...` 等格式 token | 正常 ID | `-100` | 不算 loss |
| user 回复部分 | 正常 ID | `-100` | 不算 loss |
| **assistant 回复部分** | **正常 ID** | **正常 ID** | **算 loss** |
| padding 部分 | pad_id | `-100` | 不算 loss |

→ 这就是 [第 5 章 loss_mask 设计](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/08-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%9E%84%E9%80%A0.md#-533-sftdataset%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D) 的 HF 风格表达。
**第 5 章用 loss_mask，第 6 章直接把 labels 设成 -100**，效果一样。

### 要点 3：返回字典格式

```python
return {
    "input_ids": torch.tensor(input_ids),
    "labels": torch.tensor(labels),
    "attention_mask": input_ids.ne(tokenizer.pad_token_id),
}
```

→ Trainer 自动认识这三个键，调用 `model(**batch)` 时自动算 loss。

---

## 🐍 Dataset 类封装

```python
from torch.utils.data import Dataset

class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len):
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)  # 上面的处理逻辑
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attention_mask[i],
        }
```

→ 直接传给 Trainer 即可。

---

## 🛠 SFT 训练启动（和 Pretrain 几乎一样）

```bash
deepspeed finetune.py \
    --model_name_or_path output/pretrain/checkpoint-1000 \
    --train_files data/sft.jsonl \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \                       # ⭐ 比 Pretrain 小一个数量级
    --num_train_epochs 3 \                        # SFT 一般 1~3 epoch
    --warmup_steps 100 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero2.json \
    --output_dir output/sft
```

**和 Pretrain 的核心差异**：
- `--model_name_or_path`：**加载 Pretrain checkpoint**（不是从零开始）
- `--learning_rate`：**调小**（5e-5 vs Pretrain 的 1e-4）
- `--num_train_epochs`：可以 1~3 轮（Pretrain 一般 1 轮）

---

## ⚠️ Chat Template 适配很关键

> ⭐ **不同模型有不同 Chat Template**，混用会导致效果崩盘。

| 模型 | Chat Template 风格 |
|---|---|
| **Qwen** / Yi | `<|im_start|>role\n...<|im_end|>` |
| **LLaMA-2-Chat** | `[INST] ... [/INST]` |
| **LLaMA-3** | `<|begin_of_text|><|start_header_id|>...<|end_header_id|>` |
| **ChatGLM** | `[Round 1]\n问：...\n答：` |
| **Mistral** | `[INST] ... [/INST]` |

**实战建议**：
- **微调已有 Chat 模型**：必须用**它原来的 Chat Template**
- **从 Base 模型 SFT**：可以**自己定义**（推荐用 Qwen 风格，主流）

`tokenizer.apply_chat_template(messages, tokenize=False)` 会自动按当前 tokenizer 的模板拼接。

---

## 📌 6.2 节要点回顾

- SFT = **Pretrain 训练流程 + 改数据处理 + 改 labels**
- **`labels` 中将不算 loss 的位置设为 `-100`**（PyTorch CrossEntropy 默认 ignore）
- SFT 学习率**比 Pretrain 小**（5e-6 ~ 5e-5）
- Chat Template **不能错配**

---

## 🔗 延伸阅读

- 上一节：[02-Trainer与DeepSpeed实战](02-Trainer%E4%B8%8EDeepSpeed%E5%AE%9E%E6%88%98.md)
- 下一节：[04-高效微调三大流派](04-%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E4%B8%89%E5%A4%A7%E6%B5%81%E6%B4%BE.md) ⭐ **理论重点开始**
- 第 5 章对照：[10-SFT训练](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/10-SFT%E8%AE%AD%E7%BB%83.md)
- 理论篇：[06-SFT有监督微调](../04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-SFT%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md)

---

⬅ [02-Trainer与DeepSpeed实战](02-Trainer%E4%B8%8EDeepSpeed%E5%AE%9E%E6%88%98.md)	|	➡ [04-高效微调三大流派](04-%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E4%B8%89%E5%A4%A7%E6%B5%81%E6%B4%BE.md)
