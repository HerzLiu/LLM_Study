---
tags: [Happy-LLM, 第6章, 实战速览, LoRA, peft]
chapter: 6
section: 6.3.4-6.3.5
---

# 6.3.4-6.3.5 LoRA 代码要点 + peft 使用

⬅ [05-LoRA原理深入](05-LoRA%E5%8E%9F%E7%90%86%E6%B7%B1%E5%85%A5.md)	|	➡ [07-章节小结](07-%E7%AB%A0%E8%8A%82%E5%B0%8F%E7%BB%93.md)

> ⚡ **本节速览**：理解 LoRA 内部实现机制，然后用 peft 库**一行接入**。

---

## 🔧 LoRA 内部实现 3 步走

要把 LoRA 加到一个模型上，PEFT 库内部做了 3 步：

```
Step 1: 找到要替换的层（target_modules）
Step 2: 把原层替换成 LoRA 层（原参数 + 旁路 A、B）
Step 3: 冻结原参数，只训 A 和 B
```

---

## 🐍 Step 1：找到目标层

通过**正则匹配**模块名：

```python
import re

# 用户指定要 LoRA 的层（一般是注意力的 q_proj, v_proj）
target_modules = ["q_proj", "v_proj"]

# 遍历模型，找到匹配的层
for name, module in model.named_modules():
    if any(t in name for t in target_modules):
        # 这一层要换成 LoRA 层
        ...
```

→ 不同模型的层命名不同，要查源码确认。常见名字：
- LLaMA / Qwen：`q_proj`, `k_proj`, `v_proj`, `o_proj`
- BERT：`query`, `key`, `value`, `dense`
- ChatGLM：`query_key_value`

---

## 🐍 Step 2：LoRA 层的实现（核心）

PEFT 库定义了一个 `Linear` 类，**同时继承 `nn.Linear` 和 `LoraLayer`**：

```python
class Linear(nn.Linear, LoraLayer):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16,
                 lora_dropout=0.0, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout)

        if r > 0:
            # ⭐ 低秩分解的两个矩阵
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)

            # ⭐ 缩放系数（控制 LoRA 影响大小）
            self.scaling = self.lora_alpha / self.r

            # ⭐ 冻结原参数
            self.weight.requires_grad = False

        # 初始化 A 和 B
        self.reset_parameters()
```

### 初始化策略

```python
def reset_parameters(self):
    nn.Linear.reset_parameters(self)
    if hasattr(self, 'lora_A'):
        # ⭐ A 用 Kaiming 均匀分布
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # ⭐ B 用全零（保证训练初始 BA = 0）
        nn.init.zeros_(self.lora_B.weight)
```

> ⭐ 关键设计：**B 必须零初始化**。这样训练开始时 $\Delta W = BA = 0$，模型行为完全等价于未加 LoRA。

### 前向传播

```python
def forward(self, x):
    # 原层结果
    result = F.linear(x, self.weight, bias=self.bias)

    # ⭐ LoRA 旁路：x → A → B
    if self.r > 0:
        result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

    return result
```

**核心一行**：
```python
result += self.lora_B(self.lora_A(x)) * self.scaling
```

→ 对应公式 $h = W_0 x + BA x \cdot \alpha / r$

---

## 🎯 关键超参解读

| 参数 | 含义 | 典型值 |
|---|---|---|
| **`r`** | LoRA 秩（低维度数） | **4 / 8 / 16** |
| **`lora_alpha`** | 缩放因子分子 | **通常 = 2r**（如 16, 32） |
| `lora_dropout` | LoRA 的 dropout | 0.0 ~ 0.1 |
| `target_modules` | 加 LoRA 的层名列表 | `["q_proj", "v_proj"]` |
| `bias` | bias 是否训练 | 一般 "none" |

### `lora_alpha / r` 的含义

```python
self.scaling = self.lora_alpha / self.r
```

- 这是个**缩放系数**，控制 LoRA 旁路对模型输出的影响大小
- **保持 `alpha/r ≈ 2`** 是经验最佳值
- 改 `r` 时记得同步改 `alpha`

---

## 🐍 用 peft 库（一行接入）

实际生产中**不用自己写 LoRA 层**，直接用 PEFT：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# 1. 加载基座模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

# 2. 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,    # 任务类型
    r=8,                              # 秩
    lora_alpha=32,                    # 缩放
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 加 LoRA 的层
    bias="none",
)

# 3. 一行套上 LoRA
model = get_peft_model(model, peft_config)

# 4. 查看可训练参数
model.print_trainable_parameters()
# 输出：trainable params: 4,194,304 || all params: 1,543,714,304 || trainable%: 0.27%

# 5. 像普通 model 一样训练
from transformers import Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, ...)
trainer.train()
```

→ **就这 5 步**，把全参数微调改成 LoRA 微调，**显存降 80%**。

---

## 🎯 不同模型的 LoRA 配置示例

| 模型 | 推荐 `target_modules` |
|---|---|
| **LLaMA / Qwen** | `["q_proj", "k_proj", "v_proj", "o_proj"]`（再加更激进的 `["gate_proj", "up_proj", "down_proj"]`） |
| **ChatGLM** | 不用指定，peft 自动找 |
| **Baichuan** | `["W_pack"]`（Q/K/V 合并） |
| **BERT** | `["query", "value"]` |

→ 不确定就**先打印模型结构**：
```python
print(model)
```
看哪些 `nn.Linear` 是注意力层。

---

## 💡 LoRA 微调的几个实用 Tip

### Tip 1：保存只保 LoRA 权重

```python
model.save_pretrained("./lora_weights")
```

→ 只保存 LoRA 的 A、B 矩阵，文件**只有几 MB**（而不是几 GB）。

### Tip 2：加载 LoRA 权重

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
model = PeftModel.from_pretrained(base_model, "./lora_weights")
```

### Tip 3：合并 LoRA 到原模型（部署用）

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

→ 合并后**和原模型一样大**，但已经包含了 LoRA 的微调效果。**推理零延迟**。

### Tip 4：QLoRA（极省显存）

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ⭐ 加载时直接 4bit 量化
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    trust_remote_code=True,
)

# 然后照常加 LoRA
model = get_peft_model(model, peft_config)
```

→ **单卡 24G 微调 7B 完全无压力**。

---

## ⚠️ 小白避坑

1. **`bias="none"` 是默认**
   - 一般不训 bias（参数少且效果一般）
2. **LoRA 不能学新知识**
   - 想注入知识 → 全参数 CPT
   - 想学风格/格式 → LoRA
3. **target_modules 别全选**
   - 全选反而效果可能下降
   - 经验：先 `["q_proj", "v_proj"]`，效果不好再加
4. **QLoRA 训完合并后会变 bf16**
   - 不能保留 4bit 量化效果
   - 部署还要重新量化
5. **alpha 跟着 r 改**
   - r 翻倍，alpha 也翻倍（保持 alpha/r 比例不变）

---

## 📌 6.3 节要点回顾

- LoRA 内部实现 = **冻结 W + 加 A、B 旁路 + 只训 A、B**
- **`peft.get_peft_model(model, LoraConfig)` 一行接入**
- 常用配置：`r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"]`
- **QLoRA = 4bit 量化 + LoRA**，单卡可训 7B+
- 保存的 LoRA 权重**只有几 MB**，方便分发

---

## 🔗 延伸阅读

- 上一节：[05-LoRA原理深入](05-LoRA%E5%8E%9F%E7%90%86%E6%B7%B1%E5%85%A5.md)
- 下一节：[07-章节小结](07-%E7%AB%A0%E8%8A%82%E5%B0%8F%E7%BB%93.md)
- 实战：HuggingFace PEFT 官方文档：https://huggingface.co/docs/peft

---

⬅ [05-LoRA原理深入](05-LoRA%E5%8E%9F%E7%90%86%E6%B7%B1%E5%85%A5.md)	|	➡ [07-章节小结](07-%E7%AB%A0%E8%8A%82%E5%B0%8F%E7%BB%93.md)
