---
tags: [Hello-Agents, 第11章, SFT, LoRA, 训练实战]
chapter: 11
section: 11.3
---

# 11.3 SFT 训练实战

⬅ [02-数据集与奖励函数](02-%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8E%E5%A5%96%E5%8A%B1%E5%87%BD%E6%95%B0.md)	|	➡ [04-GRPO训练实战](04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)

> **强化学习的第一步,也是最重要的基础**。
> 没有 SFT 基础,**直接 RL 经常崩**。

---

## 🎬 故事比喻:学生 vs 老师 vs 教练

| 阶段 | 比喻 |
|---|---|
| **预训练** | 学生**自学海量教材**(语言能力) |
| **SFT** | **老师手把手教**(基本格式 + 推理模式) |
| **GRPO** | **教练让学生练习 + 评分纠错**(策略优化) |

→ **不上"老师课"就直接练习**,效果**惨不忍睹**。

---

## ❓ 11.3.1 为什么必须先 SFT?

### 预训练模型的"3 不会"

```
预训练模型 = "预测下一个词"专家
   ↓ 但不会:
1. 输出指定格式
2. 完成特定任务
3. 结构化推理
```

### 对比实验

#### 预训练模型(无 SFT)

```
问题: Natalia sold clips to 48 of her friends in April,
      then she sold half as many in May. How many in total?

模型输出:
"I can help you with that. Let me think about this problem.
Natalia sold clips to her friends. In April she sold 48 clips.
In May she sold half as many. So we need to find out how many
she sold in May and then add them together. But I'm not sure how
to calculate half of 48. Maybe I should use a calculator?
Or maybe I can just estimate..."
```

**问题**:
- ❌ 冗长无章法
- ❌ 没明确答案
- ❌ 推理混乱
- ❌ **RL 无法用**(无法提取答案、无法评分)

#### SFT 模型

```
模型输出:
Let me solve this step by step.

Step 1: Calculate clips sold in May
Natalia sold half as many clips in May as in April.
Clips in May = 48 / 2 = 24

Step 2: Calculate total clips
Total = April + May = 48 + 24 = 72

Final Answer: 72
```

**优势**:
- ✅ 结构清晰(`Step 1` / `Step 2` / `Final Answer`)
- ✅ 推理正确
- ✅ 答案明确
- ✅ **RL 可以用了**(可提取答案、计算奖励)

→ **SFT 是从预训练 → RL 的桥梁**。

---

## 🔬 11.3.2 SFT 的 4 大作用

| 作用 | 解释 |
|---|---|
| **学输出格式** | "Step 1: ..." / "Final Answer: ..." |
| **学推理模式** | 通过示例学会分解问题 + 逐步推导 |
| **建立基线** | 为 RL 提供合理起点 |
| **减少探索空间** | RL 不用从零开始 |

---

## 🛠 11.3.3 LoRA:参数高效微调(必懂)

### 痛点:**全量微调贵**

| 模型 | 全量微调显存 |
|---|---|
| **Qwen3-0.6B** | 12GB(FP16) / 24GB(FP32) |
| **Qwen 7B** | **几乎不可能在消费级 GPU** |

### LoRA 核心思想

> **微调时的参数变化可以用"低秩矩阵"表示**

**原模型权重**:$W_0 \in \mathbb{R}^{d \times k}$

**微调后权重**:$W = W_0 + \Delta W$

**LoRA 假设**:$\Delta W = BA$,其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$

**前向传播**:

$$h = (W_0 + BA) x$$

**训练**:**冻结 $W_0$**,**只训 $A$ 和 $B$**。

### 参数量对比(惊人)

```
原模型:    d=k=4096 → 16,777,216 个参数
LoRA:      r=8     → 65,536 个参数
           ↓
   参数量减少 256 倍!
```

→ **训练所需显存大幅降低**,**消费级 GPU 也能玩**。

### LoRA 优势

| 优势 | 解释 |
|---|---|
| **省显存** | 可在 8GB 卡跑微调 |
| **训练快** | 参数少 → 反传快 |
| **易部署** | LoRA 权重只有几 MB |
| **防过拟合** | 低秩 → 隐式正则 |

### LoRA 关键超参

| 参数 | 含义 | 典型值 |
|---|---|---|
| **rank (r)** | 秩,控制表达能力 | 4-64,**默认 8** |
| **alpha (α)** | 缩放因子,$\Delta W = \frac{\alpha}{r} BA$ | **通常 = r 或 2r** |
| **target_modules** | 应用层 | `["q_proj", "k_proj", "v_proj", "o_proj"]` |

> 📚 详见 [Happy-LLM LoRA 详解](../../Happy-LLM/06-%E7%AC%AC6%E7%AB%A0-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%AE%9E%E8%B7%B5/05-LoRA%E5%8E%9F%E7%90%86%E6%B7%B1%E5%85%A5.md)

---

## 🚀 11.3.4 SFT 训练实战:基础示例

```python
from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()

# 基础 SFT 训练
result = rl_tool.run({
    "action": "train",
    "algorithm": "sft",

    # 模型
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./models/sft_model",

    # 数据
    "max_samples": 100,        # 快速测试用 100

    # 训练
    "num_epochs": 3,           # 训 3 轮
    "batch_size": 4,
    "learning_rate": 5e-5,

    # LoRA
    "use_lora": True,
    "lora_rank": 8,
    "lora_alpha": 16,
})

print(f"模型保存: {result['model_path']}")
print(f"最终损失: {result['final_loss']:.4f}")
```

→ **跑起来后看 loss 是否在下降**,**这是 SFT 成功的关键标志**。

---

## 🔧 11.3.5 关键参数详解

### 数据参数

| 参数 | 含义 | 建议 |
|---|---|---|
| `max_samples` | 训练样本数 | 100-1000(快速)/ None(完整 7473) |
| `split` | 数据集划分 | "train" / "train[:1000]" |

### 训练参数

| 参数 | 含义 | 建议 |
|---|---|---|
| `num_epochs` | 训练轮数 | **3 轮起步**,看 loss 调整 |
| `batch_size` | 批次大小 | 4GB → 1-2 / 8GB → 4-8 / 16GB → 8-16 |
| `learning_rate` | 学习率 | SFT **5e-5**,LoRA 可稍大 1e-4 |

### LoRA 参数

| 参数 | 含义 | 建议 |
|---|---|---|
| `use_lora` | 是否用 LoRA | **始终开启**(除非显存超大) |
| `lora_rank` | 秩 | 小任务 4-8 / 复杂 16-32 / 大规模 64 |
| `lora_alpha` | 缩放 | **= rank × 2** |

### 优化器参数

| 参数 | 含义 | 建议 |
|---|---|---|
| `optimizer` | 优化器 | "**adamw**"(默认) |
| `weight_decay` | 权重衰减 | 0.01(防过拟合) |
| `warmup_ratio` | 学习率预热 | 0.1(前 10% 步数) |

---

## 🎯 11.3.6 完整训练示例(最佳实践)

```python
# 完整 SFT 训练:全数据 + 最佳实践
result = rl_tool.run({
    "action": "train",
    "algorithm": "sft",

    # 模型
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./models/sft_full",

    # 数据 - 全部 7473 样本
    "max_samples": None,

    # 训练
    "num_epochs": 3,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,

    # LoRA - 用更大的 rank
    "use_lora": True,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

    # 监控
    "save_steps": 500,         # 每 500 步存
    "logging_steps": 100,      # 每 100 步打印
    "eval_steps": 500,         # 每 500 步评估
})
```

→ 8GB GPU 上**预计 30-60 分钟**。

---

## 📊 11.3.7 训练监控(3 大指标)

### 指标 1:**Loss(损失)**

```
✅ 正常: 逐渐下降
❌ 不下降: 学习率太小 / 数据有问题
❌ 上升: 学习率太大 / 过拟合
```

### 指标 2:**Gradient Norm(梯度范数)**

```
✅ 正常: 0.1 ~ 10
❌ 过大 (> 100): 梯度爆炸 → 降学习率
❌ 过小 (< 0.01): 梯度消失 → 检查配置
```

### 指标 3:**Learning Rate(学习率)**

```
✅ 正常: 按 warmup 策略变化
   - 前 10% 步数:线性增加
   - 之后:线性衰减到 0
```

---

## 🚨 11.3.8 常见问题与解决

| 问题 | 解决 |
|---|---|
| **显存不足 (OOM)** | 减小 `batch_size` / `max_length`,梯度累积 |
| **训练太慢** | 增大 `batch_size`,减少 logging 频率,混合精度 |
| **Loss 不下降** | 增大学习率,检查数据格式,加 epoch |
| **过拟合** | 增大 `weight_decay`,减 epoch,加数据 |

---

## 📈 11.3.9 模型评估

```python
# 评估 SFT 后的模型
eval_result = rl_tool.run({
    "action": "evaluate",
    "model_path": "./models/sft_full",
    "max_samples": 100,
    "use_lora": True,
})

import json
data = json.loads(eval_result)
print(f"准确率: {data['accuracy']}")
print(f"平均奖励: {data['average_reward']}")
```

### 对比实验:**预训练 vs SFT**

```python
# 预训练模型(未经 SFT)
base_result = rl_tool.run({
    "action": "evaluate",
    "model_path": "Qwen/Qwen3-0.6B",   # 原始模型
    "max_samples": 100,
    "use_lora": False,
})

# SFT 模型
sft_result = rl_tool.run({
    "action": "evaluate",
    "model_path": "./models/sft_full",
    "max_samples": 100,
    "use_lora": True,
})

# 典型结果:
# 预训练: 准确率 ~5%   (基本不会做)
# SFT:    准确率 ~40-50% (大幅提升)
```

→ **SFT 后准确率 40-50%** 是 Qwen3-0.6B 的正常水平。
→ **GRPO 后可提升至 60-70%**(下节展开)。

---

## ⚠️ 小白避坑

1. **不要全量微调小模型**
   - LoRA 通常**够用 + 省 90% 显存**
2. **`batch_size` 调小不会大问题**
   - 但**别小于 1**(报错)
3. **加 `warmup_ratio=0.1`**
   - 不加 → 训练初期不稳定
4. **`save_steps` 别太小**
   - 否则**硬盘塞满**
5. **训练前先评估基线**
   - 不然不知道**有没有提升**

---

## 📌 11.3 节要点

| 知识点 | 一句话 |
|---|---|
| **SFT 必要性** | 教格式 + 推理模式,**RL 的基线** |
| **LoRA** | 低秩分解,**省 256 倍参数** |
| **关键超参** | rank=8/16, alpha=2r, lr=5e-5 |
| **3 大监控** | Loss / Grad Norm / Learning Rate |
| **预期效果** | Qwen3-0.6B SFT 后 GSM8K **40-50%** |

---

## 🔗 延伸阅读

- 上一节:[02-数据集与奖励函数](02-%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8E%E5%A5%96%E5%8A%B1%E5%87%BD%E6%95%B0.md)
- 下一节:[04-GRPO训练实战](04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md) —— **重头戏**
- LoRA 详解:[05-LoRA原理深入](../../Happy-LLM/06-%E7%AC%AC6%E7%AB%A0-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%AE%9E%E8%B7%B5/05-LoRA%E5%8E%9F%E7%90%86%E6%B7%B1%E5%85%A5.md)
- SFT 详解:[06-SFT有监督微调](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-SFT%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md)

---

⬅ [02-数据集与奖励函数](02-%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8E%E5%A5%96%E5%8A%B1%E5%87%BD%E6%95%B0.md)	|	➡ [04-GRPO训练实战](04-GRPO%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)
